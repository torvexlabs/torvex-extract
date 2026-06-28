"""UniRec-0.1B ONNX formula/text recognizer.

Added 2026-06-22 — replaces the old UniMERNet-tiny recognizer. Why UniRec-0.1B: it reads
Chinese + English and emits per-line equation granularity natively (the thing that lifted
formula CDM 0.88 -> 0.96 once we stopped collapsing it). Faithful port of OpenOCR's
standalone UniRec ONNX inference (encoder + decoder + id_to_token mapping; pure
numpy / onnxruntime / PIL, no torch/transformers/cv2).
Exposes recognize_crops(crops) -> [{"latex": str}].

Assets (models/unirec-0.1b-onnx/): unirec_encoder.onnx, unirec_decoder.onnx,
unirec_tokenizer_mapping.json. Preprocess = resize max 960x1408 down to /64 multiples,
BICUBIC, normalize mean/std 0.5. The clean_special_tokens rules below are defensive
(no <|sn|> token actually exists in this vocab).

VELOX (added 2026-06-28): when unirec_decoder_velox.onnx is present, decode runs n-gram
self-speculative decoding (draft from the model's own output, verify K tokens in one
multi-token pass, roll back the KV on a miss). It is LOSSLESS — byte-identical to greedy —
and ~1.7x faster on formula-dense content (formulas legitimately repeat structure, so the
n-gram drafts hit). Toggle with TORVEX_UNIREC_VELOX (default on); falls back to single-token
greedy when the velox decoder is absent, disabled, or no-repeat-ngram is in use.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

logger = logging.getLogger(__name__)

# src/torvex_extract/unirec_recognizer.py -> repo root is parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
# TORVEX_UNIREC_MODEL_DIR lets you swap weight variants (e.g. the fp16 / int8 build) without code change.
DEFAULT_MODEL_DIR = Path(os.getenv("TORVEX_UNIREC_MODEL_DIR", str(_REPO_ROOT / "models" / "unirec-0.1b-onnx")))


class _ImageProcessor:
    def __init__(self, max_side=(960, 1408), divided_factor=(64, 64),
                 image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5)):
        self.max_side = max_side
        self.divided_factor = divided_factor
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)

    def _target_size(self, w, h):
        max_w, max_h = self.max_side
        ar = w / h
        if w > max_w or h > max_h:
            if (max_w / max_h) >= ar:
                new_h, new_w = max_h, int(max_h * ar)
            else:
                new_w, new_h = max_w, int(max_w / ar)
        else:
            new_w, new_h = w, h
        div_w, div_h = self.divided_factor
        return (max(int(new_w // div_w * div_w), 64),
                max(int(new_h // div_h * div_h), 64))

    def __call__(self, image: Image.Image) -> np.ndarray:
        w, h = image.size
        image = image.resize(self._target_size(w, h), resample=Image.BICUBIC)
        arr = np.array(image, dtype=np.float32)[:, :, :3] / 255.0
        arr = (arr - self.image_mean) / self.image_std
        arr = arr.transpose(2, 0, 1)
        return np.expand_dims(arr, axis=0).astype(np.float32)


class _Tokenizer:
    def __init__(self, mapping_file: str):
        with open(mapping_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        self.vocab_size = data["vocab_size"]
        sp = data["special_tokens"]
        self.bos_token_id = sp["bos_token_id"]
        self.eos_token_id = sp["eos_token_id"]
        self.pad_token_id = sp["pad_token_id"]

    def decode(self, token_ids) -> str:
        return "".join(self.id_to_token.get(t, f"<unk_{t}>") for t in token_ids)


_CLEAN_RULES = [
    (r"-<\|sn\|>", ""), (r" <\|sn\|>", " "), (r"<\|sn\|>", " "),
    (r"<\|unk\|>", ""), (r"<s>", ""), (r"</s>", ""), (r"￿", ""),
    (r"_{4,}", "___"), (r"\.{4,}", "..."),
]


def clean_special_tokens(text: str) -> str:
    text = text.replace("Ġ", " ").replace("Ċ", "\n")
    text = text.replace("<|bos|>", "").replace("<|eos|>", "").replace("<|pad|>", "")
    for pat, repl in _CLEAN_RULES:
        text = re.sub(pat, repl, text)
    return text


class UniRecRecognizer:
    """recognize_crops(crops) -> [{"latex": str}]."""

    def __init__(self, model_dir=DEFAULT_MODEL_DIR, device="gpu", max_length=2048):
        model_dir = Path(model_dir)
        enc = str(model_dir / "unirec_encoder.onnx")
        base_dec = str(model_dir / "unirec_decoder.onnx")
        velox_dec = str(model_dir / "unirec_decoder_velox.onnx")
        mapping = str(model_dir / "unirec_tokenizer_mapping.json")
        for p in (enc, base_dec, mapping):
            if not os.path.exists(p):
                raise FileNotFoundError(f"UniRec asset missing: {p}")

        self.max_length = max_length
        # no-repeat-ngram: ban any n-gram of token ids from recurring in the greedy decode.
        # DEFAULT OFF (0): the 106-page hard ablation proved it LOWERS CDM (0.9186 -> worse) because
        # 858/942 GT formulas legitimately repeat a 4-gram (matrix rows, index runs) and banning it
        # forces a wrong token. The targeted runaway guard below replaces it as the loop fix. Opt-in
        # only via TORVEX_UNIREC_NO_REPEAT_NGRAM for experiments.
        try:
            self._no_repeat_ngram = max(0, int(os.getenv("TORVEX_UNIREC_NO_REPEAT_NGRAM", "0")))
        except Exception:
            self._no_repeat_ngram = 0
        # runaway-loop guard: stop when the tail is one short token block repeated back-to-back
        # >= N times. This is what an actual decode loop looks like (e.g. "&{}^{-}\\" x400, gt=78
        # tokens -> pred=2954) and it is structurally distinct from a legitimate repeat: a real
        # matrix repeats STRUCTURE but the cell contents differ (a_{11} & a_{12} -> the exact token
        # block does NOT recur), so this never fires on the 858 legitimately-repeating formulas the
        # way no-repeat-ngram does. 21/942 hard formulas looped, burning ~40% of all decode tokens.
        try:
            self._loop_min_repeats = max(3, int(os.getenv("TORVEX_UNIREC_LOOP_MIN_REPEATS", "10")))
        except Exception:
            self._loop_min_repeats = 10
        self._loop_max_period = 24

        # VELOX speculative decode: use the multi-token verify decoder when present + enabled.
        # Lossless (byte-identical to greedy) so it is safe as the default. Disabled automatically
        # when no-repeat-ngram is on, because banning tokens changes the argmax the draft must match.
        self.use_velox = (os.getenv("TORVEX_UNIREC_VELOX", "1") != "0"
                          and os.path.exists(velox_dec)
                          and self._no_repeat_ngram == 0)
        try:
            self._velox_k = max(1, int(os.getenv("TORVEX_UNIREC_VELOX_K", "16")))
        except Exception:
            self._velox_k = 16
        self._velox_ngram = 3  # match the last (ngram-1)=2 tokens against earlier output

        # torvex provider helper does the Windows CUDA-DLL preload (cublasLt/cudnn).
        try:
            from torvex_extract.onnx_runtime import select_onnx_providers
            providers = select_onnx_providers(device)
        except Exception:
            avail = ort.get_available_providers()
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if device in {"gpu", "cuda"} and "CUDAExecutionProvider" in avail
                         else ["CPUExecutionProvider"])

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        dec = velox_dec if self.use_velox else base_dec
        self.decoder_session = ort.InferenceSession(dec, so, providers=providers)
        self.encoder_session = ort.InferenceSession(enc, so, providers=providers)
        self.processor = _ImageProcessor()
        self.tokenizer = _Tokenizer(mapping)

        # CUDA arena shrinkage: free the arena after the (variable-size) encoder run so VRAM
        # doesn't ratchet up to the largest crop ever seen. The model is stateless across crops,
        # so per-crop flush is safe; capped VRAM ~28x lower with negligible latency cost.
        self._run_options = None
        if any("CUDA" in p for p in providers):
            self._run_options = ort.RunOptions()
            self._run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")

        # float dtype the models expect (float32 or, for a quantized build, float16). The base
        # decoder exposes cross_k; the velox decoder exposes past_self_k_0 + encoder_hidden_states.
        enc_in = {i.name: i.type for i in self.encoder_session.get_inputs()}
        dec_in = {i.name: i.type for i in self.decoder_session.get_inputs()}
        self._pix_dtype = np.float16 if "float16" in enc_in.get("pixel_values", "") else np.float32
        _probe = dec_in.get("cross_k") or dec_in.get("past_self_k_0") or ""
        self._fdtype = np.float16 if "float16" in _probe else np.float32

        # decoder geometry from whichever decoder is loaded (base: past_key_N, velox: past_self_k_N)
        self.num_decoder_layers = 0
        self.num_heads = None
        self.head_dim = None
        for inp in self.decoder_session.get_inputs():
            nm = inp.name
            suffix = None
            if nm.startswith("past_key_"):
                suffix = nm[len("past_key_"):]
            elif nm.startswith("past_self_k_"):
                suffix = nm[len("past_self_k_"):]
            if suffix is not None and suffix.isdigit():
                self.num_decoder_layers = max(self.num_decoder_layers, int(suffix) + 1)
                if len(inp.shape) == 4:
                    if self.num_heads is None and isinstance(inp.shape[1], int):
                        self.num_heads = inp.shape[1]
                    if self.head_dim is None and isinstance(inp.shape[3], int):
                        self.head_dim = inp.shape[3]
        logger.info("UniRec loaded: providers=%s layers=%s heads=%s head_dim=%s vocab=%s velox=%s",
                    providers, self.num_decoder_layers, self.num_heads, self.head_dim,
                    self.tokenizer.vocab_size, self.use_velox)

    def preflight(self):
        return None

    def recognize_crop(self, crop: Image.Image) -> dict:
        return {"latex": self._infer(crop.convert("RGB"))}

    def recognize_crops(self, crops):
        out = []
        n = len(crops)
        for i, c in enumerate(crops):
            out.append(self.recognize_crop(c))
            if (i + 1) % 25 == 0 or (i + 1) == n:
                logger.debug("UniRec recognized %d/%d", i + 1, n)
        return out

    def _encode(self, image: Image.Image):
        pixel_values = self.processor(image).astype(self._pix_dtype)
        enc = self.encoder_session.run(None, {"pixel_values": pixel_values}, self._run_options)
        return enc[0], enc[1], enc[2]  # hidden, cross_k, cross_v

    def _is_runaway_loop(self, generated):
        """True if the tail is one short token block repeated back-to-back >= min_repeats times.

        Discriminates a decode loop (exact block recurs identically) from a legitimate repeated
        structure (matrix/array cells differ in subscripts, so the exact token block does NOT
        recur). Cheap: at most loop_max_period * min_repeats slice compares, and short-circuits.
        """
        g = generated
        n = len(g)
        reps = self._loop_min_repeats
        for p in range(1, self._loop_max_period + 1):
            if n < p * reps:
                break  # not enough tokens for `reps` blocks of any period >= p
            block = g[-p:]
            if all(g[-p * r:n - p * (r - 1)] == block for r in range(2, reps + 1)):
                return True
        return False

    def _hit_stop_guard(self, generated):
        """Shared early-stop checks (degeneracy + runaway loop) used by both decode paths."""
        if len(generated) >= 48 and len(set(generated[-48:])) <= 4:
            return True
        return self._is_runaway_loop(generated)

    def _infer(self, image: Image.Image) -> str:
        if self.use_velox:
            generated = self._decode_velox(image)
        else:
            generated = self._decode_greedy(image)
        return clean_special_tokens(self.tokenizer.decode(generated)).strip()

    # ---- base greedy decode (single token per step) --------------------------------------------

    def _decode_step(self, input_id, past_length, cross_k, cross_v, past_kv, padding_idx):
        decoder_inputs = {
            "input_ids": np.array([[input_id]], dtype=np.int64),
            "position_ids": np.array([[padding_idx + 1 + past_length]], dtype=np.int64),
            "cross_k": cross_k.astype(self._fdtype),
            "cross_v": cross_v.astype(self._fdtype),
        }
        for i, (pk, pv) in enumerate(past_kv):
            decoder_inputs[f"past_key_{i}"] = pk.astype(self._fdtype)
            decoder_inputs[f"past_value_{i}"] = pv.astype(self._fdtype)
        outs = self.decoder_session.run(None, decoder_inputs)
        logits = outs[0]
        present = [(outs[1 + i * 2], outs[1 + i * 2 + 1]) for i in range(self.num_decoder_layers)]
        return logits, present

    def _banned_ngram_tokens(self, generated, n):
        """Tokens that would complete an n-gram already seen in `generated` (no-repeat-ngram)."""
        if n <= 1 or len(generated) < n:
            return ()
        prefix = tuple(generated[-(n - 1):])
        banned = set()
        for i in range(len(generated) - n + 1):
            if tuple(generated[i:i + n - 1]) == prefix:
                banned.add(generated[i + n - 1])
        return banned

    def _decode_greedy(self, image: Image.Image):
        bos, eos, pad = (self.tokenizer.bos_token_id,
                         self.tokenizer.eos_token_id,
                         self.tokenizer.pad_token_id)
        enc_hidden, cross_k, cross_v = self._encode(image)
        batch = enc_hidden.shape[0]
        past_kv = [(np.zeros((batch, self.num_heads, 0, self.head_dim), dtype=self._fdtype),
                    np.zeros((batch, self.num_heads, 0, self.head_dim), dtype=self._fdtype))
                   for _ in range(self.num_decoder_layers)]
        generated = [bos]
        for step in range(self.max_length - 1):
            logits, past_kv = self._decode_step(
                generated[-1], step, cross_k, cross_v, past_kv, padding_idx=pad)
            scores = logits[0, -1, :]
            banned = self._banned_ngram_tokens(generated, self._no_repeat_ngram)
            if banned:
                scores = scores.copy()
                scores[list(banned)] = -np.inf
            nxt = int(np.argmax(scores))
            generated.append(nxt)
            if nxt == eos:
                break
            if self._hit_stop_guard(generated):
                break
        return generated

    # ---- VELOX n-gram self-speculative decode (lossless, ~1.7x on formulas) ---------------------

    def _velox_step(self, input_ids, enc_hidden, past_self):
        feeds = {"input_ids": np.asarray(input_ids, dtype=np.int64).reshape(1, -1),
                 "encoder_hidden_states": enc_hidden}
        for i in range(self.num_decoder_layers):
            feeds[f"past_self_k_{i}"] = past_self[2 * i]
            feeds[f"past_self_v_{i}"] = past_self[2 * i + 1]
        outs = self.decoder_session.run(None, feeds, self._run_options)
        return outs[0], list(outs[1:])  # logits [1,K,vocab], present self-KV [k0,v0,k1,v1,...]

    def _draft_ngram(self, generated):
        """Propose the next tokens by finding the most recent earlier occurrence of the current
        suffix and copying what followed it (Prompt-Lookup-style, but over our own output)."""
        n, K = self._velox_ngram, self._velox_k
        if len(generated) < n:
            return []
        suffix = tuple(generated[-(n - 1):])
        for j in range(len(generated) - (n - 1) - 1, -1, -1):
            if tuple(generated[j:j + n - 1]) == suffix:
                return generated[j + n - 1: j + n - 1 + K]
        return []

    def _decode_velox(self, image: Image.Image):
        bos, eos = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        nh, hd, nl = self.num_heads, self.head_dim, self.num_decoder_layers
        enc_hidden = self._encode(image)[0].astype(self._fdtype)
        past = [np.zeros((1, nh, 0, hd), dtype=self._fdtype) for _ in range(2 * nl)]

        # seed one token so there is something to draft against
        logits, past = self._velox_step([bos], enc_hidden, past)
        generated = [bos, int(np.argmax(logits[0, -1]))]
        if generated[-1] == eos:
            return generated

        while len(generated) < self.max_length and generated[-1] != eos:
            draft = self._draft_ngram(generated)
            block = [generated[-1]] + draft               # last committed token + drafts
            logits, present = self._velox_step(block, enc_hidden, past)
            preds = np.argmax(logits[0], axis=-1)         # model's greedy pick at each fed position
            # pos 0 predicts the true next token after generated[-1]; always accept it. Then walk
            # the drafts: keep accepting while each draft equals what greedy would have produced.
            accepted = [int(preds[0])]
            for i, d in enumerate(draft):
                if d == accepted[i]:
                    accepted.append(int(preds[i + 1]))
                else:
                    break
            generated.extend(accepted)
            # roll the self-KV cache back to the committed length (drop the rejected-draft tail)
            keep = len(generated) - 1
            past = [p[:, :, :keep, :] for p in present]
            if eos in accepted:
                generated = generated[:len(generated) - len(accepted) + accepted.index(eos) + 1]
                break
            if self._hit_stop_guard(generated):
                break
        return generated[:self.max_length]
