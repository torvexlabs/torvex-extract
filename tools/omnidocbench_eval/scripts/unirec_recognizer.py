"""UniRec-0.1B ONNX recognizer adapter for the OmniDocBench eval harness.

Faithful port of OpenOCR's standalone `tools/infer_unirec_onnx.py` (UniRecONNX),
trimmed to the encoder/decoder ONNX + id_to_token mapping path (no torch/transformers,
no auto-download, no PDF/cv2). Exposes a `recognize_crops(crops) -> [{"latex": str}]`
interface so it can drop into gen_raw_predictions.py in place of the UniMERNet extractor.

Assets (models/unirec-0.1b-onnx/):
  unirec_encoder.onnx, unirec_decoder.onnx, unirec_tokenizer_mapping.json
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = ROOT / "models" / "unirec-0.1b-onnx"


class _ImageProcessor:
    """Standalone image processor (mirrors SimpleImageProcessor)."""

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
    """Drop-in recognizer: recognize_crops(crops) -> [{"latex": str}]."""

    def __init__(self, model_dir=DEFAULT_MODEL_DIR, device="gpu", max_length=2048):
        model_dir = Path(model_dir)
        enc = str(model_dir / "unirec_encoder.onnx")
        dec = str(model_dir / "unirec_decoder.onnx")
        mapping = str(model_dir / "unirec_tokenizer_mapping.json")
        for p in (enc, dec, mapping):
            if not os.path.exists(p):
                raise FileNotFoundError(f"UniRec asset missing: {p}")

        # Use torvex's provider helper so GPU gets the Windows CUDA-DLL preload
        # (cublasLt/cudnn) that the rest of the pipeline relies on.
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
        self.decoder_session = ort.InferenceSession(dec, so, providers=providers)
        self.encoder_session = ort.InferenceSession(enc, so, providers=providers)
        self.processor = _ImageProcessor()
        self.tokenizer = _Tokenizer(mapping)
        self.max_length = max_length

        # Infer decoder layer/head/dim from input shapes (KV cache init).
        self.num_decoder_layers = 0
        self.num_heads = None
        self.head_dim = None
        for inp in self.decoder_session.get_inputs():
            if "past_key" in inp.name:
                idx = int(inp.name.split("_")[-1])
                self.num_decoder_layers = max(self.num_decoder_layers, idx + 1)
                if len(inp.shape) == 4:
                    if self.num_heads is None and isinstance(inp.shape[1], int):
                        self.num_heads = inp.shape[1]
                    if self.head_dim is None and isinstance(inp.shape[3], int):
                        self.head_dim = inp.shape[3]
        print(f"[UniRec] providers={providers} layers={self.num_decoder_layers} "
              f"heads={self.num_heads} head_dim={self.head_dim} vocab={self.tokenizer.vocab_size}")

    # --- harness interface -------------------------------------------------
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
                print(f"[UniRec] recognized {i + 1}/{n}", flush=True)
        return out

    # --- core inference (mirrors UniRecONNX._infer_single_image) -----------
    def _encode(self, image: Image.Image):
        pixel_values = self.processor(image)
        enc = self.encoder_session.run(None, {"pixel_values": pixel_values})
        return enc[0], enc[1], enc[2]  # hidden, cross_k, cross_v

    def _decode_step(self, input_id, past_length, cross_k, cross_v, past_kv, padding_idx):
        decoder_inputs = {
            "input_ids": np.array([[input_id]], dtype=np.int64),
            "position_ids": np.array([[padding_idx + 1 + past_length]], dtype=np.int64),
            "cross_k": cross_k.astype(np.float32),
            "cross_v": cross_v.astype(np.float32),
        }
        for i, (pk, pv) in enumerate(past_kv):
            decoder_inputs[f"past_key_{i}"] = pk.astype(np.float32)
            decoder_inputs[f"past_value_{i}"] = pv.astype(np.float32)
        outs = self.decoder_session.run(None, decoder_inputs)
        logits = outs[0]
        present = [(outs[1 + i * 2], outs[1 + i * 2 + 1]) for i in range(self.num_decoder_layers)]
        return logits, present

    def _is_runaway_loop(self, generated, reps=10, max_period=24):
        """True if the tail is one short token block repeated back-to-back >= reps times.

        A decode loop repeats an EXACT token block (e.g. "&{}^{-}\\" x400); a legitimate matrix
        repeats structure but cell contents differ, so the exact block does not recur. Mirrors the
        production guard in src/torvex_extract/unirec_recognizer.py.
        """
        g = generated
        n = len(g)
        for p in range(1, max_period + 1):
            if n < p * reps:
                break
            block = g[-p:]
            if all(g[-p * r:n - p * (r - 1)] == block for r in range(2, reps + 1)):
                return True
        return False

    def _infer(self, image: Image.Image) -> str:
        bos, eos, pad = (self.tokenizer.bos_token_id,
                         self.tokenizer.eos_token_id,
                         self.tokenizer.pad_token_id)
        enc_hidden, cross_k, cross_v = self._encode(image)
        batch = enc_hidden.shape[0]
        past_kv = [(np.zeros((batch, self.num_heads, 0, self.head_dim), dtype=np.float32),
                    np.zeros((batch, self.num_heads, 0, self.head_dim), dtype=np.float32))
                   for _ in range(self.num_decoder_layers)]
        generated = [bos]
        for step in range(self.max_length - 1):
            logits, past_kv = self._decode_step(
                generated[-1], step, cross_k, cross_v, past_kv, padding_idx=pad)
            nxt = int(np.argmax(logits[0, -1, :]))
            generated.append(nxt)
            if nxt == eos:
                break
            # runaway-loop guard: short token block repeating back-to-back many times = decode
            # loop, not real math -> stop before max_length (kills the 15-40s tail + garbage).
            if step >= 10 and self._is_runaway_loop(generated):
                break
        return clean_special_tokens(self.tokenizer.decode(generated)).strip()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="gpu")
    args = ap.parse_args()
    rec = UniRecRecognizer(device=args.device)
    print(rec.recognize_crop(Image.open(args.image))["latex"])
