"""
pure_onnx_unimernet.py
======================
Drop-in formula recognizer using only onnxruntime + tokenizers.
No PyTorch. No unimernet package. No transformers.

Dependencies:
    pip install onnxruntime numpy pillow opencv-python-headless tokenizers ftfy

Usage:
    from pure_onnx_unimernet import OnnxUnimerNet

    model = OnnxUnimerNet(
        artifacts_dir="artifacts",
        tokenizer_path="models/unimernet_tiny",
    )
    latex = model.predict(PIL.Image.open("formula.png"))

File layout expected:
    artifacts/
        encoder_model.onnx
        decoder_model.onnx
        decoder_with_past_model.onnx
    models/unimernet_tiny/
        tokenizer.json
        tokenizer_config.json
"""

from __future__ import annotations

import os
import site

import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image, ImageOps
from typing import Any, List, Optional, Union


# ---------------------------------------------------------------------------
# Constants - must match what was used during export
# ---------------------------------------------------------------------------
IMAGE_H       = 192
IMAGE_W       = 672
MEAN          = 0.7931
STD           = 0.1738
NUM_LAYERS    = 8
MAX_NEW_TOKENS = 1534
_DLL_DIRECTORY_HANDLES: list[Any] = []


def _provider_name(provider: object) -> str:
    if isinstance(provider, tuple) and provider:
        return str(provider[0])
    return str(provider)


def _cuda_requested(providers: list[object]) -> bool:
    return any(_provider_name(provider) == "CUDAExecutionProvider" for provider in providers)


def _nvidia_bin_dirs() -> list[Path]:
    roots: list[Path] = []

    try:
        import nvidia

        roots.append(Path(nvidia.__file__).resolve().parent)
    except Exception:
        pass

    for site_dir in site.getsitepackages():
        roots.append(Path(site_dir) / "nvidia")

    try:
        roots.append(Path(site.getusersitepackages()) / "nvidia")
    except Exception:
        pass

    component_order = (
        "cudnn",
        "cublas",
        "cuda_runtime",
        "cuda_nvrtc",
        "cufft",
        "curand",
        "nvjitlink",
    )

    seen: set[Path] = set()
    bin_dirs: list[Path] = []
    for root in roots:
        for component in component_order:
            bin_dir = root / component / "bin"
            if bin_dir.exists() and bin_dir not in seen:
                seen.add(bin_dir)
                bin_dirs.append(bin_dir)

    return bin_dirs


def _prepare_cuda_runtime(providers: list[object]) -> None:
    if not _cuda_requested(providers):
        return

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "CUDAExecutionProvider was requested, but this ONNX Runtime build does "
            "not expose CUDA. Install onnxruntime-gpu[cuda,cudnn] or use CPU."
        )

    if os.name == "nt":
        existing_path = os.environ.get("PATH", "")
        existing_parts = set(existing_path.split(os.pathsep))
        new_parts: list[str] = []

        for bin_dir in _nvidia_bin_dirs():
            bin_dir_str = str(bin_dir)
            if hasattr(os, "add_dll_directory"):
                try:
                    _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(bin_dir_str))
                except OSError:
                    pass
            if bin_dir_str not in existing_parts:
                new_parts.append(bin_dir_str)

        if new_parts:
            os.environ["PATH"] = os.pathsep.join(new_parts + [existing_path])

    preload_dlls = getattr(ort, "preload_dlls", None)
    if preload_dlls is not None:
        try:
            preload_dlls(directory="")
        except TypeError:
            preload_dlls()


def _disable_ort_fallback_if_cuda_requested(
    providers: list[object],
    sessions: list[ort.InferenceSession],
) -> None:
    if not _cuda_requested(providers):
        return

    for session in sessions:
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError(
                "CUDAExecutionProvider was requested, but ONNX Runtime did not "
                "activate it for the UniMERNet session."
            )
        if hasattr(session, "disable_fallback"):
            session.disable_fallback()


# ---------------------------------------------------------------------------
# Preprocessor
# Replicates FormulaImageEvalProcessor exactly - same as convert_to_onnx.py
# ---------------------------------------------------------------------------
def _preprocess(img: Image.Image) -> np.ndarray:
    """
    Returns float32 numpy array [1, 1, 192, 672].
    The encoder ONNX handles the 1-to-3 channel repeat internally.
    """
    import cv2

    # crop margins
    data = np.array(img.convert("L")).astype(np.uint8)
    max_val, min_val = data.max(), data.min()
    if max_val != min_val:
        data_norm = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data_norm < 200).astype(np.uint8)
        coords = cv2.findNonZero(gray)
        if coords is not None:
            a, b, w, h = cv2.boundingRect(coords)
            img = img.crop((a, b, w + a, h + b))

    img = img.convert("RGB")

    # Scale shortest side to min(192, 672) = 192
    w, h = img.size
    short = min(h, w)
    scale = min(IMAGE_H, IMAGE_W) / short
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    img.thumbnail((IMAGE_W, IMAGE_H), Image.BICUBIC)

    # center pad to exactly 192x672
    delta_w = IMAGE_W - img.width
    delta_h = IMAGE_H - img.height
    pad_w   = delta_w // 2
    pad_h   = delta_h // 2
    img = ImageOps.expand(img, (pad_w, pad_h, delta_w - pad_w, delta_h - pad_h))

    # grayscale -> normalize -> [1, 1, H, W]
    arr = np.array(img.convert("L")).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr[np.newaxis, np.newaxis, :, :]  # [1, 1, 192, 672]


# ---------------------------------------------------------------------------
# Tokenizer wrapper - uses tokenizers library directly, no transformers
# ---------------------------------------------------------------------------
class _Tokenizer:
    def __init__(self, tokenizer_path: str):
        from tokenizers import Tokenizer
        path = Path(tokenizer_path) / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"tokenizer.json not found at {path}")
        self._tok = Tokenizer.from_file(str(path))

        # Resolve special token ids
        self.bos_token_id = self._get_id("<s>")
        self.eos_token_id = self._get_id("</s>")
        self.pad_token_id = self._get_id("<pad>")

    def _get_id(self, token: str) -> int:
        id_ = self._tok.token_to_id(token)
        if id_ is None:
            raise ValueError(f"Token {token!r} not found in tokenizer vocabulary")
        return id_

    def decode(self, token_ids: List[int]) -> str:
        # Filter special tokens
        filtered = [t for t in token_ids
                    if t not in (self.bos_token_id, self.eos_token_id, self.pad_token_id)]
        text = self._tok.decode(filtered)
        try:
            from ftfy import fix_text
            text = fix_text(text)
        except ImportError:
            pass
        return text


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class OnnxUnimerNet:
    """
    Pure ONNX formula recognizer. Drop-in replacement for pytorch_unimernet_mfr.py.

    Args:
        artifacts_dir:   Path to folder containing the three .onnx files.
        tokenizer_path:  Path to folder containing tokenizer.json.
        providers:       onnxruntime execution providers.
                         Default: ["CPUExecutionProvider"]
                         GPU:     ["CUDAExecutionProvider", "CPUExecutionProvider"]
        max_new_tokens:  Hard cap on generated tokens. Default: 1534 (model limit).
        num_threads:     onnxruntime intra/inter op thread count. Default: 0 (auto).
    """

    def __init__(
        self,
        artifacts_dir: Union[str, Path] = "artifacts",
        tokenizer_path: Union[str, Path] = "models/unimernet_tiny",
        providers: Optional[List[str]] = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
        num_threads: int = 0,
    ):
        artifacts_dir = Path(artifacts_dir)
        self.max_new_tokens = max_new_tokens

        # Session options
        opts = ort.SessionOptions()
        if num_threads > 0:
            opts.inter_op_num_threads = num_threads
            opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if providers is None:
            providers = ["CPUExecutionProvider"]
        _prepare_cuda_runtime(providers)

        # Load ONNX sessions
        enc_path      = artifacts_dir / "encoder_model.onnx"
        dec_path      = artifacts_dir / "decoder_model.onnx"
        dec_past_path = artifacts_dir / "decoder_with_past_model.onnx"

        for p in [enc_path, dec_path, dec_past_path]:
            if not p.exists():
                raise FileNotFoundError(f"ONNX file not found: {p}")

        self._enc_sess      = ort.InferenceSession(str(enc_path),      opts, providers=providers)
        self._dec_sess      = ort.InferenceSession(str(dec_path),      opts, providers=providers)
        self._dec_past_sess = ort.InferenceSession(str(dec_past_path), opts, providers=providers)
        _disable_ort_fallback_if_cuda_requested(
            providers,
            [self._enc_sess, self._dec_sess, self._dec_past_sess],
        )

        # Load tokenizer
        self._tok = _Tokenizer(str(tokenizer_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _predict_tokens(
        self,
        img: Image.Image,
        *,
        max_new_tokens: Optional[int] = None,
    ) -> List[int]:
        pixel_values = _preprocess(img)              # [1, 1, 192, 672] float32
        return self._decode(pixel_values, max_new_tokens=max_new_tokens)

    def predict(self, img: Image.Image) -> str:
        """
        Recognize a formula image and return LaTeX string.

        Args:
            img: PIL Image (any mode, any size).

        Returns:
            LaTeX string.
        """
        token_ids = self._predict_tokens(img)
        return self._tok.decode(token_ids)

    def recognize(self, image: Image.Image, max_new_tokens: int = MAX_NEW_TOKENS) -> dict[str, Any]:
        token_ids = self._predict_tokens(image, max_new_tokens=max_new_tokens)
        latex = self._tok.decode(token_ids).strip()

        return {
            "latex": latex,
            "confidence": 1.0,
            "tokens": token_ids,
        }

    def predict_batch(self, imgs: List[Image.Image]) -> List[str]:
        """
        Recognize a list of formula images. Processes one at a time (batch=1).
        Provided for API compatibility with pytorch_unimernet_mfr.py.
        """
        return [self.predict(img) for img in imgs]

    # ------------------------------------------------------------------
    # Internal decode loop
    # ------------------------------------------------------------------

    def _decode(
        self,
        pixel_values: np.ndarray,
        *,
        max_new_tokens: Optional[int] = None,
    ) -> List[int]:
        """
        Full greedy decode: encoder -> decoder step 1 -> decoder with past N times.

        Args:
            pixel_values: [1, 1, 192, 672] float32 numpy array.

        Returns:
            List of token ids (including EOS if reached).
        """
        # 1. Encode
        enc_hs = self._enc_sess.run(
            ["encoder_hidden_states"],
            {"pixel_values": pixel_values},
        )[0]  # [1, 126, 512]

        # 2. Decoder step 1 (no past KV)
        input_ids = np.array([[self._tok.bos_token_id]], dtype=np.int64)
        dec_out   = self._dec_sess.run(None, {
            "input_ids":              input_ids,
            "encoder_hidden_states":  enc_hs,
        })
        logits   = dec_out[0]        # [1, 1, 50000]
        flat_pkv = dec_out[1:]       # 16 tensors: key_0, value_0, ..., key_7, value_7

        tokens: List[int] = []

        # 3. Greedy decode loop
        token_budget = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        token_budget = max(1, min(int(token_budget), MAX_NEW_TOKENS))

        for _ in range(token_budget):
            next_id = int(np.argmax(logits[0, -1]))
            tokens.append(next_id)

            if next_id == self._tok.eos_token_id:
                break

            input_ids = np.array([[next_id]], dtype=np.int64)

            # Build past KV feed
            past_feed: dict = {
                "input_ids":             input_ids,
                "encoder_hidden_states": enc_hs,
            }
            for i in range(NUM_LAYERS):
                past_feed[f"past_key_{i}"]   = flat_pkv[i * 2]
                past_feed[f"past_value_{i}"] = flat_pkv[i * 2 + 1]

            dec_past_out = self._dec_past_sess.run(None, past_feed)
            logits   = dec_past_out[0]
            flat_pkv = dec_past_out[1:]

        return tokens


# ---------------------------------------------------------------------------
# CLI - quick sanity check
# Usage: python pure_onnx_unimernet.py path/to/formula.png
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("Usage: python pure_onnx_unimernet.py <image_path> [artifacts_dir] [tokenizer_path]")
        sys.exit(1)

    img_path       = sys.argv[1]
    artifacts_dir  = sys.argv[2] if len(sys.argv) > 2 else "artifacts"
    tokenizer_path = sys.argv[3] if len(sys.argv) > 3 else "models/unimernet_tiny"

    print(f"Loading model from {artifacts_dir}...")
    model = OnnxUnimerNet(
        artifacts_dir=artifacts_dir,
        tokenizer_path=tokenizer_path,
    )

    img = Image.open(img_path)
    print(f"Running inference on {img_path}...")

    t0     = time.perf_counter()
    result = model.predict(img)
    elapsed = time.perf_counter() - t0

    print(f"\nLatex: {result}")
    print(f"Time:  {elapsed*1000:.0f}ms")
