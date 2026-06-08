from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer


class PureOnnxMfr:
    def __init__(self, model_dir: str | Path | None = None, device: str = "cpu") -> None:
        if model_dir is None:
            model_dir = os.getenv("TORVEX_FORMULA_MODEL_DIR") or "models/pix2text-mfr-1.5"

        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Formula model directory not found: {self.model_dir}. "
                "Set TORVEX_FORMULA_MODEL_DIR or place model files under models/pix2text-mfr-1.5."
            )

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device in {"gpu", "cuda"}
            else ["CPUExecutionProvider"]
        )

        self.encoder = ort.InferenceSession(
            str(self.model_dir / "encoder_model.onnx"),
            providers=providers,
        )
        self.decoder = ort.InferenceSession(
            str(self.model_dir / "decoder_model.onnx"),
            providers=providers,
        )

        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))

        config = self._read_json("config.json")
        generation_config = self._read_json("generation_config.json")
        preprocessor_config = self._read_json("preprocessor_config.json")

        self.decoder_start_token_id = int(config.get("decoder_start_token_id", 1))
        self.eos_token_id = int(config.get("eos_token_id", 2))
        self.pad_token_id = int(config.get("pad_token_id", 0))
        self.max_length = int(generation_config.get("max_length", 384))

        size = preprocessor_config.get("size", {})
        self.image_height = int(size.get("height", 384))
        self.image_width = int(size.get("width", 384))

        self.image_mean = np.asarray(
            preprocessor_config.get("image_mean", [0.5, 0.5, 0.5]),
            dtype=np.float32,
        )
        self.image_std = np.asarray(
            preprocessor_config.get("image_std", [0.5, 0.5, 0.5]),
            dtype=np.float32,
        )

    def _read_json(self, name: str) -> dict[str, Any]:
        return json.loads((self.model_dir / name).read_text(encoding="utf-8"))

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((self.image_width, self.image_height))

        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = (arr - self.image_mean) / self.image_std
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)

        return np.ascontiguousarray(arr, dtype=np.float32)

    def recognize(self, image: Image.Image, max_new_tokens: int = 128) -> dict[str, Any]:
        pixel_values = self.preprocess(image)

        encoder_hidden = self.encoder.run(
            None,
            {"pixel_values": pixel_values},
        )[0]

        input_ids = np.array([[self.decoder_start_token_id]], dtype=np.int64)

        generated: list[int] = []
        token_probs: list[float] = []

        for _ in range(min(max_new_tokens, self.max_length)):
            logits = self.decoder.run(
                None,
                {
                    "input_ids": input_ids,
                    "encoder_hidden_states": encoder_hidden,
                },
            )[0]

            next_logits = logits[0, -1, :]
            next_id = int(np.argmax(next_logits))

            shifted = next_logits - np.max(next_logits)
            probs = np.exp(shifted) / np.sum(np.exp(shifted))
            prob = float(probs[next_id])

            if next_id == self.eos_token_id:
                break

            if next_id != self.pad_token_id:
                generated.append(next_id)

                if math.isfinite(prob) and prob > 0:
                    token_probs.append(prob)

            input_ids = np.concatenate(
                [input_ids, np.array([[next_id]], dtype=np.int64)],
                axis=1,
            )

        latex = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        confidence = 0.0
        if token_probs:
            confidence = float(
                math.exp(sum(math.log(prob) for prob in token_probs) / len(token_probs))
            )

        return {
            "latex": latex,
            "confidence": confidence,
            "tokens": generated,
        }
