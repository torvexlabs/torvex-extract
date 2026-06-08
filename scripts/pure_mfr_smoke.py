from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from tokenizers import Tokenizer


MODEL_DIR = Path("models/pix2text-mfr-1.5")


class PureOnnxMfr:
    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        self.model_dir = model_dir

        if device in {"gpu", "cuda"}:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.encoder = ort.InferenceSession(
            str(model_dir / "encoder_model.onnx"),
            providers=providers,
        )
        self.decoder = ort.InferenceSession(
            str(model_dir / "decoder_model.onnx"),
            providers=providers,
        )

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        generation_config = json.loads(
            (model_dir / "generation_config.json").read_text(encoding="utf-8")
        )

        self.decoder_start_token_id = int(config.get("decoder_start_token_id", 1))
        self.eos_token_id = int(config.get("eos_token_id", 2))
        self.pad_token_id = int(config.get("pad_token_id", 0))
        self.max_length = int(generation_config.get("max_length", 384))

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((384, 384))

        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)

        return np.ascontiguousarray(arr, dtype=np.float32)

    def recognize(self, image: Image.Image, max_new_tokens: int = 128) -> dict:
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

            # confidence from softmax probability of chosen token
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

        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        confidence = 0.0
        if token_probs:
            confidence = float(
                math.exp(sum(math.log(p) for p in token_probs) / len(token_probs))
            )

        return {
            "latex": text,
            "confidence": confidence,
            "tokens": generated,
        }


def main() -> None:
    img = Image.new("RGB", (260, 80), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 25), "E = mc^2", fill="black")

    engine = PureOnnxMfr(MODEL_DIR, device="cpu")
    out = engine.recognize(img)

    print(out)


if __name__ == "__main__":
    main()
