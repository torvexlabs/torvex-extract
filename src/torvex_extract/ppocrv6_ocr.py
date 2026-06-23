import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from torvex_extract.onnx_runtime import create_onnx_session

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "models"
PPOCRV6_SMALL_BACKEND = "ppocrv6_small"
_PYCLIPPER: Any | None = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for PP-OCRv6 inference.yml parsing. "
            "Install/sync the project dependency: pyyaml>=6.0.0"
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid PP-OCRv6 YAML file: {path}")

    return data


def _clip_box_xyxy(
    box: list[float],
    *,
    width: int,
    height: int,
) -> list[float] | None:
    x0, y0, x1, y1 = box
    x0 = max(0.0, min(float(width), x0))
    y0 = max(0.0, min(float(height), y0))
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))

    if x1 <= x0 or y1 <= y0:
        return None

    return [x0, y0, x1, y1]


def _box_to_polygon(box: list[float]) -> list[list[float]]:
    x0, y0, x1, y1 = box
    return [
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
    ]


def _clip_polygon(
    points: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    clipped = points.astype(np.float32, copy=True)
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height)
    return clipped


def _polygon_to_xyxy(
    points: np.ndarray,
    *,
    width: int,
    height: int,
) -> list[float] | None:
    return _clip_box_xyxy(
        [
            float(np.min(points[:, 0])),
            float(np.min(points[:, 1])),
            float(np.max(points[:, 0])),
            float(np.max(points[:, 1])),
        ],
        width=width,
        height=height,
    )


def _sort_text_boxes(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        boxes,
        key=lambda item: (
            round(float(item["bbox_xyxy"][1]) / 10.0),
            float(item["bbox_xyxy"][0]),
        ),
    )


def _load_pyclipper() -> Any:
    global _PYCLIPPER

    if _PYCLIPPER is None:
        try:
            import pyclipper
        except ImportError as exc:
            raise RuntimeError(
                "pyclipper is required for PP-OCRv6 DB detector postprocess. "
                "Run uv sync after adding pyclipper>=1.4.0."
            ) from exc

        _PYCLIPPER = pyclipper

    return _PYCLIPPER


class PPOCRV6SmallOCR:
    """
    Raw ONNXRuntime PP-OCRv6 small detector+recognizer adapter.
    """

    def __init__(
        self,
        *,
        providers: list[str],
        model_dir: str | Path | None = None,
    ) -> None:
        self.model_dir = Path(
            model_dir
            or os.getenv(
                "TORVEX_PPOCRV6_MODEL_DIR",
                str(_DEFAULT_MODEL_DIR / "ppocrv6-small-onnx"),
            )
        )
        self.det_dir = self.model_dir / "det"
        self.rec_dir = self.model_dir / "rec"
        self.det_config = _load_yaml(self.det_dir / "inference.yml")
        self.rec_config = _load_yaml(self.rec_dir / "inference.yml")

        det_post = self.det_config.get("PostProcess", {}) or {}
        self.det_thresh = _safe_float(det_post.get("thresh"), 0.2)
        self.det_box_thresh = _safe_float(det_post.get("box_thresh"), 0.45)
        self.det_unclip_ratio = _safe_float(det_post.get("unclip_ratio"), 1.4)
        self.det_max_candidates = int(_safe_float(det_post.get("max_candidates"), 3000))

        rec_post = self.rec_config.get("PostProcess", {}) or {}
        characters = rec_post.get("character_dict") or []
        if not isinstance(characters, list) or not characters:
            raise RuntimeError("PP-OCRv6 recognition character_dict is missing.")
        self.characters = [str(ch) for ch in characters]

        self.det_session = create_onnx_session(
            self.det_dir / "inference.onnx",
            providers=providers,
            model_name="PP-OCRv6 small detector",
        )
        self.rec_session = create_onnx_session(
            self.rec_dir / "inference.onnx",
            providers=providers,
            model_name="PP-OCRv6 small recognizer",
        )

        # CUDA arena shrinkage: PP-OCR feeds variable-size det (up to det_max_long_side_px)
        # and rec (up to rec_max_width) inputs, so the ORT CUDA arena ratchets up to the
        # largest page seen (observed ~6.6GB peak over a full run). Shrinking the arena after
        # each run caps it (~237MB) with no quality/latency cost; the sessions are stateless
        # across pages so a per-run flush is safe. Gated to CUDA providers.
        self._run_options = None
        if any("CUDA" in provider for provider in providers):
            self._run_options = ort.RunOptions()
            self._run_options.add_run_config_entry(
                "memory.enable_memory_arena_shrinkage", "gpu:0"
            )

        self.det_input_name = self.det_session.get_inputs()[0].name
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        self.det_max_long_side_px = int(
            os.getenv("TORVEX_PPOCRV6_DET_MAX_LONG_SIDE_PX", "2500")
        )
        self.rec_image_height = 48
        self.rec_max_width = int(os.getenv("TORVEX_PPOCRV6_REC_MAX_WIDTH_PX", "3200"))
        self.rec_batch_size = int(os.getenv("TORVEX_PPOCRV6_REC_BATCH_SIZE", "8"))
        self.rec_batch_size = max(1, min(32, self.rec_batch_size))

        logger.info(
            "Loaded PP-OCRv6 small OCR: model_dir=%s det_providers=%s rec_providers=%s",
            self.model_dir,
            self.det_session.get_providers(),
            self.rec_session.get_providers(),
        )

    def _preprocess_det(
        self,
        image_np: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        original_h, original_w = image_np.shape[:2]
        scale = 1.0

        if self.det_max_long_side_px > 0:
            long_side = max(original_h, original_w)
            if long_side > self.det_max_long_side_px:
                scale = self.det_max_long_side_px / float(long_side)

        resized_h = max(32, int(round(original_h * scale / 32.0)) * 32)
        resized_w = max(32, int(round(original_w * scale / 32.0)) * 32)

        resized = cv2.resize(
            image_np,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # 2026-06-15: PP-OCRv6 inference.yml declares BGR decode + HWC
        # normalization. Torvex page images are RGB, so convert before feeding
        # the raw ONNX detector.
        bgr = resized[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (bgr - mean) / std
        chw = normalized.transpose(2, 0, 1)
        return (
            chw[np.newaxis, ...].astype(np.float32),
            resized_h / original_h,
            resized_w / original_w,
        )

    def _get_mini_boxes(
        self,
        contour: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda point: point[0])

        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0

        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2

        box = np.array(
            [
                points[index_1],
                points[index_2],
                points[index_3],
                points[index_4],
            ],
            dtype=np.float32,
        )
        return box, float(min(bounding_box[1]))

    def _box_score_fast(
        self,
        pred: np.ndarray,
        box: np.ndarray,
    ) -> float:
        h, w = pred.shape[:2]
        local_box = box.astype(np.float32, copy=True)
        xmin = int(np.clip(np.floor(local_box[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(local_box[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(local_box[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(local_box[:, 1].max()), 0, h - 1))

        if xmax < xmin or ymax < ymin:
            return 0.0

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        local_box[:, 0] -= xmin
        local_box[:, 1] -= ymin
        cv2.fillPoly(mask, local_box.reshape(1, -1, 2).astype(np.int32), 1)
        return float(cv2.mean(pred[ymin : ymax + 1, xmin : xmax + 1], mask)[0])

    def _unclip_box(self, box: np.ndarray) -> np.ndarray | None:
        pyclipper = _load_pyclipper()

        area = abs(float(cv2.contourArea(box.astype(np.float32))))
        perimeter = float(cv2.arcLength(box.astype(np.float32), True))
        if area <= 0.0 or perimeter <= 0.0:
            return None

        distance = area * self.det_unclip_ratio / perimeter
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(
            box.astype(np.float64).tolist(),
            pyclipper.JT_ROUND,
            pyclipper.ET_CLOSEDPOLYGON,
        )
        expanded = offset.Execute(distance)
        if len(expanded) != 1:
            return None

        return np.array(expanded[0], dtype=np.float32).reshape(-1, 1, 2)

    def _boxes_from_bitmap(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
        *,
        dest_width: int,
        dest_height: int,
    ) -> list[dict[str, Any]]:
        # 2026-06-16: PaddleOCR's DBPostProcess is not inside the ONNX model.
        # The official code uses OpenCV contours, min-area boxes, mean box
        # scores, and pyclipper unclip. We keep that behavior local for ONNX
        # numpy outputs instead of importing the full paddleocr/paddle stack.
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        pred_h, pred_w = pred.shape[:2]
        boxes: list[dict[str, Any]] = []

        for contour in contours[: self.det_max_candidates]:
            points, short_side = self._get_mini_boxes(contour)
            if short_side < 3:
                continue

            score = self._box_score_fast(pred, points.reshape(-1, 2))
            if score < self.det_box_thresh:
                continue

            expanded = self._unclip_box(points)
            if expanded is None:
                continue

            box, short_side = self._get_mini_boxes(expanded)
            if short_side < 5:
                continue

            box[:, 0] = np.clip(
                np.round(box[:, 0] / float(pred_w) * dest_width),
                0,
                dest_width,
            )
            box[:, 1] = np.clip(
                np.round(box[:, 1] / float(pred_h) * dest_height),
                0,
                dest_height,
            )

            bbox_xyxy = _polygon_to_xyxy(
                box,
                width=dest_width,
                height=dest_height,
            )
            if bbox_xyxy is None:
                continue

            boxes.append(
                {
                    "bbox": _clip_polygon(
                        box,
                        width=dest_width,
                        height=dest_height,
                    ).tolist(),
                    "bbox_xyxy": bbox_xyxy,
                    "score": score,
                }
            )

        return boxes

    def _detect_text_boxes(self, image_np: np.ndarray) -> list[dict[str, Any]]:
        original_h, original_w = image_np.shape[:2]
        det_input, _, _ = self._preprocess_det(image_np)
        output = self.det_session.run(None, {self.det_input_name: det_input}, self._run_options)[0]
        pred = output[0, 0]

        return _sort_text_boxes(
            self._boxes_from_bitmap(
                pred,
                (pred > self.det_thresh).astype(np.uint8),
                dest_width=original_w,
                dest_height=original_h,
            )
        )

    def _crop_text_region(
        self,
        image_np: np.ndarray,
        points: list[list[float]],
    ) -> np.ndarray | None:
        pts = np.array(points, dtype=np.float32)
        if pts.shape != (4, 2):
            return None

        width = int(
            round(
                max(
                    np.linalg.norm(pts[0] - pts[1]),
                    np.linalg.norm(pts[2] - pts[3]),
                )
            )
        )
        height = int(
            round(
                max(
                    np.linalg.norm(pts[0] - pts[3]),
                    np.linalg.norm(pts[1] - pts[2]),
                )
            )
        )

        if width <= 1 or height <= 1:
            return None

        target = np.array(
            [
                [0, 0],
                [width, 0],
                [width, height],
                [0, height],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(pts, target)
        crop = cv2.warpPerspective(
            image_np,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_REPLICATE,
        )

        if crop.shape[0] / max(1, crop.shape[1]) >= 1.5:
            crop = np.rot90(crop)

        return crop

    def _preprocess_rec_chw(self, crop_np: np.ndarray) -> np.ndarray | None:
        crop_h, crop_w = crop_np.shape[:2]
        if crop_h <= 0 or crop_w <= 0:
            return None

        resized_w = int(round(self.rec_image_height * crop_w / float(crop_h)))
        resized_w = max(8, min(self.rec_max_width, resized_w))
        resized = cv2.resize(
            crop_np,
            (resized_w, self.rec_image_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # 2026-06-15: Paddle RecResizeImg normalizes recognition crops to
        # CHW float values in [-1, 1]. Keep that model-specific transform here
        # so downstream Torvex OCR consumers stay backend-agnostic.
        bgr = resized[:, :, ::-1].astype(np.float32)
        chw = bgr.transpose(2, 0, 1) / 255.0
        chw = (chw - 0.5) / 0.5
        return chw.astype(np.float32)

    def _preprocess_rec(self, crop_np: np.ndarray) -> np.ndarray | None:
        chw = self._preprocess_rec_chw(crop_np)
        if chw is None:
            return None
        return chw[np.newaxis, ...]

    def _decode_recognition(
        self,
        logits: np.ndarray,
    ) -> tuple[str, float]:
        if float(np.max(logits)) > 1.0 or float(np.min(logits)) < 0.0:
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp = np.exp(shifted)
            probs_by_class = exp / np.sum(exp, axis=1, keepdims=True)
        else:
            probs_by_class = logits

        indices = np.argmax(probs_by_class, axis=1)
        probs = np.max(probs_by_class, axis=1)

        chars: list[str] = []
        scores: list[float] = []
        previous_index: int | None = None

        for index, prob in zip(indices, probs):
            index_int = int(index)
            if index_int == 0:
                previous_index = index_int
                continue

            if previous_index == index_int:
                continue

            char_index = index_int - 1
            if 0 <= char_index < len(self.characters):
                chars.append(self.characters[char_index])
                scores.append(float(prob))

            previous_index = index_int

        text = "".join(chars).strip()
        confidence = sum(scores) / len(scores) if scores else 0.0
        return text, confidence

    def _recognize_crop(self, crop_np: np.ndarray) -> tuple[str, float]:
        rec_input = self._preprocess_rec(crop_np)
        if rec_input is None:
            return "", 0.0

        logits = self.rec_session.run(None, {self.rec_input_name: rec_input}, self._run_options)[0][0]
        return self._decode_recognition(logits)

    def _recognize_crops(
        self,
        crops: list[np.ndarray],
    ) -> list[tuple[str, float]]:
        if not crops:
            return []

        preprocessed: list[tuple[int, np.ndarray]] = []
        results: list[tuple[str, float]] = [("", 0.0) for _ in crops]

        for index, crop in enumerate(crops):
            chw = self._preprocess_rec_chw(crop)
            if chw is not None:
                preprocessed.append((index, chw))

        if not preprocessed:
            return results

        preprocessed.sort(key=lambda item: item[1].shape[2])
        batch_count = 0

        for start in range(0, len(preprocessed), self.rec_batch_size):
            chunk = preprocessed[start:start + self.rec_batch_size]
            max_width = max(chw.shape[2] for _, chw in chunk)
            batch = np.zeros(
                (len(chunk), 3, self.rec_image_height, max_width),
                dtype=np.float32,
            )

            for batch_index, (_, chw) in enumerate(chunk):
                width = chw.shape[2]
                batch[batch_index, :, :, :width] = chw

            logits_batch = self.rec_session.run(
                None,
                {self.rec_input_name: batch},
                self._run_options,
            )[0]

            for (original_index, _), logits in zip(chunk, logits_batch):
                results[original_index] = self._decode_recognition(logits)

            batch_count += 1

        logger.debug(
            "PP-OCRv6 recognized %d crops in %d batches batch_size=%d",
            len(preprocessed),
            batch_count,
            self.rec_batch_size,
        )
        return results

    def ocr_image(self, image_np: np.ndarray) -> list[dict[str, Any]]:
        if image_np is None or image_np.size == 0:
            return []

        image_h, image_w = image_np.shape[:2]
        segments: list[dict[str, Any]] = []
        text_items: list[tuple[dict[str, Any], np.ndarray]] = []

        for box in self._detect_text_boxes(image_np):
            bbox_xyxy = box["bbox_xyxy"]
            crop = self._crop_text_region(image_np, box["bbox"])
            if crop is None:
                x0, y0, x1, y1 = [int(round(value)) for value in bbox_xyxy]
                x0 = max(0, min(image_w, x0))
                y0 = max(0, min(image_h, y0))
                x1 = max(0, min(image_w, x1))
                y1 = max(0, min(image_h, y1))
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = image_np[y0:y1, x0:x1]

            text_items.append((box, crop))

        logger.debug("PP-OCRv6 detected %d text crops", len(text_items))

        recognized_items = self._recognize_crops(
            [crop for _, crop in text_items]
        )

        for (box, _), (text, rec_score) in zip(text_items, recognized_items):
            if not text:
                continue

            bbox_xyxy = box["bbox_xyxy"]
            score = (
                (float(box["score"]) + rec_score) / 2.0
                if rec_score
                else float(box["score"])
            )
            segments.append(
                {
                    "bbox": box.get("bbox") or _box_to_polygon(bbox_xyxy),
                    "bbox_xyxy": [float(value) for value in bbox_xyxy],
                    "text": text,
                    "score": score,
                }
            )

        return segments
