from __future__ import annotations

import gc
import logging
import math
import os
from pathlib import Path
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from torvex_extract.onnx_runtime import select_onnx_providers
from torvex_extract.formula_pipeline import INLINE_MIN_FRAC, select_formula_boxes
from torvex_extract.unirec_recognizer import UniRecRecognizer

logger = logging.getLogger(__name__)

FormulaFallbackOcr = Callable[[np.ndarray], list[dict[str, Any]]]

_DISPLAY_FORMULA_TYPE = "display_formula"
_INLINE_FORMULA_TYPE = "inline_formula"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    return default


def _env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except Exception:
        return default

    return max(min_value, min(max_value, value))


def _env_float(name: str, default: float, *, min_value: float, max_value: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = float(raw)
    except Exception:
        return default

    return max(min_value, min(max_value, value))


@dataclass(frozen=True)
class FormulaExtractionConfig:
    # crop prep (golden path)
    padding_ratio: float = field(default_factory=lambda: _env_float("TORVEX_FORMULA_PADDING_RATIO", 0.01, min_value=0.0, max_value=0.20))
    min_padding_px: int = field(default_factory=lambda: _env_int("TORVEX_FORMULA_MIN_PADDING_PX", 2, min_value=0, max_value=64))
    white_border_px: int = field(default_factory=lambda: _env_int("TORVEX_FORMULA_WHITE_BORDER_PX", 8, min_value=0, max_value=64))
    min_crop_width_px: int = 8
    min_crop_height_px: int = 8
    blank_dark_ratio_threshold: float = 0.0005
    # box selection: promote inline_formula boxes whose height >= this fraction of page height
    # (targeted recovery of display formulas misclassified as inline; see formula_pipeline)
    inline_min_frac: float = field(default_factory=lambda: _env_float("TORVEX_FORMULA_INLINE_MIN_FRAC", INLINE_MIN_FRAC, min_value=0.0, max_value=1.0))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default

    if not math.isfinite(number):
        return default

    return number


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        number = int(value)
    except Exception:
        return default

    return number


def _normalize_image_array(img_np: np.ndarray) -> np.ndarray:
    arr = np.asarray(img_np)

    if arr.size == 0:
        return arr

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)

        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0

        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr)


def _to_rgb_pil(img_np: np.ndarray) -> Image.Image | None:
    if img_np is None:
        return None

    arr = _normalize_image_array(img_np)

    if arr.size == 0:
        return None

    try:
        image = Image.fromarray(arr)
    except Exception:
        return None

    return image.convert("RGB")


def _extract_bbox_px(formula: dict[str, Any]) -> list[float] | None:
    bbox = formula.get("bbox_px")

    if bbox is None:
        bbox = formula.get("bbox")

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x0, y0, x1, y1 = [float(value) for value in bbox]
    except Exception:
        return None

    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        return None

    if x1 <= x0 or y1 <= y0:
        return None

    return [x0, y0, x1, y1]


def _expanded_bbox(
    bbox_px: list[float],
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float,
    min_padding_px: int,
) -> list[int] | None:
    x0, y0, x1, y1 = bbox_px

    box_w = x1 - x0
    box_h = y1 - y0

    if box_w <= 0 or box_h <= 0:
        return None

    pad = max(float(min_padding_px), max(box_w, box_h) * padding_ratio)

    x0_i = max(0, int(math.floor(x0 - pad)))
    y0_i = max(0, int(math.floor(y0 - pad)))
    x1_i = min(image_width, int(math.ceil(x1 + pad)))
    y1_i = min(image_height, int(math.ceil(y1 + pad)))

    if x1_i <= x0_i or y1_i <= y0_i:
        return None

    return [x0_i, y0_i, x1_i, y1_i]


def _crop_formula_image(
    page_image: Image.Image,
    bbox_px: list[float],
    config: FormulaExtractionConfig,
    *,
    padding_ratio: float | None = None,
) -> tuple[Image.Image, list[int]] | tuple[None, None]:
    actual_padding_ratio = (
        config.padding_ratio if padding_ratio is None else padding_ratio
    )

    image_width, image_height = page_image.size

    crop_bbox = _expanded_bbox(
        bbox_px,
        image_width=image_width,
        image_height=image_height,
        padding_ratio=actual_padding_ratio,
        min_padding_px=config.min_padding_px,
    )

    if crop_bbox is None:
        return None, None

    x0, y0, x1, y1 = crop_bbox

    if (x1 - x0) < config.min_crop_width_px:
        return None, None

    if (y1 - y0) < config.min_crop_height_px:
        return None, None

    crop = page_image.crop((x0, y0, x1, y1)).convert("RGB")

    gray = np.asarray(crop.convert("L"))
    dark_ratio = float(np.mean(gray < 245))

    if dark_ratio < config.blank_dark_ratio_threshold:
        return None, None

    crop = ImageOps.expand(
        crop,
        border=config.white_border_px,
        fill="white",
    )

    return crop, crop_bbox


# --- 2026-06-11 display-formula splitter -----------------------------------
# See FormulaExtractionConfig comment for the why. Pure numpy, deterministic,
# no model involved: scan the bbox region row-by-row, find full-width blank
# (near-white) horizontal gap runs, and cut the bbox at gaps that are tall
# enough to be equation boundaries rather than intra-equation spacing.


def _join_ocr_segments(segments: list[dict[str, Any]]) -> str:
    if not segments:
        return ""

    def segment_top(segment: dict[str, Any]) -> float:
        bbox = segment.get("bbox") or []

        try:
            return min(float(point[1]) for point in bbox)
        except Exception:
            return 0.0

    def segment_left(segment: dict[str, Any]) -> float:
        bbox = segment.get("bbox") or []

        try:
            return min(float(point[0]) for point in bbox)
        except Exception:
            return 0.0

    valid = [
        segment
        for segment in segments
        if str(segment.get("text") or "").strip()
    ]

    valid.sort(key=lambda segment: (segment_top(segment), segment_left(segment)))

    return "\n".join(
        str(segment.get("text", "")).strip()
        for segment in valid
        if str(segment.get("text", "")).strip()
    ).strip()


class FormulaMfrExtractor:
    """Formula recognizer: golden-path box selection + UniRec-0.1B recognition.

    Reworked 2026-06-22 (UniRec pipeline merge). extract() runs the lean path:
    select_formula_boxes (display + display-like inline + drop-inner-keep-outer) -> crop ->
    UniRec recognize -> per-box latex. The recognizer is loaded lazily. This reached
    display_formula CDM 0.9604 (verified by driving this extract() over the full benchmark).
    Output granularity (seg_split_latex) is applied at markdown/output assembly, not here.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        config: FormulaExtractionConfig | None = None,
    ) -> None:
        self.device = (device or "cpu").strip().lower()
        self.config = config or FormulaExtractionConfig()
        self._rec: UniRecRecognizer | None = None


    def _ensure_loaded(self) -> None:
        if self._rec is not None:
            return
        self._rec = UniRecRecognizer(device=self.device)
        return
    def recognize_crop(self, crop: Image.Image) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._rec is not None
        return self._rec.recognize_crop(crop)

    def preflight(self) -> None:
        self._ensure_loaded()

    def recognize_crops(self, crops: list[Image.Image]) -> list[dict[str, Any]]:
        self._ensure_loaded()
        assert self._rec is not None
        return self._rec.recognize_crops(crops) if crops else []
    def extract(
        self,
        *,
        img_np: np.ndarray,
        formula_bboxes: list[dict[str, Any]],
        page_num: int,
        fallback_ocr: FormulaFallbackOcr | None = None,
    ) -> list[dict[str, Any]]:
        page_image = _to_rgb_pil(img_np)
        if page_image is None:
            return []
        page_h = page_image.height
        disp = [tuple(f["bbox_px"]) for f in (formula_bboxes or [])
                if f.get("type") == _DISPLAY_FORMULA_TYPE and f.get("bbox_px")]
        inline = [tuple(f["bbox_px"]) for f in (formula_bboxes or [])
                  if f.get("type") == _INLINE_FORMULA_TYPE and f.get("bbox_px")]
        selected = select_formula_boxes(disp, inline, page_h, self.config.inline_min_frac)
        crops: list[Image.Image] = []
        boxes: list[tuple] = []
        for b in selected:
            crop, crop_bbox = _crop_formula_image(page_image, list(b), self.config)
            if crop is not None:
                crops.append(crop)
                boxes.append((b, crop_bbox))
        results = self.recognize_crops(crops) if crops else []
        out: list[dict[str, Any]] = []
        for i, ((b, crop_bbox), r) in enumerate(zip(boxes, results)):
            latex = str(r.get("latex") or "").strip()
            out.append({
                "formula_id": f"formula_{page_num}_{i}",
                "type": _DISPLAY_FORMULA_TYPE,
                "bbox_px": list(b),
                "crop_bbox_px": crop_bbox,
                "latex": latex,
                "confidence": 0.90 if latex else 0.0,
                "status": "accepted" if latex else "empty",
            })
        return out

_EXTRACTORS: dict[tuple[str, FormulaExtractionConfig], FormulaMfrExtractor] = {}


def get_formula_extractor(
    *,
    device: str | None = None,
    config: FormulaExtractionConfig | None = None,
) -> FormulaMfrExtractor:
    actual_config = config or FormulaExtractionConfig()
    key = ((device or "cpu").strip().lower(), actual_config)

    if key not in _EXTRACTORS:
        _EXTRACTORS[key] = FormulaMfrExtractor(
            device=device,
            config=actual_config,
        )

    return _EXTRACTORS[key]


def ensure_formula_runtime_available(
    *,
    device: str | None = None,
    config: FormulaExtractionConfig | None = None,
) -> None:
    extractor = get_formula_extractor(
        device=device,
        config=config,
    )
    extractor.preflight()


def extract_formulas_from_bboxes(
    *,
    img_np: np.ndarray,
    formula_bboxes: list[dict[str, Any]],
    page_num: int,
    device: str | None = None,
    fallback_ocr: FormulaFallbackOcr | None = None,
    config: FormulaExtractionConfig | None = None,
) -> list[dict[str, Any]]:
    extractor = get_formula_extractor(
        device=device,
        config=config,
    )

    return extractor.extract(
        img_np=img_np,
        formula_bboxes=formula_bboxes,
        page_num=page_num,
        fallback_ocr=fallback_ocr,
    )


def shutdown_formula_extractor() -> None:
    _EXTRACTORS.clear()
    gc.collect()
