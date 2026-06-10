from __future__ import annotations

from torvex_extract.pure_onnx_mfr import PureOnnxMfr

import gc
import logging
import math
import os
from pathlib import Path
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

FormulaFallbackOcr = Callable[[np.ndarray], list[dict[str, Any]]]

_FORMULA_NUMBER_TYPE = "formula_number"
_DISPLAY_FORMULA_TYPE = "display_formula"
_INLINE_FORMULA_TYPE = "inline_formula"

_LATEX_MARKERS = (
    "\\",
    "^",
    "_",
    "{",
    "}",
    "=",
    "+",
    "-",
    "*",
    "/",
    r"\frac",
    r"\sqrt",
    r"\sum",
    r"\int",
    r"\lim",
    r"\log",
    r"\sin",
    r"\cos",
    r"\tan",
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\lambda",
    r"\mu",
    r"\sigma",
    r"\theta",
)

_WORD_RE = re.compile(r"[A-Za-z]{4,}")
_BROKEN_COMMAND_RE = re.compile(r"\\[^A-Za-z{}\s]")
_REPEATED_GARBAGE_RE = re.compile(r"(.{1,8})\1{5,}")


_ALLOWED_FORMULA_TYPES = frozenset(
    {
        _DISPLAY_FORMULA_TYPE,
        _INLINE_FORMULA_TYPE,
        _FORMULA_NUMBER_TYPE,
    }
)


def _default_enabled_formula_types() -> tuple[str, ...]:
    # Benchmark/prod default:
    # Display formulas are the only formula class currently exported/scored safely.
    # Inline formulas and formula_number remain opt-in through TORVEX_FORMULA_TYPES.
    raw = os.getenv("TORVEX_FORMULA_TYPES", _DISPLAY_FORMULA_TYPE)

    values = tuple(
        value.strip()
        for value in raw.split(",")
        if value.strip()
    )

    selected = tuple(
        value
        for value in values
        if value in _ALLOWED_FORMULA_TYPES
    )

    return selected or (_DISPLAY_FORMULA_TYPE,)


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
    model_name: str = "breezedeus/pix2text-mfr-1.5"
    enabled_formula_types: tuple[str, ...] = field(default_factory=_default_enabled_formula_types)
    accept_confidence: float = 0.75
    low_confidence: float = 0.55
    padding_ratio: float = field(default_factory=lambda: _env_float("TORVEX_FORMULA_PADDING_RATIO", 0.01, min_value=0.0, max_value=0.20))
    min_padding_px: int = field(default_factory=lambda: _env_int("TORVEX_FORMULA_MIN_PADDING_PX", 2, min_value=0, max_value=64))
    white_border_px: int = field(default_factory=lambda: _env_int("TORVEX_FORMULA_WHITE_BORDER_PX", 8, min_value=0, max_value=64))
    min_crop_width_px: int = 8
    min_crop_height_px: int = 8
    blank_dark_ratio_threshold: float = 0.0005
    enable_self_consensus: bool = field(default_factory=lambda: _env_bool("TORVEX_FORMULA_SELF_CONSENSUS", False))
    consensus_padding_ratio: float = 0.08
    consensus_similarity_threshold: float = 0.92
    max_new_tokens: int = field(default_factory=lambda: _env_int("TORVEX_FORMULA_MAX_NEW_TOKENS", 256, min_value=16, max_value=256))


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


def _balanced(text: str, left: str, right: str) -> bool:
    depth = 0
    escaped = False

    for char in text:
        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == left:
            depth += 1
        elif char == right:
            depth -= 1

        if depth < 0:
            return False

    return depth == 0


def _debug_save_formula_crop(
    crop: Image.Image,
    *,
    formula_id: str,
    formula_type: str,
) -> str:
    out_dir = os.getenv("TORVEX_FORMULA_CROP_DEBUG_DIR")
    if not out_dir:
        return ""

    try:
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)

        safe_id = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_"
            for ch in str(formula_id or "formula")
        )
        safe_type = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_"
            for ch in str(formula_type or "unknown")
        )

        out_path = root / f"{safe_id}_{safe_type}.png"
        crop.save(out_path)
        return str(out_path)
    except Exception as exc:
        return f"crop_debug_save_failed:{exc}"


_ALLOWED_SINGLE_CHAR_LATEX_COMMANDS = frozenset(
    {
        "\\",  # row break: \\
        ",",  # thin space: \,
        ";",
        ":",
        "!",
        "_",
        "#",
        "%",
        "&",
        "$",
        "{",
        "}",
        "[",
        "]",
        "(",
        ")",
        "|",
        "~",
    }
)


def _has_broken_latex_command(text: str) -> bool:
    index = 0
    length = len(text)

    while index < length:
        if text[index] != "\\":
            index += 1
            continue

        if index + 1 >= length:
            return True

        next_char = text[index + 1]

        # Standard LaTeX command: \frac, \left, \right, \begin, etc.
        if next_char.isalpha():
            cursor = index + 2
            while cursor < length and text[cursor].isalpha():
                cursor += 1
            index = cursor
            continue

        # Common valid one-character LaTeX commands:
        # \\, \,, \;, \:, \!, \#, \_, \%, \&, \{, \}, etc.
        if next_char in _ALLOWED_SINGLE_CHAR_LATEX_COMMANDS or next_char.isspace():
            index += 2
            continue

        return True

    return False


def validate_latex(latex: str, formula_type: str) -> tuple[bool, str]:
    text = (latex or "").strip()

    if not text:
        return False, "empty_latex"

    if len(text) > 1500:
        return False, "latex_too_long"

    if "\ufffd" in text:
        return False, "replacement_char"

    if _REPEATED_GARBAGE_RE.search(text):
        return False, "repeated_garbage"

    if not _balanced(text, "{", "}"):
        return False, "unbalanced_braces"

    if not _balanced(text, "[", "]"):
        return False, "unbalanced_brackets"

    if not _balanced(text, "(", ")"):
        return False, "unbalanced_parentheses"

    if _has_broken_latex_command(text):
        return False, "broken_latex_command"

    # Inline formulas can legitimately be tiny: r, x, EPS, P/E, etc.
    if formula_type == _INLINE_FORMULA_TYPE and len(text) <= 16:
        return True, ""

    has_marker = any(marker in text for marker in _LATEX_MARKERS)
    digit_count = sum(1 for char in text if char.isdigit())
    operator_count = sum(1 for char in text if char in "=+-*/<>")

    if has_marker or digit_count >= 2 or operator_count >= 1:
        return True, ""

    words = _WORD_RE.findall(text)

    if len(words) >= 2:
        return False, "plain_text_like"

    return True, ""


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


def _as_crop_np(crop: Image.Image) -> np.ndarray:
    return np.asarray(crop.convert("RGB"))


def _text_similarity(left: str, right: str) -> float:
    left_clean = re.sub(r"\s+", "", left or "")
    right_clean = re.sub(r"\s+", "", right or "")

    if not left_clean or not right_clean:
        return 0.0

    return SequenceMatcher(None, left_clean, right_clean).ratio()


def _default_formula_model_dir() -> str:
    env_value = os.getenv("TORVEX_FORMULA_MODEL_DIR")
    if env_value:
        return env_value

    model_rel = Path("models") / "pix2text-mfr-1.5"

    candidates: list[Path] = [
        Path.cwd() / model_rel,
    ]

    # Important for editable installs:
    # C:\torvex-extract\src\torvex_extract\formula_extractor.py
    # -> C:\torvex-extract\models\pix2text-mfr-1.5
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / model_rel)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Final fallback keeps old behavior.
    return str(model_rel)


class FormulaMfrExtractor:
    """
    Optional Pix2Text-MFR ONNX formula recognizer.

    Important:
    - PureOnnxMfr is loaded lazily only when a supported formula type is recognized.
    - This module does not mutate page final_text.
    - It only enriches existing formula bbox artifacts with LaTeX metadata.

    This uses the local pure ONNXRuntime MFR model:
    models/pix2text-mfr-1.5
    No heavy framework wrapper package is required.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        config: FormulaExtractionConfig | None = None,
    ) -> None:
        self.device = (device or "cpu").strip().lower()
        self.config = config or FormulaExtractionConfig()
        self._ocr: Any | None = None


    def _ensure_loaded(self) -> None:
        if self._ocr is not None:
            return

        self._ocr = PureOnnxMfr(
            model_dir=_default_formula_model_dir(),
            device=self.device,
        )

        logger.info(
            "Loaded pure ONNX MFR model: device=%s",
            self.device,
        )

    def recognize_crop(self, crop: Image.Image) -> dict[str, Any]:
        self._ensure_loaded()

        assert self._ocr is not None

        return self._ocr.recognize(
            crop,
            max_new_tokens=self.config.max_new_tokens,
        )

    def _maybe_run_self_consensus(
        self,
        *,
        page_image: Image.Image,
        bbox_px: list[float],
        first_latex: str,
        first_confidence: float,
    ) -> dict[str, Any]:
        if not self.config.enable_self_consensus:
            return {}

        crop, crop_bbox_px = _crop_formula_image(
            page_image,
            bbox_px,
            self.config,
            padding_ratio=self.config.consensus_padding_ratio,
        )

        if crop is None:
            return {
                "consensus_used": False,
                "consensus_error": "consensus_crop_empty",
            }

        try:
            second = self.recognize_crop(crop)
        except Exception as exc:
            return {
                "consensus_used": False,
                "consensus_error": f"consensus_mfr_error: {exc}",
            }

        second_latex = str(second.get("latex") or "").strip()
        second_confidence = _safe_float(second.get("confidence"), 0.0)
        similarity = _text_similarity(first_latex, second_latex)

        return {
            "consensus_used": True,
            "consensus_latex": second_latex,
            "consensus_confidence": second_confidence,
            "consensus_similarity": similarity,
            "consensus_crop_bbox_px": crop_bbox_px,
        }

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

        artifacts: list[dict[str, Any]] = []

        for index, formula in enumerate(formula_bboxes or []):
            formula_type = str(formula.get("type") or "unknown").strip()
            formula_id = str(
                formula.get("formula_id") or f"formula_{page_num}_{index}"
            )

            bbox_px = _extract_bbox_px(formula)

            artifact: dict[str, Any] = {
                "formula_id": formula_id,
                "type": formula_type,
                "latex": "",
                "confidence": 0.0,
                "status": "unknown",
                "bbox_px": bbox_px,
                "bbox_pdfium": formula.get("bbox_pdfium"),
                "bbox_plumber": formula.get("bbox_plumber"),
                "crop_bbox_px": None,
                "crop_debug_path": "",
                "layout_score": _safe_float(formula.get("score"), 0.0),
                "fallback_text": "",
                "validation_error": "",
                "consensus_used": False,
                "consensus_latex": "",
                "consensus_confidence": 0.0,
                "consensus_similarity": 0.0,
                "quality_flags": [],
            }

            if bbox_px is None:
                artifact["status"] = "crop_empty"
                artifact["validation_error"] = "missing_or_invalid_bbox_px"
                artifacts.append(artifact)
                continue

            if formula_type not in self.config.enabled_formula_types:
                artifact["status"] = "skipped_formula_type"
                artifact["quality_flags"].append(
                    f"mfr_disabled_for_type:{formula_type}"
                )
                artifacts.append(artifact)
                continue

            crop, crop_bbox_px = _crop_formula_image(
                page_image,
                bbox_px,
                self.config,
            )

            artifact["crop_bbox_px"] = crop_bbox_px

            if formula_type == _FORMULA_NUMBER_TYPE:
                if crop is not None and fallback_ocr is not None:
                    try:
                        segments = fallback_ocr(_as_crop_np(crop))
                        artifact["fallback_text"] = _join_ocr_segments(segments)
                    except Exception as exc:
                        artifact["validation_error"] = (
                            f"formula_number_ocr_failed: {exc}"
                        )

                artifact["status"] = "skipped_formula_number"
                artifacts.append(artifact)
                continue

            if formula_type not in {_DISPLAY_FORMULA_TYPE, _INLINE_FORMULA_TYPE}:
                artifact["quality_flags"].append("unknown_formula_type")

            if crop is None:
                artifact["status"] = "crop_empty"
                artifacts.append(artifact)
                continue

            crop_debug_path = _debug_save_formula_crop(
                crop,
                formula_id=formula_id,
                formula_type=formula_type,
            )
            if crop_debug_path:
                artifact["crop_debug_path"] = crop_debug_path

            try:
                recognized = self.recognize_crop(crop)
            except Exception as exc:
                artifact["status"] = "text_fallback"
                artifact["validation_error"] = f"mfr_error: {exc}"

                if fallback_ocr is not None:
                    try:
                        segments = fallback_ocr(_as_crop_np(crop))
                        artifact["fallback_text"] = _join_ocr_segments(segments)
                    except Exception as ocr_exc:
                        artifact["quality_flags"].append(
                            f"fallback_ocr_failed: {ocr_exc}"
                        )

                artifacts.append(artifact)
                continue

            latex = str(recognized.get("latex") or "").strip()
            confidence = _safe_float(recognized.get("confidence"), 0.0)

            artifact["latex"] = latex
            artifact["confidence"] = confidence

            is_valid, validation_error = validate_latex(latex, formula_type)

            if not is_valid:
                artifact["status"] = "invalid_latex"
                artifact["validation_error"] = validation_error
            elif confidence < self.config.low_confidence:
                artifact["status"] = "text_fallback"
            elif confidence < self.config.accept_confidence:
                artifact["status"] = "low_confidence"
            else:
                artifact["status"] = "accepted"

            if artifact["status"] in {"low_confidence", "invalid_latex", "text_fallback"}:
                consensus = self._maybe_run_self_consensus(
                    page_image=page_image,
                    bbox_px=bbox_px,
                    first_latex=latex,
                    first_confidence=confidence,
                )

                if consensus:
                    artifact.update(consensus)

                    consensus_latex = str(consensus.get("consensus_latex") or "")
                    consensus_confidence = _safe_float(
                        consensus.get("consensus_confidence"), 0.0
                    )
                    consensus_similarity = _safe_float(
                        consensus.get("consensus_similarity"), 0.0
                    )

                    consensus_valid, consensus_error = validate_latex(
                        consensus_latex,
                        formula_type,
                    )

                    if not consensus_valid:
                        artifact["quality_flags"].append(
                            f"consensus_invalid: {consensus_error}"
                        )

                    if (
                        consensus_valid
                        and consensus_similarity >= self.config.consensus_similarity_threshold
                        and consensus_confidence > confidence
                    ):
                        artifact["latex"] = consensus_latex
                        artifact["confidence"] = consensus_confidence

                        if consensus_confidence >= self.config.accept_confidence:
                            artifact["status"] = "accepted"
                            artifact["validation_error"] = ""
                        elif consensus_confidence >= self.config.low_confidence:
                            artifact["status"] = "low_confidence"
                            artifact["validation_error"] = ""

            if artifact["status"] in {"invalid_latex", "text_fallback"} and fallback_ocr is not None:
                try:
                    segments = fallback_ocr(_as_crop_np(crop))
                    artifact["fallback_text"] = _join_ocr_segments(segments)
                except Exception as exc:
                    artifact["quality_flags"].append(f"fallback_ocr_failed: {exc}")

            artifacts.append(artifact)

        return artifacts


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
