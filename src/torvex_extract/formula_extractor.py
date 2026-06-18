from __future__ import annotations

import gc
import importlib
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

logger = logging.getLogger(__name__)

FormulaFallbackOcr = Callable[[np.ndarray], list[dict[str, Any]]]

_FORMULA_NUMBER_TYPE = "formula_number"
_DISPLAY_FORMULA_TYPE = "display_formula"
_INLINE_FORMULA_TYPE = "inline_formula"
UNIMERNET_MAX_NEW_TOKENS = 1534

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
    model_name: str = "Sibitorvex/unimernet-tiny-onnx"
    enabled_formula_types: tuple[str, ...] = field(default_factory=_default_enabled_formula_types)
    padding_ratio: float = field(default_factory=lambda: _env_float("TORVEX_FORMULA_PADDING_RATIO", 0.01, min_value=0.0, max_value=0.20))
    min_padding_px: int = field(default_factory=lambda: _env_int("TORVEX_FORMULA_MIN_PADDING_PX", 2, min_value=0, max_value=64))
    white_border_px: int = field(default_factory=lambda: _env_int("TORVEX_FORMULA_WHITE_BORDER_PX", 8, min_value=0, max_value=64))
    min_crop_width_px: int = 8
    min_crop_height_px: int = 8
    blank_dark_ratio_threshold: float = 0.0005
    enable_fallback_ocr: bool = field(default_factory=lambda: _env_bool("TORVEX_FORMULA_FALLBACK_OCR", False))
    # UniMERNet's decoder can need a much larger budget than the old Pix2Text
    # path. Treat early non-EOS stops as a quality signal, not a success.
    max_new_tokens: int = field(
        default_factory=lambda: _env_int(
            "TORVEX_FORMULA_MAX_NEW_TOKENS",
            UNIMERNET_MAX_NEW_TOKENS,
            min_value=16,
            max_value=UNIMERNET_MAX_NEW_TOKENS,
        )
    )
    max_batch_size: int = field(
        default_factory=lambda: _env_int(
            "TORVEX_FORMULA_MAX_BATCH_SIZE",
            8,
            min_value=1,
            max_value=32,
        )
    )
    sort_by_size: bool = field(
        default_factory=lambda: _env_bool("TORVEX_FORMULA_SORT_BY_SIZE", True)
    )

    # 2026-06-11 display-formula splitter:
    # PP-DocLayoutV3 frequently emits ONE display_formula box covering a
    # vertical stack of SEPARATE equations (measured on OmniDocBench
    # equation_hard, 30 pages: 58 of 155 display zones contained 2-10 GT
    # equations each). Merged crops force the MFR to decode multiple
    # equations in one pass, blowing the token budget -> truncation ->
    # invalid_latex -> every covered GT equation scores zero. We split each
    # display_formula bbox at full-width horizontal white gaps BEFORE
    # cropping/MFR. A gap is a boundary only if it is >= split_min_gap_px
    # AND >= split_gap_ratio * the smallest substantial content-run height
    # (~one text line). Matrices self-protect: bracket verticals ink every
    # row, so they produce no blank runs at all (verified on debug crops).
    enable_display_split: bool = field(default_factory=lambda: _env_bool("TORVEX_FORMULA_SPLIT", True))
    split_min_gap_px: int = field(default_factory=lambda: _env_int("TORVEX_FORMULA_SPLIT_MIN_GAP_PX", 5, min_value=2, max_value=200))
    split_gap_ratio: float = field(default_factory=lambda: _env_float("TORVEX_FORMULA_SPLIT_GAP_RATIO", 0.25, min_value=0.05, max_value=2.0))
    split_min_content_height_px: int = field(default_factory=lambda: _env_int("TORVEX_FORMULA_SPLIT_MIN_CONTENT_PX", 20, min_value=8, max_value=200))
    split_max_segments: int = field(default_factory=lambda: _env_int("TORVEX_FORMULA_SPLIT_MAX_SEGMENTS", 12, min_value=2, max_value=64))
    # Force-bisect fallback: for bboxes taller than this threshold where the
    # main pass finds no qualifying gap (>= split_min_gap_px), retry with any
    # blank run (even 1px) that has split_min_content_height_px on both sides.
    # Catches equations that are nearly touching (1-4px gap) inside a merged bbox.
    # Set 0 to disable.
    split_force_bisect_height_px: int = field(default_factory=lambda: _env_int("TORVEX_FORMULA_SPLIT_FORCE_BISECT_PX", 150, min_value=0, max_value=1000))

    trust_model_output: bool = field(default_factory=lambda: _env_bool("TORVEX_FORMULA_TRUST_MODEL", True))


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


def _horizontal_white_gap_segments(
    page_image: Image.Image,
    bbox_px: list[float],
    config: FormulaExtractionConfig,
) -> list[tuple[float, float]]:
    """
    Return [(y0, y1), ...] page-pixel row spans, one per detected equation.

    Returns a single span (= no split) whenever anything looks risky:
    degenerate crop, single content block, no qualifying gap, or more cuts
    than split_max_segments (a wall of tiny gaps usually means a matrix or
    dotted structure we must not cut).
    """
    whole = [(float(bbox_px[1]), float(bbox_px[3]))]

    x0 = max(0, int(math.floor(bbox_px[0])))
    y0 = max(0, int(math.floor(bbox_px[1])))
    x1 = min(page_image.size[0], int(math.ceil(bbox_px[2])))
    y1 = min(page_image.size[1], int(math.ceil(bbox_px[3])))

    if (x1 - x0) < config.min_crop_width_px or (y1 - y0) < config.min_crop_height_px:
        return whole

    gray = np.asarray(page_image.crop((x0, y0, x1, y1)).convert("L"))
    if gray.size == 0:
        return whole

    height, width = gray.shape
    dark_per_row = (gray < 245).sum(axis=1)

    # Tolerate up to 0.8% of row width as dark pixels (scan noise,
    # bleed-through, light ruling lines). The old 0.4% was too strict and
    # missed real gaps that had a few stray dark pixels.
    blank_row_max_dark = max(2, int(width * 0.008))
    is_blank = dark_per_row <= blank_row_max_dark

    # Run-length encode rows into alternating blank/content runs.
    runs: list[tuple[int, int, bool]] = []
    start = 0
    for row in range(1, height + 1):
        if row == height or bool(is_blank[row]) != bool(is_blank[start]):
            runs.append((start, row, bool(is_blank[start])))
            start = row

    content_runs = [(s, e) for s, e, blank in runs if not blank]
    if len(content_runs) <= 1:
        return whole

    substantial = [e - s for s, e in content_runs if (e - s) >= 8]
    line_h = min(substantial) if substantial else min(e - s for s, e in content_runs)
    min_gap = max(float(config.split_min_gap_px), line_h * config.split_gap_ratio)

    # Build candidate cuts: interior blank runs tall enough to be equation
    # boundaries AND where both adjacent content runs look like real
    # equations (not internal whitespace within a tall fraction/matrix).
    min_content_h = float(config.split_min_content_height_px)
    cut_rows: list[int] = []
    for run_idx, (s, e, blank) in enumerate(runs):
        if not blank or s == 0 or e == height:
            continue
        if (e - s) < min_gap:
            continue

        # Find the content run heights on each side of this gap.
        above_h = 0.0
        for prev_idx in range(run_idx - 1, -1, -1):
            ps, pe, pb = runs[prev_idx]
            if not pb:
                above_h += pe - ps
            else:
                break

        below_h = 0.0
        for next_idx in range(run_idx + 1, len(runs)):
            ns, ne, nb = runs[next_idx]
            if not nb:
                below_h += ne - ns
            else:
                break

        if above_h >= min_content_h and below_h >= min_content_h:
            cut_rows.append((s + e) // 2)

    if not cut_rows:
        # Force-bisect fallback: for tall bboxes where the main pass found no
        # qualifying gap, retry with no minimum gap size. Any blank run (even
        # 1px) that has split_min_content_height_px of content on both sides
        # is accepted as a cut. This catches equations that are nearly touching
        # (1-4px gap) inside a merged detection bbox.
        force_h = config.split_force_bisect_height_px
        if force_h > 0 and height >= force_h:
            for run_idx, (s, e, blank) in enumerate(runs):
                if not blank or s == 0 or e == height:
                    continue
                above_h = 0.0
                for prev_idx in range(run_idx - 1, -1, -1):
                    ps, pe, pb = runs[prev_idx]
                    if not pb:
                        above_h += pe - ps
                    else:
                        break
                below_h = 0.0
                for next_idx in range(run_idx + 1, len(runs)):
                    ns, ne, nb = runs[next_idx]
                    if not nb:
                        below_h += ne - ns
                    else:
                        break
                if above_h >= min_content_h and below_h >= min_content_h:
                    cut_rows.append((s + e) // 2)
        if not cut_rows:
            return whole

    if len(cut_rows) + 1 > config.split_max_segments:
        return whole

    # Build segments between cuts, then tighten each to its own content
    # extents (+2px pad) so MFR crops stay snug.
    segments: list[tuple[float, float]] = []
    boundaries = [0] + cut_rows + [height]
    for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
        seg_blank = is_blank[seg_start:seg_end]
        rows_with_ink = [
            seg_start + offset
            for offset, blank in enumerate(seg_blank)
            if not blank
        ]
        if not rows_with_ink:
            continue
        top = max(seg_start, rows_with_ink[0] - 2)
        bottom = min(seg_end, rows_with_ink[-1] + 3)
        if (bottom - top) < config.min_crop_height_px:
            continue
        segments.append((float(y0 + top), float(y0 + bottom)))

    if len(segments) <= 1:
        return whole

    return segments


def _split_display_formula_bboxes(
    page_image: Image.Image,
    formula_bboxes: list[dict[str, Any]],
    config: FormulaExtractionConfig,
    *,
    _depth: int = 0,
) -> list[dict[str, Any]]:
    """
    Expands merged display_formula bboxes into one bbox per equation.

    Runs two passes: the main pixel-gap pass plus the force-bisect fallback
    (for tall bboxes with tiny gaps). After the first split, applies one
    recursive pass to child segments so that e.g. a 380px bbox split into two
    170px children can each be split again if they still contain 2 equations.
    """
    if not config.enable_display_split:
        return list(formula_bboxes or [])

    expanded: list[dict[str, Any]] = []

    for formula in formula_bboxes or []:
        if str(formula.get("type") or "") != _DISPLAY_FORMULA_TYPE:
            expanded.append(formula)
            continue

        if bool(formula.get("preserve_display_group")):
            expanded.append(formula)
            continue

        bbox_px = _extract_bbox_px(formula)
        if bbox_px is None:
            expanded.append(formula)
            continue

        segments = _horizontal_white_gap_segments(page_image, bbox_px, config)
        if len(segments) <= 1:
            expanded.append(formula)
            continue

        parent_id = str(formula.get("formula_id") or "formula")
        children: list[dict[str, Any]] = []
        for seg_index, (seg_y0, seg_y1) in enumerate(segments):
            child = dict(formula)
            child["bbox_px"] = [bbox_px[0], seg_y0, bbox_px[2], seg_y1]
            child["formula_id"] = f"{parent_id}_s{seg_index}"
            child["bbox_pdfium"] = None
            child["bbox_plumber"] = None
            child["split_from"] = parent_id
            children.append(child)

        # One recursive pass: children that are still tall enough to contain
        # multiple equations get split again (catches cases like a 380px bbox
        # initially split into two 170px halves, each still holding 2 GTs).
        if _depth == 0:
            children = _split_display_formula_bboxes(
                page_image, children, config, _depth=1
            )

        expanded.extend(children)
        logger.debug(
            "display-formula split: %s -> %d segments (depth=%d)",
            parent_id, len(children), _depth,
        )

    return expanded


def _bbox_area(bbox_px: list[float] | None) -> float:
    if bbox_px is None:
        return 0.0

    x0, y0, x1, y1 = bbox_px
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_intersection_area(
    left: list[float] | None,
    right: list[float] | None,
) -> float:
    if left is None or right is None:
        return 0.0

    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right

    ix0 = max(lx0, rx0)
    iy0 = max(ly0, ry0)
    ix1 = min(lx1, rx1)
    iy1 = min(ly1, ry1)

    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def _bbox_iou(left: list[float] | None, right: list[float] | None) -> float:
    left_area = _bbox_area(left)
    right_area = _bbox_area(right)
    if left_area <= 0.0 or right_area <= 0.0:
        return 0.0

    intersection = _bbox_intersection_area(left, right)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0

    return intersection / union


def _bbox_ioa(inner: list[float] | None, outer: list[float] | None) -> float:
    inner_area = _bbox_area(inner)
    if inner_area <= 0.0:
        return 0.0

    return _bbox_intersection_area(inner, outer) / inner_area


def _is_split_formula(formula: dict[str, Any]) -> bool:
    if formula.get("split_from"):
        return True

    formula_id = str(formula.get("formula_id") or "")
    return bool(re.search(r"_s\d+$", formula_id))


def _prefer_formula_candidate(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    left_split = _is_split_formula(left)
    right_split = _is_split_formula(right)

    if left_split != right_split:
        return right if left_split else left

    left_score = _safe_float(left.get("score"), 0.0)
    right_score = _safe_float(right.get("score"), 0.0)
    if abs(left_score - right_score) > 0.05:
        return left if left_score > right_score else right

    left_area = _bbox_area(_extract_bbox_px(left))
    right_area = _bbox_area(_extract_bbox_px(right))
    return left if left_area <= right_area else right


def _contained_display_children(
    formula_bboxes: list[dict[str, Any]],
    parent_index: int,
    display_indexes: list[int],
    parent_bbox: list[float],
    parent_area: float,
) -> list[int]:
    children: list[int] = []

    for child_index in display_indexes:
        if child_index == parent_index:
            continue

        child_bbox = _extract_bbox_px(formula_bboxes[child_index])
        child_area = _bbox_area(child_bbox)
        if child_area <= 0.0 or child_area > parent_area * 0.75:
            continue

        if _bbox_ioa(child_bbox, parent_bbox) >= 0.85:
            children.append(child_index)

    return children


def _should_preserve_display_parent(
    formula_bboxes: list[dict[str, Any]],
    parent_index: int,
    child_indexes: list[int],
) -> bool:
    if len(child_indexes) < 2:
        return False

    parent_score = _safe_float(formula_bboxes[parent_index].get("score"), 0.0)
    child_scores = [
        _safe_float(formula_bboxes[child_index].get("score"), 0.0)
        for child_index in child_indexes
    ]
    best_child_score = max(child_scores, default=0.0)

    return parent_score >= 0.65 and (parent_score - best_child_score) >= 0.15


def _suppress_duplicate_display_formula_bboxes(
    formula_bboxes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Remove display-formula parent/child duplicates before MFR.

    PP-DocLayout can emit a large display_formula group and separate child
    display_formula boxes for the same visual region. The splitter can also
    create children that overlap original child detections. IoU misses these
    cases because child boxes are intentionally much smaller than parents, so
    this uses directional containment (IoA) for parent-child suppression and
    IoU for near-identical duplicates.
    """
    if len(formula_bboxes) <= 1:
        return list(formula_bboxes or [])

    formula_bboxes = [dict(formula) for formula in formula_bboxes or []]

    display_indexes = [
        index
        for index, formula in enumerate(formula_bboxes)
        if str(formula.get("type") or "") == _DISPLAY_FORMULA_TYPE
        and _extract_bbox_px(formula) is not None
    ]
    suppressed: set[int] = set()

    # Drop parent/group display boxes when they contain multiple smaller
    # display formulas unless the parent is clearly the stronger detection.
    # Some DocLayout outputs include a high-confidence full multi-line formula
    # plus low-confidence row fragments; OmniDocBench often scores the full
    # parent as one isolated equation, so preserving that parent avoids
    # shredding the crop before UniMERNet sees it.
    for parent_index in display_indexes:
        parent_bbox = _extract_bbox_px(formula_bboxes[parent_index])
        parent_area = _bbox_area(parent_bbox)
        if parent_area <= 0.0:
            continue

        contained_children = _contained_display_children(
            formula_bboxes,
            parent_index,
            display_indexes,
            parent_bbox,
            parent_area,
        )

        if _should_preserve_display_parent(
            formula_bboxes,
            parent_index,
            contained_children,
        ):
            formula_bboxes[parent_index]["preserve_display_group"] = True
        elif len(contained_children) >= 2:
            suppressed.add(parent_index)

    # Resolve near-identical or split-vs-original duplicates among survivors.
    for left_pos, left_index in enumerate(display_indexes):
        if left_index in suppressed:
            continue

        left_bbox = _extract_bbox_px(formula_bboxes[left_index])
        left_area = _bbox_area(left_bbox)
        if left_area <= 0.0:
            continue

        for right_index in display_indexes[left_pos + 1 :]:
            if right_index in suppressed:
                continue

            right_bbox = _extract_bbox_px(formula_bboxes[right_index])
            right_area = _bbox_area(right_bbox)
            if right_area <= 0.0:
                continue

            smaller_area = min(left_area, right_area)
            larger_area = max(left_area, right_area)
            area_ratio = smaller_area / larger_area if larger_area > 0.0 else 0.0
            smaller_inside_larger = max(
                _bbox_ioa(left_bbox, right_bbox),
                _bbox_ioa(right_bbox, left_bbox),
            )
            is_duplicate = (
                _bbox_iou(left_bbox, right_bbox) >= 0.75
                or (smaller_inside_larger >= 0.85 and area_ratio >= 0.40)
            )
            if not is_duplicate:
                continue

            preferred = _prefer_formula_candidate(
                formula_bboxes[left_index],
                formula_bboxes[right_index],
            )
            if preferred is formula_bboxes[left_index]:
                suppressed.add(right_index)
            else:
                suppressed.add(left_index)
                break

    if not suppressed:
        return list(formula_bboxes)

    for index in sorted(suppressed):
        formula = formula_bboxes[index]
        logger.debug(
            "suppressed duplicate display formula candidate: %s",
            formula.get("formula_id", index),
        )

    return [
        formula
        for index, formula in enumerate(formula_bboxes)
        if index not in suppressed
    ]


# --- end 2026-06-11 display-formula splitter --------------------------------


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


def _prepare_formula_bboxes_for_mfr(
    page_image: Image.Image,
    formula_bboxes: list[dict[str, Any]],
    config: FormulaExtractionConfig,
) -> list[dict[str, Any]]:
    formula_bboxes = _suppress_duplicate_display_formula_bboxes(
        list(formula_bboxes or [])
    )
    formula_bboxes = _split_display_formula_bboxes(
        page_image,
        formula_bboxes,
        config,
    )
    return _suppress_duplicate_display_formula_bboxes(formula_bboxes)


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


def _default_unimernet_onnx_root() -> Path:
    env_value = os.getenv("TORVEX_UNIMERNET_ONNX_MODEL_DIR")
    if env_value:
        return Path(env_value)

    model_rel = Path("models") / "unimernet-tiny-onnx"
    candidates: list[Path] = [
        Path.cwd() / model_rel,
    ]

    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / model_rel)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return model_rel


def _load_unimernet_runtime() -> type[Any]:
    try:
        module = importlib.import_module("pure_onnx_unimernet")
    except ImportError as exc:
        raise RuntimeError(
            "UniMERNet ONNX runtime is not installed. Install it into the "
            "active Torvex environment with: uv pip install --no-deps "
            "\"git+https://github.com/torvexlabs/unimernet-onnx.git\""
        ) from exc

    runtime = getattr(module, "OnnxUnimerNet", None)
    if runtime is None:
        raise RuntimeError(
            "Installed pure_onnx_unimernet module does not expose OnnxUnimerNet."
        )

    return runtime


def _unimernet_quality_confidence(result: dict[str, Any]) -> float:
    latex = str(result.get("latex") or "").strip()
    if not latex:
        return 0.0

    if bool(result.get("truncated")):
        return 0.25

    if result.get("eos_reached") is False:
        return 0.50

    return 0.90


class FormulaMfrExtractor:
    """
    Optional UniMERNet ONNX formula recognizer.

    Important:
    - UniMERNet is loaded lazily only when a supported formula type is recognized.
    - This module does not mutate page final_text.
    - It only enriches existing formula bbox artifacts with LaTeX metadata.
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
        self._recognize_max_new_tokens = self.config.max_new_tokens


    def _ensure_loaded(self) -> None:
        if self._ocr is not None:
            return

        engine_name = (
            os.getenv("TORVEX_FORMULA_ENGINE", "unimernet_onnx")
            .strip()
            .lower()
        )
        if engine_name and engine_name not in {"unimernet", "unimernet_onnx"}:
            logger.warning(
                "TORVEX_FORMULA_ENGINE=%s is ignored; UniMERNet ONNX is the "
                "only formula engine.",
                engine_name,
            )
        engine_name = "unimernet_onnx"

        self._recognize_max_new_tokens = self.config.max_new_tokens
        model_root = _default_unimernet_onnx_root()
        artifacts_dir = Path(
            os.getenv(
                "TORVEX_UNIMERNET_ONNX_ARTIFACTS_DIR",
                str(model_root / "artifacts"),
            )
        )
        tokenizer_path = Path(
            os.getenv(
                "TORVEX_UNIMERNET_TOKENIZER_DIR",
                str(model_root / "models" / "unimernet_tiny"),
            )
        )
        # 2026-06-15: use the shared ONNX provider helper so UniMERNet gets
        # the same Windows CUDA DLL preload path as layout/table/OCR sessions.
        # Before this, formula-only GPU loading could miss cublas/cuDNN DLLs
        # unless another engine happened to warm CUDA first.
        providers = select_onnx_providers(self.device)
        use_iobinding = _env_bool(
            "TORVEX_UNIMERNET_IO_BINDING",
            self.device in {"gpu", "cuda"},
        )
        logger.info(
            "Formula engine: unimernet_onnx artifacts=%s tokenizer=%s providers=%s "
            "max_new_tokens=%s io_binding=%s",
            artifacts_dir,
            tokenizer_path,
            providers,
            self._recognize_max_new_tokens,
            use_iobinding,
        )

        runtime = _load_unimernet_runtime()
        self._ocr = runtime(
            artifacts_dir=artifacts_dir,
            tokenizer_path=tokenizer_path,
            providers=providers,
            max_new_tokens=self._recognize_max_new_tokens,
            use_iobinding=use_iobinding,
        )

        logger.info(
            "Loaded formula MFR model: engine=%s device=%s trust_model=%s",
            engine_name,
            self.device,
            self.config.trust_model_output,
        )

    def recognize_crop(self, crop: Image.Image) -> dict[str, Any]:
        self._ensure_loaded()

        assert self._ocr is not None

        result = dict(self._ocr.recognize(crop))
        result["latex"] = str(result.get("latex") or "").strip()
        result["confidence"] = _unimernet_quality_confidence(result)
        return result

    def preflight(self) -> None:
        self._ensure_loaded()

    def recognize_crops(self, crops: list[Image.Image]) -> list[dict[str, Any]]:
        self._ensure_loaded()

        assert self._ocr is not None

        if not crops:
            return []

        recognize_batch = getattr(self._ocr, "recognize_batch", None)
        if callable(recognize_batch):
            raw_results = list(
                recognize_batch(
                    crops,
                    max_batch_size=self.config.max_batch_size,
                    sort_by_size=self.config.sort_by_size,
                )
            )
        else:
            raw_results = [self._ocr.recognize(crop) for crop in crops]

        if len(raw_results) != len(crops):
            raise RuntimeError(
                "UniMERNet batch result count mismatch: "
                f"expected {len(crops)}, got {len(raw_results)}"
            )

        results: list[dict[str, Any]] = []
        for raw_result in raw_results:
            result = dict(raw_result)
            result["latex"] = str(result.get("latex") or "").strip()
            result["confidence"] = _unimernet_quality_confidence(result)
            results.append(result)

        return results

    def _apply_recognition_result(
        self,
        *,
        artifact: dict[str, Any],
        recognized: dict[str, Any],
    ) -> None:
        latex = str(recognized.get("latex") or "").strip()
        confidence = _safe_float(recognized.get("confidence"), 0.0)

        artifact["latex"] = latex
        artifact["token_count"] = int(
            _safe_float(recognized.get("token_count"), 0.0)
        )
        artifact["last_token"] = recognized.get("last_token")
        artifact["eos_reached"] = bool(recognized.get("eos_reached"))
        artifact["truncated"] = bool(recognized.get("truncated"))
        artifact["mfr_elapsed_ms"] = _safe_float(recognized.get("elapsed_ms"), 0.0)
        artifact["mfr_ms_per_token"] = _safe_float(
            recognized.get("ms_per_token"),
            0.0,
        )
        artifact["mfr_active_providers"] = recognized.get("active_providers") or {}
        artifact["mfr_io_binding"] = bool(recognized.get("io_binding"))
        artifact["mfr_batch_size"] = int(
            _safe_float(recognized.get("batch_size"), 1.0)
        )
        artifact["mfr_batch_group_index"] = int(
            _safe_float(recognized.get("batch_group_index"), 0.0)
        )

        # Trust the recognizer output: accept whatever LaTeX it produced.
        # trust_model_output only controls the confidence floor.
        artifact["status"] = "accepted"
        artifact["confidence"] = (
            max(confidence, 0.90) if self.config.trust_model_output else confidence
        )

        if artifact["truncated"]:
            artifact["quality_flags"].append("mfr_truncated")
        elif not artifact["eos_reached"]:
            artifact["quality_flags"].append("mfr_no_eos")

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

        if self.config.trust_model_output or not self.config.enable_fallback_ocr:
            fallback_ocr = None

        artifacts: list[dict[str, Any]] = []
        pending_mfr: list[tuple[dict[str, Any], Image.Image, str, list[float]]] = []

        formula_bboxes = _prepare_formula_bboxes_for_mfr(
            page_image,
            list(formula_bboxes or []),
            self.config,
        )

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
                "preserve_display_group": bool(formula.get("preserve_display_group")),
                "fallback_text": "",
                "validation_error": "",
                "consensus_used": False,
                "consensus_latex": "",
                "consensus_confidence": 0.0,
                "consensus_similarity": 0.0,
                "token_count": 0,
                "last_token": None,
                "eos_reached": False,
                "truncated": False,
                "mfr_elapsed_ms": 0.0,
                "mfr_ms_per_token": 0.0,
                "mfr_active_providers": {},
                "mfr_io_binding": False,
                "quality_flags": [],
            }

            artifacts.append(artifact)

            if bbox_px is None:
                artifact["status"] = "crop_empty"
                artifact["validation_error"] = "missing_or_invalid_bbox_px"
                continue

            if formula_type not in self.config.enabled_formula_types:
                artifact["status"] = "skipped_formula_type"
                artifact["quality_flags"].append(
                    f"mfr_disabled_for_type:{formula_type}"
                )
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
                continue

            if formula_type not in {_DISPLAY_FORMULA_TYPE, _INLINE_FORMULA_TYPE}:
                artifact["quality_flags"].append("unknown_formula_type")

            if crop is None:
                artifact["status"] = "crop_empty"
                continue

            crop_debug_path = _debug_save_formula_crop(
                crop,
                formula_id=formula_id,
                formula_type=formula_type,
            )
            if crop_debug_path:
                artifact["crop_debug_path"] = crop_debug_path

            pending_mfr.append((artifact, crop, formula_type, bbox_px))

        if pending_mfr:
            crops = [crop for _, crop, _, _ in pending_mfr]
            try:
                recognized_items = self.recognize_crops(crops)
            except Exception as exc:
                for artifact, crop, _, _ in pending_mfr:
                    artifact["status"] = "text_fallback"
                    artifact["validation_error"] = f"mfr_error: {exc}"
                    if fallback_ocr is None:
                        continue
                    try:
                        segments = fallback_ocr(_as_crop_np(crop))
                        artifact["fallback_text"] = _join_ocr_segments(segments)
                    except Exception as ocr_exc:
                        artifact["quality_flags"].append(
                            f"fallback_ocr_failed: {ocr_exc}"
                        )
            else:
                for (
                    artifact,
                    crop,
                    formula_type,
                    bbox_px,
                ), recognized in zip(pending_mfr, recognized_items):
                    self._apply_recognition_result(
                        artifact=artifact,
                        recognized=recognized,
                    )

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
