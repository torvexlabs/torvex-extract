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
from difflib import SequenceMatcher
from typing import Any

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

FormulaFallbackOcr = Callable[[np.ndarray], list[dict[str, Any]]]

_FORMULA_NUMBER_TYPE = "formula_number"
_DISPLAY_FORMULA_TYPE = "display_formula"
_INLINE_FORMULA_TYPE = "inline_formula"
UNIMERNET_MAX_NEW_TOKENS = 1534

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
    model_name: str = "Sibitorvex/unimernet-tiny-onnx"
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
    split_min_gap_px: int = field(default_factory=lambda: _env_int("TORVEX_FORMULA_SPLIT_MIN_GAP_PX", 12, min_value=4, max_value=200))
    split_gap_ratio: float = field(default_factory=lambda: _env_float("TORVEX_FORMULA_SPLIT_GAP_RATIO", 0.35, min_value=0.05, max_value=2.0))
    split_max_segments: int = field(default_factory=lambda: _env_int("TORVEX_FORMULA_SPLIT_MAX_SEGMENTS", 12, min_value=2, max_value=64))


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

    # A "blank" row tolerates a few dark pixels (scan noise, bleed-through)
    # but NOT a \vdots / \ddots row, which carries visibly more ink.
    blank_row_max_dark = max(1, int(width * 0.004))
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

    # 2026-06-11 threshold calibration (measured on run_042 debug crops):
    # equation boundaries are 16-18px blank runs, scan noise is 1-5px, and
    # matrices produce ZERO blank runs (bracket verticals ink every row, so
    # they self-protect). Ratio is anchored to the SMALLEST substantial
    # content run (~one text line height), NOT the median: fraction-heavy
    # equations produce 120px+ content runs that inflate the median and
    # push the threshold past real boundaries.
    substantial = [e - s for s, e in content_runs if (e - s) >= 8]
    line_h = min(substantial) if substantial else min(e - s for s, e in content_runs)
    min_gap = max(float(config.split_min_gap_px), line_h * config.split_gap_ratio)

    # Interior blank runs tall enough to be equation boundaries.
    cut_rows: list[int] = []
    for s, e, blank in runs:
        if not blank or s == 0 or e == height:
            continue
        if (e - s) >= min_gap:
            cut_rows.append((s + e) // 2)

    if not cut_rows:
        return whole

    if len(cut_rows) + 1 > config.split_max_segments:
        # Too many cuts is a red flag (dense dotted/matrix structure).
        # Refuse to split rather than shred a single tall equation.
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
) -> list[dict[str, Any]]:
    """
    2026-06-11 display-formula splitter entry point.

    Expands merged display_formula bboxes into one bbox per equation.
    Non-display formulas and unsplittable boxes pass through unchanged.
    Children inherit the parent dict (type, score, ...) but get their own
    bbox_px, a derived formula_id "<parent>_s<i>", a "split_from" tag for
    traceability, and None pdfium/plumber boxes (only the parent zone had
    coordinates in those spaces; the exporter keys off bbox_px).
    """
    if not config.enable_display_split:
        return list(formula_bboxes or [])

    expanded: list[dict[str, Any]] = []

    for formula in formula_bboxes or []:
        if str(formula.get("type") or "") != _DISPLAY_FORMULA_TYPE:
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
        for seg_index, (seg_y0, seg_y1) in enumerate(segments):
            child = dict(formula)
            child["bbox_px"] = [bbox_px[0], seg_y0, bbox_px[2], seg_y1]
            child["formula_id"] = f"{parent_id}_s{seg_index}"
            child["bbox_pdfium"] = None
            child["bbox_plumber"] = None
            child["split_from"] = parent_id
            expanded.append(child)

        logger.debug(
            "display-formula split: %s -> %d segments", parent_id, len(segments)
        )

    return expanded


# --- end 2026-06-11 display-formula splitter --------------------------------


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


# --- 2026-06-11 formula salvage ---------------------------------------------
# run_043: 56 invalid_latex formulas (18 truncations at the token cap, 23
# repeated-garbage decode loops, ~15 recognition errors) were dropped by the
# exporter, scoring 0 for their GT formulas. Most are partially correct; a
# repaired prefix scores far better than an empty prediction under both edit
# distance and CDM (a balanced prefix usually still renders). Repair instead
# of reject.

_TRAILING_HALF_COMMAND_RE = re.compile(r"\\[A-Za-z]*$")
_BEGIN_ENV_RE = re.compile(r"\\begin\{([A-Za-z*]+)\}")
_END_ENV_RE = re.compile(r"\\end\{([A-Za-z*]+)\}")


def _cut_repeated_garbage(text: str) -> str:
    """Keep everything up to and including ONE cycle of a detected repeat."""
    match = _REPEATED_GARBAGE_RE.search(text)
    if not match:
        return text
    return text[: match.start() + len(match.group(1))]


def _trim_to_balanced_prefix(text: str) -> str:
    """
    Longest prefix where {}, [] and () are all simultaneously balanced.
    Escaped delimiters (\\{ etc.) are ignored, matching _balanced().
    """
    best_end = 0
    braces = brackets = parens = 0
    escaped = False

    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "{":
            braces += 1
        elif char == "}":
            braces -= 1
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets -= 1
        elif char == "(":
            parens += 1
        elif char == ")":
            parens -= 1

        if braces < 0 or brackets < 0 or parens < 0:
            break

        if braces == 0 and brackets == 0 and parens == 0 and not escaped:
            best_end = index + 1

    return text[:best_end].rstrip()


def _close_open_environments(text: str) -> str:
    """Append \\end{env} for every \\begin{env} left open (LIFO order)."""
    stack: list[str] = []
    events = sorted(
        [(m.start(), "begin", m.group(1)) for m in _BEGIN_ENV_RE.finditer(text)]
        + [(m.start(), "end", m.group(1)) for m in _END_ENV_RE.finditer(text)]
    )
    for _, kind, name in events:
        if kind == "begin":
            stack.append(name)
        elif stack and stack[-1] == name:
            stack.pop()

    for name in reversed(stack):
        text += " \\end{" + name + "}"

    return text


def _salvage_latex(latex: str, formula_type: str) -> str:
    """
    Repair pipeline for LaTeX that failed validate_latex(). Returns the
    repaired string if it now validates, else "" (caller keeps old behavior).
    """
    text = (latex or "").strip()
    if not text:
        return ""

    text = _cut_repeated_garbage(text)
    text = _TRAILING_HALF_COMMAND_RE.sub("", text).rstrip()
    text = _trim_to_balanced_prefix(text)
    if not text:
        return ""

    text = _close_open_environments(text)

    is_valid, _ = validate_latex(text, formula_type)
    if not is_valid:
        return ""

    return text


# --- end 2026-06-11 formula salvage ------------------------------------------


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
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device in {"gpu", "cuda"}
            else ["CPUExecutionProvider"]
        )
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
            "Loaded formula MFR model: engine=%s device=%s",
            engine_name,
            self.device,
        )

    def recognize_crop(self, crop: Image.Image) -> dict[str, Any]:
        self._ensure_loaded()

        assert self._ocr is not None

        result = dict(self._ocr.recognize(crop))
        result["latex"] = str(result.get("latex") or "").strip()
        result["confidence"] = _unimernet_quality_confidence(result)
        return result

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

        # 2026-06-11 display-formula splitter: expand merged PP-DocLayoutV3
        # display boxes into one bbox per equation BEFORE cropping/MFR.
        formula_bboxes = _split_display_formula_bboxes(
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

            if artifact["truncated"]:
                artifact["quality_flags"].append("mfr_truncated")
            elif not artifact["eos_reached"]:
                artifact["quality_flags"].append("mfr_no_eos")

            is_valid, validation_error = validate_latex(latex, formula_type)

            # 2026-06-11 formula salvage: repair instead of reject. A
            # truncated/garbled output that trims to a valid prefix is worth
            # far more than an empty prediction (see patch header).
            if not is_valid:
                salvaged = _salvage_latex(latex, formula_type)
                if salvaged:
                    artifact["latex"] = salvaged
                    artifact["quality_flags"].append(
                        f"salvaged:{validation_error}"
                    )
                    latex = salvaged
                    is_valid, validation_error = True, ""

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
