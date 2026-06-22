"""Golden-path formula helpers: box selection + output granularity.

Added 2026-06-22. Single source of truth shared by the production extractor and the
benchmark harness, so the formula score can't drift between them. This exact logic reached
display_formula CDM 0.9604 (UniRec-0.1B + targeted inline promotion + seg-split):
  - select_formula_boxes: display boxes + inline boxes tall enough to actually be display
    formulas misclassified as inline (height >= 1.5% of page height), then drop-inner-keep-outer,
    then reading order.
  - seg_split_latex: split a multi-equation recognizer output ("\\[a\\]\\[b\\]") into per-equation
    blocks. UniRec emits per-line granularity; the matcher wants it, so don't collapse it.
Future me: the 1.5% threshold separates genuine embedded inline math (caps ~1.4% of page
height) from misclassified display formulas (>=~1.5%); it's resolution-independent on purpose.
"""
from __future__ import annotations

import re

# Default: promote an inline_formula box only if its height is >= this fraction of
# page height. Genuine embedded inline math caps ~1.4% of page height; display
# formulas misclassified as inline are >=~1.5%. Resolution-independent.
INLINE_MIN_FRAC = 0.015


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def drop_contained_boxes(boxes, ratio=0.7):
    """drop-inner-keep-outer: drop a box that is >= `ratio` inside a strictly larger box."""
    def contained(s, l):
        ix0, iy0 = max(s[0], l[0]), max(s[1], l[1])
        ix1, iy1 = min(s[2], l[2]), min(s[3], l[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        a = _area(s)
        return (inter / a) if a > 0 else 0.0

    return [b for i, b in enumerate(boxes)
            if not any(_area(o) > _area(b) and contained(b, o) >= ratio
                       for j, o in enumerate(boxes) if j != i)]


def display_like_inline(box, page_h, min_frac=INLINE_MIN_FRAC):
    """An inline_formula box tall enough (vs page height) to be a misclassified display formula."""
    return page_h > 0 and (box[3] - box[1]) >= min_frac * page_h


def select_formula_boxes(display_boxes, inline_boxes, page_h, inline_min_frac=INLINE_MIN_FRAC):
    """Golden-path box selection. display_boxes/inline_boxes: (x0,y0,x1,y1) tuples.

    = all display + display-like inline, drop-inner-keep-outer over the combined set,
    sorted top->bottom, left->right (reading order). Returns ordered (x0,y0,x1,y1) tuples.
    """
    selected = list(display_boxes) + [
        b for b in inline_boxes if display_like_inline(b, page_h, inline_min_frac)
    ]
    selected = drop_contained_boxes(selected)
    selected.sort(key=lambda b: (round(b[1] / 10), b[0]))
    return selected


# --- output granularity: split a multi-segment recognizer output into per-equation blocks ---
# A unified recognizer (UniRec) emits a multi-line crop as several delimited segments in ONE
# string (\[a\]\n\[b\]\n\[c\]). MGAM wants per-equation granularity -> split them.
_SEG = re.compile(r"\\\[(.*?)\\\]|\\\((.*?)\\\)|\$\$(.*?)\$\$", re.DOTALL)


def _unwrap_delims(c: str) -> str:
    c = (c or "").strip()
    for lo, hi in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if c.startswith(lo) and c.endswith(hi) and len(c) > len(lo) + len(hi):
            return c[len(lo):-len(hi)].strip()
    return c


def seg_split_latex(latex: str) -> list[str]:
    """Return individual equation latex strings (delimiters stripped). Splits only when the
    output genuinely contains >=2 delimited segments; otherwise returns the single unwrapped body."""
    segs = [next(g for g in m.groups() if g is not None).strip() for m in _SEG.finditer(latex or "")]
    segs = [s for s in segs if s]
    if len(segs) >= 2:
        return segs
    body = _unwrap_delims(latex)
    return [body] if body else []
