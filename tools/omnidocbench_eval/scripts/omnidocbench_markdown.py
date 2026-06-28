from __future__ import annotations

import argparse
import json
import os
import re
from html import escape
from pathlib import Path
from typing import Any


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def clean_latex(value: Any) -> str:
    latex = clean_text(value)

    if latex.startswith("$$") and latex.endswith("$$") and len(latex) >= 4:
        latex = latex[2:-2].strip()

    if latex.startswith("\\[") and latex.endswith("\\]"):
        latex = latex[2:-2].strip()

    return latex


def table_rows_to_html(rows: list[list[Any]]) -> str:
    if not rows:
        return ""

    lines = ["<table>"]
    for row in rows:
        cells = []
        for cell in row or []:
            text = clean_text(cell).replace("\n", " ")
            cells.append(f"<td>{escape(text)}</td>")
        lines.append(f"<tr>{''.join(cells)}</tr>")
    lines.append("</table>")

    return "\n".join(lines)


# Seg-split: a unified recognizer (UniRec) emits a multi-line crop as several delimited
# segments in ONE latex string (\[a\]\[b\]\[c\]). Collapsing them into one $$ block makes the
# matcher pair only the first line and drop the rest -> emit one $$ per delimited segment. This
# is the granularity fix the formula score depends on. Kept self-contained here (mirroring
# converter.py) so the benchmark harness stays stable.
_SEG = re.compile(r"\\\[(.*?)\\\]|\\\((.*?)\\\)|\$\$(.*?)\$\$", re.DOTALL)


def _split_segments(content: str) -> list[str]:
    segs = [
        next(g for g in m.groups() if g is not None).strip()
        for m in _SEG.finditer(content or "")
    ]
    segs = [s for s in segs if s]
    return segs if len(segs) >= 2 else []


def formula_emittable(formula: dict[str, Any]) -> bool:
    if str(formula.get("type") or "") != "display_formula":
        return False

    if str(formula.get("status") or "") not in {"accepted", "low_confidence"}:
        return False

    return bool(clean_text(formula.get("latex")))


def formula_markdown_blocks(formula: dict[str, Any]) -> list[str]:
    """One $$ block per equation. Applies seg-split so a multi-segment recognizer
    output becomes per-equation blocks (the granularity the matcher wants)."""
    if not formula_emittable(formula):
        return []

    raw = clean_text(formula.get("latex"))
    bodies = _split_segments(raw) or [clean_latex(raw)]

    return [f"$$\n{body}\n$$" for body in bodies if body]


def bbox(value: Any) -> list[float] | None:
    if value is None:
        return None

    try:
        box = [float(v) for v in value]
    except Exception:
        return None

    if len(box) != 4:
        return None

    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None

    return [x0, y0, x1, y1]


def bbox_iou(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 0.0

    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def table_bbox(table: dict[str, Any]) -> list[float] | None:
    return (
        bbox(table.get("bbox_px"))
        or bbox(table.get("bbox_plumber"))
        or bbox(table.get("bbox_pdfium"))
    )


def zone_bbox(zone: dict[str, Any]) -> list[float] | None:
    return (
        bbox(zone.get("bbox_px"))
        or bbox(zone.get("bbox_plumber"))
        or bbox(zone.get("bbox_pdfium"))
    )


def formula_bbox(formula: dict[str, Any]) -> list[float] | None:
    return (
        bbox(formula.get("bbox_px"))
        or bbox(formula.get("crop_bbox_px"))
        or bbox(formula.get("bbox_plumber"))
        or bbox(formula.get("bbox_pdfium"))
    )


def best_table_for_zone(
    zone: dict[str, Any],
    tables: list[dict[str, Any]],
    used: set[int],
) -> int | None:
    zbox = zone_bbox(zone)
    best_index = None
    best_score = 0.0

    for index, table in enumerate(tables):
        if index in used:
            continue

        score = bbox_iou(zbox, table_bbox(table))
        if score > best_score:
            best_score = score
            best_index = index

    if best_index is None or best_score < 0.05:
        return None

    return best_index


def formula_indexes_for_zone(
    zone: dict[str, Any],
    formulas: list[dict[str, Any]],
    used: set[int],
) -> list[int]:
    zbox = zone_bbox(zone)
    if zbox is None:
        return []

    matched: list[tuple[float, int]] = []

    for index, formula in enumerate(formulas):
        if index in used:
            continue

        if not formula_emittable(formula):
            continue

        fbox = formula_bbox(formula)
        if fbox is None:
            continue

        ix0 = max(zbox[0], fbox[0])
        iy0 = max(zbox[1], fbox[1])
        ix1 = min(zbox[2], fbox[2])
        iy1 = min(zbox[3], fbox[3])

        if ix1 <= ix0 or iy1 <= iy0:
            continue

        inter = (ix1 - ix0) * (iy1 - iy0)
        formula_area = (fbox[2] - fbox[0]) * (fbox[3] - fbox[1])

        inside_enough = formula_area > 0 and (inter / formula_area) >= 0.5
        iou_enough = bbox_iou(zbox, fbox) >= 0.05

        if inside_enough or iou_enough:
            matched.append((fbox[1], index))

    matched.sort()
    return [index for _, index in matched]


def fallback_page_to_markdown(page: dict[str, Any]) -> str:
    blocks: list[str] = []

    text = clean_text(page.get("text") or page.get("final_text"))
    if text:
        blocks.append(text)

    for formula in page.get("formulas") or []:
        blocks.extend(formula_markdown_blocks(formula))

    for table in page.get("tables") or []:
        table_html = table_rows_to_html(table.get("rows") or [])
        if table_html:
            blocks.append(table_html)

    return "\n\n".join(blocks).strip() + "\n"


def normalized_page_to_markdown(page: dict[str, Any]) -> str:
    tables = list(page.get("tables") or [])
    formulas = list(page.get("formulas") or [])
    zones = list(page.get("layout_zones") or page.get("zones") or [])

    if not zones:
        return fallback_page_to_markdown(page)

    blocks: list[str] = []
    used_tables: set[int] = set()
    used_formulas: set[int] = set()

    skip_zone_types = {
        "image",
        "chart",
        "figure",
        "header_image",
        "footer_image",
        "seal",
        "formula_number",
    }

    for zone in zones:
        zone_type = str(zone.get("type") or "")

        if zone_type in skip_zone_types:
            continue

        if zone_type == "table":
            table_index = best_table_for_zone(zone, tables, used_tables)
            if table_index is not None:
                table_html = table_rows_to_html(tables[table_index].get("rows") or [])
                if table_html:
                    blocks.append(table_html)
                    used_tables.add(table_index)
            continue

        if zone_type in {"display_formula", "inline_formula"}:
            # display_formula zones, plus inline_formula zones that carry a recognized
            # (display-like, promoted) formula -> emit as display $$ blocks in reading
            # order. Plain inline zones with no recognized formula emit nothing.
            for formula_index in formula_indexes_for_zone(zone, formulas, used_formulas):
                blocks.extend(formula_markdown_blocks(formulas[formula_index]))
                used_formulas.add(formula_index)
            continue

        zone_text = clean_text(
            zone.get("text")
            or zone.get("zone_text")
            or zone.get("content")
            or ""
        )
        if zone_type in {"title", "paragraph_title"}:
            blocks.append("# " + zone_text.strip("#").strip())
        else:
            blocks.append(zone_text)

    if os.getenv("TORVEX_ODB_EXPORT_UNMATCHED_FORMULAS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        for formula_index, formula in enumerate(formulas):
            if formula_index in used_formulas:
                continue
            blocks.extend(formula_markdown_blocks(formula))

    for table_index, table in enumerate(tables):
        if table_index in used_tables:
            continue
        table_html = table_rows_to_html(table.get("rows") or [])
        if table_html:
            blocks.append(table_html)

    if not blocks:
        return fallback_page_to_markdown(page)

    return "\n\n".join(block.strip() for block in blocks if block.strip()).strip() + "\n"


def load_first_page(normalized_json_path: Path) -> dict[str, Any]:
    payload = json.loads(normalized_json_path.read_text(encoding="utf-8"))

    pages = payload.get("pages") or []
    if not pages:
        raise ValueError(f"No pages found in normalized JSON: {normalized_json_path}")

    return pages[0]


def prediction_name_from_gt_image_path(image_path: str) -> str:
    return Path(Path(image_path).name).with_suffix(".md").name


def export_markdown_prediction(
    *,
    normalized_json_path: Path,
    output_dir: Path,
    gt_image_path: str,
) -> Path:
    page = load_first_page(normalized_json_path)
    markdown = normalized_page_to_markdown(page)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / prediction_name_from_gt_image_path(gt_image_path)
    output_path.write_text(markdown, encoding="utf-8")

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert one Torvex normalized JSON page to OmniDocBench prediction markdown."
    )
    parser.add_argument("--normalized-json", required=True, type=Path)
    parser.add_argument("--gt-image-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)

    args = parser.parse_args()

    output_path = export_markdown_prediction(
        normalized_json_path=args.normalized_json,
        gt_image_path=args.gt_image_path,
        output_dir=args.output_dir,
    )

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
