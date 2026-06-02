"""
Fast PDF page extractor using pypdfium2.

Returns one Phase 1 page dict per page.
Runs fast routing signals:
- bordered table detection
- OCR need classification
- optional render for OCR / spotlight
"""
import time

import logging

import pdfplumber

import numpy as np

import statistics

import pypdfium2
from pypdfium2 import raw as pdfium_c

from torvex_extract.table_extractor import (
    detect_table_fast,
    extract_tables_pdfplumber,
)

from torvex_extract.ocr_engine import (
    sort_and_join_ocr_segments,
    ocr_page,
    assign_ocr_segments_to_bboxes,
)

from torvex_extract.visual_zoning import (
    RENDER_DPI,
    engine,
    attach_zone_bboxes,
    classify_digital_page_zones,
    process_layout_zones,
    is_tier1_duplicate,
    crop_image,
    TRIGGER_ZONE_TYPES,
    SPOTLIGHT_TYPES,
)


from torvex_extract.table_structure import (
    extract_scanned_table,
    extract_table_cell_pdfplumber,
    extract_table_explicit_pdfplumber,
)

logger = logging.getLogger(__name__)

LINE_Y_SNAP = 3.0

# 2026-05-26:
# Global scanned-page OCR can leave small DocLayout zones empty because OCR words
# are assigned exclusively to the best matching zone.
# Do not downgrade a scanned page for empty header/footer/tiny text fragments.
SCANNED_MAJOR_TEXT_ZONE_TYPES = frozenset({"text", "content", "abstract"})
SCANNED_MAJOR_EMPTY_ZONE_MIN_AREA_RATIO = 0.015


_VALID_TEXT_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "$.,%-+()/: "
    "\n\r\t"
)


def _is_non_textual(text: str) -> bool:
    """
    Return True when pypdfium extracted text looks like garbage.

    This is only an OCR routing signal.
    It is not a quality scorer.
    """
    stripped = text.strip()

    if not stripped:
        return True

    valid_count = sum(1 for char in stripped if char in _VALID_TEXT_CHARS)
    valid_ratio = valid_count / len(stripped)

    return valid_ratio < 0.30


def classify_ocr_need(probe_text: str) -> tuple[bool, str]:
    """
    Decide whether this page needs OCR.

    probe_text comes from pypdfium full-page text extraction.
    It is only a routing signal.
    It must never become final_text.
    """
    stripped = probe_text.strip()

    if not stripped:
        return True, "empty"

    if _is_non_textual(stripped):
        return True, "low_ratio"

    return False, "clean"


def _is_image_dominated(page_obj, threshold: float = 0.85) -> bool:
    """
    Return True if image objects cover most of the PDF page.

    This is not part of the OCR gate in v9.
    Keep it for later quality scoring / diagnostics.
    """
    page_width = page_obj.get_width()
    page_height = page_obj.get_height()
    page_area = page_width * page_height

    if page_area <= 0:
        return False

    image_area = 0.0

    try:
        for obj in page_obj.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]):
            left, bottom, right, top = obj.get_bounds()

            obj_width = abs(right - left)
            obj_height = abs(top - bottom)

            image_area += obj_width * obj_height
    except Exception:
        return False

    image_ratio = image_area / page_area

    return image_ratio >= threshold


def _normalize_bounded_pdf_text(text: str) -> str:
    """
    Normalize pypdfium bounded text without destroying real content.

    Debug note â€” 2026-05-25:
    Apple 2024 10-K smoke showed char-by-char reconstruction broke clean
    digital words:
        "Company" -> "Com an"
        "Report"  -> "Re ort"
        "Nasdaq"  -> "Nasda"

    Cause:
        get_charbox() + manual x-gap spacing is too fragile for PDF glyph
        positioning. Some glyph boxes are missing/tight/oddly spaced, so
        reconstructing words from individual characters corrupts readable text.

    Fix:
        Use pypdfium's bounded text extraction as the primary path.
        Only do light whitespace cleanup here.
    """
    if not text:
        return ""

    lines = []

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        # Collapse repeated spaces inside a line, but keep line boundaries.
        cleaned = " ".join(raw_line.split())

        if cleaned:
            lines.append(cleaned)

    return "\n".join(lines).strip()


def extract_text_in_bbox_sorted(
    textpage,
    bbox_pdfium: tuple[float, float, float, float],
) -> str:
    """
    Extract digital PDF text inside one DocLayout SAFE zone.

    Important:
        This function is only for digital SAFE zones.
        Do not call it for:
        - table zones
        - chart/image/spotlight zones
        - scanned OCR pages

    Debug note â€” 2026-05-25:
        The earlier implementation rebuilt text character-by-character using
        get_text_range(i, 1) and get_charbox(i). That looked precise, but it
        corrupted clean Apple 10-K prose because PDF glyph boxes are not a
        reliable word reconstruction contract.

        For clean digital PDFs, pypdfium's own bounded extraction preserves
        text order and spacing better than our manual char sorter.

    Why bounded extraction is okay now:
        DocLayout already gives us small SAFE zone bboxes.
        So we do not need to visually re-sort every character inside the zone.
        The safer contract is:
            DocLayout handles zone order.
            pypdfium handles text inside the zone.
    """
    if textpage is None:
        return ""

    left, bottom, right, top = bbox_pdfium

    if right <= left or top <= bottom:
        logger.warning(
            "extract_text_in_bbox_sorted skipped invalid bbox: %s",
            bbox_pdfium,
        )
        return ""

    try:
        # Primary path: preserve pypdfium's native text reconstruction.
        # This avoids the Apple 10-K broken-word bug caused by manual char sorting.
        text = textpage.get_text_bounded(
            left=left,
            bottom=bottom,
            right=right,
            top=top,
        )
        return _normalize_bounded_pdf_text(text)

    except TypeError:
        # Some pypdfium2 versions may not accept keyword args.
        # Keep positional fallback so future version drift does not break extraction.
        try:
            text = textpage.get_text_bounded(left, bottom, right, top)
            return _normalize_bounded_pdf_text(text)
        except Exception as exc:
            logger.warning("bounded text extraction failed: %s", exc)
            return ""

    except Exception as exc:
        logger.warning("bounded text extraction failed: %s", exc)
        return ""

# 2026-05-26:
# Debugged bordered-table duplication in final_text.
# pdfplumber already stores bordered tables as structured artifacts in page["tables"],
# but DocLayout can still label the same region as SAFE text/content.
# These helpers let SAFE-zone extraction exclude only the table characters instead
# of skipping the whole SAFE zone and losing surrounding prose.
def _bbox_intersects_pdfium(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> bool:
    """
    Return True if two bbox_pdfium boxes overlap.

    bbox_pdfium format:
        [left, bottom, right, top]
    """
    ax0, ab, ax1, at = box_a
    bx0, bb, bx1, bt = box_b

    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(at, bt) - max(ab, bb))

    return inter_w > 0.0 and inter_h > 0.0


def _char_center_inside_pdfium_bbox(
    char_box: tuple[float, float, float, float],
    bbox_pdfium: tuple[float, float, float, float],
) -> bool:
    """
    Check whether a character center is inside a bbox_pdfium box.

    Using center-point prevents border padding from deleting nearby prose.
    """
    left, bottom, right, top = char_box
    box_left, box_bottom, box_right, box_top = bbox_pdfium

    cx = (left + right) / 2.0
    cy = (bottom + top) / 2.0

    return (
        box_left <= cx <= box_right
        and box_bottom <= cy <= box_top
    )


def extract_text_in_bbox_excluding_tables(
    textpage,
    bbox_pdfium: tuple[float, float, float, float],
    exclude_bboxes_pdfium: list[tuple[float, float, float, float]],
) -> str:
    """
    Extract SAFE-zone digital text while excluding bordered-table regions.

    Use only when a SAFE zone overlaps one or more pdfplumber bordered-table
    bboxes. Normal SAFE zones should still use extract_text_in_bbox_sorted()
    because pypdfium's native bounded extraction preserves cleaner spacing.

    This prevents table text from leaking into final_text after the same table
    has already been stored as a structured artifact.
    """
    if textpage is None:
        return ""

    if not exclude_bboxes_pdfium:
        return extract_text_in_bbox_sorted(
            textpage=textpage,
            bbox_pdfium=bbox_pdfium,
        )

    overlapping_excludes = [
        exclude_bbox
        for exclude_bbox in exclude_bboxes_pdfium
        if _bbox_intersects_pdfium(bbox_pdfium, exclude_bbox)
    ]
    # Fast path: no table overlap, so keep the cleaner native bounded extraction.
    if not overlapping_excludes:
        return extract_text_in_bbox_sorted(
            textpage=textpage,
            bbox_pdfium=bbox_pdfium,
        )

    zone_left, zone_bottom, zone_right, zone_top = bbox_pdfium

    if zone_right <= zone_left or zone_top <= zone_bottom:
        logger.warning(
            "extract_text_in_bbox_excluding_tables skipped invalid bbox: %s",
            bbox_pdfium,
        )
        return ""

    chars = []

    try:
        char_count = textpage.count_chars()

        for index in range(char_count):
            char_text = textpage.get_text_range(index, 1)

            if not char_text.strip() and char_text != " ":
                continue

            rect = textpage.get_charbox(index, loose=False)

            if rect is None:
                continue

            char_left, char_bottom, char_right, char_top = rect
            char_box = (char_left, char_bottom, char_right, char_top)

            if not _char_center_inside_pdfium_bbox(char_box, bbox_pdfium):
                continue

            if any(
                _char_center_inside_pdfium_bbox(char_box, exclude_bbox)
                for exclude_bbox in overlapping_excludes
            ):
                continue

            chars.append(
                {
                    "text": char_text,
                    "x0": char_left,
                    "x1": char_right,
                    "bottom": char_bottom,
                    "top": char_top,
                }
            )

    except Exception as exc:
        logger.warning("table-excluding text extraction failed: %s", exc)
        return ""

    if not chars:
        return ""

    # PDFium coords are bottom-origin. Higher top value = visually higher on page.
    chars.sort(key=lambda item: (-round(item["top"], 1), item["x0"]))

    lines: list[list[dict]] = []
    current_line: list[dict] = [chars[0]]
    current_top = chars[0]["top"]

    for char in chars[1:]:
        if abs(char["top"] - current_top) <= 2.0:
            current_line.append(char)
        else:
            lines.append(sorted(current_line, key=lambda item: item["x0"]))
            current_line = [char]
            current_top = char["top"]

    lines.append(sorted(current_line, key=lambda item: item["x0"]))

    output_lines = []

    for line in lines:
        line_text = ""
        previous_x1 = line[0]["x0"]

        for char in line:
            if (char["x0"] - previous_x1) > 2.0:
                line_text += " "

            line_text += char["text"]
            previous_x1 = char["x1"]

        cleaned = " ".join(line_text.split())

        if cleaned:
            output_lines.append(cleaned)

    return "\n".join(output_lines).strip()


def _dedupe_nearby_final_text_lines(
    text: str,
    lookback: int = 3,
) -> str:
    """
    Remove nearby duplicate lines from final_text.

    2026-05-27:
    DocLayout can label the same visual text bbox as multiple SAFE types:
        paragraph_title / doc_title / text

    Example:
        Page 17 in Doc 3 repeated:
            SUPPLEMENTAL INFORMATION
            SUPPLEMENTAL INFORMATION
            SUPPLEMENTAL INFORMATION

    Why here, not in zone suppression:
        Cross-type zone overlaps can be valid.
        Tables/images/text may overlap legitimately.
        Final text cleanup is safer than dropping zones globally.

    Scope:
        Only removes exact duplicate lines seen within a small nearby window.
        Does not touch table artifacts.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    kept: list[str] = []

    for line in lines:
        recent = kept[-lookback:]

        if line in recent:
            continue

        kept.append(line)

    return "\n".join(kept)


def assemble_final_text(zones: list[dict], page: dict) -> str:
    """
    Build final page prose from processed SAFE zones only.

    Tables are not included here.
    Charts/images/figures are not included here.
    unsafe_for_text zones are skipped.
    ocr_probe_text is never used here.
    """
    parts = []

    for zone in zones:
        zone_type = zone.get("type", "unknown")

        if zone.get("unsafe_for_text"):
            continue

        if zone_type in SPOTLIGHT_TYPES:
            continue

        if zone_type in TRIGGER_ZONE_TYPES:
            # 2026-05-27:
            # Doc 8 IBM degraded/corrupt scanned 10-K exposed a table-only page failure:
            # DocLayout marked the whole page as a TRIGGER/table zone.
            # TATR failed to build a trustworthy structured table.
            # OCR had some readable text, but final_text stayed empty because TRIGGER
            # zones are normally excluded from prose assembly.
            #
            # Keep normal table behavior unchanged:
            #   - successful table artifact -> do not duplicate table text in final_text
            #   - failed scanned table artifact + OCR fallback -> preserve degraded text
            #     so the page does not become fatal empty output.
            if zone.get("degraded_table_text_fallback"):
                text = zone.get("zone_text", "")
                if text.strip():
                    parts.append(text.strip())

            continue

        text = zone.get("zone_text", "")

        if text.strip():
            parts.append(text.strip())

    return _dedupe_nearby_final_text_lines("\n".join(parts))


def detect_interleaving(text: str) -> tuple[bool, float]:
    """
    Detect likely bad reading order in final_text.

    Debug note â€” 2026-05-26:
    Apple 10-K smoke showed clean pages like page 4 being flagged with
    reading_order_warning even though final_text was readable.

    Root cause:
    The old detector used only line-length variation:
        stdev(line_lengths) / mean_length > 0.85

    That is too sensitive for SEC filings because normal pages contain:
    - short headings
    - long wrapped paragraph lines
    - footer/page-number lines
    - occasional short continuation lines

    New rule:
    Ignore tiny heading/footer lines and only warn when there is strong evidence
    of mixed short/long body lines. This keeps the warning conservative.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) < 8:
        return False, 0.0

    # Ignore tiny lines because headings/footers/page numbers create false
    # variation but do not prove broken reading order.
    body_lines = [line for line in lines if len(line) >= 20]

    if len(body_lines) < 6:
        return False, 0.0

    line_lengths = [len(line) for line in body_lines]
    mean_length = statistics.mean(line_lengths)

    if mean_length <= 0:
        return False, 0.0

    try:
        variation_score = statistics.stdev(line_lengths) / mean_length
    except statistics.StatisticsError:
        return False, 0.0

    short_body_rate = sum(
        1 for length in line_lengths
        if length < mean_length * 0.45
    ) / len(line_lengths)

    long_body_rate = sum(
        1 for length in line_lengths
        if length > mean_length * 1.55
    ) / len(line_lengths)

    # Conservative warning:
    # variation alone is not enough. We require both short and long body-line
    # clusters, which is more representative of true interleaving/column bleed.
    is_interleaved = (
        variation_score > 1.15
        and short_body_rate >= 0.25
        and long_body_rate >= 0.15
    )

    return is_interleaved, round(variation_score, 3)


def _set_layout_grade(page: dict) -> None:
    """
    Set page-level layout_grade after zone processing.

    Phase 1 allowed values:
    POOR | FAIR | GOOD | EXCELLENT
    """
    zones = page.get("zones") or []

    if not zones:
        page["layout_grade"] = "POOR"
        return

    mean_score = sum(float(zone.get("score", 0.0)) for zone in zones) / len(zones)

    has_failed_extraction = any(
        bool(zone.get("extraction_failed"))
        for zone in zones
    )

    if mean_score >= 0.75 and not has_failed_extraction:
        page["layout_grade"] = "EXCELLENT"
    elif mean_score >= 0.55 and not has_failed_extraction:
        page["layout_grade"] = "GOOD"
    elif mean_score >= 0.35:
        page["layout_grade"] = "FAIR"
    else:
        page["layout_grade"] = "POOR"


def _full_page_sorted_text(filepath: str, page_idx: int, page: dict) -> str:
    """
    Fallback for digital pages when PP-DocLayout returns zero zones.

    This opens the PDF again, extracts full-page text using character geometry,
    then closes every pypdfium handle safely.

    Use only as degraded fallback.
    Normal extraction should use SAFE zone bboxes, not full-page text.
    """
    pdf = None
    page_obj = None
    textpage = None

    try:
        pdf = pypdfium2.PdfDocument(filepath)
        page_obj = pdf[page_idx]
        textpage = page_obj.get_textpage()

        bbox_pdfium = (
            0.0,
            0.0,
            page["page_width"],
            page["page_height"],
        )

        return extract_text_in_bbox_sorted(textpage, bbox_pdfium)

    except Exception as exc:
        logger.warning("_full_page_sorted_text failed on page %d: %s", page_idx, exc)
        return ""

    finally:
        if textpage is not None:
            textpage.close()

        if page_obj is not None:
            page_obj.close()

        if pdf is not None:
            pdf.close()



def _full_page_sorted_text_excluding_bboxes(
    filepath: str,
    page_idx: int,
    page: dict,
    exclude_bboxes_plumber: list[list[float]],
) -> str:
    """
    Fallback for digital zero-zone pages when bordered tables were already extracted.

    Extracts full-page digital text but skips characters inside excluded table bboxes.
    This prevents duplicated table text from leaking into final_text.
    """
    pdf = None
    page_obj = None
    textpage = None

    try:
        pdf = pypdfium2.PdfDocument(filepath)
        page_obj = pdf[page_idx]
        textpage = page_obj.get_textpage()

        if textpage is None:
            return ""

        excluded_pdfium = []

        for bbox_plumber in exclude_bboxes_plumber:
            x0, top, x1, bottom = bbox_plumber
            excluded_pdfium.append(
                (
                    x0,
                    page["effective_page_height_pt"] - bottom,
                    x1,
                    page["effective_page_height_pt"] - top,
                )
            )

        chars = []
        char_count = textpage.count_chars()

        for i in range(char_count):
            char_text = textpage.get_text_range(i, 1)

            if not char_text.strip() and char_text != " ":
                continue

            rect = textpage.get_charbox(i, loose=False)

            if rect is None:
                continue

            char_left, char_bottom, char_right, char_top = rect
            char_x_center = (char_left + char_right) / 2
            char_y_center = (char_bottom + char_top) / 2

            inside_excluded = any(
                left <= char_x_center <= right and bottom <= char_y_center <= top
                for left, bottom, right, top in excluded_pdfium
            )

            if inside_excluded:
                continue

            chars.append(
                {
                    "text": char_text,
                    "x0": min(char_left, char_right),
                    "x1": max(char_left, char_right),
                    "y_center": char_y_center,
                }
            )

        if not chars:
            return ""

        chars.sort(key=lambda item: (-item["y_center"], item["x0"]))

        lines: list[list[dict]] = []
        current_line = [chars[0]]

        for char in chars[1:]:
            if abs(char["y_center"] - current_line[0]["y_center"]) <= LINE_Y_SNAP:
                current_line.append(char)
            else:
                lines.append(sorted(current_line, key=lambda item: item["x0"]))
                current_line = [char]

        lines.append(sorted(current_line, key=lambda item: item["x0"]))

        output_lines = []

        for line in lines:
            line_text = ""
            previous_x1 = line[0]["x0"]

            for char in line:
                if (char["x0"] - previous_x1) > 2.0:
                    line_text += " "

                line_text += char["text"]
                previous_x1 = char["x1"]

            if line_text.strip():
                output_lines.append(line_text)

        return "\n".join(output_lines).strip()

    except Exception as exc:
        logger.warning(
            "_full_page_sorted_text_excluding_bboxes failed on page %d: %s",
            page_idx,
            exc,
        )
        return ""

    finally:
        if textpage is not None:
            textpage.close()

        if page_obj is not None:
            page_obj.close()

        if pdf is not None:
            pdf.close()


def _process_single_page(
    pdf,
    plumber_pdf,
    page_idx: int,
    filepath: str,
    is_tagged: bool,
    result_errors: list,
) -> dict:
    """
    Process one PDF page end-to-end.

    This function:
    1. Opens one pypdfium page.
    2. Uses pypdfium full-page text only as OCR probe.
    3. Runs fast bordered-table detection before closing the page.
    4. Renders the page once at 200 DPI.
    5. Runs visual zoning.
    6. Routes page to digital or scanned processing.
    7. Clears image/img_np before returning.

    It returns one Phase 1 page dict.
    """
    page_start = time.perf_counter()
    timings_ms: dict[str, float] = {}

    def _record_timing(stage: str, started_at: float) -> None:
        timings_ms[stage] = timings_ms.get(stage, 0.0) + (
            time.perf_counter() - started_at
        ) * 1000.0

    page_obj = None
    textpage_probe = None

    probe_text = ""
    needs_ocr = True
    ocr_reason = "empty"
    has_bordered_table = False

    image = None
    img_np = None

    page_width = 0.0
    page_height = 0.0
    effective_page_width_pt = 612.0
    effective_page_height_pt = 792.0

    try:
        t0 = time.perf_counter()
        page_obj = pdf[page_idx]

        page_width = page_obj.get_width()
        page_height = page_obj.get_height()
        _record_timing("page_open", t0)

        t0 = time.perf_counter()
        textpage_probe = page_obj.get_textpage()

        try:
            probe_text = textpage_probe.get_text_range().strip()
        finally:
            textpage_probe.close()
            textpage_probe = None

        _record_timing("probe_text", t0)

        t0 = time.perf_counter()
        needs_ocr, ocr_reason = classify_ocr_need(probe_text)
        _record_timing("ocr_classify", t0)

        if not needs_ocr:
            t0 = time.perf_counter()
            has_bordered_table = detect_table_fast(page_obj)
            _record_timing("detect_table_fast", t0)

        t0 = time.perf_counter()
        scale = RENDER_DPI / 72.0
        image = page_obj.render(scale=scale).to_pil().convert("RGB")
        img_np = np.array(image)

        img_h_px, img_w_px = img_np.shape[:2]
        effective_page_width_pt = img_w_px * (72.0 / RENDER_DPI)
        effective_page_height_pt = img_h_px * (72.0 / RENDER_DPI)
        _record_timing("render", t0)

    except Exception as exc:
        result_errors.append(
            {
                "stage": "page_open_probe_render",
                "page": page_idx,
                "error": str(exc),
            }
        )

    finally:
        if textpage_probe is not None:
            textpage_probe.close()

        if page_obj is not None:
            page_obj.close()

    page = {
        "page_num": page_idx,
        "is_tagged": is_tagged,
        "needs_ocr": needs_ocr,
        "ocr_reason": ocr_reason,
        "final_text": "",
        "page_width": page_width,
        "page_height": page_height,
        "effective_page_width_pt": effective_page_width_pt,
        "effective_page_height_pt": effective_page_height_pt,
        "image": image,
        "img_np": img_np,
        "has_bordered_table": has_bordered_table,
        "zones": [],
        "tier1_bboxes": [],
        "spotlight_bboxes": [],
        "tables": [],
        "metadata": {
            "ocr_probe_len": len(probe_text),
            "ocr_reason": ocr_reason,
            "timings_ms": timings_ms,
        },
        "layout_grade": "",
        "page_class": "unknown",
    }

    try:
        if img_np is None:
            page["metadata"].setdefault("warnings", []).append(
                "Page render failed; no image available for layout detection."
            )
            page["layout_grade"] = "POOR"
            return page

        t0 = time.perf_counter()
        raw_zones = engine.detect_layout(img_np)
        _record_timing("doclayout", t0)

        t0 = time.perf_counter()
        page_class = classify_digital_page_zones(raw_zones)
        page["page_class"] = page_class

        zones = attach_zone_bboxes(raw_zones, page)
        page["zones"] = zones
        _record_timing("zone_postprocess", t0)

        if not needs_ocr and page["has_bordered_table"]:
            try:
                plumber_page = plumber_pdf.pages[page_idx]

                t0 = time.perf_counter()
                artifacts, tier1_bboxes = extract_tables_pdfplumber(
                    plumber_page=plumber_page,
                    page_num=page_idx,
                    effective_page_height_pt=page["effective_page_height_pt"],
                    effective_page_width_pt=page["effective_page_width_pt"],
                    zones=zones,
                )
                _record_timing("pdfplumber_bordered", t0)

                page["tables"].extend(artifacts)
                page["tier1_bboxes"] = tier1_bboxes

            except Exception as exc:
                page["metadata"].setdefault("warnings", []).append(
                    f"bordered table extraction failed: {exc}"
                )

        if page_class == "zero_zones":
            page["metadata"]["quality_penalty"] = 0.25
            page["metadata"]["warning"] = "PP-DocLayout returned zero zones."
            page["layout_grade"] = "POOR"

            if not needs_ocr:
                if page["tier1_bboxes"]:
                    t0 = time.perf_counter()
                    page["final_text"] = _full_page_sorted_text_excluding_bboxes(
                        filepath=filepath,
                        page_idx=page_idx,
                        page=page,
                        exclude_bboxes_plumber=page["tier1_bboxes"],
                    )
                    page["metadata"]["zero_zone_fallback"] = (
                        "full_page_sorted_pypdfium_excluding_tier1_tables"
                    )
                    _record_timing("zero_zone_fallback", t0)

                else:
                    t0 = time.perf_counter()
                    page["final_text"] = _full_page_sorted_text(filepath, page_idx, page)
                    page["metadata"]["zero_zone_fallback"] = "full_page_sorted_pypdfium"
                    _record_timing("zero_zone_fallback", t0)

            else:
                t0 = time.perf_counter()
                segments = engine.ocr_image(img_np)
                page["final_text"] = sort_and_join_ocr_segments(segments)
                page["metadata"]["zero_zone_fallback"] = f"full_page_{engine.ocr_backend_name()}"
                _record_timing("zero_zone_fallback", t0)

                _mark_empty_scanned_page_discarded(
                    page=page,
                    img_np=img_np,
                    page_ocr_segments=segments,
                )

        elif not needs_ocr:
            t0 = time.perf_counter()
            _process_digital_page(page, filepath, plumber_pdf, is_tagged)
            _record_timing("digital_page_total", t0)

        else:
            t0 = time.perf_counter()
            _process_scanned_page(page, is_tagged)
            _record_timing("scanned_page_total", t0)

    except Exception as exc:
        result_errors.append(
            {
                "stage": "page_processing",
                "page": page_idx,
                "error": str(exc),
            }
        )

    finally:
        _record_timing("page_total", page_start)

        page["image"] = None
        page["img_np"] = None

    return page


TABLE_ARTIFACT_DUP_IOA_THRESHOLD = 0.85


def _table_kind(table: dict) -> str:
    """
    Return normalized table kind.

    Debug note â€” 2026-05-25:
    Apple 10-K smoke showed same physical table stored twice:
        - pdfplumber_bordered artifact
        - TATR borderless artifact

    Smoke caught this as:
        same_table_bordered_borderless_duplicate

    We only dedupe across bordered vs borderless here.
    Same-source table splitting is left alone because financial statements can
    legitimately contain multiple nearby tables.
    """
    kind = str(table.get("kind") or "").strip().lower()

    if kind:
        return kind

    source = str(table.get("source") or "").lower()

    if "bordered" in source:
        return "bordered"

    if "tatr" in source or "cell_crop" in source:
        return "borderless"

    return "unknown"


def _table_bbox_plumber(table: dict) -> list[float] | None:
    bbox = table.get("bbox_plumber")

    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    try:
        return [float(v) for v in bbox]
    except Exception:
        return None


def _plumber_bbox_area(bbox: list[float]) -> float:
    """
    bbox_plumber format:
        [x0, top, x1, bottom]
    """
    x0, top, x1, bottom = bbox
    return max(0.0, x1 - x0) * max(0.0, bottom - top)


def _symmetric_ioa_plumber(
    bbox_a: list[float],
    bbox_b: list[float],
) -> float:
    """
    Symmetric Intersection-over-Area for table artifact bboxes.

    Why symmetric:
    In Apple 10-K page 3, TATR produced one large table region while
    pdfplumber produced smaller bordered fragments inside it. Normal IoU is too
    low for that case, but symmetric IoA correctly says:
        "one table bbox fully contains the other."

    This is post-artifact dedup only.
    Do not replace the earlier tier1 routing guard with this.
    """
    ax0, at, ax1, ab = bbox_a
    bx0, bt, bx1, bb = bbox_b

    ix0 = max(ax0, bx0)
    it = max(at, bt)
    ix1 = min(ax1, bx1)
    ib = min(ab, bb)

    inter_area = max(0.0, ix1 - ix0) * max(0.0, ib - it)

    area_a = _plumber_bbox_area(bbox_a)
    area_b = _plumber_bbox_area(bbox_b)

    if area_a <= 0 or area_b <= 0:
        return 0.0

    return max(inter_area / area_a, inter_area / area_b)


def _table_has_warning(table: dict) -> bool:
    warnings = table.get("warnings") or []
    return bool(warnings)


def _table_shape(table: dict) -> tuple[int, int, float]:
    """
    Return:
        row_count, max_col_count, non_empty_rate
    """
    rows = table.get("rows") or []

    if not rows:
        return 0, 0, 0.0

    row_count = len(rows)
    max_cols = max((len(row) for row in rows), default=0)

    total = 0
    non_empty = 0

    for row in rows:
        for cell in row:
            total += 1
            if str(cell or "").strip():
                non_empty += 1

    non_empty_rate = non_empty / total if total else 0.0

    return row_count, max_cols, non_empty_rate


def _table_quality_score(table: dict) -> float:
    """
    Generic table artifact preference score.

    This is not Apple-specific.

    Preference rules:
    - reject empty / 1-row / 1-column table-shaped junk
    - penalize warning artifacts
    - penalize very flat wide artifacts like 2 rows x 19 cols
    - prefer explicit TATR grids slightly over cell-crop fallback
    - prefer denser/non-empty tables
    """
    rows, cols, non_empty_rate = _table_shape(table)

    score = 0.0

    score += non_empty_rate * 2.0
    score += min(rows, 30) * 0.05
    score += min(cols, 12) * 0.03

    source = str(table.get("source") or "").lower()

    if source == "tatr_explicit_pdfplumber":
        score += 1.0
    elif source == "pdfplumber_bordered":
        score += 0.8
    elif source == "tatr_cell_pdfplumber":
        score += 0.2

    if rows < 2 or cols < 2:
        score -= 10.0

    if _table_has_warning(table):
        score -= 5.0

    # Usually means pdfplumber flattened a visual row into many tiny columns.
    if rows <= 2 and cols >= 12:
        score -= 2.0

    return score


def _is_cross_method_duplicate(
    table_a: dict,
    table_b: dict,
    ioa_threshold: float = TABLE_ARTIFACT_DUP_IOA_THRESHOLD,
) -> bool:
    """
    True when two artifacts likely represent the same physical table.

    Important:
    Only dedupe bordered-vs-borderless pairs.
    Do not dedupe bordered-vs-bordered or borderless-vs-borderless here because
    adjacent financial statement tables can be close and same-source splitting
    may be legitimate.

    2026-05-26:
    Final artifact dedupe intentionally uses symmetric IoA only.
    Do NOT add the tier1 max-area-ratio guard here.

    Why:
    pdfplumber can split one bordered visual table into smaller fragments while
    TATR captures the full table region. Those smaller bordered fragments are
    real duplicates of the larger borderless artifact and must be removed.

    Smoke proof:
    Adding max_area_ratio=2.0 here caused 13 same_table_bordered_borderless_duplicate
    failures on Apple 10-K pages 3, 23, 27, 28, 41, 42 and 50.
    """
    kind_a = _table_kind(table_a)
    kind_b = _table_kind(table_b)

    if {kind_a, kind_b} != {"bordered", "borderless"}:
        return False

    bbox_a = _table_bbox_plumber(table_a)
    bbox_b = _table_bbox_plumber(table_b)

    if bbox_a is None or bbox_b is None:
        return False

    return _symmetric_ioa_plumber(bbox_a, bbox_b) >= ioa_threshold


def _dedupe_page_table_artifacts(page: dict) -> list[dict]:
    """
    Remove duplicate bordered/borderless table artifacts after all extraction
    paths have run.

    Why this exists:
    The pre-TATR tier1 guard prevents many duplicates, but it cannot catch all
    cases. Apple 10-K smoke showed pages where:
        - pdfplumber extracted a bordered fragment
        - TATR extracted a larger overlapping table region
        - both artifacts survived into page["tables"]

    That is bad because downstream citations/chunks would see the same table
    twice. This function performs a final page-level artifact dedup.

    Selection rule:
    Keep the artifact with the better generic quality score.
    Record what was removed in page["metadata"]["deduped_tables"] for debugging.
    """
    tables = page.get("tables") or []

    if len(tables) <= 1:
        return tables

    ranked = sorted(
        enumerate(tables),
        key=lambda item: (
            _table_quality_score(item[1]),
            -item[0],  # stable-ish tie-break: earlier table wins
        ),
        reverse=True,
    )

    kept: list[dict] = []
    removed: list[dict] = []

    for original_index, candidate in ranked:
        duplicate_of = None

        for kept_table in kept:
            if _is_cross_method_duplicate(candidate, kept_table):
                duplicate_of = kept_table
                break

        if duplicate_of is None:
            kept.append(candidate)
            continue

        removed.append(
            {
                "removed_table_id": candidate.get("table_id"),
                "kept_table_id": duplicate_of.get("table_id"),
                "removed_source": candidate.get("source"),
                "kept_source": duplicate_of.get("source"),
                "reason": "cross_method_same_table_duplicate",
            }
        )

    if removed:
        page["metadata"].setdefault("deduped_tables", []).extend(removed)
        logger.info(
            "page %s: deduped %d duplicate table artifact(s)",
            page.get("page_num"),
            len(removed),
        )

    return kept


def _extract_pdf_text_for_failed_digital_table(
    textpage,
    zone: dict,
) -> str:
    """
    Extract plain PDF text from a digital table/TRIGGER bbox.

    2026-05-27:
    Facebook 2017 10-K pages 54, 56, 57 exposed this gap:
        - bordered signal existed
        - DocLayout detected real financial table zones
        - pdfplumber bordered extraction produced no artifact
        - TATR / explicit-grid / cell-crop failed or was rejected
        - table values were lost because TRIGGER zones are skipped from final_text

    This is NOT normal prose extraction.
    This is only for preserving failed table content as a degraded table artifact.
    """
    if textpage is None:
        return ""

    bbox_pdfium = zone.get("bbox_pdfium")
    if not bbox_pdfium or len(bbox_pdfium) != 4:
        return ""

    left, bottom, right, top = [float(v) for v in bbox_pdfium]

    if right <= left or top <= bottom:
        return ""

    try:
        text = textpage.get_text_bounded(
            left=left,
            bottom=bottom,
            right=right,
            top=top,
        )
        return _normalize_bounded_pdf_text(text)

    except TypeError:
        try:
            text = textpage.get_text_bounded(left, bottom, right, top)
            return _normalize_bounded_pdf_text(text)
        except Exception as exc:
            logger.warning("digital degraded table text fallback failed: %s", exc)
            return ""

    except Exception as exc:
        logger.warning("digital degraded table text fallback failed: %s", exc)
        return ""


def _make_degraded_digital_table_artifact(
    *,
    table_id: str,
    zone: dict,
    raw_text: str,
    reason: str,
) -> dict:
    """
    Build a non-structured table artifact.

    Why this exists:
        We should not fake rows/columns when reconstruction failed.
        But we also should not lose visible financial table values.

    Contract:
        - structured=False
        - rows=[]
        - raw_text contains the preserved table text
        - downstream chunking must treat this as low-confidence table content,
          not normal prose
    """
    return {
        "table_id": table_id,
        "kind": "degraded_table_text",
        "method": "pdf_text_fallback",
        "source": "digital_table_text_fallback",
        "bbox_plumber": zone.get("bbox_plumber"),
        "bbox_pdfium": zone.get("bbox_pdfium"),
        "rows": [],
        "raw_text": raw_text,
        "structured": False,
        "warnings": [
            f"structured digital table extraction failed; preserved raw table text: {reason}"
        ],
    }


def _process_digital_page(
    page: dict,
    filepath: str,
    plumber_pdf,
    is_tagged: bool,
) -> None:
    """
    Process one digital page after visual zones are already detected.

    Digital path:
    - SAFE zones      -> pypdfium bbox text extraction
    - TRIGGER zones   -> TATR table extraction
    - SPOTLIGHT zones -> bbox only, no text

    Mutates page in place.
    """

    img_np = page.get("img_np")

    if img_np is None:
        page["metadata"].setdefault("warnings", []).append(
            "Digital page processing skipped because img_np is missing."
        )
        page["layout_grade"] = "POOR"
        return

    plumber_page = plumber_pdf.pages[page["page_num"]]

    pdf = None
    page_obj = None
    textpage = None

    try:
        pdf = pypdfium2.PdfDocument(filepath)
        page_obj = pdf[page["page_num"]]
        textpage = page_obj.get_textpage()

        zones = page.get("zones", [])

        zones = process_layout_zones(
            zones=zones,
            is_tagged=is_tagged,
        )

        # 2026-05-26:
        # Convert tier1 pdfplumber bordered-table bboxes into pdfium coords.
        # Reason: SAFE zones may overlap already-extracted bordered tables.
        # If not excluded, table text leaks into final_text and later poisons chunks.
        excluded_table_bboxes_pdfium = []

        for tier1_bbox in page.get("tier1_bboxes", []):
            x0, top, x1, bottom = tier1_bbox

            excluded_table_bboxes_pdfium.append(
                (
                    x0,
                    page["effective_page_height_pt"] - bottom,
                    x1,
                    page["effective_page_height_pt"] - top,
                )
            )

        for zone_index, zone in enumerate(zones):
            zone_type = zone.get("type", "unknown")

            if zone_type in SPOTLIGHT_TYPES:
                page["spotlight_bboxes"].append(zone["bbox_pdfium"])
                zone["zone_text"] = ""
                continue

            if zone_type in TRIGGER_ZONE_TYPES:
                zone["zone_text"] = ""
                continue

            bbox_pdfium = zone.get("bbox_pdfium")

            if bbox_pdfium is None:
                zone["zone_text"] = ""
                zone["extraction_failed"] = True
                continue

            # 2026-05-26:
            # Use table-excluding extraction for SAFE zones.
            # Normal prose is preserved, but characters inside already-extracted bordered
            # table regions are removed so tables do not appear twice.
            zone["zone_text"] = extract_text_in_bbox_excluding_tables(
                textpage=textpage,
                bbox_pdfium=tuple(bbox_pdfium),
                exclude_bboxes_pdfium=excluded_table_bboxes_pdfium,
            )

            if not zone["zone_text"]:
                zone["extraction_failed"] = True

        page["zones"] = zones

        for zone_index, zone in enumerate(zones):
            zone_type = zone.get("type", "unknown")

            if zone_type not in TRIGGER_ZONE_TYPES:
                continue

            is_duplicate = any(
                is_tier1_duplicate(
                    zone_bbox_plumber=zone["bbox_plumber"],
                    tier1_bbox_plumber=tier1_bbox,
                )
                for tier1_bbox in page["tier1_bboxes"]
            )

            if is_duplicate:
                continue

            table_id = f"borderless_{page['page_num']}_{zone_index}"

            table_crop = crop_image(img_np, zone["bbox_px"])

            if table_crop.size == 0:
                page["metadata"].setdefault("warnings", []).append(
                    f"{table_id}: empty table crop"
                )
                continue

            tatr_result = engine.detect_table_structure(table_crop)

            artifact = extract_table_explicit_pdfplumber(
                tatr_result=tatr_result,
                table_bbox_px=zone["bbox_px"],
                plumber_page=plumber_page,
                page=page,
                table_id=table_id,
            )

            if artifact is None:
                logger.warning("%s: explicit-grid failed; using cell-crop fallback", table_id)

                artifact = extract_table_cell_pdfplumber(
                    tatr_result=tatr_result,
                    table_bbox_px=zone["bbox_px"],
                    plumber_page=plumber_page,
                    page=page,
                    table_id=table_id,
                )

            # A failed borderless-table attempt is not a table artifact.
            # Store it as a page warning only. page["tables"] must contain usable tables only.
            if artifact is None:
                fallback_text = _extract_pdf_text_for_failed_digital_table(
                    textpage=textpage,
                    zone=zone,
                )

                if fallback_text:
                    page["tables"].append(
                        _make_degraded_digital_table_artifact(
                            table_id=table_id,
                            zone=zone,
                            raw_text=fallback_text,
                            reason="all digital borderless table extraction paths failed",
                        )
                    )
                    page["metadata"].setdefault("warnings", []).append(
                        f"{table_id}: all digital borderless table extraction paths failed; "
                        "preserved raw table text as degraded table artifact"
                    )
                else:
                    page["metadata"].setdefault("warnings", []).append(
                        f"{table_id}: all digital borderless table extraction paths failed"
                    )

                continue

            artifact_warnings = artifact.get("warnings") or []

            # Debug note â€” 2026-05-25:
            # Page 60 Apple 10-K overlay proved DocLayout can mark signature blocks as weak
            # table zones. One such block produced a junk 1x4 cell-crop artifact with:
            #     Quality gate failed. empty_rate=0.50
            #
            # A table artifact with warnings is not safe for downstream chunking/citation.
            # Keep the diagnostic as a page warning, but do not store it in page["tables"].
            if artifact_warnings:
                fallback_text = _extract_pdf_text_for_failed_digital_table(
                    textpage=textpage,
                    zone=zone,
                )

                if fallback_text:
                    page["tables"].append(
                        _make_degraded_digital_table_artifact(
                            table_id=table_id,
                            zone=zone,
                            raw_text=fallback_text,
                            reason=f"artifact rejected with warnings: {artifact_warnings}",
                        )
                    )
                    page["metadata"].setdefault("warnings", []).append(
                        f"{table_id}: rejected table artifact with warnings: {artifact_warnings}; "
                        "preserved raw table text as degraded table artifact"
                    )
                else:
                    page["metadata"].setdefault("warnings", []).append(
                        f"{table_id}: rejected table artifact with warnings: {artifact_warnings}"
                    )

                continue
        
            page["tables"].append(artifact)

        # 2026-05-26:
        # Run artifact-level table dedupe once after all TRIGGER zones finish.
        # This used to run inside the TRIGGER loop, causing repeated dedupe passes
        # and dead final_text assembly that was overwritten after the loop.
        #
        # Keep this after bordered + borderless extraction are both complete.
        page["tables"] = _dedupe_page_table_artifacts(page)

        page["final_text"] = assemble_final_text(
            zones=page["zones"],
            page=page,
        )
        _set_layout_grade(page)

        is_interleaved, interleave_cv = detect_interleaving(page["final_text"])

        if is_interleaved:
            page["metadata"]["reading_order_warning"] = True
            page["metadata"]["interleave_cv"] = interleave_cv
            page["metadata"]["quality_score_multiplier"] = 0.60

    finally:
        if textpage is not None:
            textpage.close()

        if page_obj is not None:
            page_obj.close()

        if pdf is not None:
            pdf.close()


def _should_count_empty_scanned_zone_as_failure(
    zone: dict,
    img_np,
) -> bool:
    """
    Decide whether an empty scanned SAFE zone should hurt layout_grade.

    Scanned global-OCR path assigns each OCR box to one best zone.
    Small overlapping zones may become empty even when final_text is readable.
    Only large primary text/content zones should count as extraction failures.
    """
    zone_type = zone.get("type", "unknown")

    if zone_type not in SCANNED_MAJOR_TEXT_ZONE_TYPES:
        return False

    bbox = zone.get("bbox_px")

    if not bbox or len(bbox) != 4:
        return False

    image_h, image_w = img_np.shape[:2]
    page_area = max(1.0, float(image_w * image_h))

    x0, y0, x1, y1 = bbox
    zone_area = max(0.0, float(x1 - x0)) * max(0.0, float(y1 - y0))

    return (zone_area / page_area) >= SCANNED_MAJOR_EMPTY_ZONE_MIN_AREA_RATIO


MIN_USABLE_OCR_CHARS = 40
MIN_USABLE_OCR_SEGMENTS = 3
BLANK_DARK_PIXEL_RATIO_MAX = 0.002  # 0.2% dark pixels


def _dark_pixel_ratio(img_np) -> float:
    """
    Blank page detector.

    blank page      -> very low dark pixel ratio
    degraded scan   -> higher dark/noise ratio
    """
    if img_np is None or getattr(img_np, "size", 0) == 0:
        return 0.0

    if len(img_np.shape) == 3:
        gray = img_np.mean(axis=2)
    else:
        gray = img_np

    return float((gray < 210).mean())


def _ocr_usable_stats(ocr_segments: list[dict]) -> tuple[int, int]:
    """
    Return usable OCR segment count and usable alnum char count.
    Ignore punctuation/noise-only OCR.
    """
    usable_segments = 0
    usable_chars = 0

    for segment in ocr_segments or []:
        text = str(segment.get("text") or "").strip()

        if not text:
            continue

        alnum_chars = sum(1 for ch in text if ch.isalnum())

        if alnum_chars == 0:
            continue

        usable_segments += 1
        usable_chars += alnum_chars

    return usable_segments, usable_chars


def _mark_empty_scanned_page_discarded(
    page: dict,
    img_np,
    page_ocr_segments: list[dict],
) -> None:
    """
    Mark scanned page non-chunkable only when it is genuinely blank
    or OCR produced no usable text.

    If OCR produced usable text but final_text is empty, do nothing.
    Smoke should FAIL that because the engine dropped usable content.
    """
    has_text = bool(str(page.get("final_text") or "").strip())
    has_tables = bool(page.get("tables") or [])
    has_spotlight = bool(page.get("spotlight_bboxes") or [])

    if has_text or has_tables or has_spotlight:
        return

    if not page.get("needs_ocr"):
        return

    usable_segments, usable_chars = _ocr_usable_stats(page_ocr_segments)

    has_usable_ocr = (
        usable_segments >= MIN_USABLE_OCR_SEGMENTS
        and usable_chars >= MIN_USABLE_OCR_CHARS
    )

    if has_usable_ocr:
        # OCR saw usable text but final output is empty.
        # That is an engine/assembly bug, not garbage.
        return

    dark_ratio = _dark_pixel_ratio(img_np)

    if dark_ratio <= BLANK_DARK_PIXEL_RATIO_MAX:
        discard_reason = "blank_page"
    else:
        discard_reason = "ocr_no_usable_output"

    metadata = page.setdefault("metadata", {})
    metadata["chunk_eligible"] = False
    metadata["discard_reason"] = discard_reason
    metadata["dark_pixel_ratio"] = dark_ratio
    metadata["usable_ocr_segments"] = usable_segments
    metadata["usable_ocr_chars"] = usable_chars

    metadata.setdefault("warnings", []).append(
        f"scanned page discarded: {discard_reason}"
    )

    page["layout_grade"] = "POOR"


def _fallback_ocr_text_for_scanned_trigger_zone(
    page_ocr_segments: list[dict],
    zone_bbox_px: list[float],
) -> str:
    """
    Degraded OCR-text fallback for scanned TRIGGER/table zones.

    Debug note â€” 2026-05-27:
    Doc 8 IBM degraded/corrupt scanned 10-K produced table-only pages where:
        - DocLayout detected only table/TRIGGER zones.
        - TATR could not build a valid structured table.
        - No SAFE text zones existed.
        - final_text became empty even though OCR saw some readable text.

    Why this exists:
        A failed scanned table artifact should not automatically make the page
        fatal-empty if OCR text exists inside the table zone.

    Contract:
        This does NOT create a table artifact.
        This does NOT rescue garbage into structured tables.
        It only stores degraded OCR text on the TRIGGER zone.
        assemble_final_text() may include it only when
        zone["degraded_table_text_fallback"] is True.
    """
    if not page_ocr_segments or not zone_bbox_px:
        return ""

    assigned = assign_ocr_segments_to_bboxes(
        segments=page_ocr_segments,
        indexed_bboxes=[(0, zone_bbox_px)],
    )

    return sort_and_join_ocr_segments(assigned.get(0, [])).strip()


def _process_scanned_page(page: dict, is_tagged: bool) -> None:
    """
    Process one scanned/OCR page after visual zones are already detected.

    Scanned path:
    - SAFE zones      -> full-page OCR once using active OCR backend, then geometry-map OCR boxes into zones
    - TRIGGER zones   -> TATR structure + table-specific OCR/grid mapping
    - SPOTLIGHT zones -> bbox only, no text

    Mutates page in place.
    """

    img_np = page.get("img_np")

    if img_np is None:
        page["metadata"].setdefault("warnings", []).append(
            "Scanned page processing skipped because img_np is missing."
        )
        page["layout_grade"] = "POOR"
        return
    
    timings_ms = page["metadata"].setdefault("timings_ms", {})

    def _add_timing(stage: str, started_at: float) -> None:
        timings_ms[stage] = timings_ms.get(stage, 0.0) + (
            time.perf_counter() - started_at
        ) * 1000.0

    zones = page.get("zones", [])

    zones = process_layout_zones(
        zones=zones,
        is_tagged=is_tagged,
    )

    # 2026-05-26:
    # Scanned-page performance fix.
    # Old path ran RapidOCR once per SAFE DocLayout zone.
    # New path runs RapidOCR once on the full page, then maps OCR boxes into SAFE zones.
    t0 = time.perf_counter()
    page_ocr_segments = ocr_page(img_np)
    _add_timing("scanned_page_ocr", t0)

    safe_zone_indices = [
        zone_index
        for zone_index, zone in enumerate(zones)
        if zone.get("type", "unknown") not in SPOTLIGHT_TYPES
        and zone.get("type", "unknown") not in TRIGGER_ZONE_TYPES
        and not zone.get("unsafe_for_text")
        and zone.get("bbox_px")
    ]

    t0 = time.perf_counter()
    safe_zone_segments = assign_ocr_segments_to_bboxes(
        segments=page_ocr_segments,
        indexed_bboxes=[
            (zone_index, zones[zone_index]["bbox_px"])
            for zone_index in safe_zone_indices
        ],
    )
    _add_timing("scanned_safe_zone_assignment", t0)

    for zone_index, zone in enumerate(zones):
        zone_type = zone.get("type", "unknown")

        if zone_type in SPOTLIGHT_TYPES:
            page["spotlight_bboxes"].append(zone["bbox_pdfium"])
            zone["zone_text"] = ""
            continue

        if zone_type in TRIGGER_ZONE_TYPES:
            table_id = f"scanned_{page['page_num']}_{len(page['tables'])}"

            table_crop = crop_image(img_np, zone["bbox_px"])

            if table_crop.size == 0:
                page["metadata"].setdefault("warnings", []).append(
                    f"{table_id}: empty scanned table crop"
                )
                zone["zone_text"] = ""
                continue

            t0 = time.perf_counter()
            tatr_result = engine.detect_table_structure(table_crop)
            _add_timing("scanned_tatr", t0)

            t0 = time.perf_counter()
            artifact = extract_scanned_table(
                tatr_result=tatr_result,
                table_bbox_px=zone["bbox_px"],
                page_ocr_segments=page_ocr_segments,
                table_id=table_id,
                page=page,
                ocr_backend=engine.ocr_backend_name(),
            )
            _add_timing("scanned_table_extract", t0)

            if artifact is None:
                fallback_text = _fallback_ocr_text_for_scanned_trigger_zone(
                    page_ocr_segments=page_ocr_segments,
                    zone_bbox_px=zone["bbox_px"],
                )

                if fallback_text:
                    page["metadata"].setdefault("warnings", []).append(
                        f"{table_id}: scanned table extraction failed; used OCR text fallback"
                    )
                    zone["zone_text"] = fallback_text
                    zone["degraded_table_text_fallback"] = True
                else:
                    page["metadata"].setdefault("warnings", []).append(
                        f"{table_id}: scanned table extraction failed"
                    )
                    zone["zone_text"] = ""

                continue

            artifact_warnings = artifact.get("warnings") or []

            if artifact_warnings:
                fallback_text = _fallback_ocr_text_for_scanned_trigger_zone(
                    page_ocr_segments=page_ocr_segments,
                    zone_bbox_px=zone["bbox_px"],
                )

                if fallback_text:
                    page["metadata"].setdefault("warnings", []).append(
                        f"{table_id}: rejected scanned table artifact with warnings: {artifact_warnings}; "
                        "used OCR text fallback"
                    )
                    zone["zone_text"] = fallback_text
                    zone["degraded_table_text_fallback"] = True
                else:
                    page["metadata"].setdefault("warnings", []).append(
                        f"{table_id}: rejected scanned table artifact with warnings: {artifact_warnings}"
                    )
                    zone["zone_text"] = ""

                continue

            page["tables"].append(artifact)

            zone["zone_text"] = ""
            continue

        if zone.get("unsafe_for_text"):
            zone["zone_text"] = ""
            continue

        # SAFE zone text is now pulled from the single full-page OCR map.
        # No per-zone OCR call here.
        zone["zone_text"] = sort_and_join_ocr_segments(
            safe_zone_segments.get(zone_index, [])
        )

        if not zone["zone_text"]:
            if _should_count_empty_scanned_zone_as_failure(zone, img_np):
                zone["extraction_failed"] = True
            else:
                zone["ocr_empty_ignored_for_grade"] = True

    page["zones"] = zones

    page["final_text"] = assemble_final_text(
        zones=page["zones"],
        page=page,
    )

    _set_layout_grade(page)

    _mark_empty_scanned_page_discarded(
        page=page,
        img_np=img_np,
        page_ocr_segments=page_ocr_segments,
    )

    is_interleaved, interleave_cv = detect_interleaving(page["final_text"])

    if is_interleaved:
        page["metadata"]["reading_order_warning"] = True
        page["metadata"]["interleave_cv"] = interleave_cv
        page["metadata"]["quality_score_multiplier"] = 0.60


def extract_with_pypdfium2(filepath: str) -> tuple[list[dict], list[dict]]:
    """
    Extract all pages from one PDF.

    Public entry point for pipeline.py.
    Returns:
        pages, errors
    """
    pdf = None
    pages: list[dict] = []
    errors: list[dict] = []

    # 2026-05-26:
    # Public extraction entry point requires TorvexExtractEngine to be warmed once
    # by app startup or the smoke script before page processing begins.
    # Do not auto-warm here: model loading is expensive and belongs to process startup.
    if not engine.is_warmed():
        return pages, [
            {
                "stage": "engine_startup",
                "error": (
                    "TorvexExtractEngine is not warmed. "
                    "Call engine.warm() once at app startup before extract_with_pypdfium2()."
                ),
            }
        ]

    try:
        pdf = pypdfium2.PdfDocument(filepath)
        is_tagged = pdf.is_tagged()

        with pdfplumber.open(filepath) as plumber_pdf:
            for page_idx in range(len(pdf)):
                page = _process_single_page(
                    pdf=pdf,
                    plumber_pdf=plumber_pdf,
                    page_idx=page_idx,
                    filepath=filepath,
                    is_tagged=is_tagged,
                    result_errors=errors,
                )
                pages.append(page)

    except Exception as exc:
        errors.append(
            {
                "stage": "document_extraction",
                "error": str(exc),
            }
        )

    finally:
        if pdf is not None:
            pdf.close()

    return pages, errors

