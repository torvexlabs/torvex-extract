import logging

from numbers import Real

from torvex_extract.ocr_engine import (
    sort_and_join_ocr_segments,
    _ocr_bbox_to_xyxy,
    _box_area_xyxy,
)

from torvex_extract.visual_zoning import (
    RENDER_DPI,
    crop_image,
    add_crop_offset_px,
    image_bbox_to_pdfium_coords,
    image_bbox_to_plumber_coords,
)

logger = logging.getLogger(__name__)

MAX_EMPTY_CELL_RATE = 0.40

GUTTER_CHECK_WIDTH_PT = 4.0


def build_table_artifact(
    table_id: str,
    source: str,
    rows: list[list[str]],
    bbox_px: list[float] | None,
    bbox_pdfium: list[float],
    bbox_plumber: list[float],
    confidence: float = 1.0,
    warnings: list[str] | None = None,
) -> dict:
    return {
        "table_id": table_id,
        "source": source,
        "bbox_px": bbox_px,
        "bbox_pdfium": bbox_pdfium,
        "bbox_plumber": bbox_plumber,
        "rows": rows,
        "confidence": confidence,
        "warnings": warnings or [],
    }


def _table_quality_gate_warning(
    rows_data: list[list[str]],
    empty_rate: float,
    table_id: str,
    stage: str,
) -> str | None:
    """
    Return detailed quality-gate warning, or None if table passes.

    2026-05-27:
    Invoice scanned pages exposed misleading warnings like:
        Quality gate failed. empty_rate=0.00

    That is incomplete because empty_rate can be good while the real failure is:
        row_count < 2
        first_row_cols < 2
        header row empty

    This function is diagnostic only.
    It does not relax or change the quality gate.
    """
    row_count = len(rows_data)
    first_row_cols = len(rows_data[0]) if rows_data else 0

    total_cells = sum(len(row) for row in rows_data)
    non_empty_cells = sum(
        1
        for row in rows_data
        for cell in row
        if str(cell).strip()
    )
    empty_cells = total_cells - non_empty_cells

    header_non_empty_cells = 0
    if rows_data:
        header_non_empty_cells = sum(
            1
            for cell in rows_data[0]
            if str(cell).strip()
        )

    body_non_empty_cells = sum(
        1
        for row in rows_data[1:]
        for cell in row
        if str(cell).strip()
    )

    reasons: list[str] = []

    # 2026-05-27:
    # Scanned invoice tables can be valid with one OCR row:
    #   - ITEMS table with one line item
    #   - SUMMARY table with one total/VAT row
    #
    # Keep this exception scanned-only.
    # Digital pdfplumber tables still require >=2 rows because digital tables
    # should preserve header/body structure more reliably.
    scanned_stage = stage.startswith("tatr_global_")

    single_row_scanned_ok = (
        scanned_stage
        and row_count == 1
        and first_row_cols >= 4
        and non_empty_cells >= 4
        and empty_rate <= 0.25
    )

    # Keep this logic exactly aligned with _passes_quality_gate().
    if row_count < 2 and not single_row_scanned_ok:
        reasons.append("row_count < 2")

    if first_row_cols < 2:
        reasons.append("first_row_cols < 2")

    if empty_rate > MAX_EMPTY_CELL_RATE:
        reasons.append(
            f"empty_rate {empty_rate:.2f} > max {MAX_EMPTY_CELL_RATE:.2f}"
        )

    if header_non_empty_cells == 0:
        reasons.append("header row empty")

    if not reasons:
        return None

    return (
        "Quality gate failed. "
        f"stage={stage} "
        f"reasons={reasons} "
        f"rows={row_count} "
        f"first_row_cols={first_row_cols} "
        f"total_cells={total_cells} "
        f"non_empty_cells={non_empty_cells} "
        f"empty_cells={empty_cells} "
        f"empty_rate={empty_rate:.2f} "
        f"header_non_empty_cells={header_non_empty_cells} "
        f"body_non_empty_cells={body_non_empty_cells}"
    )


def _passes_quality_gate(
    rows_data: list[list[str]],
    empty_rate: float,
    table_id: str,
    stage: str,
) -> bool:
    """
    Return True if extracted table is safe to store.

    2026-05-27:
    Delegates diagnostics to _table_quality_gate_warning()
    so warning logs expose the real failure reason, not only empty_rate.
    """
    warning = _table_quality_gate_warning(
        rows_data=rows_data,
        empty_rate=empty_rate,
        table_id=table_id,
        stage=stage,
    )

    if warning is not None:
        logger.warning("%s: %s", table_id, warning)
        return False

    return True

def find_gutter_line(
    proposed_coord: float,
    plumber_page,
    axis: str,
    table_bbox_plumber: list[float],
    search_range_pt: float = GUTTER_CHECK_WIDTH_PT,
) -> tuple[float, bool, str]:
    try:
        tx0, t_top, tx1, t_bottom = table_bbox_plumber
        all_chars = plumber_page.chars

        if axis not in {"vertical", "horizontal"}:
            warning = f"Invalid gutter axis: {axis}"
            logger.warning(warning)
            return proposed_coord, False, warning

        if not all_chars:
            return proposed_coord, True, ""

        if axis == "vertical":
            chars_in_band = [
                char
                for char in all_chars
                if char["top"] >= t_top
                and char["bottom"] <= t_bottom
                and char["x0"] <= proposed_coord + search_range_pt
                and char["x1"] >= proposed_coord - search_range_pt
            ]

            if not chars_in_band:
                return proposed_coord, True, ""

            intervals = [
                (
                    max(proposed_coord - search_range_pt, char["x0"]),
                    min(proposed_coord + search_range_pt, char["x1"]),
                )
                for char in chars_in_band
            ]

        else:
            chars_in_band = [
                char
                for char in all_chars
                if char["x0"] >= tx0
                and char["x1"] <= tx1
                and char["top"] <= proposed_coord + search_range_pt
                and char["bottom"] >= proposed_coord - search_range_pt
            ]

            if not chars_in_band:
                return proposed_coord, True, ""

            intervals = [
                (
                    max(proposed_coord - search_range_pt, char["top"]),
                    min(proposed_coord + search_range_pt, char["bottom"]),
                )
                for char in chars_in_band
            ]

        intervals.sort()

        merged = []

        for interval in intervals:
            if not merged:
                merged.append(interval)
            else:
                last = merged[-1]
                if interval[0] <= last[1]:
                    merged[-1] = (last[0], max(last[1], interval[1]))
                else:
                    merged.append(interval)

        band_start = proposed_coord - search_range_pt
        band_end = proposed_coord + search_range_pt

        gaps = []

        if merged[0][0] > band_start:
            gaps.append((band_start, merged[0][0]))

        for index in range(len(merged) - 1):
            gaps.append((merged[index][1], merged[index + 1][0]))

        if merged[-1][1] < band_end:
            gaps.append((merged[-1][1], band_end))

        if not gaps:
            warning = f"Gutter guard: no clear gap at {proposed_coord:.1f}"
            logger.warning(warning)
            return proposed_coord, False, warning

        largest_gap = max(gaps, key=lambda gap: gap[1] - gap[0])
        safe_coord = (largest_gap[0] + largest_gap[1]) / 2.0

        return safe_coord, True, ""

    except Exception as exc:
        warning = f"Gutter guard failed: {exc}"
        logger.warning(warning)
        return proposed_coord, False, warning
    

def extract_table_explicit_pdfplumber(
    tatr_result: dict,
    table_bbox_px: list[float],
    plumber_page,
    page: dict,
    table_id: str,
) -> dict | None:
    """
    Primary digital borderless-table extraction path.

    TATR gives visual rows/columns.
    We convert those rows/columns into explicit pdfplumber grid lines.
    pdfplumber then extracts real PDF text from those grid cells.

    Returns:
        table artifact dict on success
        None on failure, so caller can fallback to cell-crop extraction
    """
    columns = tatr_result.get("columns", [])
    rows = tatr_result.get("rows", [])

    if len(columns) < 2 or len(rows) < 2:
        logger.warning("%s: TATR <2 columns or <2 rows", table_id)
        return None

    effective_height = page["effective_page_height_pt"]
    effective_width = page["effective_page_width_pt"]

    table_bbox_plumber = image_bbox_to_plumber_coords(
        bbox_px=table_bbox_px,
        render_dpi=RENDER_DPI,
        page_width_pt=effective_width,
        page_height_pt=effective_height,
        padding_pt=2.0,
    )

    tx0, table_top, tx1, table_bottom = table_bbox_plumber

    explicit_vertical_lines = []

    for index in range(len(columns) - 1):
        left_column_right = columns[index]["bbox_px"][2]
        right_column_left = columns[index + 1]["bbox_px"][0]

        midpoint_px = (left_column_right + right_column_left) / 2.0
        midpoint_full_page_px = midpoint_px + table_bbox_px[0]
        proposed_x = midpoint_full_page_px * (72.0 / RENDER_DPI)

        safe_x, ok, warning = find_gutter_line(
            proposed_coord=proposed_x,
            plumber_page=plumber_page,
            axis="vertical",
            table_bbox_plumber=list(table_bbox_plumber),
        )

        if not ok:
            logger.warning(
                "%s: vertical gutter failed at column %d: %s",
                table_id,
                index,
                warning,
            )
            return None

        explicit_vertical_lines.append(safe_x)

    explicit_horizontal_lines = []

    for index in range(len(rows) - 1):
        upper_row_bottom = rows[index]["bbox_px"][3]
        lower_row_top = rows[index + 1]["bbox_px"][1]

        midpoint_px = (upper_row_bottom + lower_row_top) / 2.0
        midpoint_full_page_px = midpoint_px + table_bbox_px[1]
        proposed_y = midpoint_full_page_px * (72.0 / RENDER_DPI)

        safe_y, ok, warning = find_gutter_line(
            proposed_coord=proposed_y,
            plumber_page=plumber_page,
            axis="horizontal",
            table_bbox_plumber=list(table_bbox_plumber),
        )

        if not ok:
            logger.warning(
                "%s: horizontal gutter failed at row %d: %s",
                table_id,
                index,
                warning,
            )
            return None

        explicit_horizontal_lines.append(safe_y)

    explicit_vertical_lines = sorted([tx0] + explicit_vertical_lines + [tx1])
    explicit_horizontal_lines = sorted(
        [table_top] + explicit_horizontal_lines + [table_bottom]
    )

    try:
        cropped_page = plumber_page.crop(
            (tx0, table_top, tx1, table_bottom)
        )

        extracted_rows = cropped_page.extract_table(
            {
                "vertical_strategy": "explicit",
                "horizontal_strategy": "explicit",
                "explicit_vertical_lines": explicit_vertical_lines,
                "explicit_horizontal_lines": explicit_horizontal_lines,
                "snap_x_tolerance": 3,
                "snap_y_tolerance": 3,
            }
        )
    except Exception as exc:
        logger.warning("%s: pdfplumber explicit extraction failed: %s", table_id, exc)
        return None

    if not extracted_rows:
        logger.warning("%s: pdfplumber explicit extraction returned no rows", table_id)
        return None

    rows_data = [
        [str(cell).strip() if cell is not None else "" for cell in row]
        for row in extracted_rows
    ]

    total_cells = sum(len(row) for row in rows_data)
    empty_cells = sum(
        1
        for row in rows_data
        for cell in row
        if not cell.strip()
    )

    empty_rate = empty_cells / total_cells if total_cells > 0 else 1.0

    if not _passes_quality_gate(
        rows_data=rows_data,
        empty_rate=empty_rate,
        table_id=table_id,
        stage="explicit_grid",
    ):
        return None

    bbox_pdfium = image_bbox_to_pdfium_coords(
        bbox_px=table_bbox_px,
        render_dpi=RENDER_DPI,
        page_width_pt=effective_width,
        page_height_pt=effective_height,
        padding_pt=2.0,
    )

    confidence_values = [
        float(item.get("score", 0.0))
        for item in columns + rows
    ]

    confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else 0.75
    )

    return build_table_artifact(
        table_id=table_id,
        source="tatr_explicit_pdfplumber",
        rows=rows_data,
        bbox_px=table_bbox_px,
        bbox_pdfium=list(bbox_pdfium),
        bbox_plumber=list(table_bbox_plumber),
        confidence=confidence,
        warnings=[],
    )


def extract_table_cell_pdfplumber(
    tatr_result: dict,
    table_bbox_px: list[float],
    plumber_page,
    page: dict,
    table_id: str,
) -> dict | None:
    """
    Fallback digital borderless-table extraction path.

    Used when explicit-grid extraction fails.
    Instead of asking pdfplumber to extract the whole table at once,
    this crops each TATR row/column cell and extracts text cell by cell.
    """
    columns = sorted(
        tatr_result.get("columns", []),
        key=lambda column: column["bbox_px"][0],
    )
    rows = sorted(
        tatr_result.get("rows", []),
        key=lambda row: row["bbox_px"][1],
    )

    if not columns or not rows:
        logger.warning("%s: no rows or columns for cell-crop fallback", table_id)
        return None

    effective_height = page["effective_page_height_pt"]
    effective_width = page["effective_page_width_pt"]

    grid: list[list[str]] = []

    for row in rows:
        row_data: list[str] = []

        for column in columns:
            cell_bbox_crop_px = [
                column["bbox_px"][0],
                row["bbox_px"][1],
                column["bbox_px"][2],
                row["bbox_px"][3],
            ]

            cell_bbox_page_px = add_crop_offset_px(
                cell_bbox_px=cell_bbox_crop_px,
                table_bbox_px=table_bbox_px,
            )

            cell_bbox_plumber = image_bbox_to_plumber_coords(
                bbox_px=cell_bbox_page_px,
                render_dpi=RENDER_DPI,
                page_width_pt=effective_width,
                page_height_pt=effective_height,
                padding_pt=1.0,
            )

            try:
                x0, top, x1, bottom = cell_bbox_plumber
                cell_text = plumber_page.crop((x0, top, x1, bottom)).extract_text() or ""
            except Exception as exc:
                logger.debug("%s: cell crop failed: %s", table_id, exc)
                cell_text = ""

            row_data.append(cell_text.strip())

        grid.append(row_data)

    total_cells = sum(len(row) for row in grid)
    empty_cells = sum(
        1
        for row in grid
        for cell in row
        if not cell.strip()
    )

    empty_rate = empty_cells / total_cells if total_cells > 0 else 1.0

    warnings: list[str] = []

    if not _passes_quality_gate(
        rows_data=grid,
        empty_rate=empty_rate,
        table_id=table_id,
        stage="cell_crop",
    ):
        warning = f"Quality gate failed. empty_rate={empty_rate:.2f}"
        warnings.append(warning)
        logger.warning("%s: %s", table_id, warning)

    table_bbox_plumber = image_bbox_to_plumber_coords(
        bbox_px=table_bbox_px,
        render_dpi=RENDER_DPI,
        page_width_pt=effective_width,
        page_height_pt=effective_height,
        padding_pt=2.0,
    )

    bbox_pdfium = image_bbox_to_pdfium_coords(
        bbox_px=table_bbox_px,
        render_dpi=RENDER_DPI,
        page_width_pt=effective_width,
        page_height_pt=effective_height,
        padding_pt=2.0,
    )

    return build_table_artifact(
        table_id=table_id,
        source="tatr_cell_pdfplumber",
        rows=grid,
        bbox_px=table_bbox_px,
        bbox_pdfium=list(bbox_pdfium),
        bbox_plumber=list(table_bbox_plumber),
        confidence=0.5 if warnings else 0.75,
        warnings=warnings,
    )


def _segments_to_cell_text(segments: list[dict]) -> str:
    """
    Join OCR segments inside one virtual table cell.

    Segments already belong to one cell.
    This only sorts them in reading order and joins text.
    """
    if not segments:
        return ""

    return sort_and_join_ocr_segments(segments).strip()


def _make_polygon_segment_from_xyxy(
    source_segment: dict,
    bbox_xyxy: list[float],
    text: str,
) -> dict:
    """
    Build a safe OCR segment copy with polygon bbox.

    2026-05-27:
    Used by scanned-table ONNXTR line splitting.
    ONNXTR can return one OCR line spanning multiple table columns.
    We split the text logically, then create per-cell pseudo-segments so
    sort_and_join_ocr_segments() can keep using the normal OCR contract.
    """
    x0, y0, x1, y1 = bbox_xyxy

    segment = dict(source_segment)
    segment["text"] = text.strip()
    segment["bbox_xyxy"] = [x0, y0, x1, y1]
    segment["bbox"] = [
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
    ]

    return segment


def _split_tokens_evenly(
    text: str,
    parts: int,
) -> list[str]:
    """
    Split OCR line text into roughly equal token groups.

    2026-05-27:
    This is a recovery heuristic for ONNXTR line-level OCR.
    We do NOT have word-level boxes here, so this must stay conservative.
    It is better than assigning the whole line to one cell and leaving
    other columns empty, but it is still experimental.
    """
    tokens = text.split()

    if parts <= 1:
        return [" ".join(tokens).strip()]

    if not tokens:
        return [""] * parts

    result: list[str] = []

    for index in range(parts):
        start = round(index * len(tokens) / parts)
        end = round((index + 1) * len(tokens) / parts)
        result.append(" ".join(tokens[start:end]).strip())

    return result


def _candidate_columns_for_segment(
    segment_box: list[float],
    row_y0: float,
    row_y1: float,
    columns: list[dict],
    table_x0: float,
    min_horizontal_overlap: float = 0.10,
) -> list[tuple[int, list[float]]]:
    """
    Return table columns touched by one OCR segment inside a matched row.

    2026-05-27:
    Supports ONNXTR line-level OCR where one line bbox can span multiple
    table columns. We use horizontal overlap against each TATR column.
    """
    sx0, sy0, sx1, sy1 = segment_box
    segment_w = max(0.0, sx1 - sx0)

    if segment_w <= 0:
        return []

    candidates: list[tuple[int, list[float]]] = []

    for col_index, column in enumerate(columns):
        cell_x0 = column["bbox_px"][0] + table_x0
        cell_x1 = column["bbox_px"][2] + table_x0

        overlap_w = max(0.0, min(sx1, cell_x1) - max(sx0, cell_x0))
        overlap_ratio = overlap_w / segment_w

        segment_cx = (sx0 + sx1) / 2.0
        center_inside_col = cell_x0 <= segment_cx <= cell_x1

        if center_inside_col or overlap_ratio >= min_horizontal_overlap:
            candidates.append(
                (
                    col_index,
                    [cell_x0, row_y0, cell_x1, row_y1],
                )
            )

    return candidates


# 2026-05-27:
# Scanned-table performance fix.
# Old path OCR'd scanned pages once globally, then OCR'd each table crop again.
# New path reuses full-page OCR segments and maps them into TATR cells by geometry.
# TATR still uses the table crop for structure only; OCR must not run again here.
def extract_scanned_table(
    tatr_result: dict,
    table_bbox_px: list[float],
    page_ocr_segments: list[dict],
    table_id: str,
    page: dict,
    ocr_backend: str = "unknown",
) -> dict | None:
    """
    Fast scanned/image-table extraction path.

    2026-05-27:
    Reuse the one full-page OCR result from _process_scanned_page().

    Old path:
        full page OCR once for SAFE zones
        PLUS OCR again on every table crop

    New path:
        full page OCR once
        TATR gives table row/column geometry
        OCR segments are assigned into table cells by coordinates

    Why:
        scanned_page_ocr already gives full-page OCR boxes.
        Running OCR again inside scanned table extraction duplicates the main cost.
    """
    columns = sorted(
        tatr_result.get("columns", []),
        key=lambda column: column["bbox_px"][0],
    )
    rows = sorted(
        tatr_result.get("rows", []),
        key=lambda row: row["bbox_px"][1],
    )

    if not columns or not rows:
        logger.warning("%s: no rows or columns for scanned table", table_id)
        return None

    if not page_ocr_segments:
        logger.warning("%s: no page OCR segments available for scanned table", table_id)
        return None

    effective_height = page["effective_page_height_pt"]
    effective_width = page["effective_page_width_pt"]

    table_x0, table_y0, table_x1, table_y1 = table_bbox_px
    table_box = [table_x0, table_y0, table_x1, table_y1]

    normalized_segments: list[dict] = []

    for segment in page_ocr_segments:
        segment_box = segment.get("bbox_xyxy")

        if segment_box is None:
            segment_box = _ocr_bbox_to_xyxy(segment)

        if segment_box is None:
            continue

        text = str(segment.get("text", "")).strip()

        if not text:
            continue

        segment_area = _box_area_xyxy(segment_box)

        if segment_area <= 0:
            continue

        sx0, sy0, sx1, sy1 = segment_box
        segment_cx = (sx0 + sx1) / 2.0
        segment_cy = (sy0 + sy1) / 2.0

        center_inside_table = (
            table_x0 <= segment_cx <= table_x1
            and table_y0 <= segment_cy <= table_y1
        )

        inter_x0 = max(sx0, table_box[0])
        inter_y0 = max(sy0, table_box[1])
        inter_x1 = min(sx1, table_box[2])
        inter_y1 = min(sy1, table_box[3])

        overlap_area = max(0.0, inter_x1 - inter_x0) * max(
            0.0,
            inter_y1 - inter_y0,
        )
        coverage = overlap_area / segment_area

        # 2026-05-27:
        # Only reuse OCR text that belongs to this table zone.
        # Center-inside handles normal words.
        # Coverage fallback catches words whose OCR bbox crosses the table boundary.
        if not center_inside_table and coverage < 0.50:
            continue

        normalized_segment = dict(segment)
        normalized_segment["bbox_xyxy"] = [sx0, sy0, sx1, sy1]

        # Keep sort_and_join_ocr_segments() safe.
        # It expects polygon bboxes, not flat [x0, y0, x1, y1].
        normalized_segment["bbox"] = [
            [sx0, sy0],
            [sx1, sy0],
            [sx1, sy1],
            [sx0, sy1],
        ]

        normalized_segments.append(normalized_segment)

    grid_segments: list[list[list[dict]]] = [
        [[] for _ in range(len(columns))]
        for _ in range(len(rows))
    ]

    for segment in normalized_segments:
        segment_box = segment["bbox_xyxy"]
        sx0, sy0, sx1, sy1 = segment_box

        segment_area = _box_area_xyxy(segment_box)
        if segment_area <= 0:
            continue

        segment_cy = (sy0 + sy1) / 2.0

        best_row_index: int | None = None
        best_row_score = 0.0
        best_row_y0 = 0.0
        best_row_y1 = 0.0

        for row_index, row in enumerate(rows):
            # TATR row boxes are table-crop-local.
            # Convert them to full-page pixel coordinates before matching full-page OCR.
            row_y0 = row["bbox_px"][1] + table_y0
            row_y1 = row["bbox_px"][3] + table_y0

            vertical_overlap = max(0.0, min(sy1, row_y1) - max(sy0, row_y0))
            segment_h = max(1.0, sy1 - sy0)
            vertical_coverage = vertical_overlap / segment_h

            center_inside_row = row_y0 <= segment_cy <= row_y1

            # 2026-05-27:
            # Pick the row first, then handle columns.
            # ONNXTR often returns one OCR line spanning multiple columns.
            # The old best-cell-only path assigned the entire line to one cell,
            # which made other cells empty and dropped table artifacts.
            row_score = 1.0 + vertical_coverage if center_inside_row else vertical_coverage

            if row_score > best_row_score:
                best_row_score = row_score
                best_row_index = row_index
                best_row_y0 = row_y0
                best_row_y1 = row_y1

        if best_row_index is None or best_row_score < 0.25:
            continue

        candidate_columns = _candidate_columns_for_segment(
            segment_box=segment_box,
            row_y0=best_row_y0,
            row_y1=best_row_y1,
            columns=columns,
            table_x0=table_x0,
        )

        if not candidate_columns:
            continue

        text = str(segment.get("text", "")).strip()
        tokens = text.split()

        # 2026-05-27:
        # If an ONNXTR line spans multiple columns, split text across those columns.
        # This is conservative: only split when token count can support it.
        # Otherwise keep old best-column behavior to avoid inventing structure.
        if len(candidate_columns) > 1 and len(tokens) >= len(candidate_columns):
            text_parts = _split_tokens_evenly(
                text=text,
                parts=len(candidate_columns),
            )

            for (candidate_index, (col_index, cell_box)) in enumerate(candidate_columns):
                part_text = text_parts[candidate_index].strip()

                if not part_text:
                    continue

                pseudo_segment = _make_polygon_segment_from_xyxy(
                    source_segment=segment,
                    bbox_xyxy=cell_box,
                    text=part_text,
                )

                grid_segments[best_row_index][col_index].append(pseudo_segment)

            continue

        # Fallback: one segment belongs to one best column.
        best_col_index: int | None = None
        best_col_score = 0.0

        sx_center = (sx0 + sx1) / 2.0

        for col_index, cell_box in candidate_columns:
            cell_x0, _, cell_x1, _ = cell_box

            overlap_w = max(0.0, min(sx1, cell_x1) - max(sx0, cell_x0))
            segment_w = max(1.0, sx1 - sx0)
            horizontal_coverage = overlap_w / segment_w

            center_inside_col = cell_x0 <= sx_center <= cell_x1
            col_score = (
                1.0 + horizontal_coverage
                if center_inside_col
                else horizontal_coverage
            )

            if col_score > best_col_score:
                best_col_score = col_score
                best_col_index = col_index

        if best_col_index is not None and best_col_score >= 0.10:
            grid_segments[best_row_index][best_col_index].append(segment)

    grid: list[list[str]] = []

    for row_index in range(len(rows)):
        row_data: list[str] = []

        for col_index in range(len(columns)):
            row_data.append(
                _segments_to_cell_text(grid_segments[row_index][col_index])
            )

        grid.append(row_data)

    total_cells = sum(len(row) for row in grid)
    empty_cells = sum(
        1
        for row in grid
        for cell in row
        if not cell.strip()
    )

    empty_rate = empty_cells / total_cells if total_cells > 0 else 1.0

    warnings: list[str] = []

    # 2026-05-27:
    # Source label must come from the caller-provided OCR backend.
    # Do not import/use engine here; table_structure.py should stay engine-agnostic.
    # pypdfium_extractor.py already knows the active OCR backend and passes it in.
    backend_name = (ocr_backend or "unknown").strip().lower()
    source_name = f"tatr_global_{backend_name}"

    quality_warning = _table_quality_gate_warning(
        rows_data=grid,
        empty_rate=empty_rate,
        table_id=table_id,
        stage=source_name,
    )

    if quality_warning is not None:
        warnings.append(quality_warning)
        logger.warning("%s: %s", table_id, quality_warning)

    table_bbox_plumber = image_bbox_to_plumber_coords(
        bbox_px=table_bbox_px,
        render_dpi=RENDER_DPI,
        page_width_pt=effective_width,
        page_height_pt=effective_height,
        padding_pt=2.0,
    )

    bbox_pdfium = image_bbox_to_pdfium_coords(
        bbox_px=table_bbox_px,
        render_dpi=RENDER_DPI,
        page_width_pt=effective_width,
        page_height_pt=effective_height,
        padding_pt=2.0,
    )

    return build_table_artifact(
        table_id=table_id,
        source=source_name,
        rows=grid,
        bbox_px=table_bbox_px,
        bbox_pdfium=list(bbox_pdfium),
        bbox_plumber=list(table_bbox_plumber),
        confidence=0.5 if warnings else 0.80,
        warnings=warnings,
    )

