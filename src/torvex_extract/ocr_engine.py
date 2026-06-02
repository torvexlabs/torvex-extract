import logging

from numbers import Real

from torvex_extract.visual_zoning import crop_image, engine

logger = logging.getLogger(__name__)

LINE_Y_SNAP = 3.0  # pixels


def sort_and_join_ocr_segments(segments: list[dict]) -> str:
    """
    Sort RapidOCR segments by visual position and join them into readable text.

    RapidOCR segment order is not guaranteed.
    Always sort top-to-bottom, then left-to-right before joining.
    """
    if not segments:
        return ""

    valid_segments = []

    for segment in segments:
        bbox = segment.get("bbox")
        text = segment.get("text")

        if bbox is None or text is None:
            continue

        if len(bbox) < 4:
            continue

        valid_segments.append(segment)

    if not valid_segments:
        return ""

    def seg_top(segment: dict) -> float:
        return min(point[1] for point in segment["bbox"])

    def seg_left(segment: dict) -> float:
        return min(point[0] for point in segment["bbox"])

    # Rough sort all segments top-to-bottom, then left-to-right.
    sorted_segments = sorted(
        valid_segments,
        key=lambda segment: (seg_top(segment), seg_left(segment)),
    )

    lines: list[list[dict]] = []
    current_line = [sorted_segments[0]]

    for segment in sorted_segments[1:]:
        if abs(seg_top(segment) - seg_top(current_line[0])) <= LINE_Y_SNAP:
            current_line.append(segment)
        else:
            lines.append(sorted(current_line, key=seg_left))
            current_line = [segment]

    lines.append(sorted(current_line, key=seg_left))

    return "\n".join(
        " ".join(str(segment["text"]) for segment in line).strip()
        for line in lines
    ).strip()


def _ocr_bbox_to_xyxy(segment: dict) -> list[float] | None:
    """
    Normalize RapidOCR bbox to [x0, y0, x1, y1].
    Accepts polygon bbox or flat bbox.
    """
    bbox = segment.get("bbox")

    if bbox is None:
        return None

    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()

    if (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(isinstance(value, Real) for value in bbox)
    ):
        x0, y0, x1, y1 = bbox
        return [float(x0), float(y0), float(x1), float(y1)]

    try:
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
    except Exception:
        return None

    return [min(xs), min(ys), max(xs), max(ys)]


def _box_area_xyxy(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def ocr_page(full_page_img_np) -> list[dict]:
    """
    Run RapidOCR once on a full scanned page.

    Returns normalized OCR segments in full-page pixel coordinates.
    """
    if full_page_img_np is None or full_page_img_np.size == 0:
        return []

    raw_segments = engine.ocr_image(full_page_img_np)
    normalized_segments: list[dict] = []

    for segment in raw_segments:
        bbox_xyxy = _ocr_bbox_to_xyxy(segment)

        if bbox_xyxy is None:
            continue

        text = str(segment.get("text", "")).strip()

        if not text:
            continue

        normalized_segment = dict(segment)
        normalized_segment["bbox_xyxy"] = bbox_xyxy

        # Keep sort_and_join_ocr_segments() safe.
        # It expects polygon bboxes, not flat [x0, y0, x1, y1].
        x0, y0, x1, y1 = bbox_xyxy
        normalized_segment["bbox"] = [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
        ]

        normalized_segments.append(normalized_segment)

    return normalized_segments


def assign_ocr_segments_to_bboxes(
    segments: list[dict],
    indexed_bboxes: list[tuple[int, list[float]]],
    coverage_threshold: float = 0.50,
) -> dict[int, list[dict]]:
    """
    Assign each OCR segment to exactly one best bbox.

    Used for scanned-page OCR:
        full-page RapidOCR once
        then map OCR boxes into DocLayout SAFE zones.

    This avoids duplicate text when zones overlap.
    """
    assigned: dict[int, list[dict]] = {
        index: []
        for index, _ in indexed_bboxes
    }

    for segment in segments:
        segment_box = segment.get("bbox_xyxy")

        if not segment_box:
            continue

        sx0, sy0, sx1, sy1 = segment_box
        segment_area = _box_area_xyxy(segment_box)

        if segment_area <= 0:
            continue

        segment_cx = (sx0 + sx1) / 2.0
        segment_cy = (sy0 + sy1) / 2.0

        best_index: int | None = None
        best_score = 0.0
        best_center_match = False

        for index, bbox in indexed_bboxes:
            if not bbox or len(bbox) != 4:
                continue

            bx0, by0, bx1, by1 = bbox

            center_inside = (
                bx0 <= segment_cx <= bx1
                and by0 <= segment_cy <= by1
            )

            inter_x0 = max(sx0, bx0)
            inter_y0 = max(sy0, by0)
            inter_x1 = min(sx1, bx1)
            inter_y1 = min(sy1, by1)

            overlap_area = max(0.0, inter_x1 - inter_x0) * max(
                0.0,
                inter_y1 - inter_y0,
            )

            coverage = overlap_area / segment_area
            score = 1.0 + coverage if center_inside else coverage

            if score > best_score:
                best_score = score
                best_index = index
                best_center_match = center_inside

        if best_index is None:
            continue

        if best_center_match or best_score >= coverage_threshold:
            assigned[best_index].append(segment)

    return assigned


def ocr_zone(
    zone_bbox_px: list[float],
    full_page_img_np,
) -> str:
    """
    Run OCR on one SAFE scanned layout zone.

    The bbox is in full-page pixel coordinates.
    The function crops that zone, runs RapidOCR, then returns sorted readable text.
    """
    if full_page_img_np is None:
        return ""

    if not zone_bbox_px or len(zone_bbox_px) != 4:
        return ""

    crop = crop_image(full_page_img_np, zone_bbox_px)

    if crop.size == 0:
        logger.debug("ocr_zone: zero-area crop for bbox %s", zone_bbox_px)
        return ""

    segments = engine.ocr_image(crop)

    return sort_and_join_ocr_segments(segments)


def ocr_cell(cell_crop_np) -> str:
    """
    Run OCR on one scanned table-cell crop.

    Kept as legacy/fallback helper.
    Current scanned table path uses global table OCR + coordinate mapping.
    """
    if cell_crop_np is None or cell_crop_np.size == 0:
        logger.debug("ocr_cell: empty cell crop")
        return ""

    segments = engine.ocr_image(cell_crop_np)

    return sort_and_join_ocr_segments(segments)

