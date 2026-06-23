import gc
import logging
import os
import tempfile
from pathlib import Path

from PIL import Image

import numpy as np
import onnxruntime as ort

from torvex_extract.onnx_runtime import create_onnx_session, select_onnx_providers
from torvex_extract.ppocrv6_ocr import PPOCRV6SmallOCR, PPOCRV6_SMALL_BACKEND

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "models"

DOCLAYOUT_MODEL_PATH = os.getenv(
    "DOCLAYOUT_MODEL_PATH",
    str(_DEFAULT_MODEL_DIR / "PP-DocLayoutV3_ir8.onnx"),
)

TATR_MODEL_PATH = os.getenv(
    "TATR_MODEL_PATH",
    str(_DEFAULT_MODEL_DIR / "tatr-v1.1-all.onnx"),
)
# All PDF pages are rendered at 200 DPI in Phase 1.
# 2026-05-26: tested 150 DPI on scanned Enron 5-page smoke.
# Result: only ~5% faster and slightly worse layout grade, so reverted.
RENDER_DPI = 200.0

# Verified from models/PP-DocLayoutV3_config.json label_list on 2026-05-19.
# label_id is the index in label_list.
# Important IDs:
#   21 = table
#   22 = text
#   14 = image
#   3  = chart
# Do not use the old 0-12 label map; this model has 25 labels.
DOCLAYOUT_LABEL_MAP = {
    0: "abstract",
    1: "algorithm",
    2: "aside_text",
    3: "chart",
    4: "content",
    5: "display_formula",
    6: "doc_title",
    7: "figure_title",
    8: "footer",
    9: "footer_image",
    10: "footnote",
    11: "formula_number",
    12: "header",
    13: "header_image",
    14: "image",
    15: "inline_formula",
    16: "number",
    17: "paragraph_title",
    18: "reference",
    19: "reference_content",
    20: "seal",
    21: "table",
    22: "text",
    23: "vertical_text",
    24: "vision_footnote",
}


SAFE_ZONE_TYPES = frozenset(
    {
        # PP-DocLayout may classify page numbers / small numeric fragments as "number".
        # Keep "number" SAFE, not TRIGGER, because SEC filings contain numbers everywhere.
        # Directly routing "number" to TATR caused footer/page-number fake tables.
        "number",
        "abstract",
        "algorithm",
        "aside_text",
        "content",
        "doc_title",
        "figure_title",
        "footer",
        "footnote",
        "header",
        "paragraph_title",
        "reference",
        "reference_content",
        "text",
        "vertical_text",
        "vision_footnote",
    }
)

TRIGGER_ZONE_TYPES = frozenset(
    {
        "table",
        # Only real layout table zones should trigger TATR.
        # Do NOT add "number" here unless a future table-likeness gate exists.
    }
)

SPOTLIGHT_TYPES = frozenset(
    {
        "chart",
        "image",
        "seal",
        "footer_image",
        "header_image",
    }
)

FORMULA_ZONE_TYPES = frozenset(
    {
        "display_formula",
        "inline_formula",
        "formula_number",
    }
)


def collect_formula_bboxes(
    zones: list[dict],
    page_num: int,
) -> list[dict]:
    """
    Collect formula zones as bbox-only artifacts.

    This does not run formula OCR / LaTeX extraction.
    It only preserves formula detection metadata for future optional formula extraction.
    """
    formula_bboxes: list[dict] = []

    for zone_index, zone in enumerate(zones):
        zone_type = zone.get("type", "unknown")

        if zone_type not in FORMULA_ZONE_TYPES:
            continue

        formula_bboxes.append(
            {
                "formula_id": f"formula_{page_num}_{zone_index}",
                "type": zone_type,
                "score": float(zone.get("score", 0.0)),
                "bbox_px": zone.get("bbox_px"),
                "bbox_pdfium": zone.get("bbox_pdfium"),
                "bbox_plumber": zone.get("bbox_plumber"),
            }
        )

    return formula_bboxes


def image_bbox_to_pdfium_coords(
    bbox_px: list[float],
    render_dpi: float,
    page_width_pt: float,
    page_height_pt: float,
    padding_pt: float = 2.0,
) -> tuple[float, float, float, float]:
    """
    Convert image pixel bbox to pypdfium2 PDF-point bbox.

    Input:
        bbox_px = [x0, y0, x1, y1]
        Origin: top-left
        Unit: pixels

    Output:
        bbox_pdfium = [left, bottom, right, top]
        Origin: bottom-left
        Unit: PDF points
    """
    scale = 72.0 / render_dpi

    x0, y0, x1, y1 = bbox_px

    left = x0 * scale - padding_pt
    right = x1 * scale + padding_pt

    bottom = page_height_pt - (y1 * scale) - padding_pt
    top = page_height_pt - (y0 * scale) + padding_pt

    return (
        max(0.0, left),
        max(0.0, bottom),
        min(page_width_pt, right),
        min(page_height_pt, top),
    )


def image_bbox_to_plumber_coords(
    bbox_px: list[float],
    render_dpi: float,
    page_width_pt: float,
    page_height_pt: float,
    padding_pt: float = 2.0,
) -> tuple[float, float, float, float]:
    """
    Convert image pixel bbox to pdfplumber PDF-point bbox.

    Input:
        bbox_px = [x0, y0, x1, y1]
        Origin: top-left
        Unit: pixels

    Output:
        bbox_plumber = [x0, top, x1, bottom]
        Origin: top-left
        Unit: PDF points
    """
    scale = 72.0 / render_dpi

    x0, y0, x1, y1 = bbox_px

    left = x0 * scale - padding_pt
    top = y0 * scale - padding_pt
    right = x1 * scale + padding_pt
    bottom = y1 * scale + padding_pt

    return (
        max(0.0, left),
        max(0.0, top),
        min(page_width_pt, right),
        min(page_height_pt, bottom),
    )


# Mainly used for citation storage because Torvex stores table regions in pdfium-style PDF coordinates.
def plumber_to_pdfium_coords( 
    bbox_plumber: list[float],
    page_height_pt: float,
) -> tuple[float, float, float, float]:
    """
    Convert pdfplumber bbox to pypdfium2 bbox.

    Input:
        bbox_plumber = [x0, top, x1, bottom]
        Origin: top-left
        Unit: PDF points

    Output:
        bbox_pdfium = [left, bottom, right, top]
        Origin: bottom-left
        Unit: PDF points
    """
    x0, top, x1, bottom = bbox_plumber

    return (
        x0,
        page_height_pt - bottom,
        x1,
        page_height_pt - top,
    )


# Used before converting TATR crop-local boxes to pdfplumber/pdfium coords.
def add_crop_offset_px(
    cell_bbox_px: list[float],
    table_bbox_px: list[float],
) -> list[float]:
    """
    Convert crop-local TATR cell bbox into full-page pixel bbox.

    TATR sees only the table crop, not the full page.
    So its cell bbox starts from the table crop's top-left corner.
    This helper shifts the cell bbox back to full-page coordinates.
    """
    table_x0 = table_bbox_px[0]
    table_y0 = table_bbox_px[1]

    return [
        cell_bbox_px[0] + table_x0,
        cell_bbox_px[1] + table_y0,
        cell_bbox_px[2] + table_x0,
        cell_bbox_px[3] + table_y0,
    ]


def attach_zone_bboxes(
    zones: list[dict],
    page: dict,
) -> list[dict]:
    """
    Attach all coordinate versions to every DocLayout zone.

    Before:
        zone["bbox"] = image pixel bbox

    After:
        zone["bbox_px"]      = image pixel bbox
        zone["bbox_pdfium"]  = pypdfium/PDF bbox
        zone["bbox_plumber"] = pdfplumber bbox

    Important:
        zone["bbox"] is removed so later code cannot accidentally use
        ambiguous coordinates.
    """
    page_width_pt = page["effective_page_width_pt"]
    page_height_pt = page["effective_page_height_pt"]

    # 2026-06-10: Dynamic render cap support.
    # If pypdfium_extractor rendered an oversized page below base RENDER_DPI,
    # bbox conversion must use that per-page effective render DPI.
    render_dpi = float(page.get("render_dpi", RENDER_DPI) or RENDER_DPI)

    for zone in zones:
        if "bbox" not in zone:
            logger.warning(
                "Skipping zone without bbox during bbox attachment: %s",
                zone.get("type", "unknown"),
            )
            continue

        bbox_px = zone.pop("bbox")

        zone["bbox_px"] = bbox_px

        zone["bbox_pdfium"] = list(
            image_bbox_to_pdfium_coords(
                bbox_px=bbox_px,
                render_dpi=render_dpi,
                page_width_pt=page_width_pt,
                page_height_pt=page_height_pt,
            )
        )

        zone["bbox_plumber"] = list(
            image_bbox_to_plumber_coords(
                bbox_px=bbox_px,
                render_dpi=render_dpi,
                page_width_pt=page_width_pt,
                page_height_pt=page_height_pt,
            )
        )

    return zones


def crop_image(
    img_np: np.ndarray,
    bbox_px: list[float],
) -> np.ndarray:
    """
    Safely crop a page image using bbox_px.

    bbox_px format:
        [x0, y0, x1, y1]

    Coordinate system:
        image pixels
        origin = top-left
    """
    image_h, image_w = img_np.shape[:2]

    x0 = max(0, min(image_w, int(bbox_px[0])))
    y0 = max(0, min(image_h, int(bbox_px[1])))
    x1 = max(0, min(image_w, int(bbox_px[2])))
    y1 = max(0, min(image_h, int(bbox_px[3])))

    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 0, 3), dtype=img_np.dtype)

    return img_np[y0:y1, x0:x1]


def compute_iou(
    box_a: list[float],
    box_b: list[float],
) -> float:
    """
    Compute Intersection over Union.

    Use this for normal bbox overlap checks:
    - duplicate zone removal
    - NMS
    - containment-style overlap checks

    Do NOT use this for bordered-table dedup.
    Use is_tier1_duplicate() for that.
    """
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)

    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


IOA_TIER1_THRESHOLD = 0.85


def _compute_symmetric_ioa(
    box_a: list[float],
    box_b: list[float],
) -> float:
    """
    Symmetric Intersection-over-Area.

    Both boxes must be in the same coordinate system.
    For Torvex tier1 dedup, pass bbox_plumber:

        [x0, top, x1, bottom]

    Returns:
        0.0 = no overlap
        1.0 = one box fully contains the other box

    Why symmetric:
        One-direction IoA can miss when DocLayout zone is smaller but fully
        inside the pdfplumber bordered-table bbox.
    """
    ax0, at, ax1, ab = box_a
    bx0, bt, bx1, bb = box_b

    # Find overlap rectangle.
    ix0 = max(ax0, bx0)
    it = max(at, bt)
    ix1 = min(ax1, bx1)
    ib = min(ab, bb)

    # Calculate overlap area.
    inter_w = max(0.0, ix1 - ix0)
    inter_h = max(0.0, ib - it)
    inter_area = inter_w * inter_h

    # Calculate each box area.
    area_a = max(0.0, ax1 - ax0) * max(0.0, ab - at)
    area_b = max(0.0, bx1 - bx0) * max(0.0, bb - bt)

    if area_a <= 0 or area_b <= 0:
        return 0.0

    # Symmetric containment score.
    return max(
        inter_area / area_a,
        inter_area / area_b,
    )

CONFIDENCE_THRESHOLD = 0.25

def _bbox_area_plumber(
    bbox_plumber: list[float],
) -> float:
    """
    Area for bbox_plumber: [x0, top, x1, bottom].
    """
    x0, top, x1, bottom = bbox_plumber
    return max(0.0, x1 - x0) * max(0.0, bottom - top)


def is_tier1_duplicate(
    zone_bbox_plumber: list[float],
    tier1_bbox_plumber: list[float],
    ioa_threshold: float = IOA_TIER1_THRESHOLD,
    max_area_ratio: float = 2.0,
) -> bool:
    """
    Return True only when a DocLayout TRIGGER zone is already handled by
    a pdfplumber bordered-table artifact.

    Guard:
        The DocLayout zone must not be much larger than the tier1 bbox.

    Why:
        A large TRIGGER zone can contain a small bordered table plus a separate
        borderless table. Symmetric IoA alone would skip the large zone and lose
        the borderless table.
    """
    ioa = _compute_symmetric_ioa(zone_bbox_plumber, tier1_bbox_plumber)

    if ioa < ioa_threshold:
        return False

    zone_area = _bbox_area_plumber(zone_bbox_plumber)
    tier1_area = _bbox_area_plumber(tier1_bbox_plumber)

    if zone_area <= 0 or tier1_area <= 0:
        return False

    if zone_area > max_area_ratio * tier1_area:
        return False

    return True

# Kept as second guard because tests/future callers may pass zones not produced by parse_layout_zones().
def filter_zones_by_confidence(
    zones: list[dict],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """
    Drop low-confidence DocLayout zones.

    This is a second guard after parse_layout_zones().
    It protects callers that pass manually-created or test zones.
    """
    kept = [zone for zone in zones if float(zone.get("score", 0.0)) >= threshold]
    dropped = len(zones) - len(kept)

    if dropped:
        logger.debug("Dropped %d low-confidence layout zone(s)", dropped)

    return kept


NMS_IOU_THRESHOLD = 0.45


def suppress_duplicate_zones(
    zones: list[dict],
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> list[dict]:
    """
    Suppress duplicate zones of the same type.

    Uses bbox_px only.
    Cross-type zones are not suppressed because table/image/text can legitimately overlap.
    """
    if len(zones) <= 1:
        return zones

    by_type: dict[str, list[dict]] = {}

    for zone in zones:
        zone_type = zone.get("type", "unknown")
        if "bbox_px" not in zone:
            logger.warning("Skipping zone without bbox_px during NMS: %s", zone_type)
            continue

        by_type.setdefault(zone_type, []).append(zone)

    kept: list[dict] = []

    for zone_type, typed_zones in by_type.items():
        sorted_zones = sorted(
            typed_zones,
            key=lambda zone: float(zone.get("score", 0.0)),
            reverse=True,
        )

        type_kept: list[dict] = []

        for zone in sorted_zones:
            is_duplicate = any(
                compute_iou(zone["bbox_px"], kept_zone["bbox_px"]) >= iou_threshold
                for kept_zone in type_kept
            )

            if not is_duplicate:
                type_kept.append(zone)

        kept.extend(type_kept)

    return kept

SAFE_CONTAINER_OVERLAP_THRESHOLD = 0.85

# A display_formula zone is treated as a "merged container" (and dropped) when
# at least this many OTHER display_formula zones are mostly inside it.
_FORMULA_CONTAINER_MIN_CHILDREN = 2
_FORMULA_CONTAINER_CHILD_CONTAINMENT = 0.75
# A smaller display_formula zone is dropped when this much of it sits inside a
# larger one — catches single-row double-detections.
_FORMULA_REDUNDANT_CHILD_CONTAINMENT = 0.90


def _containment_ratio(
    inner_box: list[float],
    outer_box: list[float],
) -> float:
    """
    Return how much of inner_box is inside outer_box.
    Both boxes must be bbox_px.
    """
    ix0 = max(inner_box[0], outer_box[0])
    iy0 = max(inner_box[1], outer_box[1])
    ix1 = min(inner_box[2], outer_box[2])
    iy1 = min(inner_box[3], outer_box[3])

    inter_w = max(0.0, ix1 - ix0)
    inter_h = max(0.0, iy1 - iy0)
    inter_area = inter_w * inter_h

    inner_area = max(0.0, inner_box[2] - inner_box[0]) * max(
        0.0,
        inner_box[3] - inner_box[1],
    )

    if inner_area <= 0:
        return 0.0

    return inter_area / inner_area


def mark_unsafe_container_zones(
    zones: list[dict],
    containment_threshold: float = SAFE_CONTAINER_OVERLAP_THRESHOLD,
) -> list[dict]:
    """
    Mark SAFE zones unsafe when they heavily contain table/image/chart zones.

    Why:
    A large 'content' zone can contain a table or chart.
    If extracted as prose, table/chart text leaks into final_text.

    Rule:
    Keep all zones.
    Never merge content.
    Never drop table/image/chart zones.
    Only mark the SAFE container as unsafe_for_text.
    """
    trigger_or_spotlight = [
        zone
        for zone in zones
        if zone.get("type") in (TRIGGER_ZONE_TYPES | SPOTLIGHT_TYPES)
        and "bbox_px" in zone
    ]

    if not trigger_or_spotlight:
        return zones

    for zone in zones:
        zone_type = zone.get("type", "unknown")

        if zone_type not in SAFE_ZONE_TYPES:
            continue

        if "bbox_px" not in zone:
            continue

        for child in trigger_or_spotlight:
            ratio = _containment_ratio(child["bbox_px"], zone["bbox_px"])

            if ratio >= containment_threshold:
                zone["unsafe_for_text"] = True
                zone.setdefault("metadata", {})
                zone["metadata"]["unsafe_reason"] = (
                    f"SAFE zone contains {child.get('type', 'unknown')} zone"
                )
                break

    return zones

COLUMN_OVERLAP_THRESHOLD = 0.30


def _zone_x_overlap_ratio(
    zone_box: list[float],
    band_x0: float,
    band_x1: float,
) -> float:
    """
    Return how much of zone width overlaps an existing column band.
    """
    zx0, _, zx1, _ = zone_box
    zone_w = max(0.0, zx1 - zx0)

    if zone_w <= 0:
        return 0.0

    overlap = max(0.0, min(zx1, band_x1) - max(zx0, band_x0))
    return overlap / zone_w


def _xy_cut_columns(
    zones: list[dict],
) -> list[dict]:
    """
    Build simple visual column bands from bbox_px.

    This is intentionally conservative:
    - left/right page-wide elements can join bands
    - tables/images remain ordered but are not used as prose text
    - exact perfection is not assumed
    """
    sortable = [zone for zone in zones if "bbox_px" in zone]

    if not sortable:
        return []

    sorted_zones = sorted(sortable, key=lambda zone: zone["bbox_px"][0])

    bands: list[dict] = []

    for zone in sorted_zones:
        x0, _, x1, _ = zone["bbox_px"]
        x_center = (x0 + x1) / 2.0

        matched_band = None
        best_overlap = 0.0

        for band in bands:
            overlap_ratio = _zone_x_overlap_ratio(
                zone["bbox_px"],
                band["x0"],
                band["x1"],
            )
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                matched_band = band

        if best_overlap < COLUMN_OVERLAP_THRESHOLD:
            matched_band = None


        if matched_band is None:
            bands.append(
                {
                    "x0": x0,
                    "x1": x1,
                    "x_center": x_center,
                    "zones": [zone],
                }
            )
        else:
            matched_band["x0"] = min(matched_band["x0"], x0)
            matched_band["x1"] = max(matched_band["x1"], x1)
            matched_band["x_center"] = (
                matched_band["x0"] + matched_band["x1"]
            ) / 2.0
            matched_band["zones"].append(zone)

    return bands


def order_zones_for_reading(
    zones: list[dict],
    is_tagged: bool = False,
) -> list[dict]:
    """
    Layout-aware reading order.

    Tagged PDFs:
        Use conservative visual top-left order.
        Do not skip DocLayout. Tags are not trusted enough to bypass zoning.

    Untagged PDFs:
        Use simple Docling-style column bands:
        columns left-to-right, zones top-to-bottom inside each column.
    """
    if not zones:
        return zones

    zones_with_bbox = [zone for zone in zones if "bbox_px" in zone]
    zones_without_bbox = [zone for zone in zones if "bbox_px" not in zone]

    if is_tagged:
        ordered = sorted(
            zones_with_bbox,
            key=lambda zone: (
                round(float(zone["bbox_px"][1]), 2),
                float(zone["bbox_px"][0]),
            ),
        )
        return ordered + zones_without_bbox

    column_bands = _xy_cut_columns(zones_with_bbox)

    if not column_bands:
        return sorted(
            zones_with_bbox,
            key=lambda zone: (
                round(float(zone["bbox_px"][1]), 2),
                float(zone["bbox_px"][0]),
            ),
        ) + zones_without_bbox

    ordered: list[dict] = []

    for band in sorted(column_bands, key=lambda band: band["x_center"]):
        ordered.extend(
            sorted(
                band["zones"],
                key=lambda zone: (
                    round(float(zone["bbox_px"][1]), 2),
                    float(zone["bbox_px"][0]),
                ),
            )
        )

    return ordered + zones_without_bbox


def process_layout_zones(
    zones: list[dict],
    is_tagged: bool = False,
) -> list[dict]:
    """
    Torvex production layout-zone preparation pipeline.

    Expected input:
        zones already passed through attach_zone_bboxes()
        so every normal zone has bbox_px / bbox_pdfium / bbox_plumber.

    This prepares zones only.
    It does not extract text, assemble final_text, build table artifacts,
    or compute quality scores.

    Order rationale:
        1. confidence:
        drop weak/noisy zones first.
        2. NMS:
        remove same-type duplicates only.
        Cross-type overlaps are preserved because table/text/image can overlap.
        3. unsafe container marking:
        mark SAFE zones that contain TRIGGER/SPOTLIGHT zones so prose extraction
        does not swallow tables/charts/images.
        4. reading order:
        run last because ordering should not affect cleanup decisions.
    """
    zones = filter_zones_by_confidence(zones)
    zones = suppress_duplicate_zones(zones)
    # Formula-zone selection (drop-inner-keep-outer + display-like inline promotion) now
    # happens in the formula extractor via formula_pipeline.select_formula_boxes. Do NOT
    # pre-suppress formula zones here, or the extractor can't see them (the old
    # suppress_nested/inline regressed CDM by deleting real stacked/short equations).
    zones = mark_unsafe_container_zones(zones)
    zones = order_zones_for_reading(zones, is_tagged=is_tagged)

    return zones

# Verified against PP-DocLayoutV3_ir8.onnx on 2026-05-19:
# ONNX inputs are exactly:
#   im_shape:      [N, 2] float
#   image:         [N, 3, 800, 800] float
#   scale_factor:  [N, 2] float
# Do not change this back to a single image/blob input.
# scale_h / scale_w are kept for diagnostics and input contract clarity.
# Current PP-DocLayoutV3 ONNX output is already rendered-image pixel space.
# Do NOT divide output bboxes by scale_h / scale_w in parse_layout_zones().
def preprocess_for_doclayout(img_np: np.ndarray) -> tuple[dict, float, float]:
    """
    Convert rendered page image into PP-DocLayoutV3 ONNX input feed.

    This model expects 3 inputs:
    - im_shape
    - image
    - scale_factor
    """
    import cv2

    target_size = 800

    original_h, original_w = img_np.shape[:2]
    
    if original_h == 0 or original_w == 0:
        raise ValueError("DocLayout received empty page image")

    scale_h = target_size / original_h
    scale_w = target_size / original_w

    resized = cv2.resize(
        img_np,
        (target_size, target_size),
        interpolation=cv2.INTER_LINEAR,
    )

    image = resized.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = image[np.newaxis, ...]

    im_shape = np.array(
        [[target_size, target_size]],
        dtype=np.float32,
    )

    scale_factor = np.array(
        [[scale_h, scale_w]],
        dtype=np.float32,
    )

    input_feed = {
        "im_shape": im_shape,
        "image": image,
        "scale_factor": scale_factor,
    }

    return input_feed, scale_h, scale_w     


def classify_digital_page_zones(zones: list[dict]) -> str:
    """
    Classify DocLayout zones into a page routing class.

    Returns:
        "zero_zones" -> DocLayout returned nothing.
        "text_only"  -> only normal text-like zones.
        "mixed"      -> table / number / visual / unknown zones exist.

    Unknown zones are treated as mixed because silent text-only routing can lose tables.
    This function does not decide OCR.
    """
    if not zones:
        return "zero_zones"

    zone_types = {zone.get("type", "unknown") for zone in zones}

    known_types = SAFE_ZONE_TYPES | TRIGGER_ZONE_TYPES | SPOTLIGHT_TYPES | FORMULA_ZONE_TYPES
    unknown_types = zone_types - known_types

    if unknown_types:
        logger.warning("Unknown DocLayout zone type(s): %s", sorted(unknown_types))
        return "mixed"

    if zone_types & (TRIGGER_ZONE_TYPES | SPOTLIGHT_TYPES | FORMULA_ZONE_TYPES):
        return "mixed"

    return "text_only"

# Returns "bbox" (not "bbox_px") â€” attach_zone_bboxes() consumes and normalizes.
def parse_layout_zones(
    raw_output: list,
    scale_h: float,
    scale_w: float,
    confidence_threshold: float = 0.25,
) -> list[dict]:
    """
    Parse PP-DocLayoutV3 ONNX output into Torvex zone dictionaries.

    Actual model output:
        raw_output[0] = detections, shape (N, 7)
        raw_output[1] = detection count, shape (1,)
        raw_output[2] = masks, ignored for Phase 1

    Detection row:
        [label_id, score, x0, y0, x1, y1, mask_id]
    """
    zones = []

    if not raw_output or len(raw_output) < 2:
        return zones

    detections = raw_output[0]

    try:
        detection_count = int(raw_output[1][0])
    except Exception:
        logger.warning("DocLayout detection_count malformed")
        return zones
    
    detection_count = min(detection_count, len(detections))

    for row in detections[:detection_count]:
        label_id = int(row[0])
        score = float(row[1])

        if score < confidence_threshold:
            continue

        x0 = float(row[2])
        y0 = float(row[3])
        x1 = float(row[4])
        y1 = float(row[5])
        mask_id = int(row[6])

        zone_type = DOCLAYOUT_LABEL_MAP.get(label_id, "unknown")

        zones.append(
            {
                "type": zone_type,
                "label_id": label_id,
                # PP-DocLayoutV3 ONNX already returns bboxes in rendered image pixel space.
                # Do NOT divide by scale_w / scale_h here.
                # Verified this with page 22 overlay: dividing produced oversized shifted boxes.
                "bbox": [
                    x0,
                    y0,
                    x1,
                    y1,
                ],
                "score": score,
                "mask_id": mask_id,
            }
        )

    return zones

# TATR expects 800x800 normalized table-region input.
# It is called on DocLayout table-zone crops, not on the full page.
TATR_INPUT_SIZE = 800
TATR_CONFIDENCE_THRESHOLD = 0.50

TATR_LABEL_MAP = {
    0: "table",
    1: "table_column",
    2: "table_row",
    3: "table_column_header",
    4: "table_projected_row_header",
    5: "table_spanning_cell",
    6: "no_object",
}

# Verified against tatr-v1.1-all.onnx on 2026-05-19:
# ONNX input is:
#   pixel_values: [batch_size, num_channels, height, width] float
def preprocess_for_tatr(crop_np: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Convert a table crop image into TATR ONNX input.

    Input:
        crop_np: table crop image, shape (H, W, 3), RGB

    Output:
        blob: model-ready tensor, shape (1, 3, 800, 800)
        scale_h: height resize ratio
        scale_w: width resize ratio
    """
    import cv2

    original_h, original_w = crop_np.shape[:2]

    if original_h == 0 or original_w == 0:
        raise ValueError("TATR received empty table crop")

    scale_h = TATR_INPUT_SIZE / original_h
    scale_w = TATR_INPUT_SIZE / original_w

    resized = cv2.resize(
        crop_np,
        (TATR_INPUT_SIZE, TATR_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    blob = resized.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    blob = (blob - mean) / std

    blob = blob.transpose(2, 0, 1)

    blob = blob[np.newaxis, ...]

    return blob, scale_h, scale_w

def _softmax_np(x: np.ndarray) -> np.ndarray:
    """
    Convert raw model logits into probabilities.
    """
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)

# Verified against tatr-v1.1-all.onnx on 2026-05-19:
# ONNX outputs are:
#   logits:     [batch_size, num_queries, 7]
#   pred_boxes: [batch_size, num_queries, 4]
# pred_boxes are normalized center-format boxes:
#   [cx, cy, w, h]
def parse_tatr_output(
    raw_output: list,
    scale_h: float,
    scale_w: float,
) -> dict:
    """
    Parse TATR ONNX output into row/column structure.

    Output:
        {
            "rows": [...],
            "columns": [...],
            "spanning_cells": [...],
            "headers": [...],
        }

    All bbox_px values are crop-local pixel coordinates.
    """
    result = {
        "rows": [],
        "columns": [],
        "spanning_cells": [],
        "headers": [],
    }

    if not raw_output or len(raw_output) < 2:
        return result

    logits = raw_output[0][0]
    boxes = raw_output[1][0]

    probs = _softmax_np(logits)

    original_h = TATR_INPUT_SIZE / scale_h
    original_w = TATR_INPUT_SIZE / scale_w

    for prob, box in zip(probs, boxes):
        label_id = int(np.argmax(prob[:-1]))
        score = float(prob[label_id])

        if score < TATR_CONFIDENCE_THRESHOLD:
            continue

        label = TATR_LABEL_MAP.get(label_id, "no_object")

        if label in {"table", "no_object"}:
            continue

        cx, cy, w, h = box

        x0 = (float(cx) - float(w) / 2.0) * original_w
        y0 = (float(cy) - float(h) / 2.0) * original_h
        x1 = (float(cx) + float(w) / 2.0) * original_w
        y1 = (float(cy) + float(h) / 2.0) * original_h

        x0 = max(0.0, min(original_w, x0))
        y0 = max(0.0, min(original_h, y0))
        x1 = max(0.0, min(original_w, x1))
        y1 = max(0.0, min(original_h, y1))

        if x1 <= x0 or y1 <= y0:
            continue

        entry = {
            "bbox_px": [x0, y0, x1, y1],
            "score": score,
            "label": label,
        }

        if label == "table_row":
            result["rows"].append(entry)

        elif label == "table_column":
            result["columns"].append(entry)

        elif label == "table_spanning_cell":
            result["spanning_cells"].append(entry)

        elif label in {"table_column_header", "table_projected_row_header"}:
            result["headers"].append(entry)

    result["rows"].sort(key=lambda row: row["bbox_px"][1])
    result["columns"].sort(key=lambda col: col["bbox_px"][0])

    if result["spanning_cells"]:
        logger.info(
            "TATR detected %d spanning cell(s); rowspan/colspan resolution deferred",
            len(result["spanning_cells"]),
        )

    return result


def _onnxtr_geometry_to_xyxy(
    geometry,
    image_width: int,
    image_height: int,
) -> list[float] | None:
    """
    Convert ONNXTR normalized geometry into pixel [x0, y0, x1, y1].

    ONNXTR/docTR geometry is usually:
        ((x0, y0), (x1, y1))
    with values normalized from 0.0 to 1.0.
    """
    if geometry is None:
        return None

    try:
        point_a, point_b = geometry
        x0_norm, y0_norm = point_a
        x1_norm, y1_norm = point_b

        x0 = float(x0_norm) * image_width
        y0 = float(y0_norm) * image_height
        x1 = float(x1_norm) * image_width
        y1 = float(y1_norm) * image_height

        left = min(x0, x1)
        top = min(y0, y1)
        right = max(x0, x1)
        bottom = max(y0, y1)

        if right <= left or bottom <= top:
            return None

        return [left, top, right, bottom]

    except Exception:
        return None


def _onnxtr_result_to_line_segments(
    result,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """
    Convert ONNXTR output into Torvex OCR segment dicts.

    Torvex OCR contract:
        {
            "bbox": [[x0,y0], [x1,y0], [x1,y1], [x0,y1]],
            "text": "...",
            "score": float,
        }

    We emit one segment per ONNXTR line, not one per word.
    Reason: the rest of Torvex expects OCR segments roughly like text lines.
    """
    try:
        exported = result.export()
    except Exception:
        return []

    segments: list[dict] = []

    for page in exported.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                words = []
                word_boxes = []
                scores = []

                for word in line.get("words", []):
                    text = str(word.get("value", "")).strip()

                    if not text:
                        continue

                    box = _onnxtr_geometry_to_xyxy(
                        geometry=word.get("geometry"),
                        image_width=image_width,
                        image_height=image_height,
                    )

                    if box is None:
                        continue

                    words.append(text)
                    word_boxes.append(box)

                    confidence = word.get("confidence")

                    if confidence is not None:
                        try:
                            scores.append(float(confidence))
                        except Exception:
                            pass

                if not words or not word_boxes:
                    continue

                x0 = min(box[0] for box in word_boxes)
                y0 = min(box[1] for box in word_boxes)
                x1 = max(box[2] for box in word_boxes)
                y1 = max(box[3] for box in word_boxes)

                score = sum(scores) / len(scores) if scores else 1.0

                segments.append(
                    {
                        "bbox": [
                            [x0, y0],
                            [x1, y0],
                            [x1, y1],
                            [x0, y1],
                        ],
                        "text": " ".join(words),
                        "score": score,
                    }
                )

    return segments


class TorvexExtractEngine:
    """
    Long-lived extraction engine.

    Load once at startup.
    Reuse for all pages.
    Shutdown at app exit.
    """

    def __init__(self):
        self._layout = None
        self._tatr = None
        self._tatr_run_options = None
        self._ocr = None
        self._ocr_backend = "onnxtr_fast_base"
        self._providers = None

    def _select_onnx_providers(self, device: str) -> list[str]:
        # 2026-06-15: provider selection moved to a shared helper so
        # PP-DocLayoutV3, TATR, PP-OCRv6, and UniMERNet all get the same
        # Windows CUDA DLL preload and fail-fast CUDA verification behavior.
        return select_onnx_providers(device)

    def warm(self, device: str = "cpu", ocr_backend: str | None = None):
        import sys

        self._providers = self._select_onnx_providers(device)

        self._layout = create_onnx_session(
            DOCLAYOUT_MODEL_PATH,
            providers=self._providers,
            model_name="PP-DocLayoutV3",
        )

        self._tatr = create_onnx_session(
            TATR_MODEL_PATH,
            providers=self._providers,
            model_name="TATR",
        )

        # CUDA arena shrinkage for TATR only: its input H/W are fully dynamic, so a large
        # table crop ratchets the ORT CUDA arena up and it stays resident. Shrinking after each
        # run caps it with no quality cost; TATR runs once per table (not per text-line) so the
        # re-alloc cost is negligible. NOT applied to PP-DocLayout (self._layout): its input is
        # fixed 800x800, so its arena never grows -> shrinkage would only add per-page churn.
        self._tatr_run_options = None
        if any("CUDA" in provider for provider in self._providers):
            self._tatr_run_options = ort.RunOptions()
            self._tatr_run_options.add_run_config_entry(
                "memory.enable_memory_arena_shrinkage", "gpu:0"
            )

        # 2026-05-27:
        # OCR routing is page-level in pypdfium_extractor.py.
        # Digital pages do not OCR.
        # Scanned pages use ONNXTR by default.
        #
        # RapidOCR backend was removed.
        # 2026-06-15: PP-OCRv6 small is available as an opt-in second backend
        # for Chinese/OmniDocBench comparison without changing downstream
        # scanned-page routing.
        self._ocr_backend = (
            ocr_backend
            or os.getenv("TORVEX_OCR_BACKEND")
            or os.getenv("Torvex_OCR_BACKEND")
            or "onnxtr_fast_base"
        ).strip().lower()
        
        if self._ocr_backend == "onnxtr_fast_base":
            from onnxtr.models import EngineConfig, ocr_predictor

            ocr_engine_cfg = EngineConfig(
                providers=self._providers,
            )

            self._ocr = ocr_predictor(
                det_arch="fast_base",
                reco_arch="crnn_mobilenet_v3_small",
                assume_straight_pages=True,
                det_engine_cfg=ocr_engine_cfg,
                reco_engine_cfg=ocr_engine_cfg,
                clf_engine_cfg=ocr_engine_cfg,
            )
        elif self._ocr_backend == PPOCRV6_SMALL_BACKEND:
            # 2026-06-15: PP-OCRv6 is wired as a second OCR backend only.
            # The rest of Torvex still receives the same OCR segment contract
            # so scanned SAFE-zone assignment and scanned-table mapping do not
            # change while we benchmark Chinese accuracy and latency.
            self._ocr = PPOCRV6SmallOCR(
                providers=self._providers,
            )
        else:
            raise ValueError(
                "Unsupported Torvex_OCR_BACKEND="
                f"{self._ocr_backend!r}. "
                "Expected 'onnxtr_fast_base' or 'ppocrv6_small'."
            )

        paddle_mods = [module for module in sys.modules if "paddle" in module.lower()]
        if paddle_mods:
            raise RuntimeError(f"Paddle module imported unexpectedly: {paddle_mods}")

        logger.info(
            "TorvexExtractEngine warmed successfully with providers=%s OCR backend=%s",
            self._providers,
            self._ocr_backend,
        )


    def is_warmed(self) -> bool:
        """
        Return True only when all long-lived extraction engines are loaded.

        2026-05-26:
        Added for the public extraction entry point.
        extract_with_pypdfium2() depends on this singleton being warmed once
        at app/smoke startup. This makes that contract explicit instead of
        failing later inside detect_layout(), detect_table_structure(), or OCR.
        """
        return (
            self._layout is not None
            and self._tatr is not None
            and self._ocr is not None
        )
    
    
    def ocr_backend_name(self) -> str:
        """
        Return the active OCR backend name for diagnostics and table artifact labels.

        2026-05-27:
        Added after ONNXTR backend experiment.
        Smoke reports must show the real OCR backend used by scanned tables:
            tatr_global_onnxtr_fast_base or tatr_global_ppocrv6_small

        Do not read _ocr_backend directly outside this class.
        """
        return self._ocr_backend


    def detect_layout(self, img_np: np.ndarray) -> list[dict]:
        if self._layout is None:
            raise RuntimeError("TorvexExtractEngine not warmed")
        
        input_feed, scale_h, scale_w = preprocess_for_doclayout(img_np)

        raw_output = self._layout.run(None, input_feed)
        
        return parse_layout_zones(raw_output, scale_h, scale_w)

    def detect_table_structure(self, table_crop_np: np.ndarray) -> dict:
        if self._tatr is None:
            raise RuntimeError("TorvexExtractEngine not warmed")

        blob, scale_h, scale_w = preprocess_for_tatr(table_crop_np)

        raw_output = self._tatr.run(
            None,
            {"pixel_values": blob},
            self._tatr_run_options,
        )

        return parse_tatr_output(raw_output, scale_h, scale_w)


    def ocr_image(self, image_np) -> list[dict]:
        """
        Run OCR on an RGB image crop/page and return Torvex OCR segments.

        Backend:
            TORVEX_OCR_BACKEND=onnxtr_fast_base|ppocrv6_small

        Default OCR backend is ONNXTR fast_base.

            OCR routing is page-level:
            digital page -> no OCR
            scanned page -> OCR with this backend.
        """
        if self._ocr is None:
            raise RuntimeError("TorvexExtractEngine not warmed")

        if image_np is None or image_np.size == 0:
            return []

        if self._ocr_backend == "onnxtr_fast_base":
            image_h, image_w = image_np.shape[:2]

            # 2026-06-10: Cap oversized OCR input for scanned pages.
            #
            # Large rendered pages can exceed 35-40MP. Passing those directly
            # into ONNXTR causes temporary multi-GB RAM spikes and slow OCR.
            # Keep the original rendered image for layout/table coordinates,
            # but downscale only the temporary OCR image.
            #
            # TORVEX_OCR_MAX_LONG_SIDE_PX=0 disables this cap.
            max_ocr_long_side_px = int(os.getenv("TORVEX_OCR_MAX_LONG_SIDE_PX", "2500"))
            ocr_image_np = image_np

            if max_ocr_long_side_px > 0:
                long_side = max(image_w, image_h)
                if long_side > max_ocr_long_side_px:
                    scale = max_ocr_long_side_px / float(long_side)
                    ocr_w = max(1, int(round(image_w * scale)))
                    ocr_h = max(1, int(round(image_h * scale)))

                    resampling = getattr(Image, "Resampling", Image).LANCZOS
                    ocr_image_np = np.asarray(
                        Image.fromarray(image_np).resize((ocr_w, ocr_h), resampling)
                    )

            # ONNXTR stable public path is image file input.
            # This is experiment-only for now, not production-final.
            with tempfile.TemporaryDirectory() as tmpdir:
                image_path = Path(tmpdir) / "ocr_input.png"
                Image.fromarray(ocr_image_np).save(image_path)

                from onnxtr.io import DocumentFile

                doc = DocumentFile.from_images(str(image_path))
                result = self._ocr(doc)

            return _onnxtr_result_to_line_segments(
                result=result,
                image_width=image_w,
                image_height=image_h,
            )

        if self._ocr_backend == PPOCRV6_SMALL_BACKEND:
            return self._ocr.ocr_image(image_np)

        raise RuntimeError(f"Unsupported OCR backend: {self._ocr_backend}")


    def shutdown(self) -> None:
        """
        Release long-lived extraction engine references.

        2026-05-27:
        smoke_phase1_extraction.py calls engine.shutdown() after every run.
        OCR backend experiments must not break the engine lifecycle contract.
        """
        self._layout = None
        self._tatr = None
        self._tatr_run_options = None
        self._ocr = None
        self._ocr_backend = "onnxtr_fast_base"
        self._providers = None

        gc.collect()

        logger.info("TorvexExtractEngine shut down")

engine = TorvexExtractEngine()

