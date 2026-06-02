import logging



from pypdfium2 import raw as pdfium_c

from torvex_extract.visual_zoning import (
    SPOTLIGHT_TYPES,
    plumber_to_pdfium_coords,
)

logger = logging.getLogger(__name__)

# Fine-tuned layout tolerances for standard digital PDFs
SNAP_TOLERANCE = 3.0
JOIN_TOLERANCE = 3.0
EDGE_MIN_LENGTH = 3.0
INTERSECTION_TOLERANCE = 3.0
THIN_EDGE_MAX_THICKNESS = 8.0


def detect_table_fast(page) -> bool:
    """
    Fast bordered-table detector.

    Use case:
    Detect whether a digital PDF page contains a bordered/vector-line table.

    This does NOT extract table data.
    It only answers:
        "Should we spend pdfplumber time on this page?"

    Pipeline:
        PDF path objects
        -> edges
        -> snapped edges
        -> joined edges
        -> intersections
        -> cells
        -> table-like grid signal
    """
    try:
        edges = _extract_edges(page)

        if len(edges) < 4:
            return False  # Not enough edges to form a table
        
        edges = _snap_edges(edges, SNAP_TOLERANCE)
        edges = _join_edges(edges, JOIN_TOLERANCE)
        edges = [edge for edge in edges if edge["length"] >= EDGE_MIN_LENGTH]

        horizontal_edges = [e for e in edges if e["orientation"] == "h"]
        vertical_edges = [e for e in edges if e["orientation"] == "v"]

        if len(horizontal_edges) < 3 or len(vertical_edges) < 3:
            return False  # Needs a minimum of 3 horizontal lines and 3 vertical lines

        intersections = _find_intersections(edges, INTERSECTION_TOLERANCE)
    
        if len(intersections) < 4:
            return False  
    
        cells = _find_cells(intersections)
        
        unique_x = len(set(x for cell in cells for x in (cell[0], cell[2])))
        unique_y = len(set(y for cell in cells for y in (cell[1], cell[3])))

        # Strict layout validation rules to weed out false positives (headers/footers/decorative dividers)
        return (
                len(cells) >= 3
                and unique_x >= 3  # at least 2 visual columns
                and unique_y >= 2  # at least 1 visual row band
        )    
    except Exception as e:
        logger.debug("detect_table_fast failed: %s", e)
        return False


def _extract_edges(page) -> list[dict]:
    """
    Extract horizontal, vertical, and rectangle-derived edges from PDF PATH objects.
    Enforces native PDF coordinate tracking where bottom is minimum Y and top is maximum Y.
    """
    edges = []
    for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_PATH]):
        left, bottom, right, top = obj.get_bounds()
        width = abs(right - left) 
        height = abs(top - bottom)

        if width < 1 and height < 1:
            continue   
        
        # Thin horizontal line
        if width >= EDGE_MIN_LENGTH and height < THIN_EDGE_MAX_THICKNESS:
            edges.append({
                "orientation": "h",
                "x0": min(left, right),
                "x1": max(left, right),
                "top": max(top, bottom),
                "bottom": max(top, bottom),
                "length": width,
            })
        # Thin vertical line
        elif height >= EDGE_MIN_LENGTH and width < THIN_EDGE_MAX_THICKNESS:
            edges.append({
                "orientation": "v",
                "x0": (left + right) / 2,
                "x1": (left + right) / 2,
                "top": max(top, bottom),
                "bottom": min(top, bottom),
                "length": height,
            })
        # Rectangle/box object: decompose into four distinct structural edges.
        elif width >= EDGE_MIN_LENGTH and height >= EDGE_MIN_LENGTH:
            left_x = min(left, right)
            right_x = max(left, right)
            y_min = min(top, bottom)
            y_max = max(top, bottom)

            edges.append({
                "orientation": "h",
                "x0": left_x,
                "x1": right_x,
                "top": y_min,
                "bottom": y_min,
                "length": right_x - left_x,
            })

            edges.append({
                "orientation": "h",
                "x0": left_x,
                "x1": right_x,
                "top": y_max,
                "bottom": y_max,
                "length": right_x - left_x,
            })

            edges.append({
                "orientation": "v",
                "x0": left_x,
                "x1": left_x,
                "top": y_max,
                "bottom": y_min,
                "length": y_max - y_min,
            })

            edges.append({
                "orientation": "v",
                "x0": right_x,
                "x1": right_x,
                "top": y_max,
                "bottom": y_min,
                "length": y_max - y_min,
            })
    return edges


def _find_intersections(edges: list[dict], tolerance: float) -> dict:
    """
    Find intersections between horizontal and vertical edges to identify potential table grid points.
    """
    intersections = {}

    horizontal_edges = [e for e in edges if e["orientation"] == "h"]
    vertical_edges = [e for e in edges if e["orientation"] == "v"]

    for vertical in vertical_edges:
        for horizontal in horizontal_edges:
            vertical_x = vertical["x0"]
            horizontal_y = horizontal["top"]

            vertical_reaches_y = (
                vertical["bottom"] <= horizontal_y + tolerance
                and vertical["top"] >= horizontal_y - tolerance
            )

            horizontal_reaches_x = (
                vertical_x >= horizontal["x0"] - tolerance
                and vertical_x <= horizontal["x1"] + tolerance
            )

            if vertical_reaches_y and horizontal_reaches_x:
                point = (round(vertical_x, 1), round(horizontal_y, 1))

                if point not in intersections:
                    intersections[point] = {"v": [], "h": []}

                intersections[point]["v"].append(vertical)
                intersections[point]["h"].append(horizontal)

    return intersections


def _edges_share(edges_a: list[dict], edges_b: list[dict]) -> bool:
    """
    Check if two edges share the same snapped coordinate properties.
    """
    def edge_key(edge: dict) -> tuple:
        return (
            round(edge["x0"], 1),
            round(edge["top"], 1),
            round(edge["x1"], 1),
            round(edge["bottom"], 1),
        )

    return bool(
        set(edge_key(edge) for edge in edges_a)
        & set(edge_key(edge) for edge in edges_b)
    )


def _find_cells(intersections: dict) -> list[tuple[float, float, float, float]]:
    """
    Analyze the grid of intersections to identify potential table cells bounded by 4 intersections.
    Fixed for native PDF coordinate spaces where vertical downwards tracking checks for smaller Y metrics.
    """
    cells = []
    points = sorted(intersections.keys())

    for top_left in points:
        x0, y0 = top_left

        right_points = [
            point for point in points
            if abs(point[1] - y0) < 0.5 and point[0] > x0
        ]

        # Fix: Native Y coordinates decrease as you travel down the page visually
        bottom_points = [
            point for point in points
            if abs(point[0] - x0) < 0.5 and point[1] < y0
        ]

        for top_right in right_points:
            x1, _ = top_right
            if not _edges_share(intersections[top_left]["h"], intersections[top_right]["h"]):
                continue

            for bottom_left in bottom_points:
                _, y1 = bottom_left
                if not _edges_share(intersections[top_left]["v"], intersections[bottom_left]["v"]):
                    continue

                bottom_right = (top_right[0], bottom_left[1])

                if bottom_right not in intersections:
                    continue

                if (
                    _edges_share(intersections[bottom_right]["v"], intersections[top_right]["v"])
                    and _edges_share(intersections[bottom_right]["h"], intersections[bottom_left]["h"])
                ):
                    cells.append((x0, y0, x1, y1))

    return cells


def _snap_edges(edges: list[dict], tolerance: float) -> list[dict]:
    """
    Snap nearby horizontal/vertical edges onto the same shared coordinate to normalize line paths.
    """
    horizontal_edges = [edge for edge in edges if edge["orientation"] == "h"]
    vertical_edges = [edge for edge in edges if edge["orientation"] == "v"]

    snapped_horizontal = _snap_edge_group(horizontal_edges, key="top", tolerance=tolerance)
    snapped_vertical = _snap_edge_group(vertical_edges, key="x0", tolerance=tolerance)

    return snapped_horizontal + snapped_vertical


def _snap_edge_group(edges: list[dict], key: str, tolerance: float) -> list[dict]:
    """
    Snap one group of structural edges by a single coordinate key index.
    """
    if not edges:
        return []

    sorted_edges = sorted(edges, key=lambda edge: edge[key])

    groups = []
    current_group = [sorted_edges[0]]

    for edge in sorted_edges[1:]:
        if abs(edge[key] - current_group[-1][key]) <= tolerance:
            current_group.append(edge)
        else:
            groups.append(current_group)
            current_group = [edge]

    groups.append(current_group)
    snapped_edges = []

    for group in groups:
        average_position = sum(edge[key] for edge in group) / len(group)

        for edge in group:
            snapped_edge = dict(edge)
            snapped_edge[key] = average_position

            if key == "top":
                snapped_edge["bottom"] = average_position

            if key == "x0":
                snapped_edge["x1"] = average_position

            snapped_edges.append(snapped_edge)

    return snapped_edges


def _join_edges(edges: list[dict], tolerance: float) -> list[dict]:
    """
    Join broken line segments that are collinear and close together.
    """
    horizontal_edges = [edge for edge in edges if edge["orientation"] == "h"]
    vertical_edges = [edge for edge in edges if edge["orientation"] == "v"]

    joined_horizontal = _join_horizontal_edges(horizontal_edges, tolerance)
    joined_vertical = _join_vertical_edges(vertical_edges, tolerance)

    return joined_horizontal + joined_vertical


def _join_horizontal_edges(edges: list[dict], tolerance: float) -> list[dict]:
    """
    Join horizontal edges that are on the same Y coordinate and close together in X.
    """
    if not edges:
        return []

    sorted_edges = sorted(edges, key=lambda edge: (round(edge["top"], 1), edge["x0"]))

    joined = []
    current = dict(sorted_edges[0])

    for edge in sorted_edges[1:]:
        same_line = abs(edge["top"] - current["top"]) <= tolerance
        touches_or_overlaps = edge["x0"] <= current["x1"] + tolerance

        if same_line and touches_or_overlaps:
            current["x1"] = max(current["x1"], edge["x1"])
            current["length"] = current["x1"] - current["x0"]
        else:
            joined.append(current)
            current = dict(edge)

    joined.append(current)
    return joined


def _join_vertical_edges(edges: list[dict], tolerance: float) -> list[dict]:
    """
    Join vertical edges that are on the same X coordinate and close together in Y.
    Fixed to step up from the lowest Y bounds (bottom) to the highest Y bounds (top).
    """
    if not edges:
        return []

    # Fix: Sort by X first, then from lowest Y value (bottom) ascending upwards
    sorted_edges = sorted(edges, key=lambda edge: (round(edge["x0"], 1), edge["bottom"]))

    joined = []
    current = dict(sorted_edges[0])

    for edge in sorted_edges[1:]:
        same_line = abs(edge["x0"] - current["x0"]) <= tolerance
        # Fix: The bottom Y boundary of the next segment up should overlap or sit near current's top boundary
        touches_or_overlaps = edge["bottom"] <= current["top"] + tolerance

        if same_line and touches_or_overlaps:
            current["top"] = max(current["top"], edge["top"])
            current["length"] = current["top"] - current["bottom"]
        else:
            joined.append(current)
            current = dict(edge)

    joined.append(current)
    return joined


def extract_tables_pdfplumber(
    plumber_page,
    page_num: int,
    effective_page_height_pt: float,
    effective_page_width_pt: float,
    zones: list[dict] | None = None,
) -> tuple[list[dict], list[list[float]]]:
    """
    Extract bordered tables from an already-open pdfplumber page.

    Returns:
        table_artifacts:
            structured table objects with rows + bbox_pdfium + bbox_plumber

        tier1_bboxes_plumber:
            pdfplumber-space table boxes used later to prevent duplicate TATR extraction
    """
    try:
        zones = zones or []

        strict_fast_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3.0,
            "intersection_tolerance": 3.0,
        }

        tables = plumber_page.find_tables(table_settings=strict_fast_settings)

        artifacts = []
        tier1_bboxes_plumber = []

        spotlight_plumber_bboxes = [
            zone["bbox_plumber"]
            for zone in zones
            if zone.get("type") in SPOTLIGHT_TYPES and zone.get("bbox_plumber")
        ]

        for table_index, table in enumerate(tables):
            rows = table.extract()

            if not rows or len(rows) < 2:
                continue

            if not rows[0] or len(rows[0]) < 2:
                continue

            total_cells = sum(len(row) for row in rows)
            if total_cells == 0:
                continue

            empty_cells = sum(
                1
                for row in rows
                for cell in row
                if cell is None or not str(cell).strip()
            )

            empty_rate = empty_cells / total_cells
            if empty_rate > 0.80:
                continue

            bbox_plumber = list(table.bbox)

            if _overlaps_spotlight(bbox_plumber, spotlight_plumber_bboxes):
                continue

            bbox_pdfium = list(
                plumber_to_pdfium_coords(
                    bbox_plumber,
                    effective_page_height_pt,
                )
            )

            artifact = {
                "table_id": f"bordered_{page_num}_{table_index}",
                "source": "pdfplumber_bordered",
                "bbox_px": None,
                "bbox_pdfium": bbox_pdfium,
                "bbox_plumber": bbox_plumber,
                "rows": [
                    [str(cell).strip() if cell is not None else "" for cell in row]
                    for row in rows
                ],
                "confidence": 1.0,
                "warnings": [],
            }

            artifacts.append(artifact)
            tier1_bboxes_plumber.append(bbox_plumber)


        return artifacts, tier1_bboxes_plumber

    except Exception as e:
        logger.warning(
        "extract_tables_pdfplumber failed on page %d: %s",
        page_num, e
        )
        return [], []


def _overlaps_spotlight(
    table_bbox: list[float],
    spotlight_bboxes: list[list[float]],
    threshold: float = 0.10,
) -> bool:
    """
    Prevent chart/figure boxes from being falsely accepted as bordered tables.
    """
    tx0, tt, tx1, tb = table_bbox
    table_area = max(0.0, tx1 - tx0) * max(0.0, tb - tt)

    if table_area <= 0:
        logger.warning("_overlaps_spotlight: zero-area table bbox %s", table_bbox)
        return True 
    
    for sx0, st, sx1, sb in spotlight_bboxes:
        inter_w = max(0.0, min(tx1, sx1) - max(tx0, sx0))
        inter_h = max(0.0, min(tb, sb) - max(tt, st))
        inter_area = inter_w * inter_h

        if inter_area / table_area > threshold:
            return True

    return False

