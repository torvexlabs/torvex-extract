from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from torvex_extract.pypdfium_extractor import extract_with_pypdfium2
from torvex_extract.visual_zoning import (
    FORMULA_ZONE_TYPES,
    RENDER_DPI,
    SPOTLIGHT_TYPES,
    TRIGGER_ZONE_TYPES,
    engine,
)


REQUIRED_PAGE_KEYS = {
    "page_num",
    "is_tagged",
    "needs_ocr",
    "ocr_reason",
    "final_text",
    "page_width",
    "page_height",
    "effective_page_width_pt",
    "effective_page_height_pt",
    "image",
    "img_np",
    "has_bordered_table",
    "zones",
    "tier1_bboxes",
    "spotlight_bboxes",
    "formula_bboxes",
    "tables",
    "metadata",
    "layout_grade",
    "page_class",
}

REQUIRED_ZONE_KEYS = {
    "type",
    "label_id",
    "score",
    "mask_id",
    "bbox_px",
    "bbox_pdfium",
    "bbox_plumber",
}

REQUIRED_FORMULA_KEYS = {
    "formula_id",
    "type",
    "score",
    "bbox_px",
    "bbox_pdfium",
    "bbox_plumber",
}

ALLOWED_LAYOUT_GRADES = {"POOR", "FAIR", "GOOD", "EXCELLENT", ""}
ALLOWED_PAGE_CLASSES = {"zero_zones", "text_only", "mixed", "unknown"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    return value

class ResourceMonitor:
    def __init__(self, interval_sec: float = 0.25) -> None:
        self.interval_sec = interval_sec
        self.pid = os.getpid()
        self.proc = psutil.Process(self.pid)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

        self.start_ram_mb = 0.0
        self.end_ram_mb = 0.0
        self.peak_ram_mb = 0.0

        self.start_vram_used_mb = 0.0
        self.end_vram_used_mb = 0.0
        self.peak_vram_used_mb = 0.0

    def _ram_mb(self) -> float:
        total = 0

        try:
            total += self.proc.memory_info().rss
        except psutil.Error:
            pass

        try:
            for child in self.proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.Error:
                    pass
        except psutil.Error:
            pass

        return total / (1024 * 1024)

    def _total_gpu_used_mb(self) -> float:
        """
        Return total used VRAM across all NVIDIA GPUs.

        This is more reliable than per-process VRAM on Windows/WDDM,
        where nvidia-smi compute-app process reporting can miss Python.
        """
        if shutil.which("nvidia-smi") is None:
            return 0.0

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return 0.0

        total = 0.0

        for line in result.stdout.splitlines():
            try:
                total += float(line.strip())
            except ValueError:
                continue

        return total

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.peak_ram_mb = max(self.peak_ram_mb, self._ram_mb())
            self.peak_vram_used_mb = max(
                self.peak_vram_used_mb,
                self._total_gpu_used_mb(),
            )
            time.sleep(self.interval_sec)

    def start(self) -> None:
        self.start_ram_mb = self._ram_mb()
        self.peak_ram_mb = self.start_ram_mb

        self.start_vram_used_mb = self._total_gpu_used_mb()
        self.peak_vram_used_mb = self.start_vram_used_mb

        self.thread.start()

    def stop(self) -> dict[str, float]:
        self.stop_event.set()
        self.thread.join(timeout=3)

        self.end_ram_mb = self._ram_mb()
        self.peak_ram_mb = max(self.peak_ram_mb, self.end_ram_mb)

        self.end_vram_used_mb = self._total_gpu_used_mb()
        self.peak_vram_used_mb = max(
            self.peak_vram_used_mb,
            self.end_vram_used_mb,
        )

        peak_vram_delta_mb = max(
            0.0,
            self.peak_vram_used_mb - self.start_vram_used_mb,
        )

        return {
            "start_ram_mb": round(self.start_ram_mb, 2),
            "end_ram_mb": round(self.end_ram_mb, 2),
            "peak_ram_mb": round(self.peak_ram_mb, 2),

            "start_vram_used_mb": round(self.start_vram_used_mb, 2),
            "end_vram_used_mb": round(self.end_vram_used_mb, 2),
            "peak_vram_used_mb": round(self.peak_vram_used_mb, 2),
            "peak_vram_delta_mb": round(peak_vram_delta_mb, 2),
        }


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(_is_number(v) for v in value)
    )


def _bbox_positive_area(bbox: list[float]) -> bool:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return x1 > x0 and y1 > y0


def _bbox_inside_bounds(
    bbox: list[float],
    *,
    max_x: float,
    max_y: float,
    tolerance: float = 3.0,
) -> bool:
    x0, y0, x1, y1 = [float(v) for v in bbox]

    return (
        x0 >= -tolerance
        and y0 >= -tolerance
        and x1 <= max_x + tolerance
        and y1 <= max_y + tolerance
    )


def _page_pixel_bounds(page: dict[str, Any]) -> tuple[float, float]:
    width_pt = float(page.get("effective_page_width_pt") or page.get("page_width") or 0.0)
    height_pt = float(page.get("effective_page_height_pt") or page.get("page_height") or 0.0)

    width_px = width_pt * (RENDER_DPI / 72.0)
    height_px = height_pt * (RENDER_DPI / 72.0)

    return width_px, height_px


def _add_failure(failures: list[str], page_num: Any, message: str) -> None:
    failures.append(f"page {page_num}: {message}")


def _check_page_contract(page: dict[str, Any], failures: list[str]) -> None:
    page_num = page.get("page_num", "?")

    missing = REQUIRED_PAGE_KEYS - set(page)
    if missing:
        _add_failure(failures, page_num, f"missing page keys: {sorted(missing)}")

    if page.get("image") is not None:
        _add_failure(failures, page_num, "image was not cleared before return")

    if page.get("img_np") is not None:
        _add_failure(failures, page_num, "img_np was not cleared before return")

    if page.get("layout_grade") not in ALLOWED_LAYOUT_GRADES:
        _add_failure(failures, page_num, f"invalid layout_grade={page.get('layout_grade')!r}")

    if page.get("page_class") not in ALLOWED_PAGE_CLASSES:
        _add_failure(failures, page_num, f"invalid page_class={page.get('page_class')!r}")

    if not isinstance(page.get("metadata"), dict):
        _add_failure(failures, page_num, "metadata must be dict")

    for key in ("zones", "tier1_bboxes", "spotlight_bboxes", "formula_bboxes", "tables"):
        if not isinstance(page.get(key), list):
            _add_failure(failures, page_num, f"{key} must be list")

    for dim_key in ("page_width", "page_height", "effective_page_width_pt", "effective_page_height_pt"):
        value = page.get(dim_key)
        if not _is_number(value) or float(value) < 0:
            _add_failure(failures, page_num, f"{dim_key} must be non-negative number")


def _check_zone_contract(page: dict[str, Any], failures: list[str]) -> None:
    page_num = page.get("page_num", "?")
    width_px, height_px = _page_pixel_bounds(page)
    width_pt = float(page.get("effective_page_width_pt") or page.get("page_width") or 0.0)
    height_pt = float(page.get("effective_page_height_pt") or page.get("page_height") or 0.0)

    for index, zone in enumerate(page.get("zones") or []):
        if not isinstance(zone, dict):
            _add_failure(failures, page_num, f"zone {index} must be dict")
            continue

        missing = REQUIRED_ZONE_KEYS - set(zone)
        if missing:
            _add_failure(failures, page_num, f"zone {index} missing keys: {sorted(missing)}")

        zone_type = zone.get("type", "unknown")

        if not isinstance(zone_type, str) or not zone_type:
            _add_failure(failures, page_num, f"zone {index} invalid type")

        if not _is_number(zone.get("score")):
            _add_failure(failures, page_num, f"zone {index} score must be number")

        bbox_px = zone.get("bbox_px")
        bbox_pdfium = zone.get("bbox_pdfium")
        bbox_plumber = zone.get("bbox_plumber")

        if _is_bbox(bbox_px):
            if not _bbox_positive_area(bbox_px):
                _add_failure(failures, page_num, f"zone {index} bbox_px non-positive area")
            if width_px > 0 and height_px > 0 and not _bbox_inside_bounds(bbox_px, max_x=width_px, max_y=height_px):
                _add_failure(failures, page_num, f"zone {index} bbox_px outside page bounds")
        else:
            _add_failure(failures, page_num, f"zone {index} bbox_px invalid")

        if _is_bbox(bbox_pdfium):
            if not _bbox_positive_area(bbox_pdfium):
                _add_failure(failures, page_num, f"zone {index} bbox_pdfium non-positive area")
            if width_pt > 0 and height_pt > 0 and not _bbox_inside_bounds(bbox_pdfium, max_x=width_pt, max_y=height_pt):
                _add_failure(failures, page_num, f"zone {index} bbox_pdfium outside page bounds")
        else:
            _add_failure(failures, page_num, f"zone {index} bbox_pdfium invalid")

        if _is_bbox(bbox_plumber):
            if not _bbox_positive_area(bbox_plumber):
                _add_failure(failures, page_num, f"zone {index} bbox_plumber non-positive area")
            if width_pt > 0 and height_pt > 0 and not _bbox_inside_bounds(bbox_plumber, max_x=width_pt, max_y=height_pt):
                _add_failure(failures, page_num, f"zone {index} bbox_plumber outside page bounds")
        else:
            _add_failure(failures, page_num, f"zone {index} bbox_plumber invalid")

        if zone_type in SPOTLIGHT_TYPES and str(zone.get("zone_text") or "").strip():
            _add_failure(failures, page_num, f"spotlight zone {index} leaked text")

        if zone_type in FORMULA_ZONE_TYPES and str(zone.get("zone_text") or "").strip():
            _add_failure(failures, page_num, f"formula zone {index} leaked text")

        if zone_type in TRIGGER_ZONE_TYPES:
            has_allowed_fallback = bool(zone.get("degraded_table_text_fallback"))
            has_text = bool(str(zone.get("zone_text") or "").strip())

            if has_text and not has_allowed_fallback:
                _add_failure(failures, page_num, f"table zone {index} leaked text without fallback flag")

        unsafe_reason = str(zone.get("metadata", {}).get("unsafe_reason", "")).lower()
        if "formula" in unsafe_reason:
            _add_failure(failures, page_num, f"zone {index} was made unsafe because of formula")


def _check_formula_contract(page: dict[str, Any], failures: list[str]) -> None:
    page_num = page.get("page_num", "?")
    zones = page.get("zones") or []
    formula_bboxes = page.get("formula_bboxes") or []

    formula_zones = [
        zone for zone in zones
        if isinstance(zone, dict) and zone.get("type") in FORMULA_ZONE_TYPES
    ]

    if len(formula_bboxes) != len(formula_zones):
        _add_failure(
            failures,
            page_num,
            f"formula_bboxes count {len(formula_bboxes)} != formula zone count {len(formula_zones)}",
        )

    formula_ids: set[str] = set()

    for index, item in enumerate(formula_bboxes):
        if not isinstance(item, dict):
            _add_failure(failures, page_num, f"formula bbox {index} must be dict")
            continue

        missing = REQUIRED_FORMULA_KEYS - set(item)
        if missing:
            _add_failure(failures, page_num, f"formula bbox {index} missing keys: {sorted(missing)}")

        formula_id = item.get("formula_id")
        if not isinstance(formula_id, str) or not formula_id:
            _add_failure(failures, page_num, f"formula bbox {index} invalid formula_id")
        elif formula_id in formula_ids:
            _add_failure(failures, page_num, f"duplicate formula_id={formula_id}")
        else:
            formula_ids.add(formula_id)

        if item.get("type") not in FORMULA_ZONE_TYPES:
            _add_failure(failures, page_num, f"formula bbox {index} invalid type={item.get('type')!r}")

        if not _is_number(item.get("score")):
            _add_failure(failures, page_num, f"formula bbox {index} score must be number")

        for bbox_key in ("bbox_px", "bbox_pdfium", "bbox_plumber"):
            bbox = item.get(bbox_key)
            if not _is_bbox(bbox):
                _add_failure(failures, page_num, f"formula bbox {index} {bbox_key} invalid")
                continue

            if not _bbox_positive_area(bbox):
                _add_failure(failures, page_num, f"formula bbox {index} {bbox_key} non-positive area")


def _check_spotlight_contract(page: dict[str, Any], failures: list[str]) -> None:
    page_num = page.get("page_num", "?")
    spotlight_bboxes = page.get("spotlight_bboxes") or []

    spotlight_zones = [
        zone for zone in page.get("zones") or []
        if isinstance(zone, dict) and zone.get("type") in SPOTLIGHT_TYPES
    ]

    if len(spotlight_bboxes) != len(spotlight_zones):
        _add_failure(
            failures,
            page_num,
            f"spotlight_bboxes count {len(spotlight_bboxes)} != spotlight zone count {len(spotlight_zones)}",
        )

    for index, bbox in enumerate(spotlight_bboxes):
        if not _is_bbox(bbox):
            _add_failure(failures, page_num, f"spotlight bbox {index} invalid")
            continue

        if not _bbox_positive_area(bbox):
            _add_failure(failures, page_num, f"spotlight bbox {index} non-positive area")


def _check_table_contract(page: dict[str, Any], failures: list[str]) -> None:
    page_num = page.get("page_num", "?")
    seen_ids: set[str] = set()

    for index, table in enumerate(page.get("tables") or []):
        if not isinstance(table, dict):
            _add_failure(failures, page_num, f"table {index} must be dict")
            continue

        table_id = table.get("table_id")

        if not isinstance(table_id, str) or not table_id:
            _add_failure(failures, page_num, f"table {index} missing table_id")
            continue

        if table_id in seen_ids:
            _add_failure(failures, page_num, f"duplicate table_id={table_id}")
        seen_ids.add(table_id)

        if not isinstance(table.get("source"), str) or not table.get("source"):
            _add_failure(failures, page_num, f"table {table_id} missing source")

        rows = table.get("rows")
        raw_text = str(table.get("raw_text") or "").strip()

        if not isinstance(rows, list):
            _add_failure(failures, page_num, f"table {table_id} rows must be list")
            continue

        if not rows and not raw_text:
            _add_failure(failures, page_num, f"table {table_id} has neither rows nor raw_text")

        for row_index, row in enumerate(rows):
            if not isinstance(row, list):
                _add_failure(failures, page_num, f"table {table_id} row {row_index} must be list")
                continue

            for cell_index, cell in enumerate(row):
                if cell is not None and not isinstance(cell, str):
                    _add_failure(
                        failures,
                        page_num,
                        f"table {table_id} cell {row_index},{cell_index} must be str or None",
                    )

        warnings = table.get("warnings")
        if warnings is not None and not isinstance(warnings, list):
            _add_failure(failures, page_num, f"table {table_id} warnings must be list")

        for bbox_key in ("bbox_pdfium", "bbox_plumber"):
            bbox = table.get(bbox_key)
            if bbox is None:
                continue

            if not _is_bbox(bbox):
                _add_failure(failures, page_num, f"table {table_id} {bbox_key} invalid")
                continue

            if not _bbox_positive_area(bbox):
                _add_failure(failures, page_num, f"table {table_id} {bbox_key} non-positive area")


def _check_page_not_silent(page: dict[str, Any], failures: list[str]) -> None:
    page_num = page.get("page_num", "?")
    metadata = page.get("metadata") or {}

    has_text = bool(str(page.get("final_text") or "").strip())
    has_tables = bool(page.get("tables") or [])
    has_spotlight = bool(page.get("spotlight_bboxes") or [])
    has_formula = bool(page.get("formula_bboxes") or [])
    explicitly_discarded = metadata.get("chunk_eligible") is False

    if not (has_text or has_tables or has_spotlight or has_formula or explicitly_discarded):
        _add_failure(
            failures,
            page_num,
            "silent output: no text/tables/spotlight/formula and not explicitly discarded",
        )


def _check_json_serializable(payload: dict[str, Any], failures: list[str]) -> None:
    try:
        json.dumps(_json_safe(payload), allow_nan=False)
    except Exception as exc:
        failures.append(f"output is not strict JSON serializable: {exc}")


def validate_pages(pages: list[dict[str, Any]], errors: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []

    if errors:
        failures.append(f"engine returned errors: {errors}")

    if not isinstance(pages, list):
        failures.append("engine pages output is not list")
        return failures

    if not pages:
        failures.append("engine returned zero pages")
        return failures

    for page in pages:
        if not isinstance(page, dict):
            failures.append("page item is not dict")
            continue

        _check_page_contract(page, failures)
        _check_zone_contract(page, failures)
        _check_formula_contract(page, failures)
        _check_spotlight_contract(page, failures)
        _check_table_contract(page, failures)
        _check_page_not_silent(page, failures)

    _check_json_serializable({"pages": pages, "errors": errors}, failures)

    return failures


def summarize(
    pages: list[dict[str, Any]],
    elapsed_ms: float,
    device: str,
    resources: dict[str, float],
) -> dict[str, Any]:
    safe_pages = pages if isinstance(pages, list) else []

    formula_pages_1based = []
    formula_details = []

    for page in safe_pages:
        formulas = page.get("formula_bboxes") or []

        if not formulas:
            continue

        page_index = int(page.get("page_num", 0))
        page_number = page_index + 1

        formula_pages_1based.append(page_number)

        for formula in formulas:
            formula_details.append(
                {
                    "page_index_0based": page_index,
                    "page_number_1based": page_number,
                    "formula_id": formula.get("formula_id"),
                    "type": formula.get("type"),
                    "score": formula.get("score"),
                    "bbox_pdfium": formula.get("bbox_pdfium"),
                    "bbox_plumber": formula.get("bbox_plumber"),
                }
            )

    return {
        "pages": len(safe_pages),
        "device": device,
        "elapsed_ms": round(elapsed_ms, 2),
        "ms_per_page": round(elapsed_ms / max(1, len(safe_pages)), 2),
        "start_ram_mb": resources["start_ram_mb"],
        "end_ram_mb": resources["end_ram_mb"],
        "peak_ram_mb": resources["peak_ram_mb"],
        "start_vram_used_mb": resources["start_vram_used_mb"],
        "end_vram_used_mb": resources["end_vram_used_mb"],
        "peak_vram_used_mb": resources["peak_vram_used_mb"],
        "peak_vram_delta_mb": resources["peak_vram_delta_mb"],
        "text_pages": sum(1 for p in safe_pages if str(p.get("final_text") or "").strip()),
        "table_count": sum(len(p.get("tables") or []) for p in safe_pages),
        "spotlight_count": sum(len(p.get("spotlight_bboxes") or []) for p in safe_pages),
        "formula_count": sum(len(p.get("formula_bboxes") or []) for p in safe_pages),
        "formula_pages_1based": sorted(set(formula_pages_1based)),
        "formula_details": formula_details,
        "ocr_pages": sum(1 for p in safe_pages if p.get("needs_ocr")),
        "warnings_count": sum(len((p.get("metadata") or {}).get("warnings") or []) for p in safe_pages),
        "page_classes": {
            cls: sum(1 for p in safe_pages if p.get("page_class") == cls)
            for cls in ("zero_zones", "text_only", "mixed", "unknown")
        },
        "layout_grades": {
            grade: sum(1 for p in safe_pages if p.get("layout_grade") == grade)
            for grade in ("POOR", "FAIR", "GOOD", "EXCELLENT", "")
        },
    }


def run_one(pdf_path: Path, output_dir: Path, device: str) -> int:
    print(f"\n[smoke] {pdf_path}")
    print(f"[smoke] device={device}")

    pages: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    monitor = ResourceMonitor()
    monitor.start()

    started = time.perf_counter()

    try:
        pages, errors = extract_with_pypdfium2(str(pdf_path))
    except Exception as exc:
        errors = [{"exception": repr(exc)}]
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        resources = monitor.stop()

    failures = validate_pages(pages, errors)
    summary = summarize(
        pages=pages,
        elapsed_ms=elapsed_ms,
        device=device,
        resources=resources,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / f"{pdf_path.stem}_{device}_smoke_output.json"
    summary_json = output_dir / f"{pdf_path.stem}_{device}_smoke_summary.json"

    output_payload = _json_safe(
        {
            "pdf": str(pdf_path),
            "device": device,
            "errors": errors,
            "pages": pages,
        }
    )

    summary_payload = _json_safe(
        {
            "pdf": str(pdf_path),
            "device": device,
            "summary": summary,
            "failures": failures,
        }
    )

    output_json.write_text(
        json.dumps(output_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    summary_json.write_text(
        json.dumps(summary_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, allow_nan=False))

    if failures:
        print("[FAIL]")
        for failure in failures:
            print(f" - {failure}")

        print(f"output:  {output_json}")
        print(f"summary: {summary_json}")
        return 1

    print("[PASS]")
    print(f"output:  {output_json}")
    print(f"summary: {summary_json}")
    return 0


def discover_pdfs(pdf_args: list[str], pdf_dir: str) -> list[Path]:
    if pdf_args:
        return [Path(pdf) for pdf in pdf_args]

    root = Path(pdf_dir)

    if not root.exists():
        return []

    return sorted(root.glob("*.pdf"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Production-grade Torvex Extract engine smoke contract test."
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        help="PDF files to smoke-test. If omitted, all PDFs in --pdf-dir are used.",
    )
    parser.add_argument(
        "--pdf-dir",
        default="test_docs",
        help="default folder to scan when no PDF paths are provided",
    )
    parser.add_argument(
        "--output-dir",
        default="results/smoke",
        help="where to write smoke JSON outputs",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="device for layout/table ONNX inference. Default: cpu.",
    )

    args = parser.parse_args()

    pdf_paths = discover_pdfs(args.pdfs, args.pdf_dir)

    if not pdf_paths:
        print(f"[FAIL] no PDFs found. Pass paths or add PDFs to: {args.pdf_dir}")
        return 1

    missing_paths = [pdf for pdf in pdf_paths if not pdf.exists()]

    if missing_paths:
        print("[FAIL] missing PDF file(s):")
        for pdf in missing_paths:
            print(f" - {pdf}")
        return 1

    if not engine.is_warmed():
        print(f"[smoke] warming Torvex engine on {args.device}...")
        engine.warm(device=args.device)

    try:
        status = 0
        output_dir = Path(args.output_dir)

        for pdf_path in pdf_paths:
            status |= run_one(pdf_path, output_dir, args.device)

        return status

    finally:
        engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())