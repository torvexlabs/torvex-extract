from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


"""
OmniDocBench subset JSON
        ↓
find each page image
        ↓
convert image page to one-page PDF
        ↓
run Torvex engine on that PDF
        ↓
save Torvex normalized JSON
        ↓
convert Torvex normalized JSON to OmniDocBench .md prediction
"""

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from PIL import Image

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from omnidocbench_markdown import export_markdown_prediction

from torvex_extract.formula_extractor import shutdown_formula_extractor
from torvex_extract.pypdfium_extractor import extract_with_pypdfium2
from torvex_extract.visual_zoning import engine


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [json_safe(v) for v in value]

    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()

    if np is not None and isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    return value


def image_path_from_sample(sample: dict[str, Any]) -> str:
    page_info = sample.get("page_info") or {}
    image_path = str(page_info.get("image_path") or "").strip()
    if not image_path:
        raise ValueError("Sample is missing page_info.image_path")
    return image_path


def prediction_stem_from_image_path(image_path: str) -> str:
    return Path(Path(image_path).name).stem


def image_to_single_page_pdf(image_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as img:
        if img.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", img.size, "white")
            alpha = img.getchannel("A")
            background.paste(img.convert("RGB"), mask=alpha)
            img = background
        else:
            img = img.convert("RGB")

        img.save(pdf_path, "PDF", resolution=200.0)


def find_image(raw_images_dir: Path, gt_image_path: str) -> Path:
    image_name = Path(gt_image_path).name

    candidates = [
        raw_images_dir / image_name,
        raw_images_dir / gt_image_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(raw_images_dir.rglob(image_name))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Image not found for {gt_image_path!r} under {raw_images_dir}")


PROFILE_TIMING_KEYS = [
    "page_open",
    "probe_text",
    "ocr_classify",
    "detect_table_fast",
    "render",
    "doclayout",
    "zone_postprocess",
    "scanned_page_ocr",
    "scanned_safe_zone_assignment",
    "scanned_tatr",
    "scanned_table_extract",
    "scanned_page_total",
    "digital_page_total",
    "formula_mfr",
    "page_total",
]

PROFILE_COLUMNS = [
    "index",
    "gt_image_path",
    "elapsed_ms",
    "pages",
    "errors",
    "ram_before_mb",
    "ram_after_mb",
    "vram_before_mb",
    "vram_after_mb",
    *[f"{key}_ms" for key in PROFILE_TIMING_KEYS],
]


def round_optional(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def process_ram_mb() -> float | None:
    if psutil is None:
        return None

    try:
        proc = psutil.Process()
        total = proc.memory_info().rss

        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:
                pass

        return total / (1024.0 * 1024.0)
    except Exception:
        return None


def gpu_vram_used_mb() -> float | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        return None

    values: list[float] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            pass

    return max(values) if values else None


class MemorySampler:
    def __init__(self, interval_sec: float = 0.25) -> None:
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.ram_start_mb: float | None = None
        self.ram_peak_mb: float | None = None
        self.ram_end_mb: float | None = None
        self.vram_start_mb: float | None = None
        self.vram_peak_mb: float | None = None
        self.vram_end_mb: float | None = None

    def start(self) -> None:
        self.ram_start_mb = process_ram_mb()
        self.vram_start_mb = gpu_vram_used_mb()
        self.ram_peak_mb = self.ram_start_mb
        self.vram_peak_mb = self.vram_start_mb

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

        self.ram_end_mb = process_ram_mb()
        self.vram_end_mb = gpu_vram_used_mb()
        self._sample_once()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            self._sample_once()

    def _sample_once(self) -> None:
        ram = process_ram_mb()
        vram = gpu_vram_used_mb()

        if ram is not None:
            self.ram_peak_mb = ram if self.ram_peak_mb is None else max(self.ram_peak_mb, ram)

        if vram is not None:
            self.vram_peak_mb = vram if self.vram_peak_mb is None else max(self.vram_peak_mb, vram)


def timing_summary_from_pages(pages: list[dict[str, Any]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {f"{key}_ms": None for key in PROFILE_TIMING_KEYS}

    for page in pages:
        metadata = page.get("metadata") or {}
        timings = metadata.get("timings_ms") or {}
        if not isinstance(timings, dict):
            continue

        for key in PROFILE_TIMING_KEYS:
            value = timings.get(key)
            if not isinstance(value, (int, float)):
                continue

            column = f"{key}_ms"
            out[column] = float(value) if out[column] is None else float(out[column]) + float(value)

    return {key: round_optional(value) for key, value in out.items()}


def write_profile_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_normalized_payload(
    *,
    output_path: Path,
    pdf_path: Path,
    device: str,
    formula_device: str,
    ocr_backend: str,
    enable_formula: bool,
    pages: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    elapsed_ms: float,
    gt_image_path: str,
    sample_index: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "pdf": str(pdf_path),
        "gt_image_path": gt_image_path,
        "sample_index": sample_index,
        "device": device,
        "formula_device": formula_device,
        "ocr_backend": ocr_backend,
        "formula_enabled": enable_formula,
        "engine": "torvex_extract",
        "summary": {
            "pages": len(pages),
            "errors": len(errors),
            "elapsed_ms": round(elapsed_ms, 2),
            "ms_per_page": round(elapsed_ms / max(1, len(pages)), 2),
            "text_pages": sum(1 for page in pages if str(page.get("final_text") or "").strip()),
            "table_count": sum(len(page.get("tables") or []) for page in pages),
            "formula_bbox_count": sum(len(page.get("formula_bboxes") or []) for page in pages),
            "formula_artifact_count": sum(len(page.get("formulas") or []) for page in pages),
            "formula_latex_count": sum(
                1
                for page in pages
                for formula in (page.get("formulas") or [])
                if str(formula.get("latex") or "").strip()
            ),
        },
        "errors": errors,
        "pages": pages,
    }

    output_path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clean_run_dirs(*dirs: Path) -> None:
    for directory in dirs:
        if directory.exists():
            shutil.rmtree(directory)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Torvex on an OmniDocBench subset and export prediction markdown."
    )
    parser.add_argument(
        "--subset-json",
        type=Path,
        required=True,
        help="Subset GT JSON, e.g. .bench/omnidocbench/subsets/subset_3.json",
    )
    parser.add_argument(
        "--raw-images-dir",
        type=Path,
        default=Path(".bench/omnidocbench/raw/images"),
        help="Directory containing OmniDocBench page images.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(".bench/omnidocbench"),
        help="Local ignored OmniDocBench workspace root.",
    )
    parser.add_argument("--run-id", required=True, help="Run id, e.g. run_001")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument(
        "--formula-device",
        choices=["cpu", "gpu"],
        default=None,
        help="UniMERNet formula device. Defaults to --device.",
    )
    parser.add_argument(
        "--ocr-backend",
        choices=["onnxtr_fast_base", "ppocrv6_small"],
        default="ppocrv6_small",
    )
    parser.add_argument("--enable-formula", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep-pdfs", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--trust-model",
        action="store_true",
        help="Bypass all formula validation/splitting/salvage. Raw bbox -> UniMERNet -> raw LaTeX.",
    )

    args = parser.parse_args()
    formula_device = args.formula_device or args.device

    if args.trust_model:
        import os as _os
        _os.environ["TORVEX_FORMULA_TRUST_MODEL"] = "true"

    subset_json = args.subset_json
    samples = json.loads(subset_json.read_text(encoding="utf-8-sig"))
    if not isinstance(samples, list):
        raise ValueError(f"Subset JSON must contain a list: {subset_json}")

    if args.limit is not None:
        samples = samples[: args.limit]

    normalized_dir = args.work_root / "normalized" / args.run_id
    predictions_dir = args.work_root / "predictions" / args.run_id
    pdf_dir = args.work_root / "work_pdfs" / args.run_id
    logs_dir = args.work_root / "logs"
    summary_path = logs_dir / f"{args.run_id}_prediction_summary.json"
    profile_path = logs_dir / f"{args.run_id}_profile.csv"

    if args.clean:
        clean_run_dirs(normalized_dir, predictions_dir, pdf_dir)

    normalized_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    sampler = MemorySampler()
    warm_elapsed_ms: float | None = None

    print(f"[torvex-odb] subset: {subset_json}")
    print(f"[torvex-odb] samples: {len(samples)}")
    print(f"[torvex-odb] predictions: {predictions_dir}")
    print(f"[torvex-odb] normalized: {normalized_dir}")
    print(f"[torvex-odb] device: {args.device}")
    print(f"[torvex-odb] formula_device: {formula_device}")
    print(f"[torvex-odb] ocr_backend: {args.ocr_backend}")
    print(f"[torvex-odb] formula: {args.enable_formula}")
    print(f"[torvex-odb] trust_model: {args.trust_model}")

    sampler.start()

    try:
        warm_started = time.perf_counter()
        engine.warm(device=args.device, ocr_backend=args.ocr_backend)
        warm_elapsed_ms = (time.perf_counter() - warm_started) * 1000.0

        for index, sample in enumerate(samples):
            gt_image_path = image_path_from_sample(sample)
            stem = prediction_stem_from_image_path(gt_image_path)

            image_path = find_image(args.raw_images_dir, gt_image_path)
            pdf_path = pdf_dir / f"{stem}.pdf"
            normalized_path = normalized_dir / f"{stem}.json"

            print(f"[{index + 1}/{len(samples)}] {gt_image_path}")

            image_to_single_page_pdf(image_path, pdf_path)

            ram_before_mb = process_ram_mb()
            vram_before_mb = gpu_vram_used_mb()
            started = time.perf_counter()
            pages, errors = extract_with_pypdfium2(
                str(pdf_path),
                enable_formula=args.enable_formula,
                formula_device=formula_device,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            ram_after_mb = process_ram_mb()
            vram_after_mb = gpu_vram_used_mb()
            timings = timing_summary_from_pages(pages)

            write_normalized_payload(
                output_path=normalized_path,
                pdf_path=pdf_path,
                device=args.device,
                formula_device=formula_device,
                ocr_backend=args.ocr_backend,
                enable_formula=args.enable_formula,
                pages=pages,
                errors=errors,
                elapsed_ms=elapsed_ms,
                gt_image_path=gt_image_path,
                sample_index=index,
            )

            prediction_path = export_markdown_prediction(
                normalized_json_path=normalized_path,
                output_dir=predictions_dir,
                gt_image_path=gt_image_path,
            )

            row = {
                "index": index,
                "gt_image_path": gt_image_path,
                "image_path": str(image_path),
                "pdf_path": str(pdf_path),
                "normalized_path": str(normalized_path),
                "prediction_path": str(prediction_path),
                "elapsed_ms": round(elapsed_ms, 2),
                "errors": len(errors),
                "pages": len(pages),
            }
            rows.append(row)
            profile_rows.append(
                {
                    "index": index,
                    "gt_image_path": gt_image_path,
                    "elapsed_ms": round_optional(elapsed_ms),
                    "pages": len(pages),
                    "errors": len(errors),
                    "ram_before_mb": round_optional(ram_before_mb),
                    "ram_after_mb": round_optional(ram_after_mb),
                    "vram_before_mb": round_optional(vram_before_mb),
                    "vram_after_mb": round_optional(vram_after_mb),
                    **timings,
                }
            )

            print(
                f"  -> {prediction_path.name} "
                f"{elapsed_ms:.1f} ms errors={len(errors)} pages={len(pages)}"
            )

    finally:
        shutdown_formula_extractor()
        engine.shutdown()
        sampler.stop()

        if not args.keep_pdfs and pdf_dir.exists():
            shutil.rmtree(pdf_dir)

    total_elapsed_ms = (time.perf_counter() - run_started) * 1000.0
    write_profile_csv(profile_path, profile_rows)

    summary = {
        "run_id": args.run_id,
        "subset_json": str(subset_json),
        "raw_images_dir": str(args.raw_images_dir),
        "normalized_dir": str(normalized_dir),
        "predictions_dir": str(predictions_dir),
        "device": args.device,
        "formula_device": formula_device,
        "ocr_backend": args.ocr_backend,
        "formula_enabled": args.enable_formula,
        "profile_path": str(profile_path),
        "samples": len(samples),
        "warm_elapsed_ms": round_optional(warm_elapsed_ms),
        "total_elapsed_ms": round(total_elapsed_ms, 2),
        "avg_elapsed_ms": round(total_elapsed_ms / max(1, len(samples)), 2),
        "ram_start_mb": round_optional(sampler.ram_start_mb),
        "ram_peak_mb": round_optional(sampler.ram_peak_mb),
        "ram_end_mb": round_optional(sampler.ram_end_mb),
        "vram_start_mb": round_optional(sampler.vram_start_mb),
        "vram_peak_mb": round_optional(sampler.vram_peak_mb),
        "vram_end_mb": round_optional(sampler.vram_end_mb),
        "errors": sum(row["errors"] for row in rows),
        "rows": rows,
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[torvex-odb] summary: {summary_path}")
    print(
        f"[torvex-odb] done samples={len(samples)} "
        f"errors={summary['errors']} avg_ms={summary['avg_elapsed_ms']}"
    )

    return 0 if summary["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
