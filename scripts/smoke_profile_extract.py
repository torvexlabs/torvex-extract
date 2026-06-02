import argparse
import json
import os
import threading
import time
from pathlib import Path

import psutil


PROCESS = psutil.Process(os.getpid())


def rss_mb() -> float:
    return PROCESS.memory_info().rss / 1024 / 1024


class PeakRamSampler:
    def __init__(self, interval_sec: float = 0.1) -> None:
        self.interval_sec = interval_sec
        self.peak_mb = rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, rss_mb())
            time.sleep(self.interval_sec)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self.peak_mb = max(self.peak_mb, rss_mb())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    sampler = PeakRamSampler()
    sampler.start()

    total_t0 = time.perf_counter()
    ram_start = rss_mb()

    import_t0 = time.perf_counter()
    from torvex_extract.visual_zoning import engine
    from torvex_extract import extract_with_pypdfium2
    import_sec = time.perf_counter() - import_t0
    ram_after_import = rss_mb()

    warm_t0 = time.perf_counter()
    engine.warm()
    warm_sec = time.perf_counter() - warm_t0
    ram_after_warm = rss_mb()

    extract_t0 = time.perf_counter()
    pages, errors = extract_with_pypdfium2(str(pdf_path))
    extract_sec = time.perf_counter() - extract_t0
    ram_after_extract = rss_mb()

    sampler.stop()

    page_count = len(pages)
    avg_sec_per_page = extract_sec / page_count if page_count else None

    per_page = []
    for page in pages:
        timings = page.get("metadata", {}).get("timings_ms", {})
        per_page.append(
            {
                "page_num": page.get("page_num"),
                "page_total_ms": timings.get("page_total"),
                "needs_ocr": page.get("needs_ocr"),
                "ocr_used": page.get("ocr_used"),
                "ocr_reason": page.get("metadata", {}).get("ocr_reason"),
                "tables": len(page.get("tables", [])),
            }
        )

    report = {
        "pdf": str(pdf_path),
        "pages": page_count,
        "errors": len(errors),
        "import_sec": round(import_sec, 3),
        "model_warm_sec": round(warm_sec, 3),
        "extract_sec": round(extract_sec, 3),
        "overall_sec": round(time.perf_counter() - total_t0, 3),
        "avg_sec_per_page": round(avg_sec_per_page, 3) if avg_sec_per_page else None,
        "ram_mb": {
            "start": round(ram_start, 1),
            "after_import": round(ram_after_import, 1),
            "after_warm": round(ram_after_warm, 1),
            "after_extract": round(ram_after_extract, 1),
            "peak": round(sampler.peak_mb, 1),
            "model_load_delta": round(ram_after_warm - ram_after_import, 1),
            "overall_delta": round(ram_after_extract - ram_start, 1),
            "peak_delta": round(sampler.peak_mb - ram_start, 1),
        },
        "per_page": per_page,
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()