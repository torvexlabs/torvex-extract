from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from torvex_extract.pypdfium_extractor import extract_with_pypdfium2
from torvex_extract.visual_zoning import engine


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    return value


def _summary(pages: list[dict[str, Any]], errors: list[dict[str, Any]], elapsed_ms: float) -> dict[str, Any]:
    return {
        "pages": len(pages),
        "errors": len(errors),
        "elapsed_ms": round(elapsed_ms, 2),
        "ms_per_page": round(elapsed_ms / max(1, len(pages)), 2),
        "text_pages": sum(1 for page in pages if str(page.get("final_text") or "").strip()),
        "table_count": sum(len(page.get("tables") or []) for page in pages),
        "spotlight_count": sum(len(page.get("spotlight_bboxes") or []) for page in pages),
        "formula_count": sum(len(page.get("formula_bboxes") or []) for page in pages),
        "ocr_pages": sum(1 for page in pages if page.get("needs_ocr")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="torvex-extract",
        description="Extract PDF text, tables, layout zones, spotlight boxes, and formula boxes.",
    )

    parser.add_argument(
        "pdf",
        help="input PDF path",
    )

    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="output JSON path. If omitted, writes next to the PDF.",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="ONNX inference device for layout/table models. Default: cpu.",
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="write pretty indented JSON",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        print(f"[FAIL] PDF not found: {pdf_path}")
        return 1

    if not pdf_path.is_file():
        print(f"[FAIL] Not a file: {pdf_path}")
        return 1

    output_path = Path(args.out) if args.out else pdf_path.with_suffix(".torvex.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[torvex-extract] input:  {pdf_path}")
    print(f"[torvex-extract] output: {output_path}")
    print(f"[torvex-extract] device: {args.device}")

    started = time.perf_counter()

    try:
        engine.warm(device=args.device)

        pages, errors = extract_with_pypdfium2(str(pdf_path))

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        payload = {
            "pdf": str(pdf_path),
            "device": args.device,
            "engine": "torvex_extract",
            "summary": _summary(pages, errors, elapsed_ms),
            "errors": errors,
            "pages": pages,
        }

        output_path.write_text(
            json.dumps(
                _json_safe(payload),
                indent=2 if args.pretty else None,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        print(json.dumps(payload["summary"], indent=2))

        if errors:
            print("[DONE_WITH_ERRORS]")
            return 2

        print("[DONE]")
        return 0

    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1

    finally:
        engine.shutdown()