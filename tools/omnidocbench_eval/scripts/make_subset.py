from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def image_path(record: dict[str, Any]) -> str:
    page_info = record.get("page_info") or {}
    value = str(page_info.get("image_path") or "").strip()
    if not value:
        raise ValueError("record missing page_info.image_path")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create fixed OmniDocBench subset GT JSON files."
    )
    parser.add_argument(
        "--raw-json",
        type=Path,
        default=Path(".bench/omnidocbench/raw/OmniDocBench.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".bench/omnidocbench/subsets"),
    )
    parser.add_argument(
        "--limits",
        type=int,
        nargs="+",
        default=[3, 30, 100],
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index in OmniDocBench.json. Default: 0.",
    )

    args = parser.parse_args()

    records = read_json(args.raw_json)
    if not isinstance(records, list):
        raise ValueError(f"Expected list in {args.raw_json}")

    total = len(records)
    print(f"[make-subset] raw records: {total}")

    for limit in args.limits:
        if limit <= 0:
            raise ValueError(f"Limit must be positive: {limit}")

        start = args.offset
        end = start + limit
        if end > total:
            raise ValueError(
                f"Requested offset={start} limit={limit}, but dataset has {total}"
            )

        subset = records[start:end]
        out_path = args.output_dir / f"subset_{limit}.json"
        write_json(out_path, subset)

        first = image_path(subset[0])
        last = image_path(subset[-1])
        print(
            f"[make-subset] wrote {out_path} "
            f"records={len(subset)} first={first} last={last}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())