from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


CATEGORY_COLORS = {
    "table": "orange",
    "figure": "green",
    "text_block": "blue",
    "text_span": "#07689f",
    "equation_inline": "#590d82",
    "equation_isolated": "red",
    "equation_ignore": "#769fcd",
    "header": "purple",
    "page_number": "gray",
    "title": "cyan",
    "abandon": "black",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def poly_to_bbox(poly: Any) -> list[float] | None:
    if not poly or len(poly) < 8:
        return None
    try:
        left = float(poly[0])
        top = float(poly[1])
        right = float(poly[2])
        bottom = float(poly[5])
    except Exception:
        return None
    return [min(left, right), min(top, bottom), max(left, right), max(top, bottom)]


def bbox(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        box = [float(v) for v in value]
    except Exception:
        return None
    if len(box) != 4:
        return None
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def image_name_from_record(record: dict[str, Any]) -> str:
    return Path(str(record["page_info"]["image_path"])).name


def image_stem_from_record(record: dict[str, Any]) -> str:
    return Path(image_name_from_record(record)).stem


def find_result_file(result_dir: Path, suffix: str) -> Path | None:
    matches = sorted(result_dir.glob(f"*{suffix}"))
    return matches[0] if matches else None


def flatten_metric_values(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_metric_values(item, child))
    elif isinstance(value, (int, float)):
        out[prefix] = float(value)
    return out


def load_metric_summary(result_dir: Path) -> dict[str, Any]:
    metric_path = find_result_file(result_dir, "_metric_result.json")
    run_summary_path = find_result_file(result_dir, "_run_summary.json")

    metric = read_json(metric_path) if metric_path else {}
    run_summary = read_json(run_summary_path) if run_summary_path else {}

    return {
        "metric_path": str(metric_path) if metric_path else "",
        "run_summary_path": str(run_summary_path) if run_summary_path else "",
        "metric": metric,
        "run_summary": run_summary,
        "flat_metrics": flatten_metric_values(metric),
    }


def load_json_by_suffix(result_dir: Path, suffix: str) -> Any:
    path = find_result_file(result_dir, suffix)
    if not path:
        return {}
    return read_json(path)


def load_normalized_by_stem(normalized_dir: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for path in normalized_dir.glob("*.json"):
        out[path.stem] = read_json(path)
    return out


def first_page(payload: dict[str, Any]) -> dict[str, Any]:
    pages = payload.get("pages") or []
    return pages[0] if pages else {}


def count_formula_status(page: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for formula in page.get("formulas") or []:
        status = str(formula.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def formula_debug_rows(
    *,
    run_name: str,
    normalized_by_stem: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for stem, payload in sorted(normalized_by_stem.items()):
        page = first_page(payload)
        for formula in page.get("formulas") or []:
            rows.append(
                {
                    "run": run_name,
                    "image_stem": stem,
                    "formula_id": formula.get("formula_id", ""),
                    "type": formula.get("type", ""),
                    "status": formula.get("status", ""),
                    "latex_len": len(str(formula.get("latex") or "")),
                    "latex": str(formula.get("latex") or "")[:500],
                    "validation_error": formula.get("validation_error", ""),
                    "token_count": formula.get("token_count", ""),
                    "eos_reached": formula.get("eos_reached", ""),
                    "truncated": formula.get("truncated", ""),
                    "mfr_elapsed_ms": formula.get("mfr_elapsed_ms", ""),
                    "mfr_io_binding": formula.get("mfr_io_binding", ""),
                    "quality_flags": "|".join(formula.get("quality_flags") or []),
                }
            )

    return rows


def numeric_values(value: Any) -> list[float]:
    values: list[float] = []
    if isinstance(value, dict):
        for item in value.values():
            values.extend(numeric_values(item))
    elif isinstance(value, list):
        for item in value:
            values.extend(numeric_values(item))
    elif isinstance(value, (int, float)):
        values.append(float(value))
    return values


def summarize_numeric_json(value: Any) -> dict[str, Any]:
    vals = numeric_values(value)
    if not vals:
        return {"count": 0, "avg": None, "min": None, "max": None}
    return {
        "count": len(vals),
        "avg": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
    }


def get_path(value: Any, path: list[str]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def as_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        if value.lower() == "nan":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    return None


def compact_scores(label: str, metrics: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    metric = metrics.get("metric") or {}
    cdm_per_sample = load_json_by_suffix(result_dir, "_display_formula_per_sample_CDM.json")
    cdm_values = numeric_values(cdm_per_sample)

    return {
        "run": label,
        "text_edit": as_float_or_none(
            get_path(metric, ["text_block", "all", "Edit_dist", "ALL_page_avg"])
        ),
        "reading_order_edit": as_float_or_none(
            get_path(metric, ["reading_order", "all", "Edit_dist", "ALL_page_avg"])
        ),
        "formula_edit": as_float_or_none(
            get_path(metric, ["display_formula", "all", "Edit_dist", "ALL_page_avg"])
        ),
        "formula_cdm": as_float_or_none(
            get_path(metric, ["display_formula", "all", "CDM", "all"])
        ),
        "formula_cdm_count": len(cdm_values),
        "table_teds": as_float_or_none(
            get_path(metric, ["table", "all", "TEDS", "ALL_page_avg"])
        ),
        "table_edit": as_float_or_none(
            get_path(metric, ["table", "all", "Edit_dist", "ALL_page_avg"])
        ),
        "metric_path": metrics.get("metric_path", ""),
    }


def draw_gt_overlay(
    *,
    record: dict[str, Any],
    images_dir: Path,
    output_path: Path,
) -> None:
    image_name = image_name_from_record(record)
    image_path = images_dir / image_name
    if not image_path.exists():
        matches = list(images_dir.rglob(image_name))
        if not matches:
            return
        image_path = matches[0]

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for anno in record.get("layout_dets") or []:
        category = str(anno.get("category_type") or "unknown")
        if "mask" in category or category == "abandon":
            continue

        attr = anno.get("attribute") or {}
        if category == "table" and attr.get("include_photo"):
            continue

        box = poly_to_bbox(anno.get("poly"))
        if not box:
            continue

        color = CATEGORY_COLORS.get(category, "yellow")
        draw.rectangle(box, outline=color, width=3)

        order = anno.get("order")
        label = f"{order}:{category}" if order is not None else category
        draw.text((box[0] + 2, box[1] + 2), label, fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def draw_torvex_overlay(
    *,
    stem: str,
    payload: dict[str, Any],
    images_dir: Path,
    output_path: Path,
) -> None:
    page = first_page(payload)
    image_path = images_dir / f"{stem}.png"
    if not image_path.exists():
        matches = list(images_dir.rglob(f"{stem}.*"))
        if not matches:
            return
        image_path = matches[0]

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    colors = {
        "text": "blue",
        "header": "purple",
        "number": "gray",
        "paragraph_title": "cyan",
        "table": "orange",
        "display_formula": "red",
        "inline_formula": "#590d82",
        "formula_number": "pink",
    }

    for zone in page.get("layout_zones") or page.get("zones") or []:
        zone_type = str(zone.get("type") or "unknown")
        box = bbox(zone.get("bbox_px"))
        if not box:
            continue
        color = colors.get(zone_type, "yellow")
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 2, box[1] + 2), zone_type, fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def markdown_preview(path: Path, max_chars: int = 2500) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8-sig", errors="replace")[:max_chars]


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            text = str(value).replace("\n", " ").replace("|", "\\|")
            vals.append(text)
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


RUNTIME_COLUMNS = [
    "run",
    "samples",
    "engine_avg_ms",
    "engine_total_ms",
    "warm_elapsed_ms",
    "ram_peak_mb",
    "vram_peak_mb",
    "render_ms",
    "doclayout_ms",
    "scanned_page_ocr_ms",
    "scanned_tatr_ms",
    "scanned_table_extract_ms",
    "formula_mfr_ms",
    "page_total_ms",
]


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def avg_csv_column(rows: list[dict[str, Any]], column: str) -> float | None:
    values: list[float] = []

    for row in rows:
        value = as_float_or_none(row.get(column))
        if value is not None:
            values.append(value)

    if not values:
        return None

    return sum(values) / len(values)


def runtime_summary(*, bench_root: Path, run_id: str, label: str) -> dict[str, Any]:
    summary_path = bench_root / "logs" / f"{run_id}_prediction_summary.json"
    profile_path = bench_root / "logs" / f"{run_id}_profile.csv"

    summary = read_json(summary_path) if summary_path.exists() else {}
    profile_rows = read_csv_rows(profile_path)

    return {
        "run": label,
        "samples": summary.get("samples", ""),
        "engine_avg_ms": summary.get("avg_elapsed_ms", ""),
        "engine_total_ms": summary.get("total_elapsed_ms", ""),
        "warm_elapsed_ms": summary.get("warm_elapsed_ms", ""),
        "ram_peak_mb": summary.get("ram_peak_mb", ""),
        "vram_peak_mb": summary.get("vram_peak_mb", ""),
        "render_ms": avg_csv_column(profile_rows, "render_ms"),
        "doclayout_ms": avg_csv_column(profile_rows, "doclayout_ms"),
        "scanned_page_ocr_ms": avg_csv_column(profile_rows, "scanned_page_ocr_ms"),
        "scanned_tatr_ms": avg_csv_column(profile_rows, "scanned_tatr_ms"),
        "scanned_table_extract_ms": avg_csv_column(profile_rows, "scanned_table_extract_ms"),
        "formula_mfr_ms": avg_csv_column(profile_rows, "formula_mfr_ms"),
        "page_total_ms": avg_csv_column(profile_rows, "page_total_ms"),
    }


def build_report(
    *,
    out_dir: Path,
    run_a: str,
    run_b: str,
    label_a: str,
    label_b: str,
    compact: list[dict[str, Any]],
    runtime: list[dict[str, Any]],
    page_rows: list[dict[str, Any]],
    formula_rows: list[dict[str, Any]],
    overlay_count: int,
) -> str:
    lines: list[str] = []
    lines.append("# OmniDocBench Compare Report")
    lines.append("")
    lines.append(f"- Run A: `{run_a}` as `{label_a}`")
    lines.append(f"- Run B: `{run_b}` as `{label_b}`")
    lines.append(f"- Pages compared: `{len(page_rows)}`")
    lines.append("")
    lines.append("## Scores")
    lines.append("")
    score_rows = [
        {
            "run": row["run"],
            "text_edit": fmt(row.get("text_edit")),
            "reading_order_edit": fmt(row.get("reading_order_edit")),
            "formula_edit": fmt(row.get("formula_edit")),
            "formula_cdm": fmt(row.get("formula_cdm")),
            "formula_cdm_count": fmt(row.get("formula_cdm_count")),
            "table_teds": fmt(row.get("table_teds")),
            "table_edit": fmt(row.get("table_edit")),
        }
        for row in compact
    ]
    lines.extend(
        md_table(
            score_rows,
            [
                "run",
                "text_edit",
                "reading_order_edit",
                "formula_edit",
                "formula_cdm",
                "formula_cdm_count",
                "table_teds",
                "table_edit",
            ],
        )
    )
    lines.append("")
    lines.append("## Engine Runtime")
    lines.append("")
    runtime_rows = [{key: fmt(row.get(key)) for key in RUNTIME_COLUMNS} for row in runtime]
    lines.extend(md_table(runtime_rows, RUNTIME_COLUMNS))
    lines.append("")
    lines.append("## Page Comparison")
    lines.append("")
    lines.extend(
        md_table(
            page_rows,
            [
                "image_stem",
                f"{label_a}_final_text_len",
                f"{label_b}_final_text_len",
                f"{label_a}_zones",
                f"{label_b}_zones",
                f"{label_a}_formula_status",
                f"{label_b}_formula_status",
            ],
        )
    )
    lines.append("")
    lines.append("## Formula Diagnostics")
    lines.append("")
    status_counts: dict[str, int] = {}
    for row in formula_rows:
        key = f"{row.get('run')}:{row.get('status')}"
        status_counts[key] = status_counts.get(key, 0) + 1
    diag_rows = [{"status": key, "count": value} for key, value in sorted(status_counts.items())]
    lines.extend(md_table(diag_rows, ["status", "count"]))
    lines.append("")
    bad_formula_rows = [
        row
        for row in formula_rows
        if str(row.get("status")) not in {"accepted", "skipped_formula_type"}
    ][:20]
    if bad_formula_rows:
        lines.append("### First Non-Accepted Formula Rows")
        lines.append("")
        lines.extend(
            md_table(
                bad_formula_rows,
                [
                    "run",
                    "image_stem",
                    "formula_id",
                    "type",
                    "status",
                    "latex_len",
                    "validation_error",
                    "eos_reached",
                    "truncated",
                    "quality_flags",
                ],
            )
        )
        lines.append("")
    lines.append("## Debug Artifacts")
    lines.append("")
    lines.append("- `score_summary.json`: raw official metric JSON")
    lines.append("- `compact_scores.json`: compact extracted scores")
    lines.append("- `compact_scores.csv`: compact extracted scores")
    lines.append("- `runtime_summary.json`: compact engine runtime summary")
    lines.append("- `runtime_summary.csv`: compact engine runtime summary")
    lines.append("- `page_compare.csv`: per-page comparison and normalized metadata")
    lines.append("- `formula_debug.csv`: per-formula Torvex metadata")
    lines.append("- `overlays/`: GT and Torvex layout overlays")
    lines.append("- `snippets/`: prediction markdown previews")
    lines.append("")
    if overlay_count > 0:
        lines.append("## Overlay Preview")
        lines.append("")
        lines.append(f"Generated overlays for first `{overlay_count}` page(s). Open files in:")
        lines.append("")
        lines.append(f"`{out_dir / 'overlays'}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two OmniDocBench Docker-evaluated Torvex runs."
    )
    parser.add_argument("--subset-json", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=Path(".bench/omnidocbench/raw/images"))

    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--label-a", default="run_a")
    parser.add_argument("--label-b", default="run_b")

    parser.add_argument("--bench-root", type=Path, default=Path(".bench/omnidocbench"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overlay-count", type=int, default=5)

    args = parser.parse_args()

    result_a = args.bench_root / "result" / args.run_a
    result_b = args.bench_root / "result" / args.run_b
    normalized_a = args.bench_root / "normalized" / args.run_a
    normalized_b = args.bench_root / "normalized" / args.run_b
    predictions_a = args.bench_root / "predictions" / args.run_a
    predictions_b = args.bench_root / "predictions" / args.run_b

    records = read_json(args.subset_json)
    if not isinstance(records, list):
        raise ValueError("subset JSON must contain a list")

    metrics_a = load_metric_summary(result_a)
    metrics_b = load_metric_summary(result_b)

    compact = [
        compact_scores(args.label_a, metrics_a, result_a),
        compact_scores(args.label_b, metrics_b, result_b),
    ]
    runtime = [
        runtime_summary(bench_root=args.bench_root, run_id=args.run_a, label=args.label_a),
        runtime_summary(bench_root=args.bench_root, run_id=args.run_b, label=args.label_b),
    ]

    write_json(args.out_dir / "score_summary.json", {args.label_a: metrics_a, args.label_b: metrics_b})
    write_json(args.out_dir / "compact_scores.json", compact)
    write_csv(args.out_dir / "compact_scores.csv", compact)
    write_json(args.out_dir / "runtime_summary.json", runtime)
    write_csv(args.out_dir / "runtime_summary.csv", runtime)

    norm_a = load_normalized_by_stem(normalized_a)
    norm_b = load_normalized_by_stem(normalized_b)

    page_rows: list[dict[str, Any]] = []
    for record in records:
        stem = image_stem_from_record(record)
        page_a = first_page(norm_a.get(stem, {}))
        page_b = first_page(norm_b.get(stem, {}))

        page_rows.append(
            {
                "image_stem": stem,
                "image_name": image_name_from_record(record),
                f"{args.label_a}_final_text_len": len(str(page_a.get("final_text") or "")),
                f"{args.label_b}_final_text_len": len(str(page_b.get("final_text") or "")),
                f"{args.label_a}_zones": len(page_a.get("zones") or page_a.get("layout_zones") or []),
                f"{args.label_b}_zones": len(page_b.get("zones") or page_b.get("layout_zones") or []),
                f"{args.label_a}_formula_status": json.dumps(count_formula_status(page_a), ensure_ascii=False),
                f"{args.label_b}_formula_status": json.dumps(count_formula_status(page_b), ensure_ascii=False),
            }
        )

    write_csv(args.out_dir / "page_compare.csv", page_rows)

    formula_rows: list[dict[str, Any]] = []
    formula_rows.extend(formula_debug_rows(run_name=args.label_a, normalized_by_stem=norm_a))
    formula_rows.extend(formula_debug_rows(run_name=args.label_b, normalized_by_stem=norm_b))
    write_csv(args.out_dir / "formula_debug.csv", formula_rows)

    overlay_records = records[: max(0, args.overlay_count)]
    for record in overlay_records:
        stem = image_stem_from_record(record)
        draw_gt_overlay(
            record=record,
            images_dir=args.images_dir,
            output_path=args.out_dir / "overlays" / f"{stem}_gt.png",
        )
        if stem in norm_a:
            draw_torvex_overlay(
                stem=stem,
                payload=norm_a[stem],
                images_dir=args.images_dir,
                output_path=args.out_dir / "overlays" / f"{stem}_{args.label_a}.png",
            )
        if stem in norm_b:
            draw_torvex_overlay(
                stem=stem,
                payload=norm_b[stem],
                images_dir=args.images_dir,
                output_path=args.out_dir / "overlays" / f"{stem}_{args.label_b}.png",
            )

    snippets_dir = args.out_dir / "snippets"
    snippets_dir.mkdir(parents=True, exist_ok=True)
    for record in records[: max(0, args.overlay_count)]:
        stem = image_stem_from_record(record)
        (snippets_dir / f"{stem}_{args.label_a}.md").write_text(
            markdown_preview(predictions_a / f"{stem}.md"),
            encoding="utf-8",
        )
        (snippets_dir / f"{stem}_{args.label_b}.md").write_text(
            markdown_preview(predictions_b / f"{stem}.md"),
            encoding="utf-8",
        )

    report = build_report(
        out_dir=args.out_dir,
        run_a=args.run_a,
        run_b=args.run_b,
        label_a=args.label_a,
        label_b=args.label_b,
        compact=compact,
        runtime=runtime,
        page_rows=page_rows,
        formula_rows=formula_rows,
        overlay_count=args.overlay_count,
    )
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")

    print(f"[compare] wrote {args.out_dir}")
    print(f"[compare] report: {args.out_dir / 'report.md'}")
    print()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
