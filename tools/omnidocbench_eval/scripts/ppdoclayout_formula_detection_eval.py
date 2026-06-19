"""
Measure PP-DocLayout's formula DETECTION accuracy (precision/recall/F1) against
the full OmniDocBench ground truth — independent of CDM / recognition.

For every image we run PP-DocLayout (layout model only, no OCR / no MFR), take its
display_formula boxes, and match them to GT `equation_isolated` boxes by IoU.
A predicted box is a true positive if it matches an unused GT box at IoU >= thr.

    precision = TP / (TP + FP)   how many predicted formulas are real
    recall    = TP / (TP + FN)   how many real formulas were found
    F1        = harmonic mean

Run on the FULL set so false positives (pages with no GT formula) count.

Usage:
  .venv/Scripts/python.exe tools/omnidocbench_eval/scripts/ppdoclayout_formula_detection_eval.py \
      --gt .bench/omnidocbench/raw/OmniDocBench.json \
      --images-dir .bench/omnidocbench/raw/images \
      --device gpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from torvex_extract.visual_zoning import engine

GT_FORMULA_CATEGORY = "equation_isolated"
PRED_FORMULA_TYPE = "display_formula"


def poly_to_xyxy(poly: list[float]) -> tuple[float, float, float, float]:
    xs, ys = poly[0::2], poly[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def iou(a: tuple, b: tuple) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def match(preds: list[tuple], gts: list[tuple], thr: float) -> tuple[int, int, int]:
    """Greedy IoU matching. Returns (tp, fp, fn)."""
    used_gt = set()
    tp = 0
    # match each prediction to the highest-IoU unused GT above threshold
    for p in preds:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in used_gt:
                continue
            v = iou(p, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= thr:
            used_gt.add(best_j)
            tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=Path(".bench/omnidocbench/raw/OmniDocBench.json"))
    ap.add_argument("--images-dir", type=Path, default=Path(".bench/omnidocbench/raw/images"))
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--thresholds", default="0.5,0.7,0.3")
    ap.add_argument("--out", type=Path, default=Path(".bench/omnidocbench/ppdoclayout_formula_detection.json"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    samples = json.loads(args.gt.read_text(encoding="utf-8-sig"))
    if args.limit:
        samples = samples[: args.limit]

    print(f"[detect-eval] GT samples: {len(samples)}")
    print(f"[detect-eval] device: {args.device}")
    print(f"[detect-eval] warming PP-DocLayout...")
    engine.warm(device=args.device)

    # aggregate counters per threshold
    agg = {t: {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}
    per_page = []
    total_pred = total_gt = 0
    missing_img = 0
    started = time.perf_counter()

    for i, s in enumerate(samples):
        stem_name = s["page_info"]["image_path"]
        img_path = args.images_dir / Path(stem_name).name
        if not img_path.exists():
            missing_img += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            missing_img += 1
            continue

        # GT display formulas
        gts = [
            poly_to_xyxy(d["poly"])
            for d in s.get("layout_dets", [])
            if d.get("category_type") == GT_FORMULA_CATEGORY and not d.get("ignore")
        ]

        # PP-DocLayout raw display_formula detections (image-pixel coords)
        zones = engine.detect_layout(img)
        preds = [
            tuple(z["bbox"]) for z in zones if z.get("type") == PRED_FORMULA_TYPE
        ]

        total_pred += len(preds)
        total_gt += len(gts)

        row = {"image": Path(stem_name).stem, "gt": len(gts), "pred": len(preds)}
        for t in thresholds:
            tp, fp, fn = match(preds, gts, t)
            agg[t]["tp"] += tp
            agg[t]["fp"] += fp
            agg[t]["fn"] += fn
            if t == thresholds[0]:
                row[f"tp@{t}"], row[f"fp@{t}"], row[f"fn@{t}"] = tp, fp, fn
        per_page.append(row)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(samples)} processed...")

    elapsed = time.perf_counter() - started

    print()
    print(f"[detect-eval] done in {elapsed:.1f}s  ({missing_img} images missing)")
    print(f"[detect-eval] total GT formulas: {total_gt}  |  total predicted: {total_pred}")
    print()
    print(f"{'IoU':>5} | {'TP':>5} {'FP':>5} {'FN':>5} | {'Precision':>9} {'Recall':>7} {'F1':>6}")
    print("-" * 56)
    results = {}
    for t in sorted(thresholds, reverse=True):
        tp, fp, fn = agg[t]["tp"], agg[t]["fp"], agg[t]["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        results[t] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}
        print(f"{t:>5} | {tp:>5} {fp:>5} {fn:>5} | {prec:>9.4f} {rec:>7.4f} {f1:>6.4f}")

    args.out.write_text(
        json.dumps(
            {
                "gt_samples": len(samples),
                "missing_images": missing_img,
                "total_gt": total_gt,
                "total_pred": total_pred,
                "elapsed_sec": round(elapsed, 1),
                "results_by_iou": results,
                "per_page": per_page,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[detect-eval] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
