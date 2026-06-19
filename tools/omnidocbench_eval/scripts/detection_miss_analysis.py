"""
WHY does PP-DocLayout "miss" 33% of GT formulas? Categorize every unmatched
GT equation_isolated so we know if it's a real miss, a merged box, a loose
box, or a class/coord problem -- before choosing a fix.

Buckets per GT formula vs PP-DocLayout display_formula boxes:
  matched          IoU >= 0.5 with a display_formula pred
  covered_merged   GT mostly inside a LARGER display_formula box (IoA>=0.7,
                   IoU<0.5)  -> one pred box swallows several GT equations
  loose_partial    IoU 0.3-0.5, or GT center inside a pred (box exists, sloppy)
  wrong_class      a NON-display zone covers the GT (IoA>=0.5) -> mislabeled
  absent           nothing overlaps the GT at all -> true miss / no proposal
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from ppdoclayout_formula_detection_eval import iou, poly_to_xyxy  # noqa: E402

from torvex_extract.visual_zoning import engine  # noqa: E402

GT_CAT = "equation_isolated"
DISP = "display_formula"


def ioa(inner, outer) -> float:
    """Intersection over INNER area (how much of `inner` is inside `outer`)."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    a = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / a


def center_in(box, outer) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=Path(".bench/omnidocbench/raw/OmniDocBench.json"))
    ap.add_argument("--images-dir", type=Path, default=Path(".bench/omnidocbench/raw/images"))
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path(".bench/omnidocbench/detection_miss_analysis.json"))
    args = ap.parse_args()

    samples = json.loads(args.gt.read_text(encoding="utf-8-sig"))
    if args.limit:
        samples = samples[: args.limit]

    engine.warm(device=args.device)

    buckets = Counter()
    examples: dict[str, list] = {}
    n_gt = 0
    merge_pages = 0  # pages where #disp_pred < #gt (suggestive of merging)
    tot_disp = 0

    for s in samples:
        img_path = args.images_dir / Path(s["page_info"]["image_path"]).name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gts = [poly_to_xyxy(d["poly"]) for d in s.get("layout_dets", [])
               if d.get("category_type") == GT_CAT and not d.get("ignore")]
        if not gts:
            continue

        zones = engine.detect_layout(img)
        disp = [tuple(z["bbox"]) for z in zones if z.get("type") == DISP]
        other = [(z.get("type"), tuple(z["bbox"])) for z in zones if z.get("type") != DISP]
        tot_disp += len(disp)
        if len(disp) < len(gts):
            merge_pages += 1

        used = set()
        for g in gts:
            n_gt += 1
            # best display match
            best_iou, best_j = 0.0, -1
            for j, p in enumerate(disp):
                if j in used:
                    continue
                v = iou(g, p)
                if v > best_iou:
                    best_iou, best_j = v, j
            max_ioa = max((ioa(g, p) for p in disp), default=0.0)
            cin = any(center_in(g, p) for p in disp)

            if best_j >= 0 and best_iou >= 0.5:
                used.add(best_j)
                b = "matched"
            elif max_ioa >= 0.7:
                b = "covered_merged"
            elif best_iou >= 0.3 or cin:
                b = "loose_partial"
            else:
                # mislabeled as another class?
                wc = None
                for t, p in other:
                    if ioa(g, p) >= 0.5:
                        wc = t
                        break
                if wc:
                    b = "wrong_class"
                    examples.setdefault("wrong_class", [])
                    if len(examples["wrong_class"]) < args.examples:
                        examples["wrong_class"].append(
                            {"image": img_path.name, "gt": [round(x, 1) for x in g], "as": wc})
                else:
                    b = "absent"
            buckets[b] += 1
            if b in ("absent", "covered_merged") and len(examples.get(b, [])) < args.examples:
                examples.setdefault(b, []).append(
                    {"image": img_path.name, "gt": [round(x, 1) for x in g],
                     "best_iou": round(best_iou, 2), "max_ioa": round(max_ioa, 2)})

    engine.shutdown()

    print("\n" + "=" * 60)
    print(f"GT formulas analyzed: {n_gt}   (display preds total: {tot_disp})")
    print(f"pages where #display_pred < #gt: {merge_pages}")
    print("-" * 60)
    for b in ["matched", "covered_merged", "loose_partial", "wrong_class", "absent"]:
        c = buckets[b]
        print(f"  {b:<16} {c:>5}  {c/max(1,n_gt)*100:5.1f}%")
    print("=" * 60)
    print("Reading:")
    print("  matched+covered_merged+loose_partial = formula WAS found (box exists)")
    print("  absent = real detection failure;  wrong_class = label/threshold issue")

    args.out.write_text(json.dumps(
        {"n_gt": n_gt, "buckets": dict(buckets), "merge_pages": merge_pages,
         "total_display_pred": tot_disp, "examples": examples}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
