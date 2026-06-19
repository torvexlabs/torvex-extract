"""
Of the formulas R2 scores ~0 (no IoU>=0.5 match after detection+split), HOW MANY
are recoverable by a better splitter vs stuck on un-splittable tight stacks?

For each MISSED GT equation we bucket:
  merged_wide_gap   covered by a larger display box; nearest sibling GT is >=8px
                    away -> a clear whitespace gap the splitter SHOULD cut (RECOVERABLE)
  merged_mid_gap    3-8px gap (borderline; tunable)
  merged_tight      <3px gap -> pixels can't separate (needs smarter than gap-scan)
  merged_oversize   inside a big box but no sibling GT in it -> box too loose (tighten)
  loose_partial     best display IoU 0.3-0.5 (localization slack)
  wrong_class       a non-display zone covers it (label/threshold)
  absent            nothing covers it (true miss)

Loads images with PIL (cv2.imread returns None on non-ASCII Windows paths, which
would falsely zero ~97 formulas).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from ppdoclayout_formula_detection_eval import iou, poly_to_xyxy  # noqa: E402

GT_CAT = "equation_isolated"
DISP = "display_formula"


def ioa(inner, outer) -> float:
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    a = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / a


def vgap(a, b) -> float:
    """vertical whitespace gap between boxes a,b (0 if they overlap in y)."""
    if b[1] >= a[3]:
        return b[1] - a[3]
    if a[1] >= b[3]:
        return a[1] - b[3]
    return 0.0


def match_gt(split_boxes, gts, thr):
    used = set()
    matched = set()
    for gi, g in enumerate(gts):
        best, bj = 0.0, -1
        for pj, p in enumerate(split_boxes):
            if pj in used:
                continue
            v = iou(g, p)
            if v > best:
                best, bj = v, pj
        if bj >= 0 and best >= thr:
            used.add(bj)
            matched.add(gi)
    return matched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=Path(".bench/omnidocbench/raw/OmniDocBench.json"))
    ap.add_argument("--images-dir", type=Path, default=Path(".bench/omnidocbench/raw/images"))
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path(".bench/omnidocbench/split_recovery_analysis.json"))
    args = ap.parse_args()

    from torvex_extract.formula_extractor import (  # noqa: E402
        FormulaExtractionConfig,
        _crop_formula_image,
        _split_display_formula_bboxes,
        _DISPLAY_FORMULA_TYPE,
    )
    from torvex_extract.visual_zoning import engine  # noqa: E402

    config = FormulaExtractionConfig()
    samples = json.loads(args.gt.read_text(encoding="utf-8-sig"))
    engine.warm(device=args.device)

    buckets = Counter()
    gaps = []
    total_gt = total_miss = 0

    for s in samples:
        p = args.images_dir / Path(s["page_info"]["image_path"]).name
        if not p.exists():
            continue
        gts = [poly_to_xyxy(d["poly"]) for d in s.get("layout_dets", [])
               if d.get("category_type") == GT_CAT and not d.get("ignore")]
        if not gts:
            continue
        total_gt += len(gts)

        img_pil = Image.open(p).convert("RGB")
        img_bgr = np.array(img_pil)[:, :, ::-1].copy()   # PIL RGB -> BGR for detect_layout

        zones = engine.detect_layout(img_bgr)
        raw_disp = [tuple(z["bbox"]) for z in zones if z.get("type") == DISP]
        other = [tuple(z["bbox"]) for z in zones if z.get("type") != DISP]

        raw_dicts = [{"type": _DISPLAY_FORMULA_TYPE, "bbox_px": list(b), "formula_id": f"{i}"}
                     for i, b in enumerate(raw_disp)]
        split = _split_display_formula_bboxes(img_pil, raw_dicts, config)
        split_boxes = []
        for f in split:
            b = f.get("bbox_px")
            if not b:
                continue
            crop, _ = _crop_formula_image(img_pil, b, config)
            if crop is not None:
                split_boxes.append(tuple(b))

        matched = match_gt(split_boxes, gts, args.iou)

        for gi, g in enumerate(gts):
            if gi in matched:
                continue
            total_miss += 1
            ioa_raw = max((ioa(g, rb) for rb in raw_disp), default=0.0)
            if ioa_raw >= 0.7:
                cover = max(raw_disp, key=lambda rb: ioa(g, rb))
                sibs = [gg for j, gg in enumerate(gts) if j != gi and ioa(gg, cover) >= 0.5]
                if not sibs:
                    buckets["merged_oversize"] += 1
                    continue
                gap = min(vgap(g, sb) for sb in sibs)
                gaps.append(gap)
                if gap >= 8:
                    buckets["merged_wide_gap(RECOVERABLE)"] += 1
                elif gap >= 3:
                    buckets["merged_mid_gap(borderline)"] += 1
                else:
                    buckets["merged_tight(<3px,hard)"] += 1
            else:
                max_iou_disp = max((iou(g, db) for db in raw_disp), default=0.0)
                if max_iou_disp >= 0.3:
                    buckets["loose_partial"] += 1
                elif any(ioa(g, ob) >= 0.5 for ob in other):
                    buckets["wrong_class"] += 1
                else:
                    buckets["absent"] += 1

    engine.shutdown()

    print("\n" + "=" * 64)
    print(f"GT formulas: {total_gt}   |   R2 misses (score ~0): {total_miss}")
    print("-" * 64)
    order = ["merged_wide_gap(RECOVERABLE)", "merged_mid_gap(borderline)",
             "merged_tight(<3px,hard)", "merged_oversize", "loose_partial",
             "wrong_class", "absent"]
    for b in order:
        c = buckets.get(b, 0)
        print(f"  {b:<30} {c:>5}  {c/max(1,total_miss)*100:5.1f}% of misses")
    if gaps:
        gaps_sorted = sorted(gaps)
        import statistics
        print("-" * 64)
        print(f"stacking-gap px among merged misses: median={statistics.median(gaps):.1f} "
              f"p25={gaps_sorted[len(gaps)//4]:.1f} p75={gaps_sorted[3*len(gaps)//4]:.1f}")
    print("=" * 64)
    recoverable = buckets.get("merged_wide_gap(RECOVERABLE)", 0) + buckets.get("merged_mid_gap(borderline)", 0)
    print(f"RECOVERABLE by better splitter (wide+mid gap): {recoverable} "
          f"({recoverable/max(1,total_miss)*100:.1f}% of misses)")

    args.out.write_text(json.dumps(
        {"total_gt": total_gt, "total_miss": total_miss,
         "buckets": dict(buckets), "gap_count": len(gaps)}, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
