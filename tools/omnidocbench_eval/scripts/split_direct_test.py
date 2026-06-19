"""Directly test the splitter: for each raw display box that contains >=2 GT
equations (a real merge), run _split_display_formula_bboxes and compare the
number of segments it produces to the number of GT equations inside.
Tells us if the splitter UNDER-splits (the actual bug) vs whether the IoU-0.5
proxy was just mislabeling correctly-split boxes."""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(SCRIPT_DIR))
from ppdoclayout_formula_detection_eval import poly_to_xyxy  # noqa

GT_CAT, DISP = "equation_isolated", "display_formula"


def ioa(inner, outer):
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix1-ix0) * max(0.0, iy1-iy0)
    return inter / max(1e-9, (inner[2]-inner[0])*(inner[3]-inner[1]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=Path(".bench/omnidocbench/raw/OmniDocBench.json"))
    ap.add_argument("--images-dir", type=Path, default=Path(".bench/omnidocbench/raw/images"))
    ap.add_argument("--device", default="gpu")
    args = ap.parse_args()
    from torvex_extract.formula_extractor import (
        FormulaExtractionConfig, _split_display_formula_bboxes, _DISPLAY_FORMULA_TYPE)
    from torvex_extract.visual_zoning import engine
    cfg = FormulaExtractionConfig()
    engine.warm(device=args.device)
    samples = json.loads(args.gt.read_text(encoding="utf-8-sig"))

    merged_boxes = 0
    under = 0           # m < k  (under-split, the bug)
    exact = 0           # m == k
    over = 0            # m > k  (over-split, the risk)
    seg_hist = Counter()
    gt_in_box_hist = Counter()
    examples = []
    for s in samples:
        p = args.images_dir / Path(s["page_info"]["image_path"]).name
        if not p.exists():
            continue
        gts = [poly_to_xyxy(d["poly"]) for d in s.get("layout_dets", [])
               if d.get("category_type") == GT_CAT and not d.get("ignore")]
        if not gts:
            continue
        img_pil = Image.open(p).convert("RGB")
        bgr = np.array(img_pil)[:, :, ::-1].copy()
        raw = [tuple(z["bbox"]) for z in engine.detect_layout(bgr) if z.get("type") == DISP]
        for rb in raw:
            k = sum(1 for g in gts if ioa(g, rb) >= 0.5)
            if k < 2:
                continue
            merged_boxes += 1
            gt_in_box_hist[k] += 1
            d = [{"type": _DISPLAY_FORMULA_TYPE, "bbox_px": list(rb), "formula_id": "x"}]
            segs = _split_display_formula_bboxes(img_pil, d, cfg)
            m = len(segs)
            seg_hist[m] += 1
            if m < k:
                under += 1
            elif m == k:
                exact += 1
            else:
                over += 1
    engine.shutdown()
    print(f"\nmerged raw boxes (>=2 GT inside): {merged_boxes}")
    print(f"  UNDER-split (m<k, the bug):   {under}  ({under/max(1,merged_boxes)*100:.1f}%)")
    print(f"  EXACT     (m==k):             {exact}  ({exact/max(1,merged_boxes)*100:.1f}%)")
    print(f"  OVER-split (m>k, the risk):   {over}  ({over/max(1,merged_boxes)*100:.1f}%)")
    print(f"  GT-per-box distribution: {dict(sorted(gt_in_box_hist.items()))}")
    print(f"  segments-produced distribution: {dict(sorted(seg_hist.items()))}")
    print("  examples of under-split:")
    for e in examples:
        print("   ", e)


if __name__ == "__main__":
    main()
