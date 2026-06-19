"""Why does the splitter miss the wide/mid-gap merges? For each merged-but-missed
GT, look at the actual pixel gap to its nearest sibling and classify:
  full_blank      a full-width blank row exists -> splitter SHOULD cut (tuning bug)
  side_blocked    center is blank but a side column has ink (eqn number / margin)
                  -> needs column-aware blank detection
  no_gap          even the center has ink -> no real whitespace (truly hard)
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(SCRIPT_DIR))
from ppdoclayout_formula_detection_eval import iou, poly_to_xyxy  # noqa

GT_CAT, DISP = "equation_isolated", "display_formula"


def ioa(inner, outer):
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    return inter / max(1e-9, (inner[2]-inner[0])*(inner[3]-inner[1]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=Path(".bench/omnidocbench/raw/OmniDocBench.json"))
    ap.add_argument("--images-dir", type=Path, default=Path(".bench/omnidocbench/raw/images"))
    ap.add_argument("--device", default="gpu")
    args = ap.parse_args()
    from torvex_extract.visual_zoning import engine
    engine.warm(device=args.device)
    samples = json.loads(args.gt.read_text(encoding="utf-8-sig"))

    cls = Counter()
    DARK = 0.008  # current blank-row tolerance (fraction of width)
    for s in samples:
        p = args.images_dir / Path(s["page_info"]["image_path"]).name
        if not p.exists():
            continue
        gts = [poly_to_xyxy(d["poly"]) for d in s.get("layout_dets", [])
               if d.get("category_type") == GT_CAT and not d.get("ignore")]
        if not gts:
            continue
        img = np.array(Image.open(p).convert("L"))
        bgr = np.array(Image.open(p).convert("RGB"))[:, :, ::-1].copy()
        raw = [tuple(z["bbox"]) for z in engine.detect_layout(bgr) if z.get("type") == DISP]
        for gi, g in enumerate(gts):
            # only merged ones: GT inside a bigger raw display box, with a sibling
            covers = [rb for rb in raw if ioa(g, rb) >= 0.7]
            if not covers:
                continue
            cover = max(covers, key=lambda rb: ioa(g, rb))
            sibs = [(j, gg) for j, gg in enumerate(gts) if j != gi and ioa(gg, cover) >= 0.5]
            if not sibs:
                continue
            # nearest sibling + the vertical gap band between them
            best = None
            for _, gg in sibs:
                if gg[1] >= g[3]:
                    d, ya, yb = gg[1]-g[3], g[3], gg[1]      # sib below
                elif g[1] >= gg[3]:
                    d, ya, yb = g[1]-gg[3], gg[3], g[1]      # sib above
                else:
                    continue
                if best is None or d < best[0]:
                    best = (d, ya, yb)
            if best is None or not (3 <= best[0]):   # only wide/mid gaps
                continue
            _, ya, yb = best
            x0, x1 = int(cover[0]), int(cover[2])
            ya, yb = int(round(ya)), int(round(yb))
            band = img[ya:yb, x0:x1]
            if band.size == 0:
                continue
            w = band.shape[1]
            cen = band[:, int(w*0.15):int(w*0.85)]            # central 70%
            full_dark = ((band < 200).sum(axis=1) / max(1, w))
            cen_dark = ((cen < 200).sum(axis=1) / max(1, cen.shape[1]))
            if full_dark.min() <= DARK:
                cls["full_blank(tuning)"] += 1
            elif cen_dark.min() <= DARK:
                cls["side_blocked(column-aware)"] += 1
            else:
                cls["no_gap(hard)"] += 1
    engine.shutdown()
    tot = sum(cls.values())
    print(f"\nmerged wide/mid-gap cases examined: {tot}")
    for k in ["full_blank(tuning)", "side_blocked(column-aware)", "no_gap(hard)"]:
        print(f"  {k:<28} {cls.get(k,0):>5}  {cls.get(k,0)/max(1,tot)*100:5.1f}%")


if __name__ == "__main__":
    main()
