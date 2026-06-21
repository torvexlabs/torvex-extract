"""Stage A: generate RAW pipeline predictions (no splitter). For each page with GT isolated formulas,
recognize ALL display_formula boxes (display-only + drop-inner-keep-outer) and emit them in reading
order. Output feeds the docker MGAM+CDM scorer (Stage B).

Out: .bench/omnidocbench/raw_predictions.json  = {img_name: [{"content": latex, "order": k}, ...]}
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(SCRIPT_DIR))
from ppdoclayout_formula_detection_eval import poly_to_xyxy

GT_CAT, DISP = "equation_isolated", "display_formula"


def drop_contained_boxes(boxes, ratio=0.7):
    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    def contained(s, l):
        ix0, iy0 = max(s[0], l[0]), max(s[1], l[1])
        ix1, iy1 = min(s[2], l[2]), min(s[3], l[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        a = area(s)
        return (inter / a) if a > 0 else 0.0

    return [b for i, b in enumerate(boxes)
            if not any(area(o) > area(b) and contained(b, o) >= ratio for j, o in enumerate(boxes) if j != i)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=Path(".bench/omnidocbench/raw/OmniDocBench.json"))
    ap.add_argument("--images-dir", type=Path, default=Path(".bench/omnidocbench/raw/images"))
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-dedup", action="store_true",
                    help="keep ALL display boxes incl per-line inner (test if per-line crops beat merged)")
    ap.add_argument("--out", type=Path, default=Path(".bench/omnidocbench/raw_predictions.json"))
    ap.add_argument("--recognizer", choices=["unimernet", "unirec"], default="unimernet",
                    help="recognizer backend; detection (PP-DocLayout) + crop prep stay identical")
    ap.add_argument("--only-pages", type=Path, default=None,
                    help="JSON list of image names to restrict to (e.g. docker GT formula pages)")
    ap.add_argument("--include-inline", action="store_true",
                    help="also recognize inline_formula boxes; drop-inner-keep-outer across the COMBINED "
                         "set removes any inline nested inside a display (or larger) box")
    args = ap.parse_args()
    only_pages = set(json.loads(args.only_pages.read_text(encoding="utf-8"))) if args.only_pages else None

    from torvex_extract.formula_extractor import (
        FormulaExtractionConfig, _crop_formula_image, get_formula_extractor, shutdown_formula_extractor)
    from torvex_extract.visual_zoning import engine
    cfg = FormulaExtractionConfig()
    engine.warm(device=args.device)
    if args.recognizer == "unirec":
        from unirec_recognizer import UniRecRecognizer
        rec = UniRecRecognizer(device=args.device)
    else:
        rec = get_formula_extractor(device=args.device, config=cfg)
    rec.preflight()

    samples = json.loads(args.gt.read_text(encoding="utf-8-sig"))
    if args.limit:
        samples = samples[: args.limit]

    all_crops, page_boxes = [], []
    for s in samples:
        name = Path(s["page_info"]["image_path"]).name
        if only_pages is not None and name not in only_pages:
            continue
        p = args.images_dir / name
        if not p.exists():
            continue
        has_gt = any(d.get("category_type") == GT_CAT and not d.get("ignore") and str(d.get("latex") or "").strip()
                     for d in s.get("layout_dets", []))
        if not has_gt:
            continue
        try:
            img = Image.open(p).convert("RGB")
            bgr = np.array(img)[:, :, ::-1].copy()
        except Exception:
            continue
        keep_types = {DISP, "inline_formula"} if args.include_inline else {DISP}
        boxes = [tuple(z["bbox"]) for z in engine.detect_layout(bgr) if z.get("type") in keep_types]
        if not args.no_dedup:
            boxes = drop_contained_boxes(boxes)   # inline nested in display (or larger) is dropped here
        boxes.sort(key=lambda b: (round(b[1] / 10), b[0]))      # reading order: top->bottom, left->right
        idxs = []
        for b in boxes:
            c, _ = _crop_formula_image(img, list(b), cfg)
            idxs.append(len(all_crops)); all_crops.append(c)
        page_boxes.append((name, idxs))

    preds = [""] * len(all_crops)
    if all_crops:
        for j, r in enumerate(rec.recognize_crops(all_crops)):
            preds[j] = str(r.get("latex") or "")

    out = {}
    for name, idxs in page_boxes:
        out[name] = [{"content": preds[i], "order": k} for k, i in enumerate(idxs) if preds[i].strip()]
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    n_pred = sum(len(v) for v in out.values())
    print(f"pages={len(out)}  total display-formula preds={n_pred}")
    print("wrote:", args.out)
    if args.recognizer == "unimernet":
        shutdown_formula_extractor()
    engine.shutdown()


if __name__ == "__main__":
    main()
