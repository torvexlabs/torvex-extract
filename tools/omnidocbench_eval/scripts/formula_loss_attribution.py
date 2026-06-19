"""
Formula loss attribution — find WHERE the 0.94->0.7 formula CDM bleeds out.

Runs the SAME UniMERNet recognizer on progressively-degraded inputs and scores
each rung against GT LaTeX. The gap between rungs = that stage's cost.

    R0  ceiling      GT equation_isolated box, raw crop -> recognizer
                     (the recognizer's own _preprocess replicates
                     FormulaImageEvalProcessor exactly -- same conditions as
                     the ~0.94 UniMER-Test eval. This is the fair ceiling on
                     OmniDocBench's distribution.)
    R1  +crop prep   GT box -> engine _crop_formula_image (padding, white
                     border, blank-check) -> recognizer.   R0-R1 = crop tax
    R2  +detect/split engine detect_layout + _split_display_formula_bboxes,
                     IoU-matched to GT (unmatched GT scored 0) -> recognizer.
                     R1-R2 = detection-recall + splitter tax
    R3  end-to-end   (separate) run_torvex_subset.py -> OmniDocBench CDM.
                     R2-R3 = assembly/markdown tax

Scoring:
  - fast proxy (normalized-LaTeX exact-match + difflib ratio) on ALL formulas
  - true CDM on a fixed paired sample (--cdm-sample) to anchor absolute scale

Usage:
  .venv/Scripts/python.exe tools/omnidocbench_eval/scripts/formula_loss_attribution.py \
      --device gpu --cdm-sample 150
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the detection eval's tiny geometry helpers (single source of truth).
from ppdoclayout_formula_detection_eval import iou, poly_to_xyxy  # noqa: E402

GT_FORMULA_CATEGORY = "equation_isolated"
PRED_FORMULA_TYPE = "display_formula"


# --------------------------------------------------------------------------- #
# LaTeX normalization + scoring
# --------------------------------------------------------------------------- #
def strip_delims(s: str) -> str:
    s = (s or "").strip()
    for d in ("$$", "$", r"\[", r"\]", r"\(", r"\)"):
        if s.startswith(d):
            s = s[len(d):]
        if s.endswith(d):
            s = s[: -len(d)]
    return s.strip()


import re

# Collapse cosmetic LaTeX differences so the render-free metric is fair.
# (UniMERNet emits space-tokenized, heavily-grouped LaTeX that is render-
# equivalent to GT but string-different: \left[ vs \left\lbrack, \mathbf{{A}}
# vs AB, { x } vs x, thin spaces, etc. True CDM handles this by rendering;
# this canonicalization approximates it without a TeX toolchain.)
_DROP_CMDS = re.compile(
    r"\\(left|right|bigl|bigr|biggl|biggr|Big|Bigg|displaystyle|textstyle"
    r"|scriptstyle|mathbf|boldsymbol|mathrm|mathbb|mathcal|mathit|operatorname"
    r"|bf|rm|it|sf|tt)\b"
)
_DROP_SPACING = re.compile(r"\\[,!;:>\s]|\\q?quad|~")


def norm_for_proxy(s: str) -> str:
    """Render-free canonical form: strip cosmetic LaTeX, keep structure."""
    s = strip_delims(s)
    s = s.replace(r"\lbrack", "[").replace(r"\rbrack", "]")
    s = _DROP_CMDS.sub("", s)
    s = _DROP_SPACING.sub("", s)
    s = s.replace(r"\\", "")            # row breaks
    s = re.sub(r"\s+", "", s)           # all whitespace
    for ch in "{}&":                     # grouping + alignment
        s = s.replace(ch, "")
    return s


def proxy_score(pred: str, gt: str) -> tuple[float, bool]:
    np_, ng = norm_for_proxy(pred), norm_for_proxy(gt)
    if not ng:
        return (1.0 if not np_ else 0.0), (np_ == ng)
    ratio = difflib.SequenceMatcher(None, np_, ng).ratio()
    return ratio, (np_ == ng)


def load_cdm():
    """Return cdm(pred, gt)->float or None if the toolchain isn't available."""
    try:
        from src.metrics.cdm.cdm import cdm  # type: ignore

        def _safe(pred: str, gt: str) -> float | None:
            try:
                return float(cdm(strip_delims(pred), strip_delims(gt)))
            except Exception as exc:  # one bad render must not kill the run
                print(f"   [cdm] skipped one pair: {exc}")
                return None

        return _safe
    except Exception as exc:
        print(f"[attrib] CDM unavailable ({exc}); proxy-only.")
        return None


# --------------------------------------------------------------------------- #
# GT
# --------------------------------------------------------------------------- #
def load_gt(gt_path: Path, images_dir: Path, limit: int | None):
    samples = json.loads(gt_path.read_text(encoding="utf-8-sig"))
    if limit:
        samples = samples[:limit]

    pages = []  # [{image, formulas:[{box, latex}]}]
    n_formulas = 0
    for s in samples:
        stem_name = s["page_info"]["image_path"]
        img_path = images_dir / Path(stem_name).name
        if not img_path.exists():
            continue
        formulas = []
        for d in s.get("layout_dets", []):
            if d.get("category_type") != GT_FORMULA_CATEGORY or d.get("ignore"):
                continue
            latex = str(d.get("latex") or "").strip()
            if not latex:
                continue
            formulas.append({"box": poly_to_xyxy(d["poly"]), "latex": latex})
        if formulas:
            pages.append({"image": img_path, "formulas": formulas})
            n_formulas += len(formulas)
    return pages, n_formulas


def clamp_crop(img: Image.Image, box) -> Image.Image | None:
    x0, y0, x1, y1 = box
    x0 = max(0, int(round(x0)))
    y0 = max(0, int(round(y0)))
    x1 = min(img.width, int(round(x1)))
    y1 = min(img.height, int(round(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return img.crop((x0, y0, x1, y1)).convert("RGB")


def match_pairs(preds: list[tuple], gts: list[tuple], thr: float):
    """Greedy IoU matching. Returns dict gt_index -> pred_index (best, unique)."""
    used_pred = set()
    out: dict[int, int] = {}
    for gi, g in enumerate(gts):
        best_iou, best_p = 0.0, -1
        for pi, p in enumerate(preds):
            if pi in used_pred:
                continue
            v = iou(p, g)
            if v > best_iou:
                best_iou, best_p = v, pi
        if best_p >= 0 and best_iou >= thr:
            used_pred.add(best_p)
            out[gi] = best_p
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=Path(".bench/omnidocbench/raw/OmniDocBench.json"))
    ap.add_argument("--images-dir", type=Path, default=Path(".bench/omnidocbench/raw/images"))
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None, help="limit GT pages")
    ap.add_argument("--max-formulas", type=int, default=None, help="cap total formulas (quick smoke)")
    ap.add_argument("--cdm-sample", type=int, default=150, help="formulas for true CDM (0=skip)")
    ap.add_argument("--split", choices=["on", "off"], default="on",
                    help="toggle _split_display_formula_bboxes for R2")
    ap.add_argument("--out", type=Path, default=Path(".bench/omnidocbench/formula_loss_attribution.json"))
    args = ap.parse_args()

    # The splitter is env-driven in FormulaExtractionConfig; set before import-time
    # config construction below.
    os.environ["TORVEX_FORMULA_SPLIT"] = "true" if args.split == "on" else "false"

    from torvex_extract.formula_extractor import (  # noqa: E402
        FormulaExtractionConfig,
        _crop_formula_image,
        _split_display_formula_bboxes,
        get_formula_extractor,
        shutdown_formula_extractor,
        _DISPLAY_FORMULA_TYPE,
    )
    from torvex_extract.visual_zoning import engine  # noqa: E402

    config = FormulaExtractionConfig()
    cdm_fn = load_cdm() if args.cdm_sample > 0 else None

    print(f"[attrib] device={args.device} iou={args.iou} split={args.split}")
    print(f"[attrib] loading GT from {args.gt}")
    pages, n_formulas = load_gt(args.gt, args.images_dir, args.limit)
    print(f"[attrib] pages with formulas: {len(pages)}  |  GT formulas: {n_formulas}")

    print("[attrib] warming engine (PP-DocLayout) + recognizer...")
    engine.warm(device=args.device)
    extractor = get_formula_extractor(device=args.device, config=config)
    extractor.preflight()

    # Per-rung: parallel lists indexed by a global formula id.
    gt_latex: list[str] = []
    r0_crop: list[Image.Image | None] = []
    r1_crop: list[Image.Image | None] = []
    r2_crop: list[Image.Image | None] = []  # None = detection/split miss -> score 0

    started = time.perf_counter()
    for pidx, page in enumerate(pages):
        if args.max_formulas and len(gt_latex) >= args.max_formulas:
            break

        # Build a whole page's crops atomically so one bad image can't desync
        # the parallel rung lists during a long unattended run.
        try:
            img_pil = Image.open(page["image"]).convert("RGB")
            img_bgr = cv2.imread(str(page["image"]))
            gt_boxes = [f["box"] for f in page["formulas"]]

            # --- R2: engine detection + split, then match to GT ---
            zones = engine.detect_layout(img_bgr) if img_bgr is not None else []
            raw = [
                {"type": _DISPLAY_FORMULA_TYPE, "bbox_px": list(z["bbox"]), "formula_id": f"p{pidx}_{i}"}
                for i, z in enumerate(zones)
                if z.get("type") == PRED_FORMULA_TYPE
            ]
            split = _split_display_formula_bboxes(img_pil, raw, config)
            pred_boxes, pred_crops = [], []
            for f in split:
                box = f.get("bbox_px")
                if not box:
                    continue
                crop, _ = _crop_formula_image(img_pil, box, config)
                pred_boxes.append(tuple(box))
                pred_crops.append(crop)  # may be None (blank-check rejected)
            gi2pi = match_pairs(pred_boxes, [tuple(b) for b in gt_boxes], args.iou)

            pg_gt, pg_r0, pg_r1, pg_r2 = [], [], [], []
            for fi, f in enumerate(page["formulas"]):
                pg_gt.append(f["latex"])
                pg_r0.append(clamp_crop(img_pil, f["box"]))           # raw GT crop
                c1, _ = _crop_formula_image(img_pil, list(f["box"]), config)
                pg_r1.append(c1)                                       # engine crop prep
                pi = gi2pi.get(fi)
                pg_r2.append(pred_crops[pi] if pi is not None else None)  # None = miss
        except Exception as exc:
            print(f"  [skip page {pidx} {Path(page['image']).name}] {exc}")
            continue

        gt_latex.extend(pg_gt)
        r0_crop.extend(pg_r0)
        r1_crop.extend(pg_r1)
        r2_crop.extend(pg_r2)

        if args.max_formulas and len(gt_latex) >= args.max_formulas:
            n = args.max_formulas
            gt_latex, r0_crop, r1_crop, r2_crop = (
                gt_latex[:n], r0_crop[:n], r1_crop[:n], r2_crop[:n]
            )
            break

        if (pidx + 1) % 50 == 0:
            print(f"  {pidx + 1}/{len(pages)} pages  ({len(gt_latex)} formulas)")

    total = len(gt_latex)
    print(f"[attrib] crops built for {total} formulas in {time.perf_counter()-started:.1f}s")

    # --- recognize each rung (recognize_crops batches internally) ---
    def recognize(crops: list[Image.Image | None]) -> list[str]:
        idx = [i for i, c in enumerate(crops) if c is not None]
        out = [""] * len(crops)
        if idx:
            res = extractor.recognize_crops([crops[i] for i in idx])
            for j, r in zip(idx, res):
                out[j] = str(r.get("latex") or "")
        return out

    rungs = {}
    for name, crops in (("R0", r0_crop), ("R1", r1_crop), ("R2", r2_crop)):
        t = time.perf_counter()
        preds = recognize(crops)
        present = sum(1 for c in crops if c is not None)
        rungs[name] = preds
        print(f"[attrib] {name} recognized {present}/{total} crops in {time.perf_counter()-t:.1f}s")

    # --- score: proxy on all, CDM on a fixed paired sample ---
    cdm_idx = []
    if cdm_fn and total:
        step = max(1, total // args.cdm_sample)
        cdm_idx = list(range(0, total, step))[: args.cdm_sample]

    report = {"meta": {
        "device": args.device, "iou": args.iou, "split": args.split,
        "total_formulas": total, "cdm_sample": len(cdm_idx),
    }, "rungs": {}, "per_formula": []}

    for name in ("R0", "R1", "R2"):
        preds = rungs[name]
        ratios, exacts = [], 0
        present = 0
        for i in range(total):
            r, e = proxy_score(preds[i], gt_latex[i])
            ratios.append(r)
            exacts += int(e)
            present += int(bool(norm_for_proxy(preds[i])))
        agg = {
            "n": total,
            "recognized": present,
            "proxy_ratio": round(sum(ratios) / total, 4) if total else 0.0,
            "exact_match": round(exacts / total, 4) if total else 0.0,
        }
        if name == "R2":
            agg["detection_recall"] = round(
                sum(1 for c in r2_crop if c is not None) / total, 4
            ) if total else 0.0
        if cdm_idx:
            vals = [cdm_fn(preds[i], gt_latex[i]) for i in cdm_idx]
            vals = [v for v in vals if v is not None]
            agg["cdm_sample"] = round(sum(vals) / len(vals), 4) if vals else None
        rungs_meta = agg
        report["rungs"][name] = rungs_meta

    for i in range(total):
        report["per_formula"].append({
            "gt": gt_latex[i][:300],
            "R0": rungs["R0"][i][:300],
            "R1": rungs["R1"][i][:300],
            "R2": rungs["R2"][i][:300],
            "R2_detected": r2_crop[i] is not None,
        })

    # FULL untruncated predictions for every formula -> real-CDM scoring.
    full_pairs = {
        "gt": gt_latex,
        "R0": rungs["R0"],
        "R1": rungs["R1"],
        "R2": rungs["R2"],
    }
    Path(".bench/omnidocbench/rung_pairs_full.json").write_text(
        json.dumps(full_pairs, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[attrib] wrote full untruncated pairs for {total} formulas -> rung_pairs_full.json")

    shutdown_formula_extractor()
    engine.shutdown()

    # --- print the ladder ---
    print("\n" + "=" * 64)
    print("FORMULA LOSS ATTRIBUTION LADDER")
    print("=" * 64)
    hdr = f"{'rung':<5} {'proxy':>7} {'exact':>7} {'cdm':>7}  note"
    print(hdr)
    print("-" * 64)
    notes = {"R0": "recognizer ceiling (GT crop)",
             "R1": "+ engine crop prep",
             "R2": "+ detection + split"}
    prev = None
    for name in ("R0", "R1", "R2"):
        a = report["rungs"][name]
        cdm_s = a.get("cdm_sample")
        cdm_str = f"{cdm_s:.3f}" if isinstance(cdm_s, float) else "  -  "
        line = f"{name:<5} {a['proxy_ratio']:>7.3f} {a['exact_match']:>7.3f} {cdm_str:>7}  {notes[name]}"
        if prev is not None:
            drop = prev - a["proxy_ratio"]
            line += f"   (proxy drop {drop:+.3f})"
        print(line)
        prev = a["proxy_ratio"]
    r2 = report["rungs"]["R2"]
    print("-" * 64)
    print(f"R2 detection recall: {r2.get('detection_recall')}  "
          f"(matched GT formulas with a usable crop)")
    print("=" * 64)
    print("Read: the biggest proxy drop between rungs is where the points bleed.")
    print("R2 -> R3 (assembly tax) = run_torvex_subset.py end-to-end CDM, separately.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[attrib] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
