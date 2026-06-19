"""
Measure pix2text-mfd-1.5 (YOLOv8n-MFD) formula DETECTION accuracy against the
full OmniDocBench GT — same protocol as ppdoclayout_formula_detection_eval.py,
so the two numbers are directly comparable.

Self-contained ONNX detector (the package mfd_detector.py was removed in the
2026-06-18 cleanup). Reports:
  - "isolated"  (MFD class 1) vs GT equation_isolated  <- apples-to-apples with PP-DocLayout
  - "all boxes" (class 0 + 1) vs GT equation_isolated  <- recall ceiling regardless of inline/display labelling

Usage:
  .venv/Scripts/python.exe tools/omnidocbench_eval/scripts/mfd_formula_detection_eval.py --device gpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from torvex_extract.onnx_runtime import (
    create_onnx_session,
    prepare_onnx_cuda_runtime,
    select_onnx_providers,
)
from ppdoclayout_formula_detection_eval import poly_to_xyxy, iou, match  # reuse

MFD_MODEL = ROOT / "models" / "pix2text-mfd-1.5.onnx"
IOU_NMS = 0.45
# ONNX metadata names = {0: 'embedding' (inline), 1: 'isolated' (display)}
CLASS_ISOLATED = 1

GT_FORMULA_CATEGORY = "equation_isolated"


def letterbox(img, size):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pt, pl = (size - nh) // 2, (size - nw) // 2
    canvas[pt : pt + nh, pl : pl + nw] = resized
    return canvas, scale, pl, pt


def nms(boxes, scores, thr):
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix0 = np.maximum(x0[i], x0[rest]); iy0 = np.maximum(y0[i], y0[rest])
        ix1 = np.minimum(x1[i], x1[rest]); iy1 = np.minimum(y1[i], y1[rest])
        inter = np.maximum(0, ix1 - ix0) * np.maximum(0, iy1 - iy0)
        iouv = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iouv <= thr]
    return keep


def detect(session, input_name, img, size, conf_thr):
    h, w = img.shape[:2]
    padded, scale, pl, pt = letterbox(img, size)
    tensor = (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
    raw = session.run(None, {input_name: tensor})[0]
    preds = raw[0].T  # (N, 4+nc)
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    conf = class_scores.max(axis=1)
    m = conf >= conf_thr
    if not m.any():
        return [], [], []
    boxes_xywh, conf, class_ids = boxes_xywh[m], conf[m], class_ids[m]
    xyxy = np.empty_like(boxes_xywh)
    xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    keep = nms(xyxy, conf, IOU_NMS)
    isolated, embedding, allboxes = [], [], []
    for k in keep:
        x0 = (xyxy[k, 0] - pl) / scale
        y0 = (xyxy[k, 1] - pt) / scale
        x1 = (xyxy[k, 2] - pl) / scale
        y1 = (xyxy[k, 3] - pt) / scale
        x0 = max(0.0, min(w, x0)); y0 = max(0.0, min(h, y0))
        x1 = max(0.0, min(w, x1)); y1 = max(0.0, min(h, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        box = (x0, y0, x1, y1)
        allboxes.append(box)
        if int(class_ids[k]) == CLASS_ISOLATED:
            isolated.append(box)
        else:
            embedding.append(box)
    return isolated, embedding, allboxes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=Path(".bench/omnidocbench/raw/OmniDocBench.json"))
    ap.add_argument("--images-dir", type=Path, default=Path(".bench/omnidocbench/raw/images"))
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--thresholds", default="0.5,0.7,0.3")
    ap.add_argument("--imgsz", type=int, default=768, help="must be a multiple of 32")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=Path(".bench/omnidocbench/mfd_formula_detection.json"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    samples = json.loads(args.gt.read_text(encoding="utf-8-sig"))
    if args.limit:
        samples = samples[: args.limit]

    print(f"[mfd-eval] GT samples: {len(samples)}  device: {args.device}  imgsz: {args.imgsz}  conf: {args.conf}")
    prepare_onnx_cuda_runtime()
    providers = select_onnx_providers(args.device)
    session = create_onnx_session(MFD_MODEL, providers=providers, model_name="pix2text-mfd-1.5")
    input_name = session.get_inputs()[0].name
    print(f"[mfd-eval] model loaded ({providers})")

    agg = {("iso", t): {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}
    agg.update({("all", t): {"tp": 0, "fp": 0, "fn": 0} for t in thresholds})
    # diagnostic @ IoU 0.5: of GT matched by ALL boxes, how many were matched
    # only because an *embedding*(inline) box covered them (class mismatch)?
    gt_by_embedding = 0
    total_gt = total_iso = total_emb = total_all = missing = 0
    started = time.perf_counter()

    for i, s in enumerate(samples):
        img_path = args.images_dir / Path(s["page_info"]["image_path"]).name
        if not img_path.exists():
            missing += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            missing += 1
            continue

        gts = [
            poly_to_xyxy(d["poly"])
            for d in s.get("layout_dets", [])
            if d.get("category_type") == GT_FORMULA_CATEGORY and not d.get("ignore")
        ]
        iso, emb, allb = detect(session, input_name, img, args.imgsz, args.conf)
        total_gt += len(gts); total_iso += len(iso); total_emb += len(emb); total_all += len(allb)

        for t in thresholds:
            tp, fp, fn = match(iso, gts, t)
            agg[("iso", t)]["tp"] += tp; agg[("iso", t)]["fp"] += fp; agg[("iso", t)]["fn"] += fn
            tp, fp, fn = match(allb, gts, t)
            agg[("all", t)]["tp"] += tp; agg[("all", t)]["fp"] += fp; agg[("all", t)]["fn"] += fn

        # class-mismatch diagnostic @0.5: GT matched by all-boxes but NOT by isolated
        tp_iso, _, _ = match(iso, gts, 0.5)
        tp_all, _, _ = match(allb, gts, 0.5)
        gt_by_embedding += max(0, tp_all - tp_iso)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(samples)} processed...")

    elapsed = time.perf_counter() - started
    print(f"\n[mfd-eval] done in {elapsed:.1f}s ({missing} missing)")
    print(f"[mfd-eval] GT={total_gt}  isolated_pred={total_iso}  embedding_pred={total_emb}  all_pred={total_all}")
    print(f"[mfd-eval] GT formulas matched ONLY via embedding(inline) box @0.5: {gt_by_embedding} "
          f"(class-label mismatch, would be recovered if both classes count as 'formula')\n")

    def report(tag, label):
        print(f"=== {label} ===")
        print(f"{'IoU':>5} | {'TP':>5} {'FP':>5} {'FN':>5} | {'Prec':>6} {'Rec':>6} {'F1':>6}")
        print("-" * 50)
        out = {}
        for t in sorted(thresholds, reverse=True):
            c = agg[(tag, t)]
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0
            out[t] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}
            print(f"{t:>5} | {tp:>5} {fp:>5} {fn:>5} | {p:>6.4f} {r:>6.4f} {f:>6.4f}")
        print()
        return out

    res_iso = report("iso", "MFD isolated (class 1) vs GT equation_isolated")
    res_all = report("all", "MFD all boxes (class 0+1) vs GT equation_isolated")

    args.out.write_text(json.dumps({
        "gt_samples": len(samples), "missing_images": missing,
        "total_gt": total_gt, "total_isolated_pred": total_iso, "total_all_pred": total_all,
        "elapsed_sec": round(elapsed, 1),
        "isolated_vs_gt": res_iso, "allboxes_vs_gt": res_all,
    }, indent=2), encoding="utf-8")
    print(f"[mfd-eval] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
