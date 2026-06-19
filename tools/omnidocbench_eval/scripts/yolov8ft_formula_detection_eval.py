"""
Measure MinerU yolo_v8_ft.pt formula DETECTION accuracy against the full
OmniDocBench GT — same IoU matching protocol as ppdoclayout_formula_detection_eval.py.

Runs the model raw via ultralytics (no ONNX export). Classes match pix2text-mfd:
  0 = embedding (inline)   1 = isolated (display)

Reports:
  - "isolated" (class 1) vs GT equation_isolated  <- apples-to-apples with PP-DocLayout
  - "all boxes" (class 0+1) vs GT equation_isolated  <- recall ceiling ignoring class label

Usage:
  .venv/Scripts/python.exe tools/omnidocbench_eval/scripts/yolov8ft_formula_detection_eval.py --device gpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from ppdoclayout_formula_detection_eval import poly_to_xyxy, iou, match  # reuse

MODEL_PATH = ROOT / "models" / "yolo_v8_ft.pt"
GT_FORMULA_CATEGORY = "equation_isolated"
CLASS_ISOLATED = 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=Path(".bench/omnidocbench/raw/OmniDocBench.json"))
    ap.add_argument("--images-dir", type=Path, default=Path(".bench/omnidocbench/raw/images"))
    ap.add_argument("--device", default="0", help="'cpu' or GPU index e.g. '0'")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--thresholds", default="0.3,0.5,0.7")
    ap.add_argument("--out", type=Path, default=Path(".bench/omnidocbench/yolov8ft_formula_detection.json"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))

    thresholds = [float(t) for t in args.thresholds.split(",")]
    samples = json.loads(args.gt.read_text(encoding="utf-8-sig"))
    if args.limit:
        samples = samples[: args.limit]

    print(f"[yolov8ft-eval] model: {MODEL_PATH.name}  samples: {len(samples)}  device: {args.device}  imgsz: {args.imgsz}  conf: {args.conf}")

    agg = {("iso", t): {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}
    agg.update({("all", t): {"tp": 0, "fp": 0, "fn": 0} for t in thresholds})
    gt_by_embedding = 0
    total_gt = total_iso = total_emb = total_all = missing = 0
    started = time.perf_counter()

    for i, s in enumerate(samples):
        img_path = args.images_dir / Path(s["page_info"]["image_path"]).name
        if not img_path.exists():
            missing += 1
            continue

        gts = [
            poly_to_xyxy(d["poly"])
            for d in s.get("layout_dets", [])
            if d.get("category_type") == GT_FORMULA_CATEGORY and not d.get("ignore")
        ]

        results = model.predict(
            str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        r = results[0]
        boxes = r.boxes

        iso, emb, allb = [], [], []
        if boxes is not None and len(boxes):
            for box in boxes:
                x0, y0, x1, y1 = box.xyxy[0].tolist()
                cls = int(box.cls[0].item())
                b = (x0, y0, x1, y1)
                allb.append(b)
                if cls == CLASS_ISOLATED:
                    iso.append(b)
                else:
                    emb.append(b)

        total_gt += len(gts)
        total_iso += len(iso)
        total_emb += len(emb)
        total_all += len(allb)

        for t in thresholds:
            tp, fp, fn = match(iso, gts, t)
            agg[("iso", t)]["tp"] += tp; agg[("iso", t)]["fp"] += fp; agg[("iso", t)]["fn"] += fn
            tp, fp, fn = match(allb, gts, t)
            agg[("all", t)]["tp"] += tp; agg[("all", t)]["fp"] += fp; agg[("all", t)]["fn"] += fn

        tp_iso, _, _ = match(iso, gts, 0.5)
        tp_all, _, _ = match(allb, gts, 0.5)
        gt_by_embedding += max(0, tp_all - tp_iso)

        if (i + 1) % 200 == 0:
            elapsed_so_far = time.perf_counter() - started
            print(f"  {i + 1}/{len(samples)} processed... ({elapsed_so_far:.0f}s)")

    elapsed = time.perf_counter() - started
    print(f"\n[yolov8ft-eval] done in {elapsed:.1f}s  ({missing} missing)")
    print(f"[yolov8ft-eval] GT={total_gt}  isolated_pred={total_iso}  embedding_pred={total_emb}  all_pred={total_all}")
    print(f"[yolov8ft-eval] GT matched ONLY via embedding box @IoU0.5: {gt_by_embedding} (class-label mismatch)\n")

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
            out[str(t)] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}
            print(f"{t:>5} | {tp:>5} {fp:>5} {fn:>5} | {p:>6.4f} {r:>6.4f} {f:>6.4f}")
        print()
        return out

    res_iso = report("iso", "yolo_v8_ft isolated (class 1) vs GT equation_isolated")
    res_all = report("all", "yolo_v8_ft all boxes (class 0+1) vs GT equation_isolated")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": MODEL_PATH.name,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "gt_samples": len(samples),
        "missing_images": missing,
        "total_gt": total_gt,
        "total_isolated_pred": total_iso,
        "total_embedding_pred": total_emb,
        "total_all_pred": total_all,
        "elapsed_sec": round(elapsed, 1),
        "isolated_vs_gt": res_iso,
        "allboxes_vs_gt": res_all,
    }, indent=2), encoding="utf-8")
    print(f"[yolov8ft-eval] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
