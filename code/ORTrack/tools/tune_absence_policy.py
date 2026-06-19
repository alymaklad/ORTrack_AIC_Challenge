import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch_compat

torch_compat.ensure_torch_six()

from lib.test.parameter.ortrack import parameters
from lib.test.tracker.ortrack import ORTrack


def load_boxes(path):
    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in re.split(r"[\s,]+", line) if x.strip()]
            if len(vals) >= 4:
                boxes.append(vals[:4])
    return boxes


def valid_box(box):
    return len(box) >= 4 and box[2] > 0 and box[3] > 0


def center_error(pred, gt):
    pcx, pcy = pred[0] + pred[2] / 2.0, pred[1] + pred[3] / 2.0
    gcx, gcy = gt[0] + gt[2] / 2.0, gt[1] + gt[3] / 2.0
    return math.hypot(pcx - gcx, pcy - gcy)


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - inter
    return inter / union if union > 0 else 0.0


def features_for_box(box, prev_box, width, height, confidence):
    x, y, w, h = [float(v) for v in box]
    px, py, pw, ph = [float(v) for v in prev_box]
    frame_area = max(float(width * height), 1.0)
    area_ratio = max(w, 0.0) * max(h, 0.0) / frame_area
    touch_boundary = x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2
    out_of_bounds = x < -1 or y < -1 or x + w > width + 1 or y + h > height + 1
    huge_box = area_ratio >= 0.25
    pcx, pcy = px + pw / 2.0, py + ph / 2.0
    cx, cy = x + w / 2.0, y + h / 2.0
    prev_diag = math.sqrt(max(pw * pw + ph * ph, 1e-6))
    jump_obj = math.hypot(cx - pcx, cy - pcy) / prev_diag
    prev_area = max(pw * ph, 1e-9)
    area = max(w * h, 1e-9)
    scale_change = max(area / prev_area, prev_area / area)
    return {
        "confidence": float(confidence),
        "area_ratio": float(area_ratio),
        "touch_boundary": bool(touch_boundary),
        "out_of_bounds": bool(out_of_bounds),
        "huge_box": bool(huge_box),
        "jump_obj": float(jump_obj),
        "scale_change": float(scale_change),
    }


def policy_absent(row, policy):
    conf = row["confidence"]
    if conf < policy["conf_low"]:
        return True
    if row["touch_boundary"] and conf < policy["boundary_conf"]:
        return True
    if row["area_ratio"] >= policy["max_area_ratio"] and conf < policy["huge_conf"]:
        return True
    if row["scale_change"] >= policy["max_scale_change"] and conf < policy["scale_conf"]:
        return True
    if row["out_of_bounds"]:
        return True
    return False


def summarize_records(records, policy):
    precision_hits = []
    overlaps = []
    absent_tp = absent_fp = absent_fn = absent_tn = 0
    for row in records:
        pred_absent = policy_absent(row, policy)
        gt_absent = not row["gt_valid"]
        if gt_absent:
            if pred_absent:
                absent_tp += 1
                precision_hits.append(1.0)
                overlaps.append(1.0)
            else:
                absent_fn += 1
                precision_hits.append(0.0)
                overlaps.append(0.0)
            continue
        if pred_absent:
            absent_fp += 1
            precision_hits.append(0.0)
            overlaps.append(0.0)
        else:
            absent_tn += 1
            precision_hits.append(1.0 if row["center_error"] <= 20.0 else 0.0)
            overlaps.append(row["iou"])

    overlaps = np.asarray(overlaps, dtype=np.float64)
    success_thresholds = np.linspace(0, 1, 101)
    success_auc = float(np.mean([(overlaps >= t).mean() for t in success_thresholds]))
    precision = float(np.mean(precision_hits))
    score = 0.6 * success_auc + 0.4 * precision
    absent_precision = absent_tp / max(absent_tp + absent_fp, 1)
    absent_recall = absent_tp / max(absent_tp + absent_fn, 1)
    return {
        "score": score,
        "success_auc": success_auc,
        "precision_at_20px_or_absent": precision,
        "absent_tp": absent_tp,
        "absent_fp": absent_fp,
        "absent_fn": absent_fn,
        "absent_tn": absent_tn,
        "absent_precision": absent_precision,
        "absent_recall": absent_recall,
        "absent_f1": 2 * absent_precision * absent_recall / max(absent_precision + absent_recall, 1e-12),
    }


def collect_records(args):
    data_root = Path(args.data_root)
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)["train"]
    with open(args.split_file, "r", encoding="utf-8") as f:
        keys = [line.strip() for line in f if line.strip()]
    if args.limit:
        keys = keys[: args.limit]

    params = parameters(args.config)
    params.checkpoint = args.checkpoint
    params.debug = 0
    params.save_all_boxes = False
    tracker = ORTrack(params, "aic_absence_tune")

    records = []
    for seq_key in tqdm(keys, desc="Collecting tracker diagnostics"):
        item = manifest.get(seq_key)
        if item is None:
            continue
        video_path = data_root / item["video_path"]
        anno_path = data_root / item["annotation_path"]
        gt_boxes = load_boxes(anno_path)
        if not gt_boxes:
            continue
        first_idx = next((i for i, box in enumerate(gt_boxes) if valid_box(box)), None)
        if first_idx is None:
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0
        frame = None
        while frame_idx <= first_idx:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
        if frame is None:
            cap.release()
            continue
        init_box = gt_boxes[first_idx]
        tracker.initialize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), {"init_bbox": init_box})
        records.append(
            {
                "sequence": seq_key,
                "frame": first_idx,
                "gt_valid": True,
                "confidence": 1.0,
                "area_ratio": init_box[2] * init_box[3] / max(width * height, 1),
                "touch_boundary": False,
                "out_of_bounds": False,
                "huge_box": False,
                "jump_obj": 0.0,
                "scale_change": 1.0,
                "center_error": 0.0,
                "iou": 1.0,
            }
        )
        total = min(len(gt_boxes), int(item.get("n_frames", len(gt_boxes))))
        while frame_idx < total:
            ok, frame = cap.read()
            if not ok:
                break
            prev_state = list(tracker.state)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = tracker.track(rgb, {})
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pred = [float(x) for x in out["target_bbox"]]
            feat = features_for_box(pred, prev_state, width, height, getattr(tracker, "last_confidence", 0.0))
            gt = gt_boxes[frame_idx]
            gt_valid = valid_box(gt)
            records.append(
                {
                    "sequence": seq_key,
                    "frame": frame_idx,
                    "gt_valid": gt_valid,
                    **feat,
                    "center_error": center_error(pred, gt) if gt_valid else math.inf,
                    "iou": iou_xywh(pred, gt) if gt_valid else 0.0,
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                }
            )
            frame_idx += 1
        cap.release()
    return records


def tune(records):
    best = None
    results = []
    for conf_low in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.25, 0.30]:
        for boundary_conf in [0.04, 0.08, 0.12, 0.16, 0.20, 0.25, 0.35]:
            for huge_conf in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
                for max_area_ratio in [0.15, 0.20, 0.25, 0.35, 0.50]:
                    for scale_conf in [0.04, 0.08, 0.12, 0.20]:
                        policy = {
                            "conf_low": conf_low,
                            "boundary_conf": boundary_conf,
                            "huge_conf": huge_conf,
                            "max_area_ratio": max_area_ratio,
                            "scale_conf": scale_conf,
                            "max_scale_change": 4.0,
                            "freeze_on_absent": True,
                        }
                        metrics = summarize_records(records, policy)
                        result = {**policy, **metrics}
                        results.append(result)
                        if best is None or result["score"] > best["score"]:
                            best = result
    return best, sorted(results, key=lambda x: x["score"], reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"C:\AIC\Data")
    parser.add_argument("--manifest", default=r"C:\AIC\Data\metadata\contestant_manifest.json")
    parser.add_argument("--split-file", default=r"C:\AIC\ORTrack\data_specs\aic_contest_val.txt")
    parser.add_argument("--config", default="deit_tiny_aic_stage1")
    parser.add_argument(
        "--checkpoint",
        default=r"C:\AIC\ORTrack\output_aic_finetune\checkpoints\train\ortrack\deit_tiny_aic_stage1\ORTrack_ep0008.pth.tar",
    )
    parser.add_argument("--output-dir", default=r"C:\AIC\ORTrack\output_aic_finetune\absence_policy_stage1_ep0008")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = collect_records(args)
    best, results = tune(records)

    with open(output_dir / "diagnostics.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for record in records for key in record.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    with open(output_dir / "absence_policy.json", "w", encoding="utf-8") as f:
        json.dump({"best_policy": best, "top_results": results[:20], "records": len(records)}, f, indent=2)
    print(json.dumps({"best_policy": best, "records": len(records), "output": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
