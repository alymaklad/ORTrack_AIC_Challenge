import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch_compat

torch_compat.ensure_torch_six()

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aic_online_pipeline import (
    center_error,
    iou_xywh,
    is_valid_box,
    load_boxes,
    load_policy_blob,
    normalized_center_error,
    track_sequence_online,
)
from lib.test.parameter.ortrack import parameters
from lib.test.tracker.ortrack import ORTrack


def summarize(errors, norm_errors, overlaps, latencies_ms):
    errors = np.asarray(errors, dtype=np.float64)
    norm_errors = np.asarray(norm_errors, dtype=np.float64)
    overlaps = np.asarray(overlaps, dtype=np.float64)
    precision_thresholds = np.arange(0, 51, dtype=np.float64)
    success_thresholds = np.linspace(0, 1, 101, dtype=np.float64)
    norm_precision_thresholds = np.linspace(0, 0.5, 101, dtype=np.float64)

    precision_curve = [(errors <= t).mean().item() for t in precision_thresholds]
    success_curve = [(overlaps >= t).mean().item() for t in success_thresholds]
    norm_precision_curve = [(norm_errors <= t).mean().item() for t in norm_precision_thresholds]

    return {
        "frames_evaluated": int(len(errors)),
        "precision_at_20px": float((errors <= 20.0).mean()),
        "precision_auc_0_50px": float(np.mean(precision_curve)),
        "success_auc": float(np.mean(success_curve)),
        "normalized_precision_at_0_5": float((norm_errors <= 0.5).mean()),
        "normalized_precision_auc_0_0_5": float(np.mean(norm_precision_curve)),
        "mean_center_error_px": float(errors.mean()),
        "median_center_error_px": float(np.median(errors)),
        "mean_iou": float(overlaps.mean()),
        "failures_iou_eq_0": int((overlaps <= 0.0).sum()),
        "failures_iou_lt_0_1": int((overlaps < 0.1).sum()),
        "latency_ms_mean_track_only": float(np.mean(latencies_ms)) if latencies_ms else None,
        "latency_ms_median_track_only": float(np.median(latencies_ms)) if latencies_ms else None,
        "fps_track_only": float(1000.0 / np.mean(latencies_ms)) if latencies_ms else None,
    }


def evaluate_sequence(tracker, data_root, rel_key, item, pred_dir, absence_policy=None, recovery_policy=None):
    anno_path = data_root / item["annotation_path"]
    gt_boxes = load_boxes(anno_path)
    t0 = time.perf_counter()
    diagnostics = []
    predictions, _ = track_sequence_online(
        tracker,
        data_root,
        item,
        absence_policy or {"conf_low": -1e9, "boundary_conf": -1e9, "huge_conf": -1e9, "max_area_ratio": 1e9, "scale_conf": -1e9, "max_scale_change": 1e9, "freeze_on_absent": False},
        recovery_policy=recovery_policy,
        diagnostics=diagnostics,
        gt_boxes=gt_boxes,
    )
    lat_ms = (time.perf_counter() - t0) * 1000.0

    seq_pred_dir = pred_dir / item["dataset"]
    seq_pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = seq_pred_dir / f"{item['seq_name']}.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "w", "h"])
        for idx, box in enumerate(predictions):
            writer.writerow([idx, *box])

    errors = []
    norm_errors = []
    overlaps = []
    absent_gt = 0
    absent_tp = 0
    absent_fp = 0
    absent_fn = 0
    recovered_frames = 0
    for idx, pred in enumerate(predictions):
        gt = gt_boxes[idx]
        pred_zero = (abs(pred[2]) < 1e-9 and abs(pred[3]) < 1e-9)
        gt_valid = is_valid_box(gt)
        if not gt_valid:
            absent_gt += 1
            if pred_zero:
                absent_tp += 1
            else:
                absent_fn += 1
            errors.append(1e6)
            norm_errors.append(1e6)
            overlaps.append(1.0 if pred_zero else 0.0)
            continue

        if pred_zero:
            absent_fp += 1
            errors.append(1e6)
            norm_errors.append(1e6)
            overlaps.append(0.0)
        else:
            errors.append(center_error(pred, gt))
            norm_errors.append(normalized_center_error(pred, gt))
            overlaps.append(iou_xywh(pred, gt))
        if idx < len(diagnostics) and diagnostics[idx].get("recovered"):
            recovered_frames += 1

    metrics = summarize(errors, norm_errors, overlaps, [lat_ms / max(len(predictions), 1)] * max(len(predictions) - 1, 1))
    metrics.update(
        {
            "sequence": rel_key,
            "video_path": str(data_root / item["video_path"]),
            "annotation_path": str(anno_path),
            "prediction_path": str(pred_path),
            "frames_expected": int(len(gt_boxes)),
            "frames_read": int(len(predictions)),
            "absent_gt_frames": int(absent_gt),
            "absent_tp": int(absent_tp),
            "absent_fp": int(absent_fp),
            "absent_fn": int(absent_fn),
            "recovered_frames": int(recovered_frames),
            "score_0p6_auc_0p4_precision": float(0.6 * metrics["success_auc"] + 0.4 * metrics["precision_at_20px"]),
        }
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("AIC_DATA_ROOT", "/data"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--output", default="outputs/aic_train_eval")
    parser.add_argument("--config", default="deit_tiny_aic_stage1")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--policy", default="")
    parser.add_argument("--recovery-policy", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.manifest is None:
        args.manifest = str(data_root / "metadata" / "contestant_manifest.json")
    output = Path(args.output)
    pred_dir = output / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    per_seq_path = output / "per_sequence_metrics.jsonl"
    summary_path = output / "summary_metrics.json"

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    sequences = list(manifest[args.split].items())
    if args.split_file:
        with open(args.split_file, "r", encoding="utf-8") as f:
            wanted = [line.strip() for line in f if line.strip()]
        manifest_split = manifest[args.split]
        sequences = [(key, manifest_split[key]) for key in wanted if key in manifest_split]
    if args.sequence:
        sequences = [(k, v) for k, v in sequences if k == args.sequence or v["seq_name"] == args.sequence]
    if args.limit:
        sequences = sequences[: args.limit]

    done = set()
    if args.resume and per_seq_path.exists():
        with open(per_seq_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["sequence"])

    absence_policy = load_policy_blob(args.policy) if args.policy else None
    recovery_policy = load_policy_blob(args.recovery_policy) if args.recovery_policy else None

    params = parameters(args.config)
    if args.checkpoint:
        params.checkpoint = args.checkpoint
    params.debug = 0
    params.save_all_boxes = False
    tracker = ORTrack(params, "aic_contest_eval")

    all_metrics = []
    if args.resume and per_seq_path.exists():
        with open(per_seq_path, "r", encoding="utf-8") as f:
            all_metrics.extend(json.loads(line) for line in f if line.strip())

    mode = "a" if args.resume else "w"
    with open(per_seq_path, mode, encoding="utf-8") as out_f:
        for rel_key, item in tqdm(sequences, desc=f"Evaluating {args.split} online pipeline"):
            if rel_key in done:
                continue
            try:
                metrics = evaluate_sequence(
                    tracker,
                    data_root,
                    rel_key,
                    item,
                    pred_dir,
                    absence_policy=absence_policy,
                    recovery_policy=recovery_policy,
                )
                all_metrics.append(metrics)
                out_f.write(json.dumps(metrics) + "\n")
                out_f.flush()
            except Exception as exc:
                error_metrics = {"sequence": rel_key, "error": repr(exc)}
                all_metrics.append(error_metrics)
                out_f.write(json.dumps(error_metrics) + "\n")
                out_f.flush()

    usable = [m for m in all_metrics if "error" not in m]
    if not usable:
        raise RuntimeError("No sequences were evaluated successfully.")

    keys = [
        "precision_at_20px",
        "precision_auc_0_50px",
        "success_auc",
        "normalized_precision_at_0_5",
        "normalized_precision_auc_0_0_5",
        "mean_center_error_px",
        "median_center_error_px",
        "mean_iou",
        "latency_ms_mean_track_only",
        "latency_ms_median_track_only",
        "fps_track_only",
        "score_0p6_auc_0p4_precision",
    ]
    frame_weighted = {}
    for key in keys:
        vals = [m[key] for m in usable if m.get(key) is not None]
        if vals:
            denom = sum(m["frames_evaluated"] for m in usable if m.get(key) is not None)
            frame_weighted[key] = float(sum(m[key] * m["frames_evaluated"] for m in usable if m.get(key) is not None) / max(denom, 1))
    frame_weighted["frames_evaluated"] = int(sum(m["frames_evaluated"] for m in usable))
    frame_weighted["sequences_evaluated"] = len(usable)
    frame_weighted["sequences_failed_to_run"] = len([m for m in all_metrics if "error" in m])
    frame_weighted["robustness_failures_iou_eq_0"] = int(sum(m["failures_iou_eq_0"] for m in usable))
    frame_weighted["robustness_failures_iou_lt_0_1"] = int(sum(m["failures_iou_lt_0_1"] for m in usable))
    frame_weighted["absent_gt_frames"] = int(sum(m["absent_gt_frames"] for m in usable))
    frame_weighted["absent_tp"] = int(sum(m["absent_tp"] for m in usable))
    frame_weighted["absent_fp"] = int(sum(m["absent_fp"] for m in usable))
    frame_weighted["absent_fn"] = int(sum(m["absent_fn"] for m in usable))
    frame_weighted["recovered_frames"] = int(sum(m["recovered_frames"] for m in usable))
    frame_weighted["metric_notes"] = {
        "precision_at_20px": "Fraction of valid-ground-truth frames with center error <= 20 px; absent GT frames count as success only if prediction is zero.",
        "success_auc": "Mean success rate over IoU thresholds 0.00..1.00; absent GT frames count as IoU=1 only when prediction is zero.",
        "normalized_precision_at_0_5": "Fraction of valid-ground-truth frames with normalized center error <= 0.5.",
        "robustness": "Reported as both IoU == 0 and IoU < 0.1 frame counts; no GT re-initialization was used.",
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"summary": frame_weighted, "sequences": all_metrics}, f, indent=2)

    print(json.dumps(frame_weighted, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
