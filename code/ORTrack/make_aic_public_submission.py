import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

import torch_compat

torch_compat.ensure_torch_six()

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.test.parameter.ortrack import parameters
from lib.test.tracker.ortrack import ORTrack


def load_init_box(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [float(x) for x in re.split(r"[\s,]+", line) if x.strip()]
            if len(parts) < 4:
                raise ValueError(f"Expected 4 bbox values in {path}: {line}")
            return parts[:4]
    raise ValueError(f"No initialization bbox found in {path}")


def fmt_number(value):
    if value is None or not math.isfinite(value):
        return "0"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def track_sequence(tracker, data_root, item):
    video_path = data_root / item["video_path"]
    init_box = load_init_box(data_root / item["annotation_path"])
    expected_frames = int(item["n_frames"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read first frame: {video_path}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tracker.initialize(rgb, {"init_bbox": init_box})
    boxes = [init_box]
    last_box = init_box

    for _ in range(1, expected_frames):
        ok, frame = cap.read()
        if not ok:
            boxes.append(last_box)
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            out = tracker.track(rgb, {})
        last_box = [float(x) for x in out["target_bbox"]]
        boxes.append(last_box)

    cap.release()
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("AIC_DATA_ROOT", "/data"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--output", default="outputs/ortrack_aic_predictions.csv")
    parser.add_argument("--config", default="deit_tiny_patch16_224")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="public_lb")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.manifest is None:
        args.manifest = str(data_root / "metadata" / "contestant_manifest.json")
    if args.sample is None:
        args.sample = str(data_root / "metadata" / "sample_submission.csv")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    sequences = list(manifest[args.split].items())

    params = parameters(args.config)
    if args.checkpoint:
        params.checkpoint = args.checkpoint
    params.debug = 0
    params.save_all_boxes = False
    tracker = ORTrack(params, "aic_public_lb")

    predictions = {}
    for rel_key, item in tqdm(sequences, desc=f"Tracking {args.split}"):
        boxes = track_sequence(tracker, data_root, item)
        for frame_idx, box in enumerate(boxes):
            predictions[f"{rel_key}_{frame_idx}"] = box

    missing = 0
    with open(args.sample, "r", encoding="utf-8", newline="") as sample_f, open(
        output_path, "w", encoding="utf-8", newline=""
    ) as out_f:
        reader = csv.DictReader(sample_f)
        writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            box = predictions.get(row["id"])
            if box is None:
                missing += 1
                box = [0, 0, 0, 0]
            writer.writerow(
                {
                    "id": row["id"],
                    "x": fmt_number(box[0]),
                    "y": fmt_number(box[1]),
                    "w": fmt_number(box[2]),
                    "h": fmt_number(box[3]),
                }
            )

    print(f"Saved submission to {output_path}")
    print(f"Rows written: {len(predictions)} predictions, missing sample ids filled: {missing}")


if __name__ == "__main__":
    main()
