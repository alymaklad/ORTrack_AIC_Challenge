import argparse
import csv
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

import torch_compat

torch_compat.ensure_torch_six()

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aic_online_pipeline import fmt_number, load_policy_blob, track_sequence_online
from lib.test.parameter.ortrack import parameters
from lib.test.tracker.ortrack import ORTrack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("AIC_DATA_ROOT", "/data"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--output", default="outputs/ortrack_aic_absent_aware_predictions.csv")
    parser.add_argument("--diagnostics-output", default="")
    parser.add_argument("--config", default="deit_tiny_aic_stage1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--recovery-policy", default="")
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
    absence_policy = load_policy_blob(args.policy)
    recovery_policy = load_policy_blob(args.recovery_policy) if args.recovery_policy else None
    sequences = list(manifest[args.split].items())

    params = parameters(args.config)
    params.checkpoint = args.checkpoint
    params.debug = 0
    params.save_all_boxes = False
    tracker = ORTrack(params, "aic_public_lb_absent")

    predictions = {}
    diagnostics = []
    absent_total = 0
    recovered_total = 0
    for rel_key, item in tqdm(sequences, desc=f"Tracking {args.split} online recovery"):
        seq_diag = []
        boxes, absent_reasons = track_sequence_online(
            tracker,
            data_root,
            item,
            absence_policy,
            recovery_policy=recovery_policy,
            diagnostics=seq_diag,
        )
        absent_total += len(absent_reasons)
        recovered_total += sum(1 for d in seq_diag if d.get("recovered"))
        for d in seq_diag:
            d["sequence"] = rel_key
            diagnostics.append(d)
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

    if args.diagnostics_output:
        diag_path = Path(args.diagnostics_output)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "policy": absence_policy,
                    "recovery_policy": recovery_policy,
                    "absent_predictions": absent_total,
                    "recovered_predictions": recovered_total,
                    "missing": missing,
                    "diagnostics": diagnostics,
                },
                f,
                indent=2,
            )

    print(f"Saved submission to {output_path}")
    print(f"Rows written: {len(predictions)} predictions, missing sample ids filled: {missing}")
    print(f"Absent/zero predictions from policy: {absent_total}")
    print(f"Frames recovered by online recovery: {recovered_total}")


if __name__ == "__main__":
    main()
