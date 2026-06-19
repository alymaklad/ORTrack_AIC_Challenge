import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch_compat

torch_compat.ensure_torch_six()

from aic_online_pipeline import fmt_number, load_policy_blob, track_sequence_online
from lib.test.parameter.ortrack import parameters
from lib.test.tracker.ortrack import ORTrack


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"C:\AIC\Data")
    parser.add_argument("--manifest", default=r"C:\AIC\Data\metadata\contestant_manifest.json")
    parser.add_argument("--split", default="public_lb")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--config", default="deit_tiny_aic_stage1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--recovery-policy", default="")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))[args.split]
    if args.sequence not in manifest:
        raise KeyError(f"Sequence {args.sequence} not found in split {args.split}")
    item = manifest[args.sequence]

    absence_policy = load_policy_blob(args.policy)
    recovery_policy = load_policy_blob(args.recovery_policy) if args.recovery_policy else None

    params = parameters(args.config)
    params.checkpoint = args.checkpoint
    params.debug = 0
    params.save_all_boxes = False
    tracker = ORTrack(params, f"single_seq_{args.sequence.replace('/', '_')}")

    diagnostics = []
    boxes, absent_reasons = track_sequence_online(
        tracker,
        data_root,
        item,
        absence_policy,
        recovery_policy=recovery_policy,
        diagnostics=diagnostics,
    )

    pred_path = output_dir / "predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "w", "h"])
        for frame_idx, box in enumerate(boxes):
            writer.writerow([frame_idx, *(fmt_number(v) for v in box)])

    diag_path = output_dir / "diagnostics.json"
    diag_path.write_text(
        json.dumps(
            {
                "sequence": args.sequence,
                "policy": absence_policy,
                "recovery_policy": recovery_policy,
                "absent_predictions": len(absent_reasons),
                "recovered_predictions": sum(1 for d in diagnostics if d.get("recovered")),
                "diagnostics": [{**d, "sequence": args.sequence} for d in diagnostics],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "sequence": args.sequence,
        "frames": len(boxes),
        "zero_boxes": sum(1 for b in boxes if abs(b[2]) < 1e-9 and abs(b[3]) < 1e-9),
        "absent_predictions": len(absent_reasons),
        "recovered_predictions": sum(1 for d in diagnostics if d.get("recovered")),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved predictions to {pred_path}")
    print(f"Saved diagnostics to {diag_path}")


if __name__ == "__main__":
    main()
