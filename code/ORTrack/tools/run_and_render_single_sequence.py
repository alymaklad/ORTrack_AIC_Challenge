import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch_compat

torch_compat.ensure_torch_six()

from aic_online_pipeline import fmt_number, load_policy_blob, track_sequence_online
from lib.test.parameter.ortrack import parameters
from lib.test.tracker.ortrack import ORTrack


FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=r"C:\AIC\Data\metadata\contestant_manifest.json")
    parser.add_argument("--data-root", default=r"C:\AIC\Data")
    parser.add_argument("--split", default="public_lb")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--config", default="deit_tiny_aic_stage1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--recovery-policy", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-width", type=int, default=1280)
    return parser.parse_args()


def draw_text_block(frame, lines):
    x = 18
    y = 26
    pad = 8
    line_h = 22
    width = max(cv2.getTextSize(line, FONT, 0.58, 1)[0][0] for line in lines) + pad * 2
    height = line_h * len(lines) + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * line_h), FONT, 0.58, (255, 255, 255), 1, cv2.LINE_AA)


def draw_box(frame, bbox, scale, color):
    x, y, w, h = [value * scale for value in bbox]
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def render_video(video_path, meta, boxes, diagnostics, output_path, max_width):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(meta.get("native_fps") or cap.get(cv2.CAP_PROP_FPS) or 30))
    scale = min(1.0, max_width / max(1, src_w))
    out_w = int(round(src_w * scale))
    out_h = int(round(src_h * scale))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1, fps), (out_w, out_h))

    frame_idx = 0
    while frame_idx < int(meta["n_frames"]):
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        bbox = boxes[frame_idx]
        diag = diagnostics[frame_idx]
        absent = abs(bbox[2]) < 1e-9 and abs(bbox[3]) < 1e-9
        recovered = bool(diag.get("recovered"))
        if absent:
            state = "ABSENT"
            color = (0, 0, 255)
        elif recovered:
            state = "RECOVERED"
            color = (0, 215, 255)
            draw_box(frame, bbox, scale, color)
        else:
            state = "TRACKING"
            color = (80, 220, 80)
            draw_box(frame, bbox, scale, color)

        conf = float(diag.get("confidence", 0.0) or 0.0)
        reason = diag.get("reason") or "normal"
        mode = diag.get("mode") or state
        anchor_mult = diag.get("tracking_anchor_area_mult")
        proto_score = diag.get("prototype_score")
        lines = [
            meta["dataset"] + "/" + meta["seq_name"],
            f"frame {frame_idx}/{int(meta['n_frames']) - 1} fps {fps} conf {conf:.3f}",
            f"state {state} mode {mode} reason {reason}",
            f"anchor x{anchor_mult:.2f}" if anchor_mult is not None else "anchor x-",
            f"proto {proto_score:.3f}" if proto_score is not None else "proto -",
        ]
        draw_text_block(frame, lines)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args = parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    item = manifest[args.split][args.sequence]
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    absence_policy = load_policy_blob(args.policy)
    recovery_policy = load_policy_blob(args.recovery_policy) if args.recovery_policy else None

    params = parameters(args.config)
    params.checkpoint = args.checkpoint
    params.debug = 0
    params.save_all_boxes = False
    tracker = ORTrack(params, f"single_{args.sequence.replace('/', '_')}")

    diagnostics = []
    boxes, _ = track_sequence_online(
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
        for idx, box in enumerate(boxes):
            writer.writerow([idx, *(fmt_number(v) for v in box)])

    diag_path = output_dir / "diagnostics.json"
    diag_path.write_text(json.dumps({"sequence": args.sequence, "diagnostics": diagnostics}, indent=2), encoding="utf-8")

    render_path = output_dir / "render.mp4"
    video_path = data_root / item["video_path"]
    render_video(video_path, item, boxes, diagnostics, render_path, args.max_width)

    summary = {
        "sequence": args.sequence,
        "frames": len(boxes),
        "zero_boxes": sum(1 for box in boxes if abs(box[2]) < 1e-9 and abs(box[3]) < 1e-9),
        "recovered_frames": sum(1 for d in diagnostics if d.get("recovered")),
        "prototype_frames": sum(1 for d in diagnostics if d.get("prototype_score") is not None),
        "identity_mismatch_frames": sum(1 for d in diagnostics if d.get("reason") == "tracking_identity_mismatch"),
        "anchor_scale_mismatch_frames": sum(1 for d in diagnostics if d.get("reason") == "tracking_anchor_scale_mismatch"),
        "render_path": str(render_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
