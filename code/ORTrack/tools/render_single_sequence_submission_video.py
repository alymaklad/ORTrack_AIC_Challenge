import argparse
import csv
import json
import os
from pathlib import Path

import cv2


FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default=os.environ.get("AIC_DATA_ROOT", "/data"))
    parser.add_argument("--max-width", type=int, default=1280)
    return parser.parse_args()


def load_submission(path, sequence):
    boxes = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if "id" in fieldnames:
            for row in reader:
                seq, frame_idx = row["id"].rsplit("_", 1)
                if seq != sequence:
                    continue
                boxes[int(frame_idx)] = tuple(float(row[key]) for key in ("x", "y", "w", "h"))
        elif "frame" in fieldnames:
            for row in reader:
                boxes[int(row["frame"])] = tuple(float(row[key]) for key in ("x", "y", "w", "h"))
        else:
            raise ValueError(f"Unsupported prediction CSV format for {path}")
    return boxes


def load_diagnostics(path, sequence):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["diagnostics"]
    return {int(item["frame"]): item for item in raw if item.get("sequence") == sequence}


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


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["public_lb"]
    meta = manifest[args.sequence]
    data_root = Path(args.data_root)
    video_path = data_root / meta["video_path"]
    boxes = load_submission(args.submission, args.sequence)
    diagnostics = load_diagnostics(args.diagnostics, args.sequence)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(meta.get("native_fps") or cap.get(cv2.CAP_PROP_FPS) or 30))
    scale = min(1.0, args.max_width / max(1, src_w))
    out_w = int(round(src_w * scale))
    out_h = int(round(src_h * scale))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1, fps), (out_w, out_h))

    frame_idx = 0
    while frame_idx < int(meta["n_frames"]):
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        bbox = boxes.get(frame_idx, (0.0, 0.0, 0.0, 0.0))
        diag = diagnostics.get(frame_idx, {})
        absent = all(abs(v) < 1e-9 for v in bbox)
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
        identity_sim = diag.get("tracking_identity_similarity")
        lines = [
            args.sequence,
            f"frame {frame_idx}/{int(meta['n_frames']) - 1} fps {fps} conf {conf:.3f}",
            f"state {state} mode {mode} reason {reason}",
            f"anchor area x{anchor_mult:.2f}" if anchor_mult is not None else "anchor area x-",
            f"id sim {identity_sim:.3f}" if identity_sim is not None else "id sim -",
        ]
        draw_text_block(frame, lines)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved full-sequence render to {output_path}")


if __name__ == "__main__":
    main()
