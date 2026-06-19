import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AIC_DATA_ROOT", "/data"))
MANIFEST_PATH = DATA_ROOT / "metadata" / "contestant_manifest.json"
CLIP_SECONDS = 12
MAX_WIDTH = 1280
FONT = cv2.FONT_HERSHEY_SIMPLEX

SELECTIONS = [
    {"sequence": "dataset5/person19_3", "bucket": "absence-heavy", "why": "many absent outputs, huge boxes, boundary pressure"},
    {"sequence": "dataset3/couple", "bucket": "absence-heavy", "why": "frequent absent outputs with crowded target interaction"},
    {"sequence": "dataset4/person19", "bucket": "absence-heavy", "why": "long sequence with many boundary touches and some absences"},
    {"sequence": "dataset5/person19_1", "bucket": "absence-heavy", "why": "repeated low-confidence drops with aggressive scale swings"},
    {"sequence": "dataset4/group2", "bucket": "absence-heavy", "why": "group scene with low-confidence disappearances"},
    {"sequence": "dataset1/person_3", "bucket": "absence-heavy", "why": "huge-box suppression episodes"},
    {"sequence": "dataset3/truck", "bucket": "absence-heavy", "why": "jumps and moderate absence spans"},
    {"sequence": "dataset3/jogging2", "bucket": "absence-heavy", "why": "runner drift and confidence collapses"},
    {"sequence": "dataset1/surfer_1", "bucket": "absence-heavy", "why": "repeated huge-box absent suppression"},
    {"sequence": "dataset5/person19_2", "bucket": "absence-heavy", "why": "boundary and jump instability with absences"},
    {"sequence": "dataset4/person20", "bucket": "challenging", "why": "mixed normal and absent behavior near image edge"},
    {"sequence": "dataset3/tennis_player1_2", "bucket": "challenging", "why": "brief absent suppression mixed with boundary touches"},
    {"sequence": "dataset1/basketball", "bucket": "challenging", "why": "short clip with quick loss and recovery"},
    {"sequence": "dataset3/car4", "bucket": "challenging", "why": "mostly stable with sharp jump events"},
    {"sequence": "dataset3/parterre2", "bucket": "challenging", "why": "few absences but unstable motion spikes"},
    {"sequence": "dataset2/RcCar4", "bucket": "challenging", "why": "small fast object with one huge-box suppression"},
    {"sequence": "dataset5/person16", "bucket": "challenging", "why": "low-confidence dip without many zero outputs"},
    {"sequence": "dataset5/uav7", "bucket": "challenging", "why": "low mean confidence and fast scale changes"},
    {"sequence": "dataset2/RcCar9", "bucket": "challenging", "why": "jump-heavy but no zero outputs"},
    {"sequence": "dataset5/group2_1", "bucket": "challenging", "why": "crowded motion with confidence wobble"},
    {"sequence": "dataset5/boat5", "bucket": "normal", "why": "very stable clean track"},
    {"sequence": "dataset5/person4_2", "bucket": "normal", "why": "stable person track"},
    {"sequence": "dataset3/car7_2", "bucket": "normal", "why": "long, high-confidence vehicle tracking"},
    {"sequence": "dataset5/person5_1", "bucket": "normal", "why": "high-confidence person motion"},
    {"sequence": "dataset5/boat3", "bucket": "normal", "why": "steady scale and clean localization"},
    {"sequence": "dataset3/bike9_1", "bucket": "normal", "why": "short stable bicycle clip"},
    {"sequence": "dataset3/car7_1", "bucket": "normal", "why": "stable car tracking"},
    {"sequence": "dataset4/person5", "bucket": "normal", "why": "long normal person tracking"},
    {"sequence": "dataset5/building1", "bucket": "normal", "why": "easy static-looking target"},
    {"sequence": "dataset1/Car_video", "bucket": "normal", "why": "baseline healthy example"},
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--submission", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Submission Gallery")
    parser.add_argument("--intro", default="Thirty annotated clips mixing absence-heavy failures, challenging edge cases, and normal healthy tracks.")
    return parser.parse_args()


def load_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["public_lb"]


def load_submission(path):
    boxes = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seq, frame_idx = row["id"].rsplit("_", 1)
            boxes[(seq, int(frame_idx))] = tuple(float(row[key]) for key in ("x", "y", "w", "h"))
    return boxes


def load_diagnostics(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["diagnostics"]
    by_seq = defaultdict(dict)
    for item in raw:
        by_seq[item["sequence"]][int(item["frame"])] = item
    return by_seq


def seq_stats(diag_by_frame):
    values = list(diag_by_frame.values())
    absent_count = sum(1 for item in values if item.get("absent"))
    huge_count = sum(1 for item in values if item.get("reason") == "huge_low_conf")
    boundary_count = sum(1 for item in values if item.get("touch_boundary"))
    recovered_count = sum(1 for item in values if item.get("recovered"))
    min_conf = min(float(item.get("confidence", 1.0)) for item in values) if values else 1.0
    mean_conf = sum(float(item.get("confidence", 1.0)) for item in values) / len(values) if values else 1.0
    return {
        "frames": len(values),
        "absent_count": absent_count,
        "huge_count": huge_count,
        "boundary_count": boundary_count,
        "recovered_count": recovered_count,
        "mean_conf": mean_conf,
        "min_conf": min_conf,
    }


def choose_center_frame(bucket, diag_by_frame, n_frames):
    if bucket == "normal":
        recovered = [frame for frame, item in diag_by_frame.items() if item.get("recovered")]
        if recovered:
            return recovered[0]
        return max(0, min(n_frames - 1, n_frames // 2))

    recovered = [frame for frame, item in diag_by_frame.items() if item.get("recovered")]
    if recovered:
        return recovered[0]
    absent_frames = [frame for frame, item in diag_by_frame.items() if item.get("absent")]
    if absent_frames:
        return absent_frames[0]
    huge_frames = [frame for frame, item in diag_by_frame.items() if item.get("reason") == "huge_low_conf"]
    if huge_frames:
        return huge_frames[0]
    low_conf_frame = None
    low_conf_value = 10.0
    for frame, item in diag_by_frame.items():
        conf = float(item.get("confidence", 1.0))
        if conf < low_conf_value:
            low_conf_value = conf
            low_conf_frame = frame
    if low_conf_frame is not None:
        return low_conf_frame
    return max(0, min(n_frames - 1, n_frames // 2))


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


def draw_box(frame, bbox, color):
    x, y, w, h = bbox
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def render_clip(entry, manifest, boxes, diagnostics, output_root):
    seq = entry["sequence"]
    meta = manifest[seq]
    video_path = DATA_ROOT / meta["video_path"]
    diag_by_frame = diagnostics.get(seq, {})
    stats = seq_stats(diag_by_frame)
    fps = int(round(meta.get("native_fps") or 30))
    center_frame = choose_center_frame(entry["bucket"], diag_by_frame, meta["n_frames"])
    half_window = max(1, fps * CLIP_SECONDS // 2)
    start_frame = max(0, center_frame - half_window)
    end_frame = min(meta["n_frames"] - 1, center_frame + half_window)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, MAX_WIDTH / max(1, src_w))
    out_w = int(round(src_w * scale))
    out_h = int(round(src_h * scale))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    safe_name = seq.replace("/", "__")
    clip_name = f"{entry['bucket']}_{safe_name}.mp4"
    out_path = output_root / "videos" / clip_name
    writer = cv2.VideoWriter(str(out_path), fourcc, max(1, fps), (out_w, out_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        bbox = boxes.get((seq, frame_idx), (0.0, 0.0, 0.0, 0.0))
        diag = diag_by_frame.get(frame_idx, {})
        absent = all(abs(v) < 1e-9 for v in bbox)
        display_box = tuple(value * scale for value in bbox)
        recovered = bool(diag.get("recovered"))
        if absent:
            label = "ABSENT"
            color = (0, 0, 255)
        elif recovered:
            label = "RECOVERED"
            color = (0, 215, 255)
            draw_box(frame, display_box, color)
        else:
            label = "TRACKING"
            color = (80, 220, 80)
            draw_box(frame, display_box, color)

        conf = float(diag.get("confidence", 1.0))
        reason = diag.get("reason") or "normal"
        mode = diag.get("mode") or ("ABSENT" if absent else "TRACKING")
        reacquire_streak = int(diag.get("reacquire_streak", 0) or 0)
        rec_score = diag.get("recovery_score")
        rec_template = diag.get("recovery_template") or "-"
        tid_checked = bool(diag.get("tracking_identity_checked"))
        tid_sim = diag.get("tracking_identity_similarity")
        tid_init = diag.get("tracking_identity_initial_similarity")
        lines = [
            f"{seq} [{entry['bucket']}]",
            f"frame {frame_idx}/{meta['n_frames'] - 1}  fps {fps}  conf {conf:.3f}",
            f"state {label}  mode {mode}  reason {reason}",
            f"recovered {stats['recovered_count']}  absent {stats['absent_count']}  huge {stats['huge_count']}",
            f"rec score {rec_score:.3f} tpl {rec_template}  reacq {reacquire_streak}" if rec_score is not None else f"boundary {stats['boundary_count']}  mean conf {stats['mean_conf']:.3f}  reacq {reacquire_streak}",
            f"idchk {int(tid_checked)} sim {tid_sim:.3f} init {tid_init:.3f}" if tid_checked and tid_sim is not None and tid_init is not None else "idchk 0",
            entry["why"],
        ]
        draw_text_block(frame, lines)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return {
        "sequence": seq,
        "bucket": entry["bucket"],
        "why": entry["why"],
        "video_path": str(video_path),
        "clip_path": str(out_path),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "fps": fps,
        **stats,
    }


def write_index(results, output_root, title, intro):
    rows = []
    for item in results:
        clip_rel = Path(item["clip_path"]).relative_to(output_root).as_posix()
        rows.append(
            f"""
            <section class="card {item['bucket']}">
              <h2>{item['sequence']}</h2>
              <p class="meta">{item['bucket']} | frames {item['start_frame']} - {item['end_frame']} | absent {item['absent_count']} | recovered {item['recovered_count']} | huge {item['huge_count']} | boundary {item['boundary_count']} | mean conf {item['mean_conf']:.3f}</p>
              <p class="why">{item['why']}</p>
              <video controls preload="metadata" src="{clip_rel}"></video>
            </section>
            """
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --panel: #fffaf1;
      --ink: #222222;
      --muted: #695f55;
      --accent: #0d6b6b;
      --absence: #9d2f2f;
      --challenge: #8c5a11;
      --normal: #2e6b2a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f6f3ea 0%, #ece6d8 100%);
      color: var(--ink);
      font-family: "Segoe UI", Tahoma, sans-serif;
    }}
    main {{
      width: min(1500px, calc(100vw - 32px));
      margin: 24px auto 48px;
    }}
    h1 {{ margin: 0 0 10px; font-size: 34px; }}
    .intro {{ margin: 0 0 24px; color: var(--muted); font-size: 16px; line-height: 1.45; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
    .card {{ background: var(--panel); border-radius: 8px; padding: 14px; border-top: 6px solid var(--accent); box-shadow: 0 10px 24px rgba(40, 34, 26, 0.08); }}
    .card.absence-heavy {{ border-top-color: var(--absence); }}
    .card.challenging {{ border-top-color: var(--challenge); }}
    .card.normal {{ border-top-color: var(--normal); }}
    h2 {{ margin: 0 0 8px; font-size: 20px; }}
    .meta, .why {{ margin: 0 0 10px; font-size: 14px; line-height: 1.4; color: var(--muted); }}
    video {{ width: 100%; display: block; background: #000; border-radius: 6px; }}
  </style>
</head>
<body>
  <main>
    <h1>{title}</h1>
    <p class="intro">{intro}</p>
    <div class="grid">{''.join(rows)}</div>
  </main>
</body>
</html>
"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    (output_root / "videos").mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    boxes = load_submission(args.submission)
    diagnostics = load_diagnostics(args.diagnostics)

    results = [render_clip(entry, manifest, boxes, diagnostics, output_root) for entry in SELECTIONS]
    (output_root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_index(results, output_root, args.title, args.intro)
    print(f"Rendered {len(results)} clips to {output_root}")
    print(output_root / "index.html")


if __name__ == "__main__":
    main()
