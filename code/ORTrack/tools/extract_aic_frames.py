import argparse
import json
import os
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_sequence(data_root, out_root, key, item, jpeg_quality):
    video_path = data_root / item["video_path"]
    seq_dir = out_root / key.replace("/", "__")
    done_marker = seq_dir / ".done"
    expected = int(item["n_frames"])
    if done_marker.exists() and len(list(seq_dir.glob("*.jpg"))) >= expected:
        return "cached"
    seq_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "failed_open"
    idx = 0
    params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    while idx < expected:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(seq_dir / f"{idx + 1:08d}.jpg"), frame, params)
        idx += 1
    cap.release()
    if idx == expected:
        done_marker.write_text(str(idx), encoding="utf-8")
        return "ok"
    return f"short_{idx}_of_{expected}"


def main():
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("AIC_DATA_ROOT", "/data"))
    parser.add_argument("--manifest", default=str(data_root / "metadata" / "contestant_manifest.json"))
    parser.add_argument("--data-root", default=str(data_root))
    parser.add_argument("--output", default=str(data_root / "frames_cache"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "r", encoding="utf-8") as f:
        entries = json.load(f)[args.split]

    statuses = {}
    for key, item in tqdm(entries.items(), desc=f"Extracting {args.split}"):
        statuses[key] = extract_sequence(data_root, out_root, key, item, args.jpeg_quality)
    report = out_root / f"{args.split}_extract_report.json"
    report.write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    print(f"Saved report to {report}")


if __name__ == "__main__":
    main()
