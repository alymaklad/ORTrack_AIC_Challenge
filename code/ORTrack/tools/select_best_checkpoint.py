import argparse
import json
import re
from pathlib import Path


LOG_RE = re.compile(
    r"\[(train|val):\s*(\d+),\s*(\d+)\s*/\s*(\d+)\].*?"
    r"Loss/total:\s*([0-9.]+).*?"
    r"Loss/giou:\s*([0-9.]+).*?"
    r"Loss/l1:\s*([0-9.]+).*?"
    r"Loss/location:\s*([0-9.]+).*?"
    r"IoU:\s*([0-9.]+)"
)


def parse_log(log_path, mode="val"):
    records = {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            match = LOG_RE.search(line)
            if not match or match.group(1) != mode:
                continue
            step = int(match.group(3))
            total_steps = int(match.group(4))
            if step != total_steps:
                continue
            epoch = int(match.group(2))
            records[epoch] = {
                "epoch": epoch,
                "line": line_no,
                "loss_total": float(match.group(5)),
                "loss_giou": float(match.group(6)),
                "loss_l1": float(match.group(7)),
                "loss_location": float(match.group(8)),
                "iou": float(match.group(9)),
            }
    return [records[k] for k in sorted(records)]


def checkpoint_for_epoch(checkpoint_dir, epoch):
    matches = sorted(Path(checkpoint_dir).glob(f"*ep{epoch:04d}.pth.tar"))
    return matches[-1] if matches else None


def rank_checkpoints(log_path, checkpoint_dir, top_k=3):
    records = parse_log(log_path, mode="val")
    records = [r for r in records if checkpoint_for_epoch(checkpoint_dir, r["epoch"]) is not None]
    ranked = sorted(records, key=lambda r: (r["iou"], -r["loss_total"]), reverse=True)
    candidates = []
    seen = set()
    for record in ranked[:top_k]:
        seen.add(record["epoch"])
        candidates.append(record)
    if records:
        best_loss = min(records, key=lambda r: r["loss_total"])
        if best_loss["epoch"] not in seen:
            candidates.append(best_loss)
        last = records[-1]
        if last["epoch"] not in seen:
            candidates.append(last)
    for record in candidates:
        record["checkpoint"] = str(checkpoint_for_epoch(checkpoint_dir, record["epoch"]))
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    candidates = rank_checkpoints(args.log, args.checkpoint_dir, args.top_k)
    result = {"candidates": candidates, "best": candidates[0] if candidates else None}
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
