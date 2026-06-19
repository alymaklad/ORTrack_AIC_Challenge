import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=r"C:\AIC\Data\metadata\contestant_manifest.json")
    parser.add_argument("--output-dir", default=r"C:\AIC\ORTrack\data_specs")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        train = json.load(f)["train"]

    grouped = defaultdict(list)
    for key, item in train.items():
        grouped[item["dataset"]].append(key)

    rng = random.Random(args.seed)
    train_keys = []
    val_keys = []
    for dataset, keys in sorted(grouped.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * args.val_ratio))
        val_keys.extend(keys[:n_val])
        train_keys.extend(keys[n_val:])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "aic_contest_train.txt").write_text("\n".join(sorted(train_keys)) + "\n", encoding="utf-8")
    (out / "aic_contest_val.txt").write_text("\n".join(sorted(val_keys)) + "\n", encoding="utf-8")
    summary = {
        "train_sequences": len(train_keys),
        "val_sequences": len(val_keys),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    (out / "aic_contest_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
