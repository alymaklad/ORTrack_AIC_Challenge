import argparse
import glob
import os
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    args = parser.parse_args()

    checkpoints = glob.glob(os.path.join(args.checkpoint_dir, "**", "*.pth.tar"), recursive=True)
    fixed = 0
    for checkpoint in checkpoints:
        try:
            data = torch.load(checkpoint, map_location="cpu")
            if "model" in data and "net" not in data:
                print(f"Fixing {checkpoint}...")
                data["net"] = data.pop("model")
                torch.save(data, checkpoint)
                fixed += 1
        except Exception as exc:
            print(f"Error processing {checkpoint}: {exc}")

    print(f"Fixed {fixed} checkpoints.")


if __name__ == "__main__":
    main()
