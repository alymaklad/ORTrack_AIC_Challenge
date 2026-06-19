import argparse
import glob
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    args = parser.parse_args()

    pattern = str(Path(args.checkpoint_dir) / "**" / "*.pth.tar")
    for checkpoint in glob.glob(pattern, recursive=True):
        data = torch.load(checkpoint, map_location="cpu")
        print(
            f"{checkpoint}: stage={data.get('stage')} "
            f"epoch={data.get('epoch')} avg_loss={data.get('avg_loss')}"
        )


if __name__ == "__main__":
    main()
