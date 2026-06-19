import torch
import glob
import os

ckpt_dir = r"C:\AIC\ORTrack\output_aic_finetune_v3\checkpoints"
checkpoints = glob.glob(os.path.join(ckpt_dir, "**", "*.pth.tar"), recursive=True)

fixed = 0
for cp in checkpoints:
    try:
        data = torch.load(cp, map_location="cpu")
        if "model" in data and "net" not in data:
            print(f"Fixing {cp}...")
            data["net"] = data.pop("model")
            torch.save(data, cp)
            fixed += 1
    except Exception as e:
        print(f"Error processing {cp}: {e}")

print(f"Fixed {fixed} checkpoints.")
