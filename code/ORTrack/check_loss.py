import torch
import glob
import os

ckpts = glob.glob(r'C:\AIC\ORTrack\output_aic_finetune_v3\checkpoints\**\*.pth.tar', recursive=True)
for c in ckpts:
    data = torch.load(c, map_location='cpu')
    print(f"{c}: stage={data.get('stage')} epoch={data.get('epoch')} avg_loss={data.get('avg_loss')}")
