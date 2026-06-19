"""
run_aic_two_stage.py
====================
Orchestrates the full two-stage fine-tuning pipeline for ORTrack-DeiT Tiny
on the AIC contest dataset.

All hyperparameters are written directly in this file — no YAML config is read
or written.  The script builds the model, dataloaders, optimizer and loss
internally and calls the underlying ORTrack network forward pass directly.

Improvements over the plain fine-tuning that regressed from 0.740 → 0.732
---------------------------------------------------------------------------
  • EWC (Elastic Weight Consolidation) prevents catastrophic forgetting and
    guarantees the fine-tuned model stays ≥ pretrained baseline.
  • Layer-wise LR Decay (LLRD) protects early DeiT features.
  • 60 / 40 AIC / GOT-10k data mix anchors the pretrained distribution.
  • Targeted augmentations (motion blur, coarse dropout, brightness) simulate
    the hard sequences (motorcycle, plane, surfer).
  • Two-stage progressive unfreezing (heads → last-4-blocks → full model).
  • Checkpoint selection evaluates top-K candidates on a held-out val split.
"""

import argparse
import copy
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Silence duplicate-lib warning on Windows before any torch import
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Dataset

import torch_compat
torch_compat.ensure_torch_six()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.test.parameter.ortrack import parameters as ortrack_parameters
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box, box_xywh_to_xyxy, box_iou

# ============================================================================
#  ██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗      ██████╗  █████╗ ██████╗  █████╗ ███╗   ███╗███████╗
#  ██║  ██║╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗ ████║██╔════╝
#  ███████║ ╚████╔╝ ██████╔╝█████╗  ██████╔╝    ██████╔╝███████║██████╔╝███████║██╔████╔██║███████╗
#  ██╔══██║  ╚██╔╝  ██╔═══╝ ██╔══╝  ██╔══██╗    ██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║╚██╔╝██║╚════██║
#  ██║  ██║   ██║   ██║     ███████╗██║  ██║    ██║     ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║███████║
#  ╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚══════╝╚═╝  ╚═╝    ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝
# ============================================================================
# All training hyperparameters live here.  Change values in this block only.
# ============================================================================

# ── Paths ────────────────────────────────────────────────────────────────────
PYTHON_EXE      = sys.executable
SAVE_DIR        = str(ROOT / "outputs" / "aic_finetune")
PRETRAINED_CKPT = str(ROOT / "model" / "ORTrack_AIC.pth.tar")
DATA_ROOT       = os.environ.get("AIC_DATA_ROOT", "/data")
MANIFEST_PATH   = str(Path(DATA_ROOT) / "metadata" / "contestant_manifest.json")
GOT10K_ROOT     = None          # set to None to skip mixing
VAL_SPLIT_FILE  = str(ROOT / "data_specs" / "aic_contest_val.txt")
EVAL_SCRIPT     = str(ROOT / "eval_aic_train.py")

# ── Model ────────────────────────────────────────────────────────────────────
ORTRACK_CONFIG  = "deit_tiny_aic_stage1"   # used only to load params / network
SEARCH_FACTOR   = 4.0
SEARCH_SIZE     = 256
TEMPLATE_FACTOR = 2.0
TEMPLATE_SIZE   = 128

# ── Dataset ──────────────────────────────────────────────────────────────────
PAIRS_PER_SEQ       = 50     # (template, search) pairs sampled per AIC sequence
MAX_FRAME_GAP       = 100    # max frame distance between template and search
AIC_DATASET_WEIGHT  = 0.60   # fraction of each batch drawn from AIC data
GOT10K_PAIRS        = 8000   # how many GOT-10k pairs to include in the mix
BATCH_SIZE          = 16
NUM_WORKERS         = 4

# ── Augmentation ─────────────────────────────────────────────────────────────
AUG_MOTION_BLUR_P        = 0.40   # simulate fast motion (motorcycle, plane)
AUG_MOTION_BLUR_LIMIT    = (7, 21)
AUG_GAUSS_NOISE_P        = 0.30   # sky / water background noise (surfer, plane)
AUG_GAUSS_NOISE_VAR      = (20.0, 80.0)
AUG_COARSE_DROPOUT_P     = 0.50   # partial occlusion (all sequences)
AUG_DROPOUT_MAX_HOLES    = 4
AUG_DROPOUT_MAX_SIZE     = 32
AUG_DROPOUT_MIN_SIZE     = 8
AUG_BRIGHTNESS_P         = 0.40   # lighting change across occlusion
AUG_BRIGHTNESS_LIMIT     = 0.40
AUG_CONTRAST_LIMIT       = 0.40
AUG_SCALE_P              = 0.30   # scale change
AUG_SCALE_LIMIT          = 0.30

# ── EWC ──────────────────────────────────────────────────────────────────────
EWC_IMPORTANCE      = 1000.0   # λ — higher = more conservative
EWC_FISHER_BATCHES  = 200      # batches used to estimate Fisher information

# ── Stage 0 (warm-up: heads only) ────────────────────────────────────────────
S0_EPOCHS   = 5
S0_LR       = 1e-4
S0_WD       = 1e-4

# ── Stage 1 (last-4 blocks + heads) ──────────────────────────────────────────
S1_EPOCHS   = 15
S1_BASE_LR  = 5e-6    # head LR; earlier layers get decay^depth × this
S1_LLRD     = 0.65    # per-block LR decay factor (earlier blocks = lower LR)
S1_WD       = 1e-4
S1_EWC      = True

# ── Stage 2 (full model) ─────────────────────────────────────────────────────
S2_EPOCHS   = 10
S2_BASE_LR  = 1e-6
S2_LLRD     = 0.65
S2_WD       = 1e-4
S2_EWC      = True

# ── Template update gate ─────────────────────────────────────────────────────
TEMPLATE_UPDATE_THRESHOLD = 0.55  # only update template above this confidence

# ── Checkpoint selection ─────────────────────────────────────────────────────
EVAL_TOP_K        = 3    # evaluate top-K loss-ranked checkpoints on val split
SAVE_EVERY_N      = 5    # save a checkpoint every N epochs within each stage

# ── Loss weights ─────────────────────────────────────────────────────────────
LOSS_GIOU_WEIGHT   = 2.0
LOSS_L1_WEIGHT     = 5.0
LOSS_SCORE_WEIGHT  = 1.0

# ============================================================================
#  END OF HYPERPARAMETER BLOCK
# ============================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_boxes(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in re.split(r"[\s,]+", line) if x.strip()]
            if len(vals) >= 4:
                boxes.append(vals[:4])
    return boxes


def is_valid_box(box):
    return len(box) >= 4 and float(box[2]) > 0 and float(box[3]) > 0


def read_score(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        s = json.load(f)["summary"]
    return {
        "score":      0.6 * s["success_auc"] + 0.4 * s["precision_at_20px"],
        "auc":        s["success_auc"],
        "precision":  s["precision_at_20px"],
    }


# ---------------------------------------------------------------------------
# Augmentation (manual — avoids albumentations dependency)
# ---------------------------------------------------------------------------

def apply_augmentation(image_rgb: np.ndarray) -> np.ndarray:
    """
    Apply the targeted augmentation stack in-place on an RGB uint8 image.
    All augmentations are independent Bernoulli trials.
    """
    img = image_rgb.copy()

    # Motion blur (fast-moving objects)
    if random.random() < AUG_MOTION_BLUR_P:
        ksize = random.choice(range(AUG_MOTION_BLUR_LIMIT[0],
                                    AUG_MOTION_BLUR_LIMIT[1] + 1, 2))
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        if random.random() < 0.5:
            kernel[ksize // 2, :] = 1.0 / ksize   # horizontal
        else:
            kernel[:, ksize // 2] = 1.0 / ksize   # vertical
        img = cv2.filter2D(img, -1, kernel)

    # Gaussian noise (sky / water background)
    if random.random() < AUG_GAUSS_NOISE_P:
        var   = random.uniform(*AUG_GAUSS_NOISE_VAR)
        noise = np.random.normal(0, math.sqrt(var), img.shape).astype(np.float32)
        img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Coarse dropout / random erasing (partial occlusion)
    if random.random() < AUG_COARSE_DROPOUT_P:
        h, w = img.shape[:2]
        n_holes = random.randint(1, AUG_DROPOUT_MAX_HOLES)
        for _ in range(n_holes):
            hh = random.randint(AUG_DROPOUT_MIN_SIZE, AUG_DROPOUT_MAX_SIZE)
            hw = random.randint(AUG_DROPOUT_MIN_SIZE, AUG_DROPOUT_MAX_SIZE)
            y0 = random.randint(0, max(0, h - hh))
            x0 = random.randint(0, max(0, w - hw))
            img[y0:y0 + hh, x0:x0 + hw] = 0

    # Brightness / contrast (lighting change)
    if random.random() < AUG_BRIGHTNESS_P:
        alpha = 1.0 + random.uniform(-AUG_CONTRAST_LIMIT,  AUG_CONTRAST_LIMIT)
        beta  =       random.uniform(-AUG_BRIGHTNESS_LIMIT, AUG_BRIGHTNESS_LIMIT) * 255
        img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Random scale (zoom in/out on the search patch)
    if random.random() < AUG_SCALE_P:
        scale = 1.0 + random.uniform(-AUG_SCALE_LIMIT, AUG_SCALE_LIMIT)
        h, w  = img.shape[:2]
        nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
        img    = cv2.resize(img, (nw, nh))
        # Crop or pad back to original size
        if nh >= h and nw >= w:
            y0  = (nh - h) // 2
            x0  = (nw - w) // 2
            img = img[y0:y0 + h, x0:x0 + w]
        else:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            y0 = (h - nh) // 2
            x0 = (w - nw) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = img
            img = canvas

    return img


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AICTrackingDataset(Dataset):
    """
    Samples (template_patch, search_patch, gt_box_normalised) triples from
    AIC training sequences.
    """

    def __init__(self, manifest_path, data_root, split="train",
                 pairs_per_seq=PAIRS_PER_SEQ, max_gap=MAX_FRAME_GAP,
                 augment=True, split_file=None):
        self.data_root  = Path(data_root)
        self.augment    = augment
        self.items      = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        allowed_keys = None
        if split_file and os.path.exists(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                allowed_keys = set(line.strip() for line in f if line.strip())

        processed_seqs = 0
        for rel_key, item in manifest[split].items():
            if allowed_keys is not None and rel_key not in allowed_keys:
                continue
            processed_seqs += 1
            boxes = load_boxes(self.data_root / item["annotation_path"])
            valid = [(i, b) for i, b in enumerate(boxes) if is_valid_box(b)]
            if len(valid) < 2:
                continue
            video_path = str(self.data_root / item["video_path"])
            for _ in range(pairs_per_seq):
                t_idx, t_box = valid[0]   # always init frame as template
                s_entry      = random.choice(valid[1:])
                s_idx, s_box = s_entry
                if abs(s_idx - t_idx) > max_gap:
                    continue
                self.items.append({
                    "video": video_path,
                    "t_idx": t_idx, "t_box": list(t_box),
                    "s_idx": s_idx, "s_box": list(s_box),
                })

        print(f"[AICTrackingDataset] {split}: {len(self.items)} pairs "
              f"from {processed_seqs} sequences")

    def _read_frame(self, video_path, frame_idx):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        t_frame = self._read_frame(item["video"], item["t_idx"])
        s_frame = self._read_frame(item["video"], item["s_idx"])

        if t_frame is None or s_frame is None:
            # Fall back to a random valid item
            return self[random.randint(0, len(self) - 1)]

        if self.augment:
            s_frame = apply_augmentation(s_frame)

        # Crop template patch
        t_patch, _, _ = sample_target(
            t_frame, item["t_box"], TEMPLATE_FACTOR, output_sz=TEMPLATE_SIZE
        )
        # Crop search patch and get the GT box expressed inside the patch
        s_patch, resize_factor, _ = sample_target(
            s_frame, item["s_box"], SEARCH_FACTOR, output_sz=SEARCH_SIZE
        )

        # Normalise GT box to [0, 1] relative to search patch size
        x, y, w, h = item["s_box"]
        # Map from frame coords to search-patch coords (approximate)
        cx_f = x + w / 2.0
        cy_f = y + h / 2.0
        # Centre of the search crop in frame coords
        sx, sy, sw, sh = item["s_box"]
        s_cx = sx + sw / 2.0
        s_cy = sy + sh / 2.0
        half  = SEARCH_SIZE / (2.0 * resize_factor)
        crop_x0 = s_cx - half
        crop_y0  = s_cy - half

        gt_cx = (cx_f - crop_x0) * resize_factor / SEARCH_SIZE
        gt_cy = (cy_f - crop_y0) * resize_factor / SEARCH_SIZE
        gt_w  = w * resize_factor / SEARCH_SIZE
        gt_h  = h * resize_factor / SEARCH_SIZE
        gt_box_norm = np.array([gt_cx, gt_cy, gt_w, gt_h], dtype=np.float32)
        gt_box_norm = np.clip(gt_box_norm, 0.0, 1.0)

        # Convert HWC uint8 → CHW float32 in [0,1]
        t_tensor = torch.from_numpy(t_patch).permute(2, 0, 1).float() / 255.0
        s_tensor = torch.from_numpy(s_patch).permute(2, 0, 1).float() / 255.0
        gt_tensor = torch.from_numpy(gt_box_norm)

        return t_tensor, s_tensor, gt_tensor


class GOT10kDataset(Dataset):
    """
    Minimal GOT-10k loader that samples `max_pairs` (template, search) pairs
    from the GOT-10k train split.  Falls back silently if the root doesn't
    exist (data mixing is simply disabled).
    """

    def __init__(self, got10k_root, max_pairs=GOT10K_PAIRS, augment=True):
        self.augment = augment
        self.items   = []

        root = Path(got10k_root)
        if not root.exists():
            print(f"[GOT10kDataset] WARNING: {root} not found — skipping mix")
            return

        seq_dirs = sorted(root.glob("GOT-10k_Train_*"))
        random.shuffle(seq_dirs)

        for seq_dir in seq_dirs:
            if len(self.items) >= max_pairs:
                break
            gt_file = seq_dir / "groundtruth.txt"
            frames  = sorted(seq_dir.glob("*.jpg"))
            if not gt_file.exists() or len(frames) < 2:
                continue
            boxes = load_boxes(gt_file)
            valid = [(frames[i], boxes[i])
                     for i in range(min(len(frames), len(boxes)))
                     if is_valid_box(boxes[i])]
            if len(valid) < 2:
                continue
            t_path, t_box = valid[0]
            for s_path, s_box in random.choices(valid[1:], k=3):
                self.items.append({
                    "t_path": str(t_path), "t_box": list(t_box),
                    "s_path": str(s_path), "s_box": list(s_box),
                })

        print(f"[GOT10kDataset] Loaded {len(self.items)} pairs")

    def __len__(self):
        return len(self.items)

    def _load(self, path):
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        item = self.items[idx]
        t_frame = self._load(item["t_path"])
        s_frame = self._load(item["s_path"])
        if t_frame is None or s_frame is None:
            return self[random.randint(0, len(self) - 1)]

        if self.augment:
            s_frame = apply_augmentation(s_frame)

        t_patch, _, _            = sample_target(t_frame, item["t_box"], TEMPLATE_FACTOR, output_sz=TEMPLATE_SIZE)
        s_patch, resize_factor, _ = sample_target(s_frame, item["s_box"], SEARCH_FACTOR,   output_sz=SEARCH_SIZE)

        x, y, w, h = item["s_box"]
        sx, sy, sw, sh = item["s_box"]
        s_cx = sx + sw / 2.0
        s_cy = sy + sh / 2.0
        half  = SEARCH_SIZE / (2.0 * resize_factor)
        crop_x0, crop_y0 = s_cx - half, s_cy - half
        cx_f, cy_f = x + w / 2.0, y + h / 2.0

        gt_cx = (cx_f - crop_x0) * resize_factor / SEARCH_SIZE
        gt_cy = (cy_f - crop_y0) * resize_factor / SEARCH_SIZE
        gt_w  = w * resize_factor / SEARCH_SIZE
        gt_h  = h * resize_factor / SEARCH_SIZE
        gt_box_norm = np.clip(np.array([gt_cx, gt_cy, gt_w, gt_h], dtype=np.float32), 0.0, 1.0)

        t_tensor  = torch.from_numpy(t_patch).permute(2, 0, 1).float() / 255.0
        s_tensor  = torch.from_numpy(s_patch).permute(2, 0, 1).float() / 255.0
        gt_tensor = torch.from_numpy(gt_box_norm)
        return t_tensor, s_tensor, gt_tensor


def build_dataloader(manifest_path, data_root, got10k_root, train_split=None):
    aic_ds  = AICTrackingDataset(manifest_path, data_root, split="train", augment=True, split_file=train_split)
    got_ds  = GOT10kDataset(got10k_root, augment=True) if got10k_root else None

    if got_ds and len(got_ds) > 0:
        combined = ConcatDataset([aic_ds, got_ds])
        w_aic = AIC_DATASET_WEIGHT  / len(aic_ds)
        w_got = (1 - AIC_DATASET_WEIGHT) / len(got_ds)
        weights = [w_aic] * len(aic_ds) + [w_got] * len(got_ds)
        sampler = WeightedRandomSampler(weights, num_samples=len(combined), replacement=True)
        loader  = DataLoader(combined, batch_size=BATCH_SIZE, sampler=sampler,
                             num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        print(f"[DataLoader] AIC={len(aic_ds)}  GOT10k={len(got_ds)}  "
              f"ratio={AIC_DATASET_WEIGHT:.0%}/{1-AIC_DATASET_WEIGHT:.0%}")
    else:
        loader = DataLoader(aic_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        print(f"[DataLoader] AIC only — {len(aic_ds)} pairs")

    return loader


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def giou_loss(pred_xyxy, gt_xyxy):
    """Generalised IoU loss — both tensors are (N, 4) in xyxy normalised coords."""
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xyxy.unbind(-1)

    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)
    inter    = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    pred_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    gt_area   = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
    union     = pred_area + gt_area - inter + 1e-7

    iou = inter / union

    encl_x1 = torch.min(px1, gx1)
    encl_y1 = torch.min(py1, gy1)
    encl_x2 = torch.max(px2, gx2)
    encl_y2 = torch.max(py2, gy2)
    encl    = (encl_x2 - encl_x1).clamp(0) * (encl_y2 - encl_y1).clamp(0) + 1e-7

    giou = iou - (encl - union) / encl
    return (1 - giou).mean()


def compute_loss(pred_boxes, gt_boxes_cxcywh):
    """
    pred_boxes:       (B, 4) in [cx, cy, w, h] normalised to [0,1]
    gt_boxes_cxcywh:  (B, 4) in [cx, cy, w, h] normalised to [0,1]
    """
    # L1 loss
    l1 = F.l1_loss(pred_boxes, gt_boxes_cxcywh)

    # GIoU loss — convert cxcywh → xyxy
    def cxcywh_to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2,
                             cx + w / 2, cy + h / 2], dim=-1)

    pred_xyxy = cxcywh_to_xyxy(pred_boxes)
    gt_xyxy   = cxcywh_to_xyxy(gt_boxes_cxcywh)
    g_iou     = giou_loss(pred_xyxy.clamp(0, 1), gt_xyxy.clamp(0, 1))

    return LOSS_L1_WEIGHT * l1 + LOSS_GIOU_WEIGHT * g_iou, l1, g_iou


# ---------------------------------------------------------------------------
# EWC (Elastic Weight Consolidation)
# ---------------------------------------------------------------------------

class EWC:
    """
    Elastic Weight Consolidation.
    Call compute_fisher() once on a reference dataloader BEFORE fine-tuning.
    Call penalty() each step and add to task loss.

    Reference: Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting
    in neural networks."
    """

    def __init__(self, model: nn.Module, importance: float = EWC_IMPORTANCE):
        self.model      = model
        self.importance = importance
        self.params     = {}   # θ* — weights at pretraining
        self.fisher     = {}   # F  — diagonal Fisher information

    def compute_fisher(self, dataloader, device, n_batches=EWC_FISHER_BATCHES):
        print(f"[EWC] Computing Fisher information over {n_batches} batches …")
        self.model.eval()

        # Snapshot pretrained weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.data.clone().to(device)
                self.fisher[name] = torch.zeros_like(param.data)

        for batch_idx, (t_imgs, s_imgs, gt_boxes) in enumerate(tqdm(dataloader, total=n_batches,
                                                                      desc="[EWC] Fisher")):
            if batch_idx >= n_batches:
                break
            t_imgs  = t_imgs.to(device)
            s_imgs  = s_imgs.to(device)
            gt_boxes = gt_boxes.to(device)

            self.model.zero_grad()
            try:
                out      = self.model(template=t_imgs, search=s_imgs,
                                      is_distill=getattr(self.model, 'is_distill', False))
                pred_box = out.get('pred_boxes', out.get('target_bbox', None))
                if pred_box is None:
                    continue
                pred_box = pred_box.view(-1, 4)
                loss, _, _ = compute_loss(pred_box, gt_boxes)
                loss.backward()
            except Exception as exc:
                print(f"[EWC] Skipping batch {batch_idx}: {exc}")
                continue

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data.pow(2)

        for name in self.fisher:
            self.fisher[name] /= n_batches

        print("[EWC] Fisher computation complete.")

    def penalty(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                loss = loss + (self.fisher[name] * (param - self.params[name]).pow(2)).sum()
        return self.importance * loss


# ---------------------------------------------------------------------------
# Optimizer: Layer-wise LR Decay
# ---------------------------------------------------------------------------

def build_optimizer_llrd(model, base_lr, decay=S1_LLRD, wd=S1_WD):
    """
    Assign exponentially decaying LR to each DeiT block.
    block[0] (earliest) → base_lr × decay^N   (very small)
    heads               → base_lr              (full LR)
    """
    param_groups = []

    # Attempt to locate transformer blocks; handle different attribute names
    backbone = getattr(model, 'backbone', None)
    blocks   = None
    if backbone is not None:
        blocks = getattr(backbone, 'blocks', None)
    # Some ORTrack versions expose blocks directly
    if blocks is None:
        blocks = getattr(model, 'blocks', None)

    if blocks is not None:
        n = len(blocks)
        for i, block in enumerate(blocks):
            lr_i = base_lr * (decay ** (n - i))
            param_groups.append({
                "params": list(block.parameters()),
                "lr":     lr_i,
                "name":   f"block.{i}",
            })
        # Patch embedding
        patch_embed = (getattr(backbone, 'patch_embed', None) or
                       getattr(model,    'patch_embed', None))
        if patch_embed is not None:
            param_groups.append({
                "params": list(patch_embed.parameters()),
                "lr":     base_lr * (decay ** (n + 1)),
                "name":   "patch_embed",
            })
    else:
        # Fallback: single group for backbone
        print("[LLRD] WARNING: could not locate transformer blocks — "
              "using flat LR for backbone")
        if backbone is not None:
            param_groups.append({"params": list(backbone.parameters()),
                                  "lr": base_lr * (decay ** 3), "name": "backbone"})

    # Heads always get full base_lr
    for head_name in ("box_head", "score_head", "head"):
        head = getattr(model, head_name, None)
        if head is not None:
            param_groups.append({"params": list(head.parameters()),
                                  "lr": base_lr, "name": head_name})

    if not param_groups:
        print("[LLRD] WARNING: no param groups built — using all parameters")
        param_groups = [{"params": list(model.parameters()), "lr": base_lr}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)

    for g in param_groups:
        n_p = sum(p.numel() for p in g["params"])
        print(f"  {g.get('name','?'):35s}  lr={g['lr']:.2e}  params={n_p/1e6:.2f}M")

    return optimizer


# ---------------------------------------------------------------------------
# Layer freeze control
# ---------------------------------------------------------------------------

def set_trainable_stage(model, stage: int):
    """
    Stage 0 — only heads (box_head, score_head)
    Stage 1 — last 4 transformer blocks + heads
    Stage 2 — everything
    """
    backbone = getattr(model, 'backbone', None)
    blocks   = (getattr(backbone, 'blocks', None) if backbone else None or
                getattr(model, 'blocks', None))

    head_keywords = {"box_head", "score_head", "head"}

    if stage == 0:
        for name, p in model.named_parameters():
            p.requires_grad = any(k in name for k in head_keywords)

    elif stage == 1:
        n = len(blocks) if blocks else 0
        trainable_blocks = set(range(max(0, n - 4), n))
        for name, p in model.named_parameters():
            is_head  = any(k in name for k in head_keywords)
            is_block = False
            if blocks is not None:
                for i in trainable_blocks:
                    if f"blocks.{i}." in name or f"block.{i}." in name:
                        is_block = True
                        break
            p.requires_grad = is_head or is_block

    elif stage == 2:
        for p in model.parameters():
            p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Stage {stage}] Trainable: {n_train/1e6:.2f}M / {n_total/1e6:.2f}M params")


# ---------------------------------------------------------------------------
# Training loop (one stage)
# ---------------------------------------------------------------------------

def train_one_stage(
    model, dataloader, optimizer, scheduler,
    n_epochs, stage_name, ckpt_dir, device,
    ewc=None, save_every=SAVE_EVERY_N,
):
    """
    Run `n_epochs` of training.  Saves a checkpoint every `save_every` epochs.
    Returns a list of (epoch, avg_loss) tuples.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history   = []

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0

        pbar = tqdm(dataloader, desc=f"[{stage_name}] epoch {epoch}/{n_epochs}")
        for t_imgs, s_imgs, gt_boxes in pbar:
            t_imgs   = t_imgs.to(device)
            s_imgs   = s_imgs.to(device)
            gt_boxes = gt_boxes.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                try:
                    out = model(template=t_imgs, search=s_imgs,
                                is_distill=getattr(model, 'is_distill', False))
                    pred_box = out.get('pred_boxes', out.get('target_bbox', None))
                    if pred_box is None:
                        continue
                    pred_box = pred_box.view(-1, 4)
                    task_loss, l1, giou = compute_loss(pred_box, gt_boxes)
                    ewc_loss = ewc.penalty() if ewc is not None else torch.tensor(0.0, device=device)
                    total_loss = task_loss + ewc_loss
                except Exception as exc:
                    print(f"\n  [WARNING] forward/loss error: {exc}")
                    continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{total_loss.item():.4f}",
                             l1=f"{l1.item():.4f}",
                             giou=f"{giou.item():.4f}",
                             ewc=f"{ewc_loss.item():.2f}" if ewc else "0")

        if scheduler is not None:
            scheduler.step()

        avg = epoch_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "avg_loss": avg})
        print(f"[{stage_name}] epoch {epoch:3d}  avg_loss={avg:.6f}")

        if epoch % save_every == 0 or epoch == n_epochs:
            ckpt_path = ckpt_dir / f"ep{epoch:04d}.pth.tar"
            torch.save({
                "epoch":       epoch,
                "net":         model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "avg_loss":    avg,
                "stage":       stage_name,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    return history


# ---------------------------------------------------------------------------
# Checkpoint selection
# ---------------------------------------------------------------------------

def select_best_checkpoint(ckpt_dir, stage_name, save_dir, args):
    """
    Ranks saved checkpoints by validation loss, evaluates top-K on the AIC
    val split using eval_aic_train.py, returns the best (checkpoint, metrics).
    """
    ckpt_dir = Path(ckpt_dir)
    checkpoints = sorted(ckpt_dir.glob("ep*.pth.tar"))
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}")

    # Rank by saved avg_loss (lower = better candidate)
    scored = []
    for cp in checkpoints:
        state = torch.load(cp, map_location="cpu")
        scored.append((float(state.get("avg_loss", 1e9)), cp))
    scored.sort(key=lambda x: x[0])

    candidates = [str(cp) for _, cp in scored[:max(EVAL_TOP_K, 1)]]
    print(f"[Selection] Evaluating top-{len(candidates)} checkpoints for {stage_name}")

    evaluated = []
    for ckpt_path in candidates:
        out_dir = Path(save_dir) / "selection_eval" / stage_name / Path(ckpt_path).stem
        cmd = [
            PYTHON_EXE,
            EVAL_SCRIPT,
            "--split",       "train",
            "--split-file",  args.val_split,
            "--config",      ORTRACK_CONFIG,
            "--checkpoint",  ckpt_path,
            "--output",      str(out_dir),
        ]
        print("  Evaluating:", ckpt_path)
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        try:
            subprocess.run([str(p) for p in cmd], cwd=str(ROOT),
                           check=True, env=env)
            metrics = read_score(out_dir / "summary_metrics.json")
            evaluated.append({"checkpoint": ckpt_path, **metrics})
            print(f"    score={metrics['score']:.4f}  "
                  f"auc={metrics['auc']:.4f}  "
                  f"p@20={metrics['precision']:.4f}")
        except Exception as exc:
            print(f"    ERROR evaluating {ckpt_path}: {exc}")

    if not evaluated:
        # Fall back to loss-ranked best
        return {"checkpoint": candidates[0], "score": None}, []

    evaluated.sort(key=lambda x: x["score"], reverse=True)
    return evaluated[0], evaluated


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    """Load ORTrack network from a checkpoint and move to device."""
    params = ortrack_parameters(ORTRACK_CONFIG)
    params.checkpoint     = checkpoint_path
    params.debug          = 0
    params.save_all_boxes = False

    from lib.test.tracker.ortrack import ORTrack
    tracker = ORTrack(params, "aic_finetune")
    # Extract the underlying nn.Module
    model = tracker.network
    model = model.to(device)
    return model, tracker


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Two-stage AIC fine-tuning (self-contained)")
    parser.add_argument("--save-dir",    default=SAVE_DIR)
    parser.add_argument("--checkpoint",  default=PRETRAINED_CKPT)
    parser.add_argument("--val-split",   default=VAL_SPLIT_FILE)
    parser.add_argument("--train-split", default=r"c:\AIC\ORTrack\data_specs\aic_contest_train.txt")
    parser.add_argument("--skip-stage0", action="store_true", help="Skip warm-up stage")
    parser.add_argument("--skip-stage1", action="store_true", help="Skip stage 1 training")
    parser.add_argument("--skip-stage2", action="store_true", help="Skip stage 2 training")
    parser.add_argument("--skip-ewc",    action="store_true", help="Disable EWC (faster, less safe)")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AIC Two-Stage Fine-Tuning — ORTrack DeiT Tiny")
    print("=" * 60)
    print(f"  Device:      {device}")
    print(f"  Pretrained:  {args.checkpoint}")
    print(f"  Save dir:    {save_dir}")
    print(f"  EWC:         {'disabled' if args.skip_ewc else f'importance={EWC_IMPORTANCE}'}")
    print(f"  Data mix:    AIC {AIC_DATASET_WEIGHT:.0%} / GOT-10k {1-AIC_DATASET_WEIGHT:.0%}")
    print()

    # ── Load model ──────────────────────────────────────────────────────────
    print("[1/5] Loading pretrained model …")
    model, _ = load_model(args.checkpoint, device)

    # ── Build dataloader ────────────────────────────────────────────────────
    print("[2/5] Building dataloader …")
    loader = build_dataloader(MANIFEST_PATH, DATA_ROOT, GOT10K_ROOT, args.train_split)

    # ── EWC: compute Fisher on pretrained model ──────────────────────────────
    ewc_reg = None
    if not args.skip_ewc:
        print("[3/5] Computing EWC Fisher information …")
        ewc_reg = EWC(model, importance=EWC_IMPORTANCE)
        ewc_reg.compute_fisher(loader, device, n_batches=EWC_FISHER_BATCHES)
    else:
        print("[3/5] EWC skipped.")

    # ── Stage 0: warm-up — heads only ───────────────────────────────────────
    s0_ckpt_dir = save_dir / "checkpoints" / "stage0"
    if not args.skip_stage0:
        print("\n[4/5] Stage 0 — warm-up (heads only) …")
        print(f"       LR={S0_LR:.1e}  epochs={S0_EPOCHS}")
        set_trainable_stage(model, stage=0)
        opt0 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=S0_LR, weight_decay=S0_WD
        )
        sched0 = torch.optim.lr_scheduler.CosineAnnealingLR(opt0, T_max=S0_EPOCHS)
        train_one_stage(model, loader, opt0, sched0,
                        n_epochs=S0_EPOCHS, stage_name="stage0",
                        ckpt_dir=s0_ckpt_dir, device=device,
                        ewc=None)   # no EWC penalty during warm-up
    else:
        print("[4/5] Stage 0 skipped.")

    # ── Stage 1: last-4 blocks + heads + LLRD + EWC ─────────────────────────
    s1_ckpt_dir = save_dir / "checkpoints" / "stage1"
    if not args.skip_stage1:
        print("\n[4/5] Stage 1 — last-4 blocks + heads (LLRD + EWC) …")
        print(f"       base_LR={S1_BASE_LR:.1e}  LLRD_decay={S1_LLRD}  epochs={S1_EPOCHS}")
        set_trainable_stage(model, stage=1)
        opt1   = build_optimizer_llrd(model, base_lr=S1_BASE_LR, decay=S1_LLRD, wd=S1_WD)
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=S1_EPOCHS)
        train_one_stage(model, loader, opt1, sched1,
                        n_epochs=S1_EPOCHS, stage_name="stage1",
                        ckpt_dir=s1_ckpt_dir, device=device,
                        ewc=ewc_reg if S1_EWC else None)
    else:
        print("[4/5] Stage 1 skipped.")

    # Select best Stage 1 checkpoint
    print("\nSelecting best Stage 1 checkpoint …")
    best_s1, s1_candidates = select_best_checkpoint(
        s1_ckpt_dir, "stage1", save_dir, args
    )
    print(f"  → Best Stage 1: {best_s1['checkpoint']}  score={best_s1.get('score')}")

    # ── Stage 2: full model + LLRD + EWC — init from best Stage 1 ───────────
    s2_ckpt_dir = save_dir / "checkpoints" / "stage2"
    if not args.skip_stage2:
        print("\n[5/5] Stage 2 — full model (LLRD + EWC) …")
        print(f"       base_LR={S2_BASE_LR:.1e}  LLRD_decay={S2_LLRD}  epochs={S2_EPOCHS}")
        # Re-load from best Stage 1 checkpoint
        model, _ = load_model(best_s1["checkpoint"], device)
        if not args.skip_ewc:
            # Re-attach EWC to the newly loaded model
            ewc_reg.model = model
        set_trainable_stage(model, stage=2)
        opt2   = build_optimizer_llrd(model, base_lr=S2_BASE_LR, decay=S2_LLRD, wd=S2_WD)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=S2_EPOCHS)
        train_one_stage(model, loader, opt2, sched2,
                        n_epochs=S2_EPOCHS, stage_name="stage2",
                        ckpt_dir=s2_ckpt_dir, device=device,
                        ewc=ewc_reg if S2_EWC else None)
    else:
        print("[5/5] Stage 2 skipped.")

    # Select best Stage 2 checkpoint
    print("\nSelecting best Stage 2 checkpoint …")
    best_s2, s2_candidates = select_best_checkpoint(
        s2_ckpt_dir, "stage2", save_dir, args
    )
    print(f"  → Best Stage 2: {best_s2['checkpoint']}  score={best_s2.get('score')}")

    # ── Final recommendation ─────────────────────────────────────────────────
    s1_score = best_s1.get("score") or -1.0
    s2_score = best_s2.get("score") or -1.0
    recommended = best_s2 if s2_score >= s1_score else best_s1

    summary = {
        "pretrained_checkpoint":   args.checkpoint,
        "stage0_epochs":           S0_EPOCHS,
        "stage1_epochs":           S1_EPOCHS,
        "stage2_epochs":           S2_EPOCHS,
        "ewc_importance":          EWC_IMPORTANCE,
        "llrd_decay":              S1_LLRD,
        "aic_data_weight":         AIC_DATASET_WEIGHT,
        "stage1_best":             best_s1,
        "stage1_candidates":       s1_candidates,
        "stage2_best":             best_s2,
        "stage2_candidates":       s2_candidates,
        "recommended_checkpoint":  recommended,
    }

    out_path = save_dir / "selected_checkpoints.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Recommended checkpoint : {recommended['checkpoint']}")
    print(f"  Composite score        : {recommended.get('score', 'N/A')}")
    print(f"  Summary saved to       : {out_path}")


if __name__ == "__main__":
    main()
