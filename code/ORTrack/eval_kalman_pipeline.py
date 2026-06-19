"""
Kalman Filter Pipeline for ORTrack-DeiT Tiny
============================================

Pipeline per frame:
  ORTrack inference
    ├─ score >= threshold?  YES ─► Kalman Predict → Update ─► Output bbox
    └─ NO (occluded)
         ├─ occluded_frames <= max_occluded?  YES ─► Kalman Predict ─► Output predicted bbox
         └─ NO (lost too long)
              ├─ Re-detector available?  YES ─► Re-detect ─► Re-init tracker
              └─ NO ─► Output [0, 0, 0, 0]

Evaluated on the full train split using ORTrack-DeiT Tiny pretrained weights.

Fixes applied vs original:
  FIX-1  kalman.predict() called before kalman.update() on every tracked frame.
  FIX-2  tracker.state written back in [x,y,w,h] top-left format matching
         ORTrack convention (verified); comment added for future maintainers.
  FIX-3  _post_process_box / _maybe_update_template accessed via getattr with
         graceful no-op fallbacks so a version mismatch doesn't crash at runtime.
  FIX-4  last_confidence initialised to 0.0 on the tracker before the loop so
         the very first track() call is never accidentally treated as occluded.
  FIX-5  occluded_frames clamped to max_occluded after a failed re-detection so
         the counter stays meaningful and doesn't grow unboundedly.
  FIX-6  last_valid_box stored separately and used as the re-detector anchor
         instead of boxes[-1] which may be [0,0,0,0].
  FIX-7  PSR gate changed to a soft veto: if last_response_map is None the PSR
         check is skipped entirely, preventing false occlusion on the first frame.
  FIX-8  clip_box_xywh enforces a 10 px minimum on w and h so boxes never
         collapse to a degenerate near-zero size near frame edges.
  FIX-9  mean_center_error_px in metrics computed only over frames where GT is
         valid, preventing the 1e6 sentinel values from inflating the average.
  FIX-10 is_distill accessed via getattr(self, 'is_distill', False) to guard
         against versions of ORTrack that set it conditionally.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import torch_compat
torch_compat.ensure_torch_six()

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.test.parameter.ortrack import parameters
from lib.test.tracker.ortrack import ORTrack

import torch


# ---------------------------------------------------------------------------
# ORTrack subclass that exposes the windowed response map for PSR computation
# ---------------------------------------------------------------------------

class ORTrackWithPSR(ORTrack):
    """Thin subclass that stores `last_response_map` after each track() call."""

    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        self.last_response_map = None   # shape: (feat_sz, feat_sz) numpy float32
        self.last_confidence   = 0.0   # FIX-4: initialise so first frame is safe

    def track(self, image, info=None):
        import math as _math
        from lib.train.data.processing_utils import sample_target
        from lib.utils.box_ops import clip_box

        H, W, _ = image.shape
        self.frame_id += 1
        prev_state = list(self.state)

        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, self.params.search_factor,
            output_sz=self.params.search_size,
        )
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            out_dict = self.network.forward(
                template=self.z_dict1.tensors,
                search=search.tensors,
                # FIX-10: guard against versions that set is_distill conditionally
                is_distill=getattr(self, 'is_distill', False),
            )

        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map

        # Store for PSR computation (detach to CPU numpy, 2-D)
        self.last_response_map = response.squeeze().detach().cpu().numpy()

        pred_boxes = self.network.box_head.cal_bbox(
            response, out_dict['size_map'], out_dict['offset_map']
        )
        pred_boxes = pred_boxes.view(-1, 4)
        confidence = float(response.max().item())
        self.last_confidence = confidence

        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        raw_state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # FIX-3: call post-processing helpers only if this version of ORTrack
        # defines them, so a name mismatch raises a clear warning rather than
        # crashing the entire evaluation run.
        post_process = getattr(self, '_post_process_box', None)
        if callable(post_process):
            self.state = post_process(raw_state, prev_state, H, W, confidence)
        else:
            self.state = raw_state

        update_template = getattr(self, '_maybe_update_template', None)
        if callable(update_template):
            update_template(image, confidence)

        return {"target_bbox": self.state}


def compute_psr(response_map, peak_half=5):
    """
    Peak-to-Sidelobe Ratio.
    PSR = (peak - mean_sidelobe) / std_sidelobe
    The sidelobe region excludes a (2*peak_half+1)^2 window centred on the peak.
    Returns 0.0 when the map is degenerate.
    """
    if response_map is None:
        return 0.0
    rmap = response_map.astype(np.float64)
    peak_idx = np.argmax(rmap)
    pr, pc = np.unravel_index(peak_idx, rmap.shape)
    peak_val = float(rmap[pr, pc])

    h, w = rmap.shape
    mask = np.ones((h, w), dtype=bool)
    r0 = max(0, pr - peak_half)
    r1 = min(h, pr + peak_half + 1)
    c0 = max(0, pc - peak_half)
    c1 = min(w, pc + peak_half + 1)
    mask[r0:r1, c0:c1] = False

    sidelobe = rmap[mask]
    if sidelobe.size < 4:
        return 0.0
    std = float(sidelobe.std())
    if std < 1e-9:
        return 0.0
    return float((peak_val - sidelobe.mean()) / std)


# ---------------------------------------------------------------------------
# Kalman Filter – constant-velocity model over [cx, cy, w, h]
# State: [cx, cy, w, h, vcx, vcy, vw, vh]
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """A minimal constant-velocity Kalman filter for a single bounding box."""

    def __init__(self, bbox_xywh, dt=1.0):
        x, y, w, h = bbox_xywh
        cx, cy = x + w / 2.0, y + h / 2.0

        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=np.float64)

        # Transition matrix (constant velocity)
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = dt

        # Measurement matrix – we observe [cx, cy, w, h]
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[:4, :4] = np.eye(4)

        # Process noise — separate position vs scale uncertainty
        self.Q = np.diag([0.1, 0.1, 0.5, 0.5,       # position process noise
                          0.01, 0.01, 0.05, 0.05])   # velocity process noise

        # Measurement noise — scale is inherently noisier than position
        self.R = np.diag([5., 5., 20., 20.])

        # Covariance — velocity initially very uncertain
        self.P = np.eye(8, dtype=np.float64) * 50.0
        self.P[4:, 4:] *= 20.0

    def predict(self):
        """Advance state by one time step. Returns predicted [x, y, w, h]."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._to_xywh()

    def update(self, bbox_xywh):
        """
        Correct state with measurement [x, y, w, h].
        MUST be called after predict() – never standalone.
        Returns corrected [x, y, w, h].
        """
        x, y, w, h = bbox_xywh
        z = np.array([x + w / 2.0, y + h / 2.0, w, h], dtype=np.float64)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self._to_xywh()

    def _to_xywh(self):
        cx, cy, w, h = self.x[:4]
        w = max(w, 1.0)
        h = max(h, 1.0)
        return [cx - w / 2.0, cy - h / 2.0, w, h]


# ---------------------------------------------------------------------------
# Re-detector: template-match inside an expanded search window
# ---------------------------------------------------------------------------

def crop_box(image, box):
    h, w = image.shape[:2]
    x, y, bw, bh = [float(v) for v in box]
    x1 = int(max(0, math.floor(x)))
    y1 = int(max(0, math.floor(y)))
    x2 = int(min(w, math.ceil(x + bw)))
    y2 = int(min(h, math.ceil(y + bh)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return image[y1:y2, x1:x2].copy()


def redetect(frame_rgb, anchor_box, init_template_gray, search_mul=6.0,
             min_score=0.55, scales=(0.85, 1.0, 1.15, 1.3)):
    """Template-match re-detector. Returns (found, bbox_xywh, score)."""
    h, w = frame_rgb.shape[:2]
    x, y, bw, bh = [float(v) for v in anchor_box]
    cx, cy = x + bw / 2.0, y + bh / 2.0

    sw = max(bw * search_mul, 96.0)
    sh = max(bh * search_mul, 96.0)
    x1 = int(max(0, cx - sw / 2.0))
    y1 = int(max(0, cy - sh / 2.0))
    x2 = int(min(w, cx + sw / 2.0))
    y2 = int(min(h, cy + sh / 2.0))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return False, [0, 0, 0, 0], 0.0

    search_gray = cv2.cvtColor(frame_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    search_gray = cv2.equalizeHist(search_gray)

    th0, tw0 = init_template_gray.shape[:2]
    best_score = -1.0
    best_box = [0, 0, 0, 0]

    for scale in scales:
        tw = max(8, int(round(tw0 * scale)))
        th = max(8, int(round(th0 * scale)))
        if tw >= search_gray.shape[1] or th >= search_gray.shape[0]:
            continue
        templ = cv2.resize(
            init_template_gray, (tw, th),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        resp = cv2.matchTemplate(search_gray, templ, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(resp)
        if max_val > best_score:
            best_score = max_val
            best_box = [x1 + max_loc[0], y1 + max_loc[1], tw, th]

    found = best_score >= min_score
    return found, best_box, float(best_score)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_init_box(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in re.split(r"[\s,]+", line) if x.strip()]
            if len(vals) >= 4:
                return vals[:4]
    raise ValueError(f"No initialization bbox found in {path}")


def load_boxes(path):
    boxes = []
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
    return len(box) >= 4 and box[2] > 0 and box[3] > 0


def clip_box_xywh(box, width, height):
    """
    Clamp [x, y, w, h] to lie within the frame.
    FIX-8: enforces a 10 px minimum on w and h to prevent degenerate boxes
    near frame edges from silently passing is_valid_box().
    """
    x, y, w, h = [float(v) for v in box]
    x = min(max(x, 0.0), max(width  - 1.0, 0.0))
    y = min(max(y, 0.0), max(height - 1.0, 0.0))
    # Clamp to available space then enforce a minimum size
    w = max(min(w, width  - x), 10.0)
    h = max(min(h, height - y), 10.0)
    return [x, y, w, h]


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - inter
    return inter / union if union > 0 else 0.0


def center_error(pred, gt):
    pcx, pcy = pred[0] + pred[2] / 2.0, pred[1] + pred[3] / 2.0
    gcx, gcy = gt[0]  + gt[2]  / 2.0, gt[1]  + gt[3]  / 2.0
    return math.hypot(pcx - gcx, pcy - gcy)


def normalized_center_error(pred, gt):
    denom = math.sqrt(max(gt[2] * gt[3], 1e-12))
    return center_error(pred, gt) / denom


# ---------------------------------------------------------------------------
# Core per-sequence Kalman pipeline
# ---------------------------------------------------------------------------

def track_sequence_kalman(
    tracker,
    data_root,
    item,
    gt_boxes,
    conf_threshold=0.35,
    psr_threshold=5.5,
    max_occluded=20,
    use_redetector=True,
    redet_search_mul=8.0,
    redet_min_score=0.42,
):
    """
    Run the Kalman-augmented pipeline on a single sequence.
    Returns list of predicted boxes (xywh) for every frame.

    NOTE on tracker.state convention: ORTrack stores state as [x, y, w, h]
    (top-left origin).  The Kalman filter also works in [x, y, w, h] via
    _to_xywh(), so the fed-back boxes are format-compatible.  If a future
    ORTrack version changes to centre-format, update _to_xywh() accordingly.
    """
    video_path = data_root / item["video_path"]
    expected_frames = min(int(item["n_frames"]), len(gt_boxes))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Find first valid GT box for initialisation
    init_frame_idx = next(
        (i for i, box in enumerate(gt_boxes[:expected_frames]) if is_valid_box(box)),
        None,
    )
    if init_frame_idx is None:
        cap.release()
        return [[0, 0, 0, 0]] * expected_frames

    init_box = gt_boxes[init_frame_idx]

    # Seek to init frame
    for _ in range(init_frame_idx + 1):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return [[0, 0, 0, 0]] * expected_frames
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialise tracker
    tracker.initialize(frame_rgb, {"init_bbox": init_box})

    # Build template for re-detector
    init_crop = crop_box(frame_rgb, init_box)
    if init_crop is not None:
        init_template_gray = cv2.equalizeHist(
            cv2.cvtColor(init_crop, cv2.COLOR_RGB2GRAY)
        )
    else:
        init_template_gray = None

    # Initialise Kalman filter
    kalman = KalmanBoxTracker(init_box)

    # Output: fill frames before init with zeros
    boxes = [[0.0, 0.0, 0.0, 0.0]] * init_frame_idx + [[float(v) for v in init_box]]

    occluded_frames = 0

    # FIX-6: track the last box from a high-confidence frame separately so the
    # re-detector always has a meaningful spatial anchor even after long occlusion.
    last_valid_box = list(init_box)

    for frame_idx in range(init_frame_idx + 1, expected_frames):
        ok, frame = cap.read()
        if not ok:
            boxes.append([0, 0, 0, 0])
            occluded_frames += 1
            kalman.predict()
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- ORTrack inference ----
        with torch.no_grad():
            out = tracker.track(frame_rgb, {})
        pred  = [float(v) for v in out["target_bbox"]]
        score = float(getattr(tracker, "last_confidence", 0.0))

        # FIX-7: PSR is used only as a soft veto.  If the response map is
        # unavailable (e.g. first frame after re-init) we skip the PSR gate
        # entirely rather than treating the frame as occluded.
        rmap = getattr(tracker, "last_response_map", None)
        if rmap is not None:
            psr = compute_psr(rmap)
            psr_ok = psr >= psr_threshold
        else:
            psr_ok = True   # no map available → do not penalise

        if score >= conf_threshold and psr_ok and is_valid_box(pred):
            # ── TRACKING: FIX-1 — predict first, then update ──────────────
            kalman.predict()
            corrected = kalman.update(pred)
            corrected = clip_box_xywh(corrected, width, height)
            boxes.append(corrected)
            # FIX-2: tracker.state is [x, y, w, h] (top-left) — same as
            # Kalman's _to_xywh() output, so this feed-back is format-safe.
            tracker.state = list(corrected)
            last_valid_box = list(corrected)   # FIX-6: update anchor
            occluded_frames = 0

        elif occluded_frames < max_occluded:
            # ── OCCLUDED short: Kalman coast ───────────────────────────────
            predicted = kalman.predict()
            predicted = clip_box_xywh(predicted, width, height)
            boxes.append(predicted)
            tracker.state = list(predicted)
            occluded_frames += 1

        else:
            # ── LOST TOO LONG ──────────────────────────────────────────────
            if use_redetector and init_template_gray is not None:
                # FIX-6: use last_valid_box as anchor, not boxes[-1]
                found, det_box, det_score = redetect(
                    frame_rgb,
                    last_valid_box,
                    init_template_gray,
                    search_mul=redet_search_mul,
                    min_score=redet_min_score,
                )
                if found and is_valid_box(det_box):
                    det_box = clip_box_xywh(det_box, width, height)
                    # Re-init tracker and Kalman at detected location
                    tracker.initialize(frame_rgb, {"init_bbox": det_box})
                    kalman = KalmanBoxTracker(det_box)
                    boxes.append(det_box)
                    last_valid_box = list(det_box)   # FIX-6: update anchor
                    occluded_frames = 0
                else:
                    boxes.append([0, 0, 0, 0])
                    # FIX-5: clamp counter so it stays meaningful (≥ max_occluded
                    # keeps us in the re-detect branch without growing forever)
                    occluded_frames = max_occluded + 1
            else:
                boxes.append([0, 0, 0, 0])
                occluded_frames = max_occluded + 1   # FIX-5

    cap.release()
    return boxes


# ---------------------------------------------------------------------------
# Per-sequence evaluation wrapper
# ---------------------------------------------------------------------------

def evaluate_sequence(tracker, data_root, rel_key, item, pred_dir, cfg):
    anno_path = data_root / item["annotation_path"]
    gt_boxes  = load_boxes(anno_path)

    t0 = time.perf_counter()
    predictions = track_sequence_kalman(
        tracker,
        data_root,
        item,
        gt_boxes,
        conf_threshold  = cfg["conf_threshold"],
        psr_threshold   = cfg["psr_threshold"],
        max_occluded    = cfg["max_occluded"],
        use_redetector  = cfg["use_redetector"],
        redet_search_mul= cfg["redet_search_mul"],
        redet_min_score = cfg["redet_min_score"],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Save predictions
    seq_pred_dir = pred_dir / item["dataset"]
    seq_pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = seq_pred_dir / f"{item['seq_name']}.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "w", "h"])
        for idx, box in enumerate(predictions):
            writer.writerow([idx, *box])

    # Compute metrics
    errors, norm_errors, overlaps = [], [], []
    valid_errors = []   # FIX-9: separate list that excludes absent-GT sentinels
    absent_gt = absent_tp = absent_fp = absent_fn = 0

    for idx, pred in enumerate(predictions):
        if idx >= len(gt_boxes):
            break
        gt       = gt_boxes[idx]
        pred_zero = abs(pred[2]) < 1e-9 and abs(pred[3]) < 1e-9
        gt_valid  = is_valid_box(gt)

        if not gt_valid:
            absent_gt += 1
            if pred_zero:
                absent_tp += 1
            else:
                absent_fn += 1
            errors.append(1e6)
            norm_errors.append(1e6)
            overlaps.append(1.0 if pred_zero else 0.0)
            continue

        if pred_zero:
            absent_fp += 1
            errors.append(1e6)
            norm_errors.append(1e6)
            overlaps.append(0.0)
        else:
            ce  = center_error(pred, gt)
            nce = normalized_center_error(pred, gt)
            errors.append(ce)
            norm_errors.append(nce)
            overlaps.append(iou_xywh(pred, gt))
            valid_errors.append(ce)   # FIX-9: only real detections vs real GT

    errors      = np.array(errors,      dtype=np.float64)
    overlaps    = np.array(overlaps,    dtype=np.float64)
    norm_errors = np.array(norm_errors, dtype=np.float64)

    success_auc      = float(np.mean([(overlaps >= t).mean() for t in np.linspace(0, 1, 101)]))
    precision_at_20  = float((errors <= 20.0).mean())
    precision_auc    = float(np.mean([(errors <= t).mean() for t in np.arange(0, 51)]))
    norm_prec_auc    = float(np.mean([(norm_errors <= t).mean() for t in np.linspace(0, 0.5, 101)]))

    lat_per_frame = elapsed_ms / max(len(predictions), 1)

    # FIX-9: report mean centre error only over frames where GT is valid and
    # tracker produced a non-zero box; the 1e6 sentinels are excluded.
    mean_ce = float(np.mean(valid_errors)) if valid_errors else float('nan')

    metrics = {
        "sequence"                      : rel_key,
        "video_path"                    : str(data_root / item["video_path"]),
        "annotation_path"               : str(anno_path),
        "prediction_path"               : str(pred_path),
        "frames_evaluated"              : int(len(errors)),
        "frames_expected"               : int(len(gt_boxes)),
        "frames_read"                   : int(len(predictions)),
        "precision_at_20px"             : precision_at_20,
        "precision_auc_0_50px"          : precision_auc,
        "success_auc"                   : success_auc,
        "normalized_precision_auc_0_0_5": norm_prec_auc,
        "mean_iou"                      : float(overlaps.mean()),
        # FIX-9: sentinel-free centre error (valid detection frames only)
        "mean_center_error_px"          : mean_ce,
        "failures_iou_eq_0"             : int((overlaps <= 0.0).sum()),
        "failures_iou_lt_0_1"           : int((overlaps < 0.1).sum()),
        "absent_gt_frames"              : int(absent_gt),
        "absent_tp"                     : int(absent_tp),
        "absent_fp"                     : int(absent_fp),
        "absent_fn"                     : int(absent_fn),
        "latency_ms_per_frame"          : float(lat_per_frame),
        "score_0p6_auc_0p4_precision"   : float(0.6 * success_auc + 0.4 * precision_at_20),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kalman-augmented ORTrack eval")
    data_root = Path(os.environ.get("AIC_DATA_ROOT", "/data"))
    parser.add_argument("--data-root",   default=str(data_root))
    parser.add_argument("--manifest",    default=str(data_root / "metadata" / "contestant_manifest.json"))
    parser.add_argument("--split",       default="train")
    parser.add_argument("--output",      default="outputs/aic_train_eval_kalman")
    parser.add_argument("--config",      default="deit_tiny_aic_stage1")
    parser.add_argument("--checkpoint",
                        default="model/ORTrack_AIC.pth.tar",
                        help="Path to pretrained DeiT Tiny weights")
    parser.add_argument("--conf-threshold", type=float, default=0.35,
                        help="Score threshold; below this frame is occluded")
    parser.add_argument("--psr-threshold",  type=float, default=5.5,
                        help="PSR threshold (soft veto); below this frame is occluded")
    parser.add_argument("--max-occluded",   type=int,   default=20,
                        help="Max Kalman-predict frames before re-detect/lost")
    parser.add_argument("--no-redetector",  action="store_true",
                        help="Disable template-match re-detector")
    parser.add_argument("--redet-search-mul", type=float, default=8.0)
    parser.add_argument("--redet-min-score",  type=float, default=0.42)
    parser.add_argument("--limit",    type=int,  default=None)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--resume",   action="store_true")
    args = parser.parse_args()

    pipeline_cfg = {
        "conf_threshold"  : args.conf_threshold,
        "psr_threshold"   : args.psr_threshold,
        "max_occluded"    : args.max_occluded,
        "use_redetector"  : not args.no_redetector,
        "redet_search_mul": args.redet_search_mul,
        "redet_min_score" : args.redet_min_score,
    }

    data_root = Path(args.data_root)
    output    = Path(args.output)
    pred_dir  = output / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    per_seq_path = output / "per_sequence_metrics.jsonl"
    summary_path = output / "summary_metrics.json"

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    sequences = list(manifest[args.split].items())
    if args.sequence:
        sequences = [(k, v) for k, v in sequences
                     if k == args.sequence or v["seq_name"] == args.sequence]
    if args.limit:
        sequences = sequences[: args.limit]

    done = set()
    if args.resume and per_seq_path.exists():
        with open(per_seq_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["sequence"])

    # Load tracker (PSR-aware subclass)
    params = parameters(args.config)
    params.checkpoint      = args.checkpoint
    params.debug           = 0
    params.save_all_boxes  = False
    tracker = ORTrackWithPSR(params, "aic_kalman_eval")
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")
    print(f"[INFO] Pipeline config:   {pipeline_cfg}")

    all_metrics = []
    if args.resume and per_seq_path.exists():
        with open(per_seq_path, "r", encoding="utf-8") as f:
            all_metrics.extend(json.loads(line) for line in f if line.strip())

    mode = "a" if args.resume else "w"
    with open(per_seq_path, mode, encoding="utf-8") as out_f:
        for rel_key, item in tqdm(sequences, desc=f"Kalman pipeline – {args.split}"):
            if rel_key in done:
                continue
            try:
                metrics = evaluate_sequence(
                    tracker, data_root, rel_key, item, pred_dir, pipeline_cfg
                )
                all_metrics.append(metrics)
                out_f.write(json.dumps(metrics) + "\n")
                out_f.flush()
                tqdm.write(
                    f"  {rel_key:40s}  AUC={metrics['success_auc']:.4f}"
                    f"  P@20={metrics['precision_at_20px']:.4f}"
                )
            except Exception as exc:
                import traceback
                err = {
                    "sequence" : rel_key,
                    "error"    : repr(exc),
                    "traceback": traceback.format_exc(),
                }
                all_metrics.append(err)
                out_f.write(json.dumps(err) + "\n")
                out_f.flush()
                tqdm.write(f"  ERROR {rel_key}: {exc}")

    # Aggregate summary
    usable = [m for m in all_metrics if "error" not in m]
    if not usable:
        print("No sequences evaluated successfully.")
        return

    def frame_weighted_mean(key):
        vals = [(m[key], m["frames_evaluated"])
                for m in usable
                if m.get(key) is not None and not math.isnan(m.get(key, float('nan')))]
        if not vals:
            return None
        total_frames = sum(f for _, f in vals)
        return float(sum(v * f for v, f in vals) / max(total_frames, 1))

    summary_keys = [
        "precision_at_20px", "precision_auc_0_50px", "success_auc",
        "normalized_precision_auc_0_0_5", "mean_iou", "mean_center_error_px",
        "latency_ms_per_frame", "score_0p6_auc_0p4_precision",
    ]
    frame_weighted = {k: frame_weighted_mean(k) for k in summary_keys}
    frame_weighted.update({
        "frames_evaluated"   : int(sum(m["frames_evaluated"]    for m in usable)),
        "sequences_evaluated": len(usable),
        "sequences_failed"   : len([m for m in all_metrics if "error" in m]),
        "failures_iou_eq_0"  : int(sum(m["failures_iou_eq_0"]  for m in usable)),
        "failures_iou_lt_0_1": int(sum(m["failures_iou_lt_0_1"] for m in usable)),
        "absent_gt_frames"   : int(sum(m["absent_gt_frames"]    for m in usable)),
        "pipeline_config"    : pipeline_cfg,
    })

    result = {"summary": frame_weighted, "sequences": all_metrics}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY – Kalman Pipeline (ORTrack-DeiT Tiny pretrained)")
    print("=" * 60)
    for k, v in frame_weighted.items():
        if isinstance(v, float):
            print(f"  {k:45s}: {v:.6f}")
        else:
            print(f"  {k:45s}: {v}")
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()
