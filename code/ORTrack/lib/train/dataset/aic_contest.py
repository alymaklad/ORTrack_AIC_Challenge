import json
import math
import os
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader


def _read_boxes(path):
    boxes = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in re.split(r"[\s,]+", line) if x.strip()]
            if len(vals) >= 4:
                boxes.append(vals[:4])
    return torch.tensor(boxes, dtype=torch.float32)


class AICContest(BaseVideoDataset):
    """AIC contest tracking data backed by contestant_manifest.json."""

    def __init__(
        self,
        manifest_path,
        data_root,
        split_file,
        frame_cache_dir="",
        image_loader=jpeg4py_loader,
        curriculum_stage=1,
        temporal_strategy="near",
        velocity_aware=False,
        velocity_aware_prob=0.0,
        hard_sample_prob=0.0,
        min_area_ratio=0.0,
        max_motion_obj=0.0,
        max_scale_change=0.0,
        min_pair_area_ratio=0.0,
        size_bucket_weights=None,
        min_valid_frames=20,
        is_validation=False,
    ):
        super().__init__("AICContest", data_root, image_loader)
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.frame_cache_dir = Path(frame_cache_dir) if frame_cache_dir else None
        self.curriculum_stage = int(curriculum_stage)
        self.temporal_strategy = temporal_strategy
        self.velocity_aware = bool(velocity_aware)
        self.velocity_aware_prob = float(velocity_aware_prob)
        self.hard_sample_prob = float(hard_sample_prob)
        self.min_area_ratio = float(min_area_ratio)
        self.max_motion_obj = float(max_motion_obj)
        self.max_scale_change = float(max_scale_change)
        self.min_pair_area_ratio = float(min_pair_area_ratio)
        self.size_bucket_weights = list(size_bucket_weights or [])
        self.min_valid_frames = int(min_valid_frames)
        self.is_validation = bool(is_validation)
        self._video_caps = {}

        with self.manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)["train"]
        with open(split_file, "r", encoding="utf-8") as f:
            split_keys = [line.strip() for line in f if line.strip()]

        self.sequence_list = []
        self.entries = []
        self.annos = []
        self.visible = []
        self.valid = []
        self.area_ratio = []
        self.motion_obj = []

        for key in split_keys:
            if key not in manifest:
                continue
            entry = manifest[key]
            anno_path = self.data_root / entry["annotation_path"]
            video_path = self.data_root / entry["video_path"]
            if not anno_path.exists() or not video_path.exists():
                continue
            bbox = _read_boxes(anno_path)
            if bbox.numel() == 0:
                continue
            width, height = self._probe_video_size(video_path)
            valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
            if width > 0 and height > 0 and self.min_area_ratio > 0:
                area_ratio = (bbox[:, 2] * bbox[:, 3]) / float(width * height)
                valid = valid & (area_ratio >= self.min_area_ratio)
            if valid.type(torch.int64).sum().item() < self.min_valid_frames:
                continue

            self.sequence_list.append(key)
            self.entries.append({**entry, "key": key, "width": width, "height": height})
            self.annos.append(bbox)
            self.valid.append(valid)
            self.visible.append(valid.clone())
            self.area_ratio.append(self._compute_area_ratio(bbox, width, height))
            self.motion_obj.append(self._compute_motion_obj(bbox, valid))

    def get_name(self):
        return "aic_contest"

    def _probe_video_size(self, path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return 0, 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return width, height

    def _compute_area_ratio(self, bbox, width, height):
        if width <= 0 or height <= 0:
            return torch.zeros((bbox.shape[0],), dtype=torch.float32)
        return (bbox[:, 2] * bbox[:, 3]) / float(width * height)

    def _compute_motion_obj(self, bbox, valid):
        motion = torch.zeros((bbox.shape[0],), dtype=torch.float32)
        for i in range(1, bbox.shape[0]):
            if not (valid[i] and valid[i - 1]):
                continue
            prev = bbox[i - 1]
            cur = bbox[i]
            c0 = prev[:2] + 0.5 * prev[2:4]
            c1 = cur[:2] + 0.5 * cur[2:4]
            obj_diag = torch.sqrt((prev[2] ** 2) + (prev[3] ** 2)).clamp_min(1e-6)
            motion[i] = torch.norm(c1 - c0) / obj_diag
        return motion

    def _gap_choices(self, max_gap):
        if self.temporal_strategy == "curriculum":
            if self.curriculum_stage <= 1:
                return [(1, min(5, max_gap), 1.0)]
            if self.curriculum_stage == 2:
                return [(1, min(5, max_gap), 0.60), (6, min(30, max_gap), 0.40)]
            return [(1, min(5, max_gap), 0.45), (6, min(30, max_gap), 0.40), (31, max_gap, 0.15)]
        if self.temporal_strategy == "mixed":
            return [(1, min(5, max_gap), 0.45), (6, min(30, max_gap), 0.40), (31, max_gap, 0.15)]
        return [(1, min(5, max_gap), 1.0)]

    def _weighted_choice(self, items):
        items = [item for item in items if item[1] >= item[0]]
        if not items:
            return None
        weights = [max(float(item[2]), 0.0) for item in items]
        if sum(weights) <= 0:
            weights = None
        return random.choices(items, weights=weights, k=1)[0]

    def _sample_search_id(self, seq_id, valid_ids, visible):
        if self.is_validation:
            return random.choice(valid_ids)

        if self.velocity_aware and random.random() < self.velocity_aware_prob:
            motion = self.motion_obj[seq_id]
            visible_motion = motion[visible]
            if visible_motion.numel() > 0:
                threshold = torch.quantile(visible_motion, 0.85)
                hard = torch.nonzero((motion >= threshold) & visible, as_tuple=False).view(-1).tolist()
                if hard:
                    return random.choice(hard)

        if self.size_bucket_weights and random.random() < self.hard_sample_prob:
            area = self.area_ratio[seq_id]
            buckets = [
                [i for i in valid_ids if area[i] < 0.001],
                [i for i in valid_ids if 0.001 <= area[i] < 0.01],
                [i for i in valid_ids if 0.01 <= area[i] < 0.05],
                [i for i in valid_ids if area[i] >= 0.05],
            ]
            weighted_buckets = []
            for bucket, weight in zip(buckets, self.size_bucket_weights):
                if bucket and weight > 0:
                    weighted_buckets.append((bucket, float(weight)))
            if weighted_buckets:
                bucket = random.choices(
                    [item[0] for item in weighted_buckets],
                    weights=[item[1] for item in weighted_buckets],
                    k=1,
                )[0]
                return random.choice(bucket)

        return random.choice(valid_ids)

    def _pair_is_usable(self, seq_id, template_id, search_id):
        if template_id is None or search_id is None or template_id >= search_id:
            return False
        area = self.area_ratio[seq_id]
        if self.min_pair_area_ratio > 0:
            if area[template_id] < self.min_pair_area_ratio or area[search_id] < self.min_pair_area_ratio:
                return False
        if self.max_scale_change > 0:
            a0 = max(float(area[template_id]), 1e-12)
            a1 = max(float(area[search_id]), 1e-12)
            if abs(math.log(a1 / a0)) > self.max_scale_change:
                return False
        if self.max_motion_obj > 0 and search_id > template_id + 1:
            motion = self.motion_obj[seq_id][template_id + 1 : search_id + 1]
            if motion.numel() > 0 and float(torch.max(motion)) > self.max_motion_obj:
                return False
        return True

    def get_sequence_info(self, seq_id):
        bbox = self.annos[seq_id]
        valid = self.valid[seq_id]
        visible = self.visible[seq_id]
        return {
            "bbox": bbox,
            "valid": valid,
            "visible": visible,
            "visible_ratio": visible.float(),
            "area_ratio": self.area_ratio[seq_id],
            "motion_obj": self.motion_obj[seq_id],
        }

    def sample_pair_ids(self, seq_id, visible, max_gap):
        valid_ids = torch.nonzero(visible, as_tuple=False).view(-1).tolist()
        if len(valid_ids) < 2:
            return None, None
        valid_set = set(valid_ids)
        gap_choices = self._gap_choices(max_gap)

        for _ in range(200):
            search_id = self._sample_search_id(seq_id, valid_ids, visible)
            gap_choice = self._weighted_choice(gap_choices)
            if gap_choice is None:
                return None, None
            g0, g1, _ = gap_choice
            if g1 < g0:
                g1 = g0
            gap = random.randint(g0, min(g1, max_gap))
            lo = max(0, search_id - gap)
            candidates = [i for i in range(lo, search_id) if i in valid_set]
            if candidates:
                template_id = random.choice(candidates)
                if self._pair_is_usable(seq_id, template_id, search_id):
                    return [template_id], [search_id]
        return None, None

    def _cache_frame_path(self, seq_key, frame_id):
        if self.frame_cache_dir is None:
            return None
        return self.frame_cache_dir / seq_key / f"{frame_id + 1:08d}.jpg"

    def _get_frame_path(self, seq_id, frame_id):
        key = self.sequence_list[seq_id].replace("/", "__")
        return self._cache_frame_path(key, frame_id)

    def _get_frame_from_video(self, seq_id, frame_id):
        entry = self.entries[seq_id]
        video_path = str(self.data_root / entry["video_path"])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_id} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _get_frame(self, seq_id, frame_id):
        frame_path = self._get_frame_path(seq_id, frame_id)
        if frame_path is not None and frame_path.exists():
            return self.image_loader(str(frame_path))
        return self._get_frame_from_video(seq_id, frame_id)

    def get_frames(self, seq_id, frame_ids, anno=None):
        frame_list = [self._get_frame(seq_id, f_id) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if torch.is_tensor(value):
                anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        meta = {
            "object_class_name": self.entries[seq_id]["seq_name"],
            "dataset": self.entries[seq_id]["dataset"],
            "sequence": self.sequence_list[seq_id],
        }
        return frame_list, anno_frames, meta
