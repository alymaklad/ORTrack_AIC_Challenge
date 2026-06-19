import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.train.data.processing_utils import sample_target


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


def fmt_number(value):
    if value is None or not math.isfinite(value):
        return "0"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def is_zero_box(box):
    return len(box) >= 4 and abs(box[0]) < 1e-9 and abs(box[1]) < 1e-9 and abs(box[2]) < 1e-9 and abs(box[3]) < 1e-9


def is_valid_box(box):
    return len(box) >= 4 and box[2] > 0 and box[3] > 0


def clip_box_xywh(box, width, height):
    x, y, w, h = [float(v) for v in box]
    x = min(max(x, 0.0), max(width - 1.0, 0.0))
    y = min(max(y, 0.0), max(height - 1.0, 0.0))
    w = min(max(w, 0.0), max(width - x, 0.0))
    h = min(max(h, 0.0), max(height - y, 0.0))
    return [x, y, w, h]


def center_error(pred, gt):
    pcx, pcy = pred[0] + pred[2] / 2.0, pred[1] + pred[3] / 2.0
    gcx, gcy = gt[0] + gt[2] / 2.0, gt[1] + gt[3] / 2.0
    return math.hypot(pcx - gcx, pcy - gcy)


def normalized_center_error(pred, gt):
    denom = math.sqrt(max(gt[2] * gt[3], 1e-12))
    return center_error(pred, gt) / denom


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


def box_features(box, prev_box, width, height, confidence):
    x, y, w, h = [float(v) for v in box]
    px, py, pw, ph = [float(v) for v in prev_box]
    frame_area = max(float(width * height), 1.0)
    area_ratio = max(w, 0.0) * max(h, 0.0) / frame_area
    touch_boundary = x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2
    out_of_bounds = x < -1 or y < -1 or x + w > width + 1 or y + h > height + 1
    prev_diag = math.sqrt(max(pw * pw + ph * ph, 1e-6))
    cx, cy = x + w / 2.0, y + h / 2.0
    pcx, pcy = px + pw / 2.0, py + ph / 2.0
    jump_obj = math.hypot(cx - pcx, cy - pcy) / prev_diag
    prev_area = max(pw * ph, 1e-9)
    area = max(w * h, 1e-9)
    scale_change = max(area / prev_area, prev_area / area)
    return {
        "confidence": float(confidence),
        "area_ratio": float(area_ratio),
        "touch_boundary": bool(touch_boundary),
        "out_of_bounds": bool(out_of_bounds),
        "jump_obj": float(jump_obj),
        "scale_change": float(scale_change),
    }


def policy_absent(feat, policy):
    conf = feat["confidence"]
    if conf < policy["conf_low"]:
        return True, "low_conf"
    jump_limit = policy.get("max_center_jump_obj")
    if jump_limit is not None and feat["jump_obj"] > float(jump_limit):
        return True, "jump_too_large"
    if feat["touch_boundary"] and conf < policy["boundary_conf"]:
        return True, "boundary_low_conf"
    if feat["area_ratio"] >= policy["max_area_ratio"] and conf < policy["huge_conf"]:
        return True, "huge_low_conf"
    if feat["scale_change"] >= policy["max_scale_change"] and conf < policy["scale_conf"]:
        return True, "scale_low_conf"
    if feat["out_of_bounds"]:
        return True, "out_of_bounds"
    return False, ""


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


def normalized_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.equalizeHist(gray)


@dataclass
class RecoveryResult:
    found: bool
    bbox: list
    score: float
    template_name: str
    search_region: list


class OnlineRecovery:
    def __init__(self, frame_rgb, init_box, cfg):
        self.cfg = cfg or {}
        self.init_box = [float(v) for v in init_box]
        self.initial_template = crop_box(frame_rgb, init_box)
        self.recent_template = self.initial_template.copy() if self.initial_template is not None else None
        self.last_good_box = [float(v) for v in init_box]
        self.last_good_conf = 1.0

    def maybe_refresh_recent_template(self, frame_rgb, box, confidence):
        if confidence < float(self.cfg.get("template_conf", 0.55)):
            return
        patch = crop_box(frame_rgb, box)
        if patch is None:
            return
        min_side = int(self.cfg.get("min_template_side", 12))
        if min(patch.shape[:2]) < min_side:
            return
        self.recent_template = patch
        self.last_good_box = [float(v) for v in box]
        self.last_good_conf = float(confidence)

    def _match_candidate(self, frame_rgb, fallback_box, absent_run, cfg_overrides=None):
        h, w = frame_rgb.shape[:2]
        anchor = self.last_good_box if is_valid_box(self.last_good_box) else fallback_box
        if not is_valid_box(anchor):
            anchor = self.init_box
        x, y, bw, bh = [float(v) for v in anchor]
        cx = x + bw / 2.0
        cy = y + bh / 2.0

        cfg = dict(self.cfg)
        if cfg_overrides:
            cfg.update(cfg_overrides)

        radius_mul = float(cfg.get("search_mul", 5.0))
        expand_step = float(cfg.get("expand_per_absent", 1.0))
        max_mul = float(cfg.get("max_search_mul", 12.0))
        search_mul = min(max_mul, radius_mul + max(0, absent_run - 1) * expand_step)
        sw = max(bw * search_mul, float(cfg.get("min_search_px", 96)))
        sh = max(bh * search_mul, float(cfg.get("min_search_px", 96)))

        x1 = int(max(0, math.floor(cx - sw / 2.0)))
        y1 = int(max(0, math.floor(cy - sh / 2.0)))
        x2 = int(min(w, math.ceil(cx + sw / 2.0)))
        y2 = int(min(h, math.ceil(cy + sh / 2.0)))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return RecoveryResult(False, [0, 0, 0, 0], 0.0, "", [x1, y1, x2 - x1, y2 - y1])

        search = frame_rgb[y1:y2, x1:x2]
        search_gray = normalized_gray(search)

        best = RecoveryResult(False, [0, 0, 0, 0], -1.0, "", [x1, y1, x2 - x1, y2 - y1])
        template_pool = []
        if self.recent_template is not None:
            template_pool.append(("recent", self.recent_template))
        if self.initial_template is not None:
            template_pool.append(("initial", self.initial_template))

        scales = cfg.get("scales", [0.85, 1.0, 1.15, 1.3])
        for template_name, template in template_pool:
            gray_template = normalized_gray(template)
            th0, tw0 = gray_template.shape[:2]
            for scale in scales:
                tw = max(8, int(round(tw0 * scale)))
                th = max(8, int(round(th0 * scale)))
                if tw >= search_gray.shape[1] or th >= search_gray.shape[0]:
                    continue
                templ = cv2.resize(gray_template, (tw, th), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
                response = cv2.matchTemplate(search_gray, templ, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(response)
                cand_box = [x1 + max_loc[0], y1 + max_loc[1], tw, th]
                if max_val > best.score:
                    best = RecoveryResult(True, cand_box, float(max_val), template_name, [x1, y1, x2 - x1, y2 - y1])

        return best

    def _validate_candidate(self, candidate, anchor, frame_shape, absent_run, cfg_overrides=None):
        if not candidate.found:
            return candidate

        h, w = frame_shape[:2]
        cfg = dict(self.cfg)
        if cfg_overrides:
            cfg.update(cfg_overrides)

        min_score = float(cfg.get("min_score", 0.58))
        if absent_run >= int(cfg.get("long_absence_after", 6)):
            min_score = float(cfg.get("min_score_long", min_score + 0.04))
        if candidate.score < min_score:
            return RecoveryResult(False, [0, 0, 0, 0], candidate.score, candidate.template_name, candidate.search_region)

        cand = clip_box_xywh(candidate.bbox, w, h)
        area_ratio = (cand[2] * cand[3]) / max(float(w * h), 1.0)
        if area_ratio > float(cfg.get("max_area_ratio", 0.12)):
            return RecoveryResult(False, [0, 0, 0, 0], candidate.score, candidate.template_name, candidate.search_region)

        prev_area = max(anchor[2] * anchor[3], 1e-9)
        cand_area = max(cand[2] * cand[3], 1e-9)
        scale_change = max(cand_area / prev_area, prev_area / cand_area)
        if scale_change > float(cfg.get("max_scale_change", 3.2)):
            return RecoveryResult(False, [0, 0, 0, 0], candidate.score, candidate.template_name, candidate.search_region)

        return RecoveryResult(True, cand, candidate.score, candidate.template_name, candidate.search_region)

    def recover(self, frame_rgb, fallback_box, absent_run):
        candidate = self._match_candidate(frame_rgb, fallback_box, absent_run)
        anchor = self.last_good_box if is_valid_box(self.last_good_box) else fallback_box
        if not is_valid_box(anchor):
            anchor = self.init_box
        return self._validate_candidate(candidate, anchor, frame_rgb.shape, absent_run)

    def pre_recover(self, frame_rgb, fallback_box, absent_run):
        cfg_overrides = {
            "search_mul": self.cfg.get("pre_search_mul", self.cfg.get("search_mul", 5.0)),
            "expand_per_absent": self.cfg.get("pre_expand_per_absent", self.cfg.get("expand_per_absent", 1.0)),
            "max_search_mul": self.cfg.get("pre_max_search_mul", self.cfg.get("max_search_mul", 12.0)),
            "min_search_px": self.cfg.get("pre_min_search_px", self.cfg.get("min_search_px", 96)),
            "min_score": self.cfg.get("pre_min_score", max(0.62, self.cfg.get("min_score", 0.58))),
            "min_score_long": self.cfg.get("pre_min_score_long", self.cfg.get("min_score_long", 0.64)),
            "long_absence_after": self.cfg.get("pre_long_absence_after", self.cfg.get("long_absence_after", 6)),
            "max_area_ratio": self.cfg.get("pre_max_area_ratio", self.cfg.get("max_area_ratio", 0.12)),
            "max_scale_change": self.cfg.get("pre_max_scale_change", min(2.8, self.cfg.get("max_scale_change", 3.2))),
            "scales": self.cfg.get("pre_scales", self.cfg.get("scales", [0.85, 1.0, 1.15, 1.3])),
        }
        candidate = self._match_candidate(frame_rgb, fallback_box, absent_run, cfg_overrides=cfg_overrides)
        anchor = self.last_good_box if is_valid_box(self.last_good_box) else fallback_box
        if not is_valid_box(anchor):
            anchor = self.init_box
        return self._validate_candidate(candidate, anchor, frame_rgb.shape, absent_run, cfg_overrides=cfg_overrides)


class StrongProposalVerifier:
    def __init__(self, tracker, frame_rgb, init_box, cfg):
        self.tracker = tracker
        self.cfg = cfg or {}
        self.initial_feature = self._extract_search_feature(frame_rgb, init_box)
        self.recent_feature = self.initial_feature.clone() if self.initial_feature is not None else None
        self.prototype_feature = self.initial_feature.clone() if self.initial_feature is not None else None

    def _extract_search_feature(self, frame_rgb, box):
        if not is_valid_box(box):
            return None
        try:
            patch, _, amask = sample_target(
                frame_rgb,
                box,
                self.tracker.params.search_factor,
                output_sz=self.tracker.params.search_size,
            )
            nested = self.tracker.preprocessor.process(patch, amask)
            with torch.no_grad():
                tokens, _ = self.tracker.network.backbone.forward_features(
                    z=self.tracker.z_dict1.tensors,
                    x=nested.tensors,
                )
            lens_x = self.tracker.network.backbone.pos_embed_x.shape[1]
            search_tokens = tokens[:, -lens_x:, :]
            feat = F.normalize(search_tokens.mean(dim=1), dim=1)
            return feat
        except Exception:
            return None

    def maybe_refresh_recent(self, frame_rgb, box, confidence):
        if confidence < float(self.cfg.get("feature_update_conf", 0.25)):
            return
        feat = self._extract_search_feature(frame_rgb, box)
        if feat is not None:
            self.recent_feature = feat
            if self.prototype_feature is None:
                self.prototype_feature = feat.clone()
            else:
                alpha = float(self.cfg.get("prototype_ema_alpha", 0.15))
                mixed = (1.0 - alpha) * self.prototype_feature + alpha * feat
                self.prototype_feature = F.normalize(mixed, dim=1)

    def _score_feature(self, feat):
        if feat is None:
            return None, {}
        scores = {}
        if self.prototype_feature is not None:
            scores["prototype_similarity"] = float(F.cosine_similarity(feat, self.prototype_feature, dim=1).item())
        if self.initial_feature is not None:
            scores["initial_similarity"] = float(F.cosine_similarity(feat, self.initial_feature, dim=1).item())
        if self.recent_feature is not None:
            scores["recent_similarity"] = float(F.cosine_similarity(feat, self.recent_feature, dim=1).item())
        if not scores:
            return None, scores
        weights = {
            "prototype_similarity": float(self.cfg.get("prototype_weight", 0.60)),
            "initial_similarity": float(self.cfg.get("initial_weight", 0.30)),
            "recent_similarity": float(self.cfg.get("recent_weight", 0.10)),
        }
        score = 0.0
        norm = 0.0
        for key, value in scores.items():
            w = weights.get(key, 0.0)
            score += w * value
            norm += w
        return (score / norm) if norm > 0 else None, scores

    def compute_similarity_info(self, frame_rgb, box):
        cand_feat = self._extract_search_feature(frame_rgb, box)
        if cand_feat is None:
            return {
                "feature_similarity": None,
                "initial_similarity": None,
                "recent_similarity": None,
            }

        recent_similarity = None
        initial_similarity = None
        if self.recent_feature is not None:
            recent_similarity = float(F.cosine_similarity(cand_feat, self.recent_feature, dim=1).item())
        if self.initial_feature is not None:
            initial_similarity = float(F.cosine_similarity(cand_feat, self.initial_feature, dim=1).item())

        sims = [v for v in (recent_similarity, initial_similarity) if v is not None]
        return {
            "feature_similarity": max(sims) if sims else None,
            "initial_similarity": initial_similarity,
            "recent_similarity": recent_similarity,
        }

    def search_prototype(self, frame_rgb, anchor_box, absent_run):
        if not self.cfg.get("enable_prototype_search", False):
            return RecoveryResult(False, [0, 0, 0, 0], 0.0, "prototype", [0, 0, 0, 0]), {}
        if not is_valid_box(anchor_box):
            return RecoveryResult(False, [0, 0, 0, 0], 0.0, "prototype", [0, 0, 0, 0]), {}

        h, w = frame_rgb.shape[:2]
        x, y, bw, bh = [float(v) for v in anchor_box]
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        radius_mul = float(self.cfg.get("prototype_search_mul", 4.0))
        expand_step = float(self.cfg.get("prototype_expand_per_absent", 0.8))
        max_mul = float(self.cfg.get("prototype_max_search_mul", 10.0))
        search_mul = min(max_mul, radius_mul + max(0, absent_run - 1) * expand_step)
        sw = max(bw * search_mul, float(self.cfg.get("prototype_min_search_px", 96)))
        sh = max(bh * search_mul, float(self.cfg.get("prototype_min_search_px", 96)))
        x1 = max(0.0, cx - sw / 2.0)
        y1 = max(0.0, cy - sh / 2.0)
        x2 = min(float(w), cx + sw / 2.0)
        y2 = min(float(h), cy + sh / 2.0)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return RecoveryResult(False, [0, 0, 0, 0], 0.0, "prototype", [x1, y1, x2 - x1, y2 - y1]), {}

        stride = max(8.0, min(bw, bh) * float(self.cfg.get("prototype_stride_frac", 0.5)))
        scales = self.cfg.get("prototype_scales", [0.85, 1.0, 1.15, 1.3])
        best = None
        best_info = {}
        cy_scan = y1
        while cy_scan <= y2:
            cx_scan = x1
            while cx_scan <= x2:
                for scale in scales:
                    cw = max(8.0, bw * float(scale))
                    ch = max(8.0, bh * float(scale))
                    cand = clip_box_xywh([cx_scan - cw / 2.0, cy_scan - ch / 2.0, cw, ch], w, h)
                    feat = self._extract_search_feature(frame_rgb, cand)
                    score, info = self._score_feature(feat)
                    if score is None:
                        continue
                    if best is None or score > best.score:
                        best = RecoveryResult(True, cand, float(score), "prototype", [x1, y1, x2 - x1, y2 - y1])
                        best_info = dict(info)
                cx_scan += stride
            cy_scan += stride

        if best is None:
            return RecoveryResult(False, [0, 0, 0, 0], 0.0, "prototype", [x1, y1, x2 - x1, y2 - y1]), {}

        min_score = float(self.cfg.get("prototype_min_score", 0.90))
        if absent_run >= int(self.cfg.get("prototype_long_absence_after", 6)):
            min_score = float(self.cfg.get("prototype_min_score_long", min_score + 0.02))
        if best.score < min_score:
            return RecoveryResult(False, [0, 0, 0, 0], best.score, "prototype", best.search_region), best_info
        return best, best_info

    def tracking_identity_mismatch(self, frame_rgb, box, feat):
        if not self.cfg.get("enable_tracking_identity_check", False):
            return False, {}
        suspicious = (
            feat["confidence"] <= float(self.cfg.get("tracking_identity_conf_max", 0.42))
            and (
                feat["area_ratio"] >= float(self.cfg.get("tracking_identity_min_area_ratio", 0.004))
                or feat["scale_change"] >= float(self.cfg.get("tracking_identity_min_scale_change", 1.8))
                or feat["jump_obj"] >= float(self.cfg.get("tracking_identity_min_jump_obj", 0.35))
                or (self.cfg.get("tracking_identity_check_boundary", True) and feat["touch_boundary"])
            )
        )
        if not suspicious:
            return False, {
                "tracking_identity_checked": False,
                "tracking_identity_suspicious": False,
                "tracking_identity_similarity": None,
                "tracking_identity_initial_similarity": None,
                "tracking_identity_recent_similarity": None,
            }

        sim_info = self.compute_similarity_info(frame_rgb, box)
        initial_similarity = sim_info["initial_similarity"]
        recent_similarity = sim_info["recent_similarity"]
        mismatch = False
        if initial_similarity is not None and initial_similarity < float(self.cfg.get("tracking_identity_initial_min", 0.88)):
            recent_floor = float(self.cfg.get("tracking_identity_recent_min", 0.86))
            mismatch = recent_similarity is None or recent_similarity < recent_floor

        return mismatch, {
            "tracking_identity_checked": True,
            "tracking_identity_suspicious": True,
            "tracking_identity_similarity": sim_info["feature_similarity"],
            "tracking_identity_initial_similarity": initial_similarity,
            "tracking_identity_recent_similarity": recent_similarity,
        }

    def verify(self, frame_rgb, proposal_box, prev_state, proposal_score):
        if not is_valid_box(proposal_box):
            return False, None, {}

        cand_feat = self._extract_search_feature(frame_rgb, proposal_box)
        feature_similarity = None
        if cand_feat is not None:
            sims = []
            if self.recent_feature is not None:
                sims.append(float(F.cosine_similarity(cand_feat, self.recent_feature, dim=1).item()))
            if self.initial_feature is not None:
                sims.append(float(F.cosine_similarity(cand_feat, self.initial_feature, dim=1).item()))
            feature_similarity = max(sims) if sims else None
        if feature_similarity is not None and feature_similarity < float(self.cfg.get("feature_sim_min", 0.88)):
            return False, None, {"feature_similarity": feature_similarity}

        saved_state = list(self.tracker.state)
        saved_conf = float(getattr(self.tracker, "last_confidence", 1.0))
        saved_z_patch = getattr(self.tracker, "z_patch_arr", None)
        saved_z_dict = getattr(self.tracker, "z_dict1", None)
        saved_box_mask_z = getattr(self.tracker, "box_mask_z", None)
        try:
            self.tracker.state = list(proposal_box)
            with torch.no_grad():
                out = self.tracker.track(frame_rgb, {})
            refined_box = [float(v) for v in out["target_bbox"]]
            refined_conf = float(getattr(self.tracker, "last_confidence", 0.0))
            agreement_iou = iou_xywh(refined_box, proposal_box)
            motion = box_features(refined_box, prev_state, frame_rgb.shape[1], frame_rgb.shape[0], refined_conf)
            if refined_conf < float(self.cfg.get("verify_conf_min", 0.16)):
                self.tracker.state = saved_state
                self.tracker.last_confidence = saved_conf
                self.tracker.z_patch_arr = saved_z_patch
                self.tracker.z_dict1 = saved_z_dict
                self.tracker.box_mask_z = saved_box_mask_z
                return False, None, {
                    "feature_similarity": feature_similarity,
                    "verify_conf": refined_conf,
                    "agreement_iou": agreement_iou,
                }
            if agreement_iou < float(self.cfg.get("agreement_iou_min", 0.30)):
                self.tracker.state = saved_state
                self.tracker.last_confidence = saved_conf
                self.tracker.z_patch_arr = saved_z_patch
                self.tracker.z_dict1 = saved_z_dict
                self.tracker.box_mask_z = saved_box_mask_z
                return False, None, {
                    "feature_similarity": feature_similarity,
                    "verify_conf": refined_conf,
                    "agreement_iou": agreement_iou,
                }
            if motion["jump_obj"] > float(self.cfg.get("verify_max_jump_obj", 4.0)):
                self.tracker.state = saved_state
                self.tracker.last_confidence = saved_conf
                self.tracker.z_patch_arr = saved_z_patch
                self.tracker.z_dict1 = saved_z_dict
                self.tracker.box_mask_z = saved_box_mask_z
                return False, None, {
                    "feature_similarity": feature_similarity,
                    "verify_conf": refined_conf,
                    "agreement_iou": agreement_iou,
                    "verify_jump_obj": motion["jump_obj"],
                }
            if motion["scale_change"] > float(self.cfg.get("verify_max_scale_change", 3.0)):
                self.tracker.state = saved_state
                self.tracker.last_confidence = saved_conf
                self.tracker.z_patch_arr = saved_z_patch
                self.tracker.z_dict1 = saved_z_dict
                self.tracker.box_mask_z = saved_box_mask_z
                return False, None, {
                    "feature_similarity": feature_similarity,
                    "verify_conf": refined_conf,
                    "agreement_iou": agreement_iou,
                    "verify_scale_change": motion["scale_change"],
                }
            return True, refined_box, {
                "feature_similarity": feature_similarity,
                "verify_conf": refined_conf,
                "agreement_iou": agreement_iou,
                "proposal_score": proposal_score,
            }
        except Exception:
            self.tracker.state = saved_state
            self.tracker.last_confidence = saved_conf
            self.tracker.z_patch_arr = saved_z_patch
            self.tracker.z_dict1 = saved_z_dict
            self.tracker.box_mask_z = saved_box_mask_z
            return False, None, {"feature_similarity": feature_similarity}


DEFAULT_RECOVERY_POLICY = {
    "enable": True,
    "enable_pre_recover": False,
    "enable_pre_verify": False,
    "reacquire_start_after": 2,
    "reacquire_confirm_frames": 2,
    "pre_recover_conf": 0.18,
    "pre_recover_after_absent": 1,
    "pre_search_mul": 4.0,
    "pre_expand_per_absent": 1.0,
    "pre_max_search_mul": 10.0,
    "pre_min_search_px": 96,
    "pre_min_score": 0.64,
    "pre_min_score_long": 0.68,
    "pre_long_absence_after": 4,
    "pre_max_area_ratio": 0.12,
    "pre_max_scale_change": 2.8,
    "pre_scales": [0.9, 1.0, 1.1, 1.25],
    "feature_sim_min": 0.88,
    "feature_update_conf": 0.25,
    "enable_prototype_search": True,
    "prototype_ema_alpha": 0.15,
    "prototype_search_mul": 4.0,
    "prototype_expand_per_absent": 0.8,
    "prototype_max_search_mul": 10.0,
    "prototype_min_search_px": 96,
    "prototype_stride_frac": 0.5,
    "prototype_scales": [0.85, 1.0, 1.15, 1.3],
    "prototype_min_score": 0.90,
    "prototype_min_score_long": 0.92,
    "prototype_long_absence_after": 6,
    "prototype_weight": 0.60,
    "initial_weight": 0.30,
    "recent_weight": 0.10,
    "enable_tracking_identity_check": True,
    "tracking_identity_conf_max": 0.42,
    "tracking_identity_min_area_ratio": 0.004,
    "tracking_identity_min_scale_change": 1.8,
    "tracking_identity_min_jump_obj": 0.35,
    "tracking_identity_check_boundary": True,
    "tracking_identity_initial_min": 0.88,
    "tracking_identity_recent_min": 0.86,
    "tracking_anchor_conf_max": 0.45,
    "tracking_anchor_area_mult": 8.0,
    "verify_conf_min": 0.16,
    "agreement_iou_min": 0.30,
    "verify_max_jump_obj": 4.0,
    "verify_max_scale_change": 3.0,
    "search_mul": 5.0,
    "expand_per_absent": 1.2,
    "max_search_mul": 12.0,
    "min_search_px": 96,
    "template_conf": 0.55,
    "min_score": 0.58,
    "min_score_long": 0.64,
    "long_absence_after": 6,
    "max_area_ratio": 0.12,
    "max_center_jump_obj": 4.5,
    "max_scale_change": 3.2,
    "scales": [0.85, 1.0, 1.15, 1.3],
    "min_template_side": 12,
}


def load_policy_blob(path):
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    return blob.get("best_policy", blob)


def load_csv_predictions(path):
    preds = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seq, frame_idx = row["id"].rsplit("_", 1)
            preds[(seq, int(frame_idx))] = [float(row[k]) for k in ("x", "y", "w", "h")]
    return preds


def track_sequence_online(tracker, data_root, item, absence_policy, recovery_policy=None, diagnostics=None, gt_boxes=None):
    video_path = data_root / item["video_path"]
    init_box = load_init_box(data_root / item["annotation_path"]) if gt_boxes is None else None
    expected_frames = min(int(item["n_frames"]), len(gt_boxes)) if gt_boxes is not None else int(item["n_frames"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    init_frame_idx = 0
    frame = None
    if gt_boxes is not None:
        init_frame_idx = next((i for i, box in enumerate(gt_boxes[:expected_frames]) if is_valid_box(box)), None)
        if init_frame_idx is None:
            raise RuntimeError(f"No valid initialization box found in ground truth for {video_path}")
        init_box = gt_boxes[init_frame_idx]
    while init_frame_idx >= 0:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read initialization frame for {video_path}")
        if init_frame_idx == 0:
            break
        init_frame_idx -= 1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    first_valid_idx = 0 if gt_boxes is None else next((i for i, box in enumerate(gt_boxes[:expected_frames]) if is_valid_box(box)), 0)
    tracker.initialize(frame_rgb, {"init_bbox": init_box})
    if recovery_policy is None:
        recovery_cfg = {"enable": False}
    else:
        recovery_cfg = dict(DEFAULT_RECOVERY_POLICY)
        recovery_cfg.update(recovery_policy)
    recovery = OnlineRecovery(frame_rgb, init_box, recovery_cfg)
    verifier = StrongProposalVerifier(tracker, frame_rgb, init_box, recovery_cfg)

    boxes = [[0.0, 0.0, 0.0, 0.0] for _ in range(first_valid_idx)] + [[float(v) for v in init_box]]
    absent_reasons = {}
    absent_run = 0
    if diagnostics is not None:
        for idx in range(first_valid_idx):
            diagnostics.append(
                {
                    "frame": idx,
                    "absent": True,
                    "reason": "pre_init_invalid_gt",
                    "confidence": 0.0,
                    "area_ratio": 0.0,
                    "touch_boundary": False,
                    "out_of_bounds": False,
                    "jump_obj": 0.0,
                    "scale_change": 1.0,
                    "recovered": False,
                    "pre_recovered": False,
                    "recovery_score": None,
                    "recovery_template": "",
                    "recovery_search_region": None,
                    "pre_recovery_score": None,
                    "pre_recovery_template": "",
                    "pre_recovery_search_region": None,
                    "pre_verify_accepted": False,
                    "pre_verify_conf": None,
                    "pre_verify_iou": None,
                    "pre_verify_similarity": None,
                    "prototype_score": None,
                    "prototype_initial_similarity": None,
                    "prototype_recent_similarity": None,
                    "tracking_identity_checked": False,
                    "tracking_identity_suspicious": False,
                    "tracking_identity_similarity": None,
                    "tracking_identity_initial_similarity": None,
                    "tracking_identity_recent_similarity": None,
                }
            )
        diagnostics.append(
            {
                "frame": first_valid_idx,
                "absent": False,
                "reason": "",
                "confidence": 1.0,
                "area_ratio": (init_box[2] * init_box[3]) / max(float(width * height), 1.0),
                "touch_boundary": False,
                "out_of_bounds": False,
                "jump_obj": 0.0,
                "scale_change": 1.0,
                "recovered": False,
                "pre_recovered": False,
                "recovery_score": None,
                "recovery_template": "",
                "recovery_search_region": None,
                "pre_recovery_score": None,
                "pre_recovery_template": "",
                "pre_recovery_search_region": None,
                "pre_verify_accepted": False,
                "pre_verify_conf": None,
                "pre_verify_iou": None,
                "pre_verify_similarity": None,
                "prototype_score": None,
                "prototype_initial_similarity": None,
                "prototype_recent_similarity": None,
                "tracking_identity_checked": False,
                "tracking_identity_suspicious": False,
                "tracking_identity_similarity": None,
                "tracking_identity_initial_similarity": None,
                "tracking_identity_recent_similarity": None,
            }
        )

    tracking_mode = "TRACKING"
    reacquire_streak = 0
    reacquire_confirm_frames = int(recovery_cfg.get("reacquire_confirm_frames", 2))
    reacquire_start_after = int(recovery_cfg.get("reacquire_start_after", 2))

    for frame_idx in range(first_valid_idx + 1, expected_frames):
        ok, frame = cap.read()
        if not ok:
            boxes.append([0, 0, 0, 0])
            absent_reasons[frame_idx] = "read_failed"
            tracking_mode = "ABSENT"
            absent_run += 1
            reacquire_streak = 0
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prev_state = list(tracker.state)
        state_before = tracking_mode
        pre_recovered = False
        pre_recovery_score = None
        pre_recovery_template = ""
        pre_recovery_search_region = None
        pre_verify_accepted = False
        pre_verify_conf = None
        pre_verify_iou = None
        pre_verify_similarity = None
        prototype_score = None
        prototype_initial_similarity = None
        prototype_recent_similarity = None
        tracking_identity_checked = False
        tracking_identity_suspicious = False
        tracking_identity_similarity = None
        tracking_identity_initial_similarity = None
        tracking_identity_recent_similarity = None
        tracking_anchor_area_mult = None
        recovered = False
        recovery_score = None
        recovery_template = ""
        recovery_search_region = None
        reason = "normal"
        feat = None

        if tracking_mode == "TRACKING":
            with torch.no_grad():
                out = tracker.track(frame_rgb, {})
            pred = [float(x) for x in out["target_bbox"]]
            feat = box_features(pred, prev_state, width, height, getattr(tracker, "last_confidence", 0.0))
            is_absent, reason = policy_absent(feat, absence_policy)
            mismatch, identity_info = verifier.tracking_identity_mismatch(frame_rgb, pred, feat)
            tracking_identity_checked = bool(identity_info.get("tracking_identity_checked"))
            tracking_identity_suspicious = bool(identity_info.get("tracking_identity_suspicious"))
            tracking_identity_similarity = identity_info.get("tracking_identity_similarity")
            tracking_identity_initial_similarity = identity_info.get("tracking_identity_initial_similarity")
            tracking_identity_recent_similarity = identity_info.get("tracking_identity_recent_similarity")
            if is_valid_box(recovery.last_good_box):
                anchor_area = max(float(recovery.last_good_box[2] * recovery.last_good_box[3]), 1e-9)
                pred_area = max(float(pred[2] * pred[3]), 1e-9)
                tracking_anchor_area_mult = pred_area / anchor_area
                if (
                    feat["confidence"] <= float(recovery_cfg.get("tracking_anchor_conf_max", 0.45))
                    and tracking_anchor_area_mult >= float(recovery_cfg.get("tracking_anchor_area_mult", 8.0))
                ):
                    is_absent = True
                    reason = "tracking_anchor_scale_mismatch"
            if mismatch:
                is_absent = True
                reason = "tracking_identity_mismatch"
            if is_absent:
                tracking_mode = "ABSENT"
                reacquire_streak = 0
                if absence_policy.get("freeze_on_absent", True):
                    tracker.state = prev_state
                boxes.append([0, 0, 0, 0])
                absent_reasons[frame_idx] = reason
                absent_run += 1
                feat = box_features(prev_state, prev_state, width, height, 0.0)
            else:
                boxes.append(pred)
                absent_run = 0
                reacquire_streak = 0
                recovery.maybe_refresh_recent_template(frame_rgb, pred, feat["confidence"])
                verifier.maybe_refresh_recent(frame_rgb, pred, feat["confidence"])
        else:
            # Stay absent until a candidate is verified; do not trust raw tracker outputs here.
            if absent_run >= reacquire_start_after and recovery_cfg.get("enable", True):
                pre = recovery.recover(frame_rgb, prev_state, absent_run + 1)
                if not pre.found and recovery_cfg.get("enable_prototype_search", False):
                    proto_candidate, proto_info = verifier.search_prototype(frame_rgb, prev_state, absent_run + 1)
                    prototype_score = proto_candidate.score
                    prototype_initial_similarity = proto_info.get("initial_similarity")
                    prototype_recent_similarity = proto_info.get("recent_similarity")
                    if proto_candidate.found:
                        pre = proto_candidate
                pre_recovery_score = pre.score
                pre_recovery_template = pre.template_name
                pre_recovery_search_region = pre.search_region
                if pre.found:
                    if recovery_cfg.get("enable_pre_verify", False):
                        accepted, refined_box, verify_info = verifier.verify(
                            frame_rgb,
                            pre.bbox,
                            prev_state,
                            pre.score,
                        )
                        pre_verify_accepted = bool(accepted)
                        pre_verify_conf = verify_info.get("verify_conf")
                        pre_verify_iou = verify_info.get("agreement_iou")
                        pre_verify_similarity = verify_info.get("feature_similarity")
                        if accepted and refined_box is not None:
                            reacquire_streak += 1
                            if reacquire_streak >= reacquire_confirm_frames:
                                tracking_mode = "TRACKING"
                                pred = list(refined_box)
                                tracker.state = list(pred)
                                boxes.append(pred)
                                absent_run = 0
                                recovered = True
                                pre_recovered = True
                                reason = f"recovered_{pre.template_name}"
                                recovery_score = float(max(pre.score, pre_verify_conf or 0.0))
                                recovery_template = pre.template_name
                                recovery_search_region = pre.search_region
                                feat = box_features(pred, prev_state, width, height, max(pre.score, pre_verify_conf or 0.0))
                                recovery.maybe_refresh_recent_template(frame_rgb, pred, max(pre.score, pre_verify_conf or 0.0))
                                verifier.maybe_refresh_recent(frame_rgb, pred, max(pre.score, pre_verify_conf or 0.0))
                                reacquire_streak = 0
                            else:
                                boxes.append([0, 0, 0, 0])
                                absent_reasons[frame_idx] = "reacquire_pending"
                                absent_run += 1
                                reason = "reacquire_pending"
                                tracker.state = prev_state
                                feat = box_features(prev_state, prev_state, width, height, 0.0)
                        else:
                            reacquire_streak = 0
                            boxes.append([0, 0, 0, 0])
                            absent_reasons[frame_idx] = "reacquire_rejected"
                            absent_run += 1
                            reason = "reacquire_rejected"
                            tracker.state = prev_state
                            feat = box_features(prev_state, prev_state, width, height, 0.0)
                    else:
                        reacquire_streak += 1
                        if reacquire_streak >= reacquire_confirm_frames:
                            tracking_mode = "TRACKING"
                            pred = list(pre.bbox)
                            tracker.state = list(pred)
                            boxes.append(pred)
                            absent_run = 0
                            recovered = True
                            pre_recovered = True
                            reason = f"recovered_{pre.template_name}"
                            recovery_score = float(pre.score)
                            recovery_template = pre.template_name
                            recovery_search_region = pre.search_region
                            feat = box_features(pred, prev_state, width, height, pre.score)
                            recovery.maybe_refresh_recent_template(frame_rgb, pred, pre.score)
                            verifier.maybe_refresh_recent(frame_rgb, pred, pre.score)
                            reacquire_streak = 0
                        else:
                            boxes.append([0, 0, 0, 0])
                            absent_reasons[frame_idx] = "reacquire_pending"
                            absent_run += 1
                            reason = "reacquire_pending"
                            tracker.state = prev_state
                            feat = box_features(prev_state, prev_state, width, height, 0.0)
                else:
                    reacquire_streak = 0
                    boxes.append([0, 0, 0, 0])
                    absent_reasons[frame_idx] = "absent_lockout"
                    absent_run += 1
                    reason = "absent_lockout"
                    tracker.state = prev_state
                    feat = box_features(prev_state, prev_state, width, height, 0.0)
            else:
                boxes.append([0, 0, 0, 0])
                absent_reasons[frame_idx] = "absent_lockout"
                absent_run += 1
                reason = "absent_lockout"
                tracker.state = prev_state
                feat = box_features(prev_state, prev_state, width, height, 0.0)

        if diagnostics is not None:
            diagnostics.append(
                {
                    "frame": frame_idx,
                    "absent": tracking_mode == "ABSENT" and not recovered,
                    "reason": reason,
                    "mode": tracking_mode,
                    "reacquire_streak": reacquire_streak,
                    **feat,
                    "recovered": recovered,
                    "pre_recovered": pre_recovered,
                    "recovery_score": recovery_score,
                    "recovery_template": recovery_template,
                    "recovery_search_region": recovery_search_region,
                    "pre_recovery_score": pre_recovery_score,
                    "pre_recovery_template": pre_recovery_template,
                    "pre_recovery_search_region": pre_recovery_search_region,
                    "pre_verify_accepted": pre_verify_accepted,
                    "pre_verify_conf": pre_verify_conf,
                    "pre_verify_iou": pre_verify_iou,
                    "pre_verify_similarity": pre_verify_similarity,
                    "prototype_score": prototype_score,
                    "prototype_initial_similarity": prototype_initial_similarity,
                    "prototype_recent_similarity": prototype_recent_similarity,
                    "tracking_identity_checked": tracking_identity_checked,
                    "tracking_identity_suspicious": tracking_identity_suspicious,
                    "tracking_identity_similarity": tracking_identity_similarity,
                    "tracking_identity_initial_similarity": tracking_identity_initial_similarity,
                    "tracking_identity_recent_similarity": tracking_identity_recent_similarity,
                    "tracking_anchor_area_mult": tracking_anchor_area_mult,
                }
            )

    cap.release()
    return boxes, absent_reasons
