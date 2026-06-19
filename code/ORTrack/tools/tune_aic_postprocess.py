import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


VARIANTS = [
    {
        "name": "pp_clamp_only",
        "post": {
            "ENABLE": True,
            "SMOOTH_ALPHA": 1.0,
            "LOW_CONF_ALPHA": 1.0,
            "LOW_CONF_THRESHOLD": 0.0,
            "MAX_CENTER_JUMP_OBJ": 5.0,
            "MAX_SCALE_CHANGE": 2.0,
            "MIN_BOX_SIZE": 2.0,
            "TEMPLATE_UPDATE": False,
            "TEMPLATE_UPDATE_CONF": 0.60,
            "TEMPLATE_UPDATE_INTERVAL": 25,
        },
    },
    {
        "name": "pp_light_smooth",
        "post": {
            "ENABLE": True,
            "SMOOTH_ALPHA": 0.90,
            "LOW_CONF_ALPHA": 0.65,
            "LOW_CONF_THRESHOLD": 0.20,
            "MAX_CENTER_JUMP_OBJ": 5.0,
            "MAX_SCALE_CHANGE": 2.0,
            "MIN_BOX_SIZE": 2.0,
            "TEMPLATE_UPDATE": False,
            "TEMPLATE_UPDATE_CONF": 0.60,
            "TEMPLATE_UPDATE_INTERVAL": 25,
        },
    },
    {
        "name": "pp_mid_smooth",
        "post": {
            "ENABLE": True,
            "SMOOTH_ALPHA": 0.85,
            "LOW_CONF_ALPHA": 0.55,
            "LOW_CONF_THRESHOLD": 0.20,
            "MAX_CENTER_JUMP_OBJ": 5.0,
            "MAX_SCALE_CHANGE": 2.0,
            "MIN_BOX_SIZE": 2.0,
            "TEMPLATE_UPDATE": False,
            "TEMPLATE_UPDATE_CONF": 0.60,
            "TEMPLATE_UPDATE_INTERVAL": 25,
        },
    },
    {
        "name": "pp_strict_jump",
        "post": {
            "ENABLE": True,
            "SMOOTH_ALPHA": 0.90,
            "LOW_CONF_ALPHA": 0.65,
            "LOW_CONF_THRESHOLD": 0.20,
            "MAX_CENTER_JUMP_OBJ": 3.0,
            "MAX_SCALE_CHANGE": 1.7,
            "MIN_BOX_SIZE": 2.0,
            "TEMPLATE_UPDATE": False,
            "TEMPLATE_UPDATE_CONF": 0.60,
            "TEMPLATE_UPDATE_INTERVAL": 25,
        },
    },
]


def load_score(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)["summary"]
    return {
        "score": 0.6 * summary["success_auc"] + 0.4 * summary["precision_at_20px"],
        "auc": summary["success_auc"],
        "precision": summary["precision_at_20px"],
        "normalized_precision": summary["normalized_precision_at_0_5"],
        "failures_iou_eq_0": summary["robustness_failures_iou_eq_0"],
        "failures_iou_lt_0_1": summary["robustness_failures_iou_lt_0_1"],
        "frames_evaluated": summary["frames_evaluated"],
        "sequences_evaluated": summary["sequences_evaluated"],
    }


def write_variant_config(base_config, output_config, post):
    with open(base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("TEST", {})["POST_PROCESS"] = post
    output_config.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_eval(args, variant):
    config_name = f"{args.config_prefix}_{variant['name']}"
    config_path = ROOT / "experiments" / "ortrack" / f"{config_name}.yaml"
    write_variant_config(ROOT / "experiments" / "ortrack" / f"{args.base_config}.yaml", config_path, variant["post"])

    output_dir = Path(args.output_dir) / variant["name"]
    cmd = [
        args.python,
        "eval_aic_train.py",
        "--split",
        "train",
        "--split-file",
        args.val_split,
        "--config",
        config_name,
        "--checkpoint",
        args.checkpoint,
        "--output",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print("Running:", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], cwd=str(ROOT), check=True, env=env)
    return {
        "variant": variant["name"],
        "config": str(config_path),
        "output": str(output_dir),
        "checkpoint": args.checkpoint,
        **variant["post"],
        **load_score(output_dir / "summary_metrics.json"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--base-config", default="deit_tiny_aic_stage1")
    parser.add_argument(
        "--checkpoint",
        default="model/ORTrack_AIC.pth.tar",
    )
    parser.add_argument("--val-split", default="data_specs/aic_contest_val.txt")
    parser.add_argument("--output-dir", default="outputs/postprocess_tuning")
    parser.add_argument("--config-prefix", default="deit_tiny_aic_stage1")
    parser.add_argument("--limit-variants", type=int, default=0)
    args = parser.parse_args()

    variants = VARIANTS[: args.limit_variants] if args.limit_variants > 0 else VARIANTS
    results = []
    for variant in variants:
        results.append(run_eval(args, variant))

    results = sorted(results, key=lambda item: item["score"], reverse=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "postprocess_tuning_results.json", "w", encoding="utf-8") as f:
        json.dump({"best": results[0], "results": results}, f, indent=2)
    with open(output_dir / "postprocess_tuning_results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"best": results[0], "results": results}, indent=2))


if __name__ == "__main__":
    main()
