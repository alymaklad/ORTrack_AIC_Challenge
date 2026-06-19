#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/output/eval_train}"
CHECKPOINT="${CHECKPOINT:-/workspace/model/ORTrack_ep0008.pth.tar}"
CONFIG="${CONFIG:-deit_tiny_aic_stage1}"
SPLIT_FILE="${SPLIT_FILE:-/workspace/ORTrack/data_specs/aic_contest_val.txt}"

mkdir -p "${OUTPUT_DIR}"

python -B /workspace/ORTrack/eval_aic_train.py \
  --data-root "${DATA_ROOT}" \
  --manifest "${DATA_ROOT}/metadata/contestant_manifest.json" \
  --split train \
  --split-file "${SPLIT_FILE}" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT_DIR}"
