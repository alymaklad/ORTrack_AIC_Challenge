#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
CHECKPOINT="${CHECKPOINT:-/workspace/model/ORTrack_ep0008.pth.tar}"
CONFIG="${CONFIG:-deit_tiny_aic_stage1}"

mkdir -p "${OUTPUT_DIR}"

python -B /workspace/ORTrack/make_aic_public_submission.py \
  --data-root "${DATA_ROOT}" \
  --manifest "${DATA_ROOT}/metadata/contestant_manifest.json" \
  --sample "${DATA_ROOT}/metadata/sample_submission.csv" \
  --split public_lb \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT_DIR}/ortrack_deit_aic_stage1_ep0008_public_lb_submission.csv"
