# ORTrack AIC Challenge

![ORTrack architecture](code/ORTrack/assets/ORTrack.png)

This repository contains an ORTrack-based UAV object tracking project prepared for the AIC Challenge. It packages the ORTrack source snapshot, a DeiT-Tiny tracking checkpoint, Docker inference scripts, local evaluation helpers, and an example prediction file.

ORTrack is a transformer-based single-object tracker. For the AIC workflow, each video sequence is initialized with the first-frame bounding box from the challenge metadata. The tracker then processes frames online, one frame at a time, and writes bounding-box predictions for the requested sample IDs.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `code/ORTrack/` | ORTrack source code and AIC-specific inference/evaluation tools. |
| `model/ORTrack_AIC.pth.tar` | Trained DeiT-Tiny ORTrack checkpoint. |
| `submission/ortrack_aic_predictions.csv` | Example prediction CSV generated with the packaged model. |
| `Dockerfile` | Reproducible CUDA/PyTorch inference environment. |
| `run_inference.sh` | Container entrypoint for prediction generation. |
| `run_evaluation.sh` | Optional local evaluation entrypoint for annotated splits. |
| `technical_report.md` | Short method and reproducibility report. |

## Model Workflow

The inference pipeline expects the AIC dataset directory to contain:

- `metadata/contestant_manifest.json`
- `metadata/sample_submission.csv`
- the video/image data referenced by the manifest

At inference time, the tracker:

1. Reads the challenge metadata and sample-submission IDs.
2. Initializes each sequence from its first annotated bounding box.
3. Runs ORTrack online across the sequence at the original frame rate.
4. Reuses the previous prediction if an individual frame cannot be read.
5. Writes predictions as `id,x,y,w,h`.

## Docker Usage

Build the image:

```bash
docker build -t ortrack-aic .
```

Run inference:

```bash
docker run --gpus all --rm \
  -v /path/to/aic-data:/data:ro \
  -v /path/to/output:/output \
  ortrack-aic
```

The default output file is:

```text
/output/ortrack_aic_predictions.csv
```

You can override the default paths and config with environment variables:

```bash
docker run --gpus all --rm \
  -e DATA_ROOT=/data \
  -e OUTPUT_DIR=/output \
  -e CONFIG=deit_tiny_aic_stage1 \
  -e CHECKPOINT=/workspace/model/ORTrack_AIC.pth.tar \
  -v /path/to/aic-data:/data:ro \
  -v /path/to/output:/output \
  ortrack-aic
```

## Native Python Usage

Install dependencies in your preferred Python environment, then run the inference script from the project root:

```bash
pip install -r requirements_docker.txt
python -B code/ORTrack/make_aic_public_submission.py \
  --data-root /path/to/aic-data \
  --manifest /path/to/aic-data/metadata/contestant_manifest.json \
  --sample /path/to/aic-data/metadata/sample_submission.csv \
  --split public_lb \
  --config deit_tiny_aic_stage1 \
  --checkpoint model/ORTrack_AIC.pth.tar \
  --output /path/to/output/ortrack_aic_predictions.csv
```

On Windows PowerShell, use the same arguments with variables for your local folders:

```powershell
$AIC_DATA_ROOT = "<path-to-aic-data>"
$OUTPUT_DIR = "<path-to-output>"
python -B code/ORTrack/make_aic_public_submission.py `
  --data-root $AIC_DATA_ROOT `
  --manifest "$AIC_DATA_ROOT\metadata\contestant_manifest.json" `
  --sample "$AIC_DATA_ROOT\metadata\sample_submission.csv" `
  --split public_lb `
  --config deit_tiny_aic_stage1 `
  --checkpoint "model\ORTrack_AIC.pth.tar" `
  --output "$OUTPUT_DIR\ortrack_aic_predictions.csv"
```

## Local Evaluation

If annotated train/validation data is available, run the evaluation entrypoint:

```bash
docker run --gpus all --rm \
  -v /path/to/aic-data:/data:ro \
  -v /path/to/output:/output \
  ortrack-aic /workspace/run_evaluation.sh
```

Optional environment variables:

- `DATA_ROOT`: mounted dataset root, default `/data`
- `OUTPUT_DIR`: evaluation output directory, default `/output/eval_train`
- `CHECKPOINT`: checkpoint path, default `/workspace/model/ORTrack_AIC.pth.tar`
- `CONFIG`: ORTrack config name, default `deit_tiny_aic_stage1`
- `SPLIT_FILE`: evaluation split file, default `/workspace/ORTrack/data_specs/aic_contest_val.txt`

## Notes

- Inference is online-only and does not use future frames.
- The tracker is initialized only from the first available annotated frame for each sequence.
- Predictions are produced at the original frame rate.
- Checksums for the main deliverables are recorded in `checksums_sha256.txt`.

## License

The bundled ORTrack source code is MIT licensed by its original author; see `code/ORTrack/LICENSE`. This repository preserves that source snapshot and adds AIC Challenge packaging around it.
