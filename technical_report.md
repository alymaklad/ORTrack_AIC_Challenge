# Technical Report: ORTrack-DeiT Tiny for UAV Object Tracking

## Overview

This project uses ORTrack with a DeiT-Tiny backbone for single-object UAV tracking. The packaged checkpoint is trained from the ORTrack-DeiT Tiny model and is intended for online inference on AIC Challenge video sequences.

## Method

Each sequence is initialized from the first-frame bounding box provided in the challenge metadata. The tracker then processes frames sequentially at the original frame rate. No future frames are used and no ground-truth re-initialization is performed after the first frame.

The model architecture follows the ORTrack tracking-by-transformer design. A template crop is extracted from the first frame and paired with each subsequent search crop. The DeiT-Tiny backbone extracts visual features, and the tracking head predicts the target bounding box in the current frame. The final output is a CSV containing one bounding box per required frame id.

## Training Summary

The packaged model comes from AIC fine-tuning using the contest training data split. The model was fine-tuned with conservative settings to avoid overfitting and to preserve the pretrained ORTrack representation. Additional experiments explored absence-aware post-processing, recovery heuristics, and prototype-style object search, but the packaged inference path keeps the tracker behavior simple and reproducible.

## Inference Pipeline

The inference pipeline is implemented in `make_aic_public_submission.py`. For each sequence:

1. Read sequence metadata and the first-frame initialization box.
2. Initialize ORTrack with the first-frame RGB image and bounding box.
3. Run the tracker online for every frame in order.
4. If a video read fails for a frame, reuse the last predicted box.
5. Write predictions matching the sample submission ids.

The pipeline preserves the original video frame rate and frame count. It does not use downsampled or reframed videos during inference.

## Reproducibility

The repository includes a Dockerfile and run scripts. The Docker image installs the required PyTorch runtime, copies the ORTrack source code and checkpoint, and exposes an inference entrypoint. The intended execution mode is inference-only, with the data mounted at `/data` and outputs written to `/output`.

## Limitations

The main known weakness is long-term disappearance or heavy occlusion. ORTrack can sometimes continue predicting a plausible box even after the original object leaves the frame, especially in scenes with similar distractors. More aggressive post-processing can reduce this behavior, but it may also suppress visible targets.
