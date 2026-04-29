# DeepFakeDetection Repository Overview

## Purpose

This repository implements and evaluates a deepfake detection algorithm centered on `HSF-CVIT`: a hybrid spatial-frequency detector with transformer-style fusion. The project contains the full practical workflow around the model:

- dataset download and frame extraction utilities,
- configurable training and validation,
- checkpointing and test-set reporting,
- image/video inference,
- cross-dataset evaluation on Celeb-DF v2.

The default training path is FaceForensics++ frame-level training. Celeb-DF v2 is currently supported as a cross-dataset evaluation target through `evaluate_celeb_df.py`.

## Current Implementation Snapshot

The code currently implements:

- `tf_efficientnet_b7` spatial backbone through `timm`;
- a two-level Haar stationary wavelet transform frequency branch (`SWTFrequencyBranch`);
- a compact 3-token transformer encoder fusion head (`[CLS]`, spatial token, frequency token);
- binary output logits interpreted as `P(fake)` after sigmoid;
- FaceForensics++ dataloaders using extracted frames and official split files in the default config;
- weighted sampling for class imbalance during training;
- deterministic validation/test frame selection;
- optional true video-level test evaluation through `train.py --video-eval`;
- a consumer inference CLI through `predict.py`;
- a Celeb-DF v2 test-set evaluator through `evaluate_celeb_df.py`.

Important implementation note: older SRM-related code still exists in `src/models/srm_filter.py`, but the active top-level model in `src/models/hsf_cvit.py` imports and uses `SWTFrequencyBranch` from `src/models/swt_filter.py`.

## High-Level Pipeline

The normal project workflow is:

1. Place raw datasets under `data/`.
2. Extract uniformly sampled frames with `scripts/extract_frames.py`.
3. Train with `train.py` using `configs/train_config.yaml`.
4. Save best and periodic checkpoints under `outputs/checkpoints/`.
5. Evaluate on the FaceForensics++ test split with `train.py --eval-only`.
6. Run single-file inference with `predict.py`.
7. Optionally evaluate cross-dataset generalization on Celeb-DF v2 with `evaluate_celeb_df.py`.

## Main Entry Points

### `train.py`

Main training and FaceForensics++ evaluation entrypoint. It:

- loads `configs/train_config.yaml`;
- loads the dataset config referenced by `data.config`;
- builds train/validation/test dataloaders;
- builds the HSF-CVIT model;
- trains with AMP when CUDA is available;
- saves checkpoints;
- writes TensorBoard, CSV, and evaluation reports.

It also supports:

- `--dummy` for no-I/O smoke tests;
- `--resume` for checkpoint resume;
- `--eval-only` for test evaluation;
- `--video-eval` for clip-level evaluation by averaging per-frame probabilities.

### `predict.py`

Consumer inference CLI for single images or videos. It loads a checkpoint and config, then returns a real/fake label, fake probability, threshold, and video frame statistics when applicable.

### `evaluate_celeb_df.py`

Cross-dataset evaluator for Celeb-DF v2. It consumes raw Celeb-DF videos listed in `List_of_testing_videos.txt`, runs `DeepFakeDetector.predict_video`, and writes:

- `outputs/celeb_df_eval/per_video.csv`
- `outputs/celeb_df_eval/metrics.json`

### `scripts/extract_frames.py`

Dataset preprocessing script for FaceForensics++ and Celeb-DF style video folders. It extracts a fixed number of frames per video, optionally resizing frames before saving.

## Repository Layout

```text
configs/                    YAML experiment and dataset settings
configs/ffpp_splits/         Official FF++ train/val/test pair splits
scripts/                    Dataset download, extraction, audit, plotting utilities
src/data/                   Dataset classes, transforms, dataloader factory
src/models/                 Spatial branch, SWT branch, fusion head, top-level model
src/training/               Losses, metrics, trainer, checkpoint/report logic
src/inference/              DeepFakeDetector inference wrapper
src/utils/                  Frame decoding, seeding, script logging helpers
outputs/checkpoints/        Saved model checkpoints
outputs/evaluation*/        Test reports from train.py evaluation
outputs/celeb_df_eval/      Celeb-DF evaluator outputs
data/                       Local datasets and extracted frames
docs/                       Project documentation
```

## Dataset Support

### FaceForensics++

The default training configuration uses the FaceForensics++ standard profile:

- real source: `original_sequences/youtube`;
- fake methods: `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`;
- compression setting documented as `c23`;
- extracted frame output: `data/frames_ffpp_standard`;
- split mode: official split JSON files in `configs/ffpp_splits/`.

The extended config includes additional methods/sources:

- `FaceShifter`;
- `DeepFakeDetection`;
- `original_sequences/actors`;
- output under `data/frames_ffpp_extended`.

### Celeb-DF v2

Celeb-DF v2 is not merged into the default training dataloader. It is configured for preprocessing and is now supported for evaluation through:

- `scripts/download_celeb_df_test.py`
- `evaluate_celeb_df.py`

Expected structure:

```text
data/Celeb-DF-v2/
├── Celeb-real/
├── YouTube-real/
├── Celeb-synthesis/
└── List_of_testing_videos.txt
```

Celeb-DF labels are inverted internally by the evaluator because the dataset list uses `1=real` and `0=fake`, while this project uses `1=fake`.

## Model Summary

The active HSF-CVIT model consists of:

1. `EfficientNetSpatialBranch`
   - `timm` backbone, default `tf_efficientnet_b7`;
   - ImageNet-pretrained by default;
   - projects pooled features to `spatial_out_dim=512`.

2. `SWTFrequencyBranch`
   - converts RGB to grayscale;
   - computes two levels of fixed Haar SWT high-frequency subbands;
   - concatenates `LH`, `HL`, and `HH` subbands from each level, giving 6 maps;
   - encodes them with a small residual CNN;
   - projects to `freq_out_dim=256`.

3. `CrossAttentionViT`
   - projects spatial and frequency vectors to `fusion_dim=256`;
   - constructs `[CLS] + spatial + frequency`;
   - applies a 2-layer `nn.TransformerEncoder`;
   - classifies the `[CLS]` representation with a linear layer.

The model returns raw logits. Metrics and inference apply sigmoid to obtain `P(fake)`.

## Training and Evaluation Behavior

The trainer implements:

- `BCEWithLogitsLoss` with optional label smoothing and optional `pos_weight`;
- AdamW, Adam, or SGD;
- linear warm-up plus cosine/step/plateau scheduling;
- spatial-backbone freezing for early epochs;
- gradient clipping;
- CUDA AMP through `torch.amp`;
- validation-threshold optimization over a configurable grid;
- best checkpoint saving by validation ROC-AUC;
- periodic checkpoint saving and pruning;
- early stopping;
- per-method test reports.

Evaluation reports from `train.py` are written as:

```text
outputs/evaluation/
├── test_metrics.json
├── test_confusion_matrix.csv
├── test_per_method.csv
└── test_predictions.csv
```

## Current Practical Status

The repository is usable end to end for:

- FaceForensics++ frame extraction;
- FaceForensics++ training;
- FaceForensics++ checkpoint evaluation;
- single image/video inference;
- Celeb-DF v2 evaluation once the gated dataset is placed locally.

The strongest run artifacts currently present are associated with the SWT+B7 iteration. A final log (`outputs/logs/final_20260427_0322_clean.log`) reports FaceForensics++ test ROC-AUC around `0.906`, while the latest `outputs/evaluation/test_metrics.json` may reflect a later evaluation command and should be treated as the most recent report file rather than necessarily the best historical model.

## Documentation Map

- `docs/UserGuide.md`: consumer-facing run guide.
- `docs/model_architecture_design.md`: current model architecture.
- `docs/dataset_configuration_report.md`: dataset config and active data paths.
- `docs/training_run_summary.md`: observed run artifacts and evaluation status.
- `docs/repo_technical_reference.md`: implementation-level reference.
- `docs/improvement_notes.md`: roadmap and known next steps.
