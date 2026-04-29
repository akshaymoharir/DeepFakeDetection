# DeepFakeDetection Technical Reference

## Scope

This reference describes the implemented code paths in the repository. It is intended for developers or reviewers who need to understand how the project works internally.

It intentionally does not replace `docs/UserGuide.md`, which is the consumer-facing run guide.

## Current System Summary

The repository implements a configurable deepfake detection pipeline:

- FaceForensics++ frame extraction and training;
- HSF-CVIT model construction;
- checkpointed training and test evaluation;
- image/video inference;
- Celeb-DF v2 cross-dataset evaluation.

The active model is:

```text
tf_efficientnet_b7 spatial branch
+ two-level Haar SWT frequency branch
+ 3-token transformer encoder fusion head
```

The active default training config is:

```text
configs/train_config.yaml
```

The active default dataset config is:

```text
configs/dataset_config.yaml
```

## Top-Level Entrypoints

### `train.py`

Responsibilities:

- parse CLI arguments;
- load training config;
- load dataset config referenced by `train_cfg["data"]["config"]`;
- apply CLI overrides;
- seed random sources;
- select device;
- build dataloaders;
- build model;
- create `Trainer`;
- optionally resume checkpoint;
- train or run eval-only;
- after training, load `best.pt` and evaluate on the test set.

Important CLI arguments:

```text
--config
--dummy
--epochs
--batch-size
--lr
--no-amp
--workers
--methods
--real-dir
--items-per-clip
--eval-frames
--eval-frame-strategy
--video-eval
--report-dir
--resume
--eval-only
--device
```

Implementation notes:

- `--video-eval` is only accepted with `--eval-only`.
- `--eval-only` can run without `--resume`, but it then evaluates a randomly initialized model and prints a warning.
- `setup_script_logging("train")` mirrors stdout/stderr to `outputs/logs/`.

### `predict.py`

Responsibilities:

- load one image or video path;
- infer file type by extension, with PIL fallback for unknown extensions;
- load config and checkpoint;
- instantiate `DeepFakeDetector`;
- run image or video prediction;
- print human-readable output or JSON.

Important CLI arguments:

```text
--input
--checkpoint
--config
--frames
--aggregation {mean,max}
--no-face-detect
--face-margin
--device
--json
```

### `evaluate_celeb_df.py`

Responsibilities:

- parse `List_of_testing_videos.txt`;
- invert Celeb-DF label convention to the project convention;
- load a trained checkpoint through `DeepFakeDetector`;
- run per-video inference on raw Celeb-DF videos;
- write per-video CSV and metrics JSON.

Default inputs:

```text
checkpoint:   outputs/checkpoints/best_iteration_swt_b7.pt
config:       configs/train_config.yaml
dataset root: data/Celeb-DF-v2
frames:       16
aggregation:  mean
output dir:   outputs/celeb_df_eval
```

The script lazy-loads ML dependencies after parsing arguments so `--help` can work in a minimal host Python environment.

### `scripts/extract_frames.py`

Responsibilities:

- extract frames from FaceForensics++ and/or Celeb-DF video folders;
- uniformly sample a configurable number of frames per video;
- optionally resize frames;
- write frames into class/video folder structure.

Important CLI arguments:

```text
--config
--dataset {ff++,celeb-df,all}
--ff-methods
--ff-real-dirs
--max-frames
--resize
```

## Configuration Files

### `configs/train_config.yaml`

Default model/training config. Key settings:

```yaml
model:
  spatial_backbone: tf_efficientnet_b7
  pretrained_spatial: true
  freeze_spatial_epochs: 2
  spatial_out_dim: 512
  freq_out_dim: 256
  fusion_heads: 4
  fusion_dim: 256
  dropout: 0.4
data:
  config: configs/dataset_config.yaml
  frames_per_clip: 1
  train_items_per_clip: 8
  image_size: 380
  balance_strategy: weighted_sampler
training:
  epochs: 60
  batch_size: 16
  optimizer: adamw
  lr: 1.0e-4
  weight_decay: 5.0e-4
  warmup_epochs: 8
  lr_schedule: cosine
  gradient_clip: 1.0
  label_smoothing: 0.10
  amp: true
evaluation:
  decision_threshold: 0.5
  deterministic_eval: true
  eval_frames_per_clip: 1
  eval_frame_strategy: center
  optimize_threshold: true
  threshold_metric: balanced_accuracy
checkpoints:
  dir: outputs/checkpoints
```

### `configs/train_config_extended.yaml`

Extended experiment profile. Differences include:

- dataset config: `configs/dataset_config_extended.yaml`;
- methods include `FaceShifter` and `DeepFakeDetection`;
- image size `224`;
- batch size `8`;
- output directories under `outputs/*_extended`.

### `configs/dataset_config.yaml`

Default dataset config. It defines:

- FaceForensics++ standard profile;
- official split files;
- four default manipulation methods;
- Celeb-DF v2 paths for preprocessing/evaluation;
- common preprocessing settings.

### `configs/dataset_config_extended.yaml`

Extended dataset config. It defines:

- six manipulation methods;
- actor source support;
- numeric fallback split behavior;
- output under `data/frames_ffpp_extended`.

## Data Layer

### `src/data/dataset.py`

Key classes/functions:

- `FaceForensicsDataset`
- `DummyFaceForensicsDataset`
- `build_dataloaders`

### `FaceForensicsDataset`

Expected extracted frame layout:

```text
<frames_dir>/
├── real/
│   └── <video_id>/
│       └── frame_*.jpg
├── Deepfakes/
├── Face2Face/
├── FaceSwap/
└── NeuralTextures/
```

Labels:

```text
0 = real
1 = fake
```

The sample tuple internally stores:

```text
(video_dir, label, method, video_id)
```

Metadata returned with each item:

```text
method
video_id
video_dir
split
num_frames_used
video_eval
```

### Split Inference

The dataset supports:

- `official` split mode using JSON split files;
- `numeric` split mode using configured ID ranges;
- deterministic hash fallback for actor-style IDs.

Default training uses official FaceForensics++ split files.

### Frame Selection

Training:

- random frame selection;
- stochastic transforms;
- optional `items_per_clip` repetition.

Validation/test:

- deterministic frame selection by default;
- `center`, `uniform`, or `random` strategy;
- optional clip stack return for video-level evaluation.

### `build_dataloaders`

Builds train/val/test dataloaders.

Important behavior:

- dummy mode creates in-memory tensors;
- missing fake method folders are skipped with a warning;
- if no selected fake method exists, a runtime error is raised;
- weighted sampler is applied only to the training loader when configured;
- validation/test loaders are deterministic and unshuffled.

## Transform Layer

### `src/data/transforms.py`

Training transform:

```text
Resize(image_size + 16)
RandomCrop(image_size)
RandomHorizontalFlip
ColorJitter
RandomRotation
RandomApply(GaussianBlur)
RandomJpegCompression(quality 60-95, p=0.3)
ToTensor
ImageNet Normalize
```

Validation/test transform:

```text
Resize(image_size)
CenterCrop(image_size)
ToTensor
ImageNet Normalize
```

The model internally denormalizes tensors before sending them to the SWT frequency branch.

## Model Layer

### `src/models/hsf_cvit.py`

Defines:

- `HSF_CVIT`
- `build_model(train_cfg)`

Composition:

```text
EfficientNetSpatialBranch
SWTFrequencyBranch
CrossAttentionViT
```

Forward behavior:

1. spatial branch receives normalized input;
2. input is denormalized to `[0, 1]`-like RGB for SWT;
3. frequency branch receives denormalized image;
4. fusion head combines both vectors;
5. model returns `(B, 1)` logits.

### `src/models/efficientnet_branch.py`

Defines `EfficientNetSpatialBranch`.

Responsibilities:

- create `timm` backbone;
- remove classifier through `num_classes=0`;
- use global average pooling;
- project features to `spatial_out_dim`;
- support backbone freeze/unfreeze.

Default backbone:

```text
tf_efficientnet_b7
```

### `src/models/swt_filter.py`

Defines `SWTFrequencyBranch`.

Responsibilities:

- RGB to grayscale conversion;
- two-level Haar SWT high-frequency extraction;
- concatenate `LH`, `HL`, `HH` subbands from each level;
- residual CNN encoding;
- projection to `freq_out_dim`.

Output maps before encoder:

```text
levels * 3 = 6 channels
```

### `src/models/cross_attention_vit.py`

Defines `CrossAttentionViT`.

Current fusion design:

```text
[CLS] + spatial token + frequency token
```

Implementation details:

- linear projection for each branch vector;
- learned `cls_token`;
- learned positional embedding of length 3;
- 2-layer `nn.TransformerEncoder`;
- classifier on encoded CLS token.

## Training Layer

### `src/training/losses.py`

Builds binary classification loss with:

- `BCEWithLogitsLoss`;
- optional label smoothing;
- optional `pos_weight`.

The project convention is:

```text
0 = real
1 = fake
```

### `src/training/metrics.py`

Computes:

- ROC-AUC;
- average precision;
- F1;
- precision;
- recall;
- specificity;
- balanced accuracy;
- accuracy;
- MCC;
- Youden's J;
- confusion matrix counts;
- predicted/true fake rates.

Also provides threshold search through `find_best_threshold`.

### `src/training/trainer.py`

Main training engine.

Features:

- AdamW/Adam/SGD;
- cosine/step/plateau LR schedule;
- linear warm-up for non-plateau schedulers;
- AMP with `torch.amp`;
- gradient clipping;
- spatial backbone freeze/unfreeze;
- validation threshold optimization;
- best checkpoint saving by validation ROC-AUC;
- periodic checkpoint saving;
- old checkpoint pruning;
- early stopping;
- TensorBoard logging;
- CSV logging;
- test metrics/report export.

Checkpoint state includes:

```text
epoch
model
optimizer
scheduler
scaler
best_val_auc
best_threshold
global_step
```

Evaluation report files:

```text
test_metrics.json
test_confusion_matrix.csv
test_per_method.csv
test_predictions.csv
```

## Inference Layer

### `src/inference/detector.py`

Defines `DeepFakeDetector`.

Responsibilities:

- load checkpoint model state;
- load `best_threshold` from checkpoint when available;
- build validation transform;
- optionally crop largest face with OpenCV Haar cascade;
- predict one image;
- predict one video by sampling frames and aggregating probabilities.

Video aggregation:

- `mean`: average sampled frame probabilities;
- `max`: use the highest sampled frame probability.

Video decoding uses `extract_frames(...)` from `src/utils/helpers.py`.

### Error Behavior

If video decoding fails, `predict_video` returns:

```python
{
    "label": "error",
    "probability": nan,
    "frame_probs": [],
    "num_frames_used": 0,
    "error": "..."
}
```

If no frames are available after decoding, it returns label `"unknown"` and probability `0.5`.

The Celeb-DF evaluator treats zero decoded frames as failed rows rather than valid predictions.

## Utility Layer

### `src/utils/helpers.py`

Important helpers:

- `setup_script_logging`;
- `load_config`;
- `get_video_paths`;
- `get_image_paths`;
- `extract_frames`;
- `seed_everything`.

Frame decoding tries:

1. OpenCV;
2. decord;
3. PyAV.

If all backends fail, `extract_frames` raises an `IOError` with backend error details.

## Script Layer

### `scripts/download_FaceForensicsPP.py`

FaceForensics++ downloader based on the public release download logic. Supports dataset/method, compression, file type, server, and partial video count.

### `scripts/download_celeb_df_test.py`

Celeb-DF v2 helper for gated dataset access.

Modes:

- `--list-only`: parse local test list and report folder counts;
- default download mode: list Google Drive folders and download matched test videos;
- `--full-folders`: download full subfolders and prune non-test videos.

Requires `gdown` for Drive download modes.

### `scripts/create_dummy_datasets.py`

Creates synthetic FaceForensics++ and Celeb-DF-style directory layouts for smoke testing.

### `scripts/explore_dataset.py`, `scripts/analyse_quality.py`, `scripts/plot_samples.py`

Exploration and reporting utilities for dataset inspection.

### `scripts/audit_faceforensics.py`

Audits extracted FaceForensics++ frame folders before training/evaluation.

## Runtime Environment

### Docker

Recommended launcher:

```bash
./start_container.sh
```

The script:

- builds `deepfake-detect` when requested or missing;
- mounts the project at `/workspace`;
- mounts `data/` and `outputs/`;
- enables GPU access with `--gpus all`;
- exposes TensorBoard and Jupyter ports.

### Dockerfile

Base image:

```text
nvcr.io/nvidia/pytorch:24.12-py3
```

Additional system packages include video and OpenCV-related dependencies such as `ffmpeg`, `libgl1`, and `libglib2.0-0`.

Python dependencies are installed from `requirements.txt`.

## Output Directories

Common outputs:

```text
outputs/checkpoints/              model checkpoints
outputs/evaluation/               default FaceForensics++ test reports
outputs/evaluation_*              alternate evaluation runs
outputs/celeb_df_eval/            Celeb-DF evaluation outputs
outputs/logs/                     mirrored script logs
outputs/runs/                     TensorBoard events
outputs/training_log.csv          epoch-level train/validation CSV
```

## Known Implementation Boundaries

- Default `train.py` does not train on Celeb-DF v2.
- The model has no temporal architecture; video support is probability aggregation over sampled frames.
- `srm_learnable` is accepted by the model constructor but has no effect on the active SWT branch.
- Report files in `outputs/evaluation/` can be overwritten by later eval runs.
- `src/models/srm_filter.py` is retained but not active in the current top-level model.

## Recommended Developer Checks

Useful smoke tests:

```bash
python train.py --dummy --epochs 1 --batch-size 4 --device cpu
python predict.py --help
python evaluate_celeb_df.py --help
python scripts/extract_frames.py --help
```

Useful final-evaluation pattern:

```bash
python train.py \
  --config configs/train_config.yaml \
  --resume outputs/checkpoints/best_iteration_swt_b7.pt \
  --eval-only \
  --report-dir outputs/evaluation_final_swt_b7 \
  --device cuda
```

## Summary

The implemented repository is an end-to-end FaceForensics++ training and inference framework with a dedicated Celeb-DF v2 evaluator. The active model is SWT+B7 HSF-CVIT with compact transformer fusion, not the older SRM+B4 design described in earlier drafts.
