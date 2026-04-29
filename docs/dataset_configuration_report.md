# Dataset Configuration Report

## Purpose

This report explains what dataset-related configuration exists in the repository and which parts are actively used by the implemented training and evaluation code.

The important distinction is:

- `train.py` currently trains and evaluates through the FaceForensics++ extracted-frame dataloader.
- Celeb-DF v2 is configured and supported for preprocessing/evaluation, but it is not merged into the default `train.py` training dataset.
- `evaluate_celeb_df.py` is the active Celeb-DF v2 evaluation path.

## Relevant Files

```text
configs/train_config.yaml
configs/train_config_extended.yaml
configs/dataset_config.yaml
configs/dataset_config_extended.yaml
configs/ffpp_splits/train.json
configs/ffpp_splits/val.json
configs/ffpp_splits/test.json
src/data/dataset.py
src/data/transforms.py
scripts/extract_frames.py
scripts/download_FaceForensicsPP.py
scripts/download_celeb_df_test.py
evaluate_celeb_df.py
```

## Default Training Data Flow

When running:

```bash
python train.py --config configs/train_config.yaml
```

the code path is:

1. `train.py` loads `configs/train_config.yaml`.
2. It reads `train_cfg["data"]["config"]`.
3. The default value points to `configs/dataset_config.yaml`.
4. `train.py` loads that dataset config.
5. `build_dataloaders(...)` in `src/data/dataset.py` builds FaceForensics++ train/val/test dataloaders.

The default trainer does not build a mixed FaceForensics++ + Celeb-DF training set.

## Default FaceForensics++ Configuration

From `configs/dataset_config.yaml`:

```yaml
faceforensics:
  profile: ffpp_standard
  root_dir: data
  compression: c23
  split_mode: official
  split_files:
    train: configs/ffpp_splits/train.json
    val: configs/ffpp_splits/val.json
    test: configs/ffpp_splits/test.json
  manipulation_methods:
    - Deepfakes
    - Face2Face
    - FaceSwap
    - NeuralTextures
  frame_extraction:
    output_dir: data/frames_ffpp_standard
```

From `configs/train_config.yaml`:

```yaml
data:
  config: configs/dataset_config.yaml
  frames_per_clip: 1
  train_items_per_clip: 8
  image_size: 380
  methods:
    - Deepfakes
    - Face2Face
    - FaceSwap
    - NeuralTextures
  real_dir: real
```

Therefore the active default training set is:

```text
data/frames_ffpp_standard/
├── real/
├── Deepfakes/
├── Face2Face/
├── FaceSwap/
└── NeuralTextures/
```

Each subdirectory should contain one directory per video/clip, and each clip directory should contain extracted image frames.

## FaceForensics++ Split Behavior

The default config uses:

```yaml
split_mode: official
```

In this mode, `src/data/dataset.py` loads the official split JSON files and assigns:

- real videos by video ID;
- fake videos by source-target pair IDs.

If the official split files are missing or malformed, the dataloader raises an error. This is intentional because the default profile is meant to reproduce the standard FaceForensics++ split behavior.

The extended dataset config uses:

```yaml
split_mode: numeric
```

That mode falls back to numeric ID ranges and deterministic hashing for actor-style IDs that do not map cleanly to the standard 0-999 YouTube ID ranges.

## Frame-Level Sampling Behavior

`FaceForensicsDataset` samples frames from extracted frame folders.

Training behavior:

- `frames_per_clip` defaults to `1`.
- a frame is randomly sampled for each item;
- `train_items_per_clip=8` repeats each clip entry during training so an epoch can see more independently sampled frames per clip;
- train transforms are stochastic.

Validation/test behavior:

- `evaluation.eval_frames_per_clip` defaults to `1`;
- `evaluation.eval_frame_strategy` defaults to `center`;
- deterministic sampling is enabled by default;
- no weighted sampler is used for validation/test.

If `--video-eval` is used with `train.py --eval-only`, validation/test datasets can return a frame stack shaped like:

```text
(frames_per_clip, C, H, W)
```

The trainer then runs the frames independently and averages their probabilities per clip.

## Class Balance Handling

The default config enables:

```yaml
balance_strategy: weighted_sampler
```

When this is active, `build_dataloaders(...)` computes class weights from the train dataset samples and uses `WeightedRandomSampler` with replacement. This reduces the effect of the typical FaceForensics++ fake-heavy class imbalance during training.

The validation and test loaders preserve their natural class distributions.

## Transform Pipeline

`src/data/transforms.py` defines transforms.

Training transforms:

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

Validation/test transforms:

```text
Resize(image_size)
CenterCrop(image_size)
ToTensor
ImageNet Normalize
```

The current default image size for training is `380`.

## Extended FaceForensics++ Profile

`configs/train_config_extended.yaml` and `configs/dataset_config_extended.yaml` define a broader experiment profile.

Extended fake methods:

```text
Deepfakes
Face2Face
FaceSwap
NeuralTextures
FaceShifter
DeepFakeDetection
```

Extended real sources:

```text
original_sequences/youtube
original_sequences/actors
```

Extended extracted frame output:

```text
data/frames_ffpp_extended
```

Extended training outputs:

```text
outputs/checkpoints_extended
outputs/evaluation_extended
outputs/runs_extended
outputs/training_log_extended.csv
```

Use the extended profile when the extra source folders and methods have been downloaded and extracted.

## Celeb-DF v2 Configuration

Both dataset configs include:

```yaml
celeb_df:
  root_dir: data/Celeb-DF-v2
  real_dirs:
    - Celeb-real
    - YouTube-real
  fake_dir: Celeb-synthesis
  test_list: List_of_testing_videos.txt
  frame_extraction:
    output_dir: data/Celeb-DF-v2/frames
```

This config supports exploration and frame extraction. It does not mean `train.py` trains on Celeb-DF by default.

## Active Celeb-DF Evaluation Path

Celeb-DF v2 evaluation is handled by `evaluate_celeb_df.py`.

Expected local layout:

```text
data/Celeb-DF-v2/
├── Celeb-real/
├── YouTube-real/
├── Celeb-synthesis/
└── List_of_testing_videos.txt
```

The evaluator:

- parses `List_of_testing_videos.txt`;
- converts Celeb-DF labels from `1=real, 0=fake` to project labels `0=real, 1=fake`;
- loads a checkpoint through `DeepFakeDetector`;
- samples frames directly from raw videos;
- writes per-video predictions and metrics.

Output:

```text
outputs/celeb_df_eval/
├── per_video.csv
└── metrics.json
```

## Celeb-DF Download Helper

`scripts/download_celeb_df_test.py` is a helper for gated Celeb-DF access. It cannot bypass dataset access requirements.

It can:

- verify and count entries in `List_of_testing_videos.txt`;
- use authorized Google Drive folder URLs with `gdown`;
- try to download only videos referenced by the test list;
- fall back to full-folder download plus pruning.

For offline/manual download, place the dataset under `data/Celeb-DF-v2/` and use:

```bash
python scripts/download_celeb_df_test.py --out-dir data/Celeb-DF-v2 --list-only
```

as a parser/check step.

## Dummy Data Mode

`train.py --dummy` bypasses real dataset loading and uses `DummyFaceForensicsDataset`.

Dummy sizes:

```text
train: 512
val:   128
test:  128
```

This is useful for validating the training loop, model construction, logging, and checkpoint mechanics without dataset I/O. It is not meaningful for model quality.

## Practical Status

Current dataset support by workflow:

| Workflow | Status | Entrypoint |
|---|---:|---|
| FaceForensics++ download | Implemented | `scripts/download_FaceForensicsPP.py` |
| FaceForensics++ frame extraction | Implemented | `scripts/extract_frames.py` |
| FaceForensics++ training | Implemented | `train.py` |
| FaceForensics++ test evaluation | Implemented | `train.py --eval-only` |
| Celeb-DF v2 frame extraction | Configured | `scripts/extract_frames.py --dataset celeb-df` |
| Celeb-DF v2 test evaluation | Implemented | `evaluate_celeb_df.py` |
| Mixed FF++ + Celeb-DF training | Not implemented | N/A |

## Summary

With the default config, `train.py` trains on the four-method FaceForensics++ standard extracted-frame dataset using official splits. Celeb-DF v2 is available as a configured dataset target and a dedicated cross-dataset evaluator, but it is not part of the default training dataloader.
