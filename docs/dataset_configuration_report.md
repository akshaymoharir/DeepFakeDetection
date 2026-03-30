# Dataset Configuration Report

## High-Level Summary

This repository defines a broader deepfake data setup than the trainer currently uses.

- The broader dataset definition lives in `configs/dataset_config.yaml` and includes both FaceForensics++ and Celeb-DF v2.
- The current `train.py` pipeline does **not** build a mixed FaceForensics++ plus Celeb-DF training set.
- With `configs/train_config.yaml`, `train.py` builds dataloaders only from **FaceForensics++ extracted frames** under `data/frames`.
- The active training subset uses:
  - `real`
  - `Deepfakes`
  - `Face2Face`
  - `FaceShifter`
  - `DeepFakeDetection`
- The effective split sizes for the current full training config are:
  - train: `4,703`
  - val: `820`
  - test: `904`
- A smaller ablation setup also exists in `configs/train_config_ablation.yaml`. Its effective sizes are:
  - train: `192`
  - val: `48`
  - test: `48`

In short, the repo is configured around a **larger dataset universe**, but the **current trainer run path is a smaller FaceForensics++-only subset** of that universe.

## Files That Control the Dataset

The dataset behavior is spread across four main places:

- `train.py`
- `configs/train_config.yaml`
- `configs/dataset_config.yaml`
- `src/data/dataset.py`

The flow is:

1. `train.py` loads `configs/train_config.yaml`.
2. That config points to `configs/dataset_config.yaml` through `data.config`.
3. `train.py` then calls `build_dataloaders(...)` from `src/data/dataset.py`.
4. `build_dataloaders(...)` only instantiates `FaceForensicsDataset` for real training.
5. The dataset loader reads from the extracted frame directory defined at `faceforensics.frame_extraction.output_dir`, which is currently `data/frames`.

That means `dataset_config.yaml` contains more than the current trainer actually consumes. The FaceForensics++ section is active for training; the Celeb-DF section is currently only a repository-level dataset definition and preprocessing target.

## Complete Larger Dataset Defined in the Repo

The complete larger dataset definition in `configs/dataset_config.yaml` covers two benchmarks.

### 1. FaceForensics++

Configured FaceForensics++ settings:

- root directory: `data`
- compression level: `c23`
- manipulation methods:
  - `Deepfakes`
  - `Face2Face`
  - `FaceShifter`
  - `DeepFakeDetection`
- real sources:
  - `original_sequences/youtube`
  - `original_sequences/actors`
- extracted frame output: `data/frames`
- extraction cap: `32` frames per video

Configured split ranges:

- train: video indices `[0, 720)`
- val: video indices `[720, 860)`
- test: video indices `[860, 1000)`

Important detail:

- YouTube-style clips with IDs in the standard FF++ `0..999` range are assigned by the explicit index ranges above.
- Actor-style names, especially from `DeepFakeDetection` and actor-source real clips, use a deterministic hash split because their IDs do not naturally map into the standard FF++ numeric ranges.

### 2. Celeb-DF v2

Configured Celeb-DF v2 settings:

- root directory: `data/Celeb-DF-v2`
- real directories:
  - `Celeb-real`
  - `YouTube-real`
- fake directory: `Celeb-synthesis`
- test list: `List_of_testing_videos.txt`
- extracted frame output: `data/Celeb-DF-v2/frames`
- extraction cap: `32` frames per video

### What "Complete Larger Dataset" Means Here

At the repository level, the project is designed to support:

- FaceForensics++ preprocessing and training
- Celeb-DF preprocessing and exploration
- dummy data smoke tests

But the current trainer implementation does not merge these into a single training dataset. The larger dataset is therefore best understood as the **full data scope the repo knows about**, not the exact dataset `train.py` currently trains on.

## Smaller Currently Configured Dataset Used by `train.py`

When `train.py` runs with the default config `configs/train_config.yaml`, it uses:

- dataset config: `configs/dataset_config.yaml`
- extracted frame root: `data/frames`
- image size: `224`
- frames sampled per clip per item: `8`
- real class directory: `real`
- fake methods:
  - `Deepfakes`
  - `Face2Face`
  - `FaceShifter`
  - `DeepFakeDetection`

The loader is clip-based, not frame-list-based:

- each clip is stored as its own directory of extracted images
- each dataset item corresponds to one clip
- on each access, the loader randomly samples up to `8` frames from that clip
- those sampled frames are averaged into one tensor before being returned

This means dataset length is measured in **usable clip directories**, not in extracted frame count.

## Effective Current Dataset Size

Using the local extracted frames and the repository audit script, the current full training config resolves to the following usable clip counts.

### Train split

- real: `955`
- Deepfakes: `635`
- Face2Face: `635`
- FaceShifter: `275`
- DeepFakeDetection: `2,203`
- total: `4,703`

### Validation split

- real: `192`
- Deepfakes: `89`
- Face2Face: `89`
- FaceShifter: `34`
- DeepFakeDetection: `416`
- total: `820`

### Test split

- real: `216`
- Deepfakes: `96`
- Face2Face: `96`
- FaceShifter: `47`
- DeepFakeDetection: `449`
- total: `904`

These totals exactly match the currently observed training run summary in `docs/training_run_summary.md`.

## Class Balance of the Current Training Dataset

The current `train_config.yaml` dataset is fake-heavy.

Per split:

- train: `955` real vs `3,748` fake, about `3.92:1`
- val: `192` real vs `628` fake, about `3.27:1`
- test: `216` real vs `688` fake, about `3.19:1`

The biggest reason is that `DeepFakeDetection` contributes many more clips than the other fake methods. It is the single largest fake source in every split.

This matters because:

- accuracy and F1 can look better than the model calibration really is
- AUC is a safer headline metric
- per-class precision and recall would be useful for reporting

## Extracted Frame Scale on Disk

Looking at the current `data/frames` directory, the extracted dataset on disk is large enough to support repeated frame sampling without re-decoding videos.

Raw frame-folder counts observed locally:

- `real`: `1,365` clip folders, `43,616` frames
- `Deepfakes`: `1,013` clip folders, `32,000` frames
- `Face2Face`: `1,000` clip folders, `32,000` frames
- `FaceShifter`: `426` clip folders, `13,632` frames
- `DeepFakeDetection`: `3,068` clip folders, `97,838` frames

Average frames per clip are all about `32`, which matches the extraction cap in `dataset_config.yaml`.

Important note:

- The audited usable counts are slightly lower than some raw on-disk counts because the filesystem contains a few hidden directories such as `._03__hugging_happy` and `._033_097`.
- The loader and audit logic ignore hidden directories, so those do not affect training.

## Why Some Clips Are Excluded

Not every extracted fake clip ends up in train, validation, or test.

From the audit:

- `Deepfakes`: `180` clips unassigned
- `Face2Face`: `180` clips unassigned
- `FaceShifter`: `70` clips unassigned

These are mostly paired-ID clip names that do not map cleanly into a single split under the repository’s split inference logic. The loader drops them rather than risking split leakage.

That is a good safety choice, because it avoids ambiguous train/val/test membership.

## Smaller Compact Dataset for Quick Experiments

The repository also includes `configs/train_config_ablation.yaml`, which defines a much smaller, intentionally balanced subset for quick studies.

It changes the active dataset in three main ways:

- it keeps only two fake methods:
  - `Deepfakes`
  - `Face2Face`
- it reduces `frames_per_clip` from `8` to `4`
- it applies deterministic per-split clip caps

Subset caps:

- real videos per split:
  - train: `64`
  - val: `16`
  - test: `16`
- fake videos per method per split:
  - train: `64`
  - val: `16`
  - test: `16`

Effective ablation dataset sizes:

- train: `64 + 64 + 64 = 192`
- val: `16 + 16 + 16 = 48`
- test: `16 + 16 + 16 = 48`

This subset is much smaller than the default training setup, but it is easier to iterate on and much less dominated by `DeepFakeDetection`.

## Practical Interpretation

There are really three different dataset scopes in this repo:

### 1. Full repository dataset scope

- FaceForensics++
- Celeb-DF v2
- dummy synthetic data

### 2. Current default training dataset

- FaceForensics++ only
- extracted frames in `data/frames`
- real plus four fake methods
- effective split sizes: `4,703 / 820 / 904`

### 3. Compact ablation dataset

- FaceForensics++ only
- real plus two fake methods
- deterministic subset caps
- effective split sizes: `192 / 48 / 48`

If the question is "what does the repo support?", the answer is the larger multi-dataset definition in `dataset_config.yaml`.

If the question is "what does `train.py` train on right now with the default config?", the answer is the smaller FaceForensics++ extracted-frame subset described by `train_config.yaml`.

## Key Takeaways

- The active training pipeline is narrower than the full repository dataset definition.
- `train.py` currently trains only on FaceForensics++ extracted frames.
- Celeb-DF v2 is configured in the repo, but not wired into the default dataloader path used by `train.py`.
- The current default dataset is moderately large in clip count but strongly fake-heavy.
- `DeepFakeDetection` is the dominant fake source in the default training setup.
- The ablation config provides a much smaller and more controlled subset for quick experiments.

## Source Basis for This Report

This report was derived from:

- `train.py`
- `configs/train_config.yaml`
- `configs/train_config_ablation.yaml`
- `configs/dataset_config.yaml`
- `src/data/dataset.py`
- `scripts/extract_frames.py`
- `scripts/audit_faceforensics.py`
- `docs/training_run_summary.md`
- the current local `data/frames` directory
