# Dataset Configuration Report

## High-Level Summary

This repository defines a broader deepfake data setup than the trainer currently uses.

- The broader dataset definition lives in `configs/dataset_config.yaml` and includes both FaceForensics++ and Celeb-DF v2.
- The current `train.py` pipeline does **not** build a mixed FaceForensics++ plus Celeb-DF training set.
- With `configs/train_config.yaml`, `train.py` builds dataloaders only from **FaceForensics++ extracted frames** under `data/frames_ffpp_standard`.
- The active training subset uses:
  - `real`
  - `Deepfakes`
  - `Face2Face`
  - `FaceSwap`
  - `NeuralTextures`
- Split assignment uses **official FF++ pair-based JSON files** (`configs/ffpp_splits/{train,val,test}.json`) rather than numeric index ranges.
- A broader extended setup also exists in `configs/train_config_extended.yaml` which includes all six methods: `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `FaceShifter`, `DeepFakeDetection`.

In short, the repo is configured around a **larger dataset universe**, but the **current trainer run path is a FaceForensics++-only subset** using the four standard FF++ c23 methods.

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
5. The dataset loader reads from the extracted frame directory defined at `faceforensics.frame_extraction.output_dir`, which is currently `data/frames_ffpp_standard`.

That means `dataset_config.yaml` contains more than the current trainer actually consumes. The FaceForensics++ section is active for training; the Celeb-DF section is currently only a repository-level dataset definition and preprocessing target.

## Complete Larger Dataset Defined in the Repo

The complete larger dataset definition in `configs/dataset_config.yaml` covers two benchmarks.

### 1. FaceForensics++

Configured FaceForensics++ settings:

- root directory: `data`
- compression level: `c23`
- split mode: `official` — uses official pair-based JSON files in `configs/ffpp_splits/`
- manipulation methods:
  - `Deepfakes`
  - `Face2Face`
  - `FaceSwap`
  - `NeuralTextures`
- real sources:
  - `original_sequences/youtube`
- extracted frame output: `data/frames_ffpp_standard`
- extraction cap: `32` frames per video

Configured split ranges (used as fallback for numeric-mode):

- train: video indices `[0, 720)`
- val: video indices `[720, 860)`
- test: video indices `[860, 1000)`

Split assignment detail:

- The active `split_mode: official` uses the official FF++ pair JSON files. Each clip pair is assigned to exactly one split based on the pair membership tables.
- The numeric fallback (hash-based for actor IDs, index-range for youtube IDs) is available when `split_mode: numeric`.

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
- extracted frame root: `data/frames_ffpp_standard`
- image size: `224`
- `frames_per_clip: 1` — one frame sampled per training item (averaging multiple frames washes artifacts)
- `train_items_per_clip: 8` — each clip directory is repeated 8× in the training sample list, so the model sees 8 independently-sampled random frames per video per epoch
- real class directory: `real`
- fake methods:
  - `Deepfakes`
  - `Face2Face`
  - `FaceSwap`
  - `NeuralTextures`
- `balance_strategy: weighted_sampler` — enforces 50/50 class balance per mini-batch

The loader is clip-based, not frame-list-based:

- each clip is stored as its own directory of extracted images
- each dataset item corresponds to one (clip, frame) sample
- training dataset length = number of usable clip directories × `train_items_per_clip`
- validation and test datasets are not expanded; each clip appears once with a fixed center frame

This means training dataset length is measured in **expanded sample entries** (clips × 8), while val/test length is in **usable clip directories**.

## Effective Current Dataset Size

The exact usable clip counts can be obtained by running `scripts/audit_faceforensics.py` against `data/frames_ffpp_standard`.

The current config uses four standard FF++ c23 methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures) with official split files. Based on the standard FF++ c23 distribution:

- train split is approximately 720 source videos × 4 methods + corresponding real clips
- val split is approximately 140 source videos × 4 methods + real
- test split is approximately 140 source videos × 4 methods + real

Training effective dataset size = clip count × `train_items_per_clip` (8), giving approximately 28,800 training items per epoch for a typical 3,600-clip raw training set.

Note: the older dataset analysis in `docs/training_run_summary.md` was based on a different configuration using `FaceShifter` and `DeepFakeDetection` methods from `data/frames`. Those numbers no longer apply to the current setup.

## Class Balance of the Current Training Dataset

With four fake methods and one real class, the raw clip ratio is approximately 4:1 fake-heavy.

The training pipeline mitigates this with:

- `balance_strategy: weighted_sampler` — `WeightedRandomSampler` enforces 50/50 real/fake per mini-batch
- `pos_weight: 1.3` — applies a 1.3× gradient multiplier to the fake class in the BCE loss, nudging output probabilities higher and keeping the calibrated threshold near 0.5

Val and test loaders use the natural imbalance. AUC is therefore the primary headline metric; accuracy and F1 should be interpreted relative to the optimized threshold reported alongside them.

## Extracted Frame Scale on Disk

The extracted frames live in `data/frames_ffpp_standard/`. Each method subdirectory has one folder per source video, containing up to 32 uniformly-sampled JPEG frames.

Run `scripts/audit_faceforensics.py` against `data/frames_ffpp_standard` to get current clip and frame counts.

Important note:

- The filesystem may contain hidden directories (e.g., `._033_097` from macOS Finder).
- The loader and audit logic skip entries whose names start with `.`.

## Why Some Clips Are Excluded

With `split_mode: official`, only clips whose pair ID appears in `configs/ffpp_splits/train.json`, `val.json`, or `test.json` are assigned to a split. Clips absent from all three files are excluded.

This is the safest policy: it strictly follows the official benchmark split rather than inferring membership from numeric indices.

## Broader Extended Dataset Config

The repository also includes `configs/train_config_extended.yaml`, which trains on all six available FF++ methods:

- `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `FaceShifter`, `DeepFakeDetection`

This uses `configs/dataset_config_extended.yaml` and writes to separate output directories (`outputs/checkpoints_extended`, etc.). Use it for broader cross-method generalization experiments.

## Practical Interpretation

There are three different dataset scopes in this repo:

### 1. Full repository dataset scope

- FaceForensics++ (four standard c23 methods by default, six with extended config)
- Celeb-DF v2 (configured for preprocessing, not wired into training)
- In-memory dummy data for smoke tests (`--dummy` flag)

### 2. Current default training dataset

- FaceForensics++ only
- extracted frames in `data/frames_ffpp_standard`
- real plus four fake methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
- official pair-based split assignment

### 3. Extended training dataset

- FaceForensics++ with all six available methods
- separate outputs and config file (`configs/train_config_extended.yaml`)

If the question is "what does the repo support?", the answer is the larger multi-dataset definition across both dataset configs.

If the question is "what does `train.py` train on right now with the default config?", the answer is the four-method FaceForensics++ extracted-frame subset using the official splits.

## Key Takeaways

- The active training pipeline uses FaceForensics++ c23 with official pair-based splits.
- Active fake methods: Deepfakes, Face2Face, FaceSwap, NeuralTextures.
- Frames extracted to `data/frames_ffpp_standard`; each clip has up to 32 frames.
- `train_items_per_clip: 8` multiplies training dataset size 8× by drawing independent random frames per pass.
- `WeightedRandomSampler` + `pos_weight: 1.3` address class imbalance.
- Celeb-DF v2 is configured for preprocessing but not included in the default training path.

## Source Basis for This Report

This report was derived from:

- `train.py`
- `configs/train_config.yaml`
- `configs/train_config_extended.yaml`
- `configs/dataset_config.yaml`
- `configs/ffpp_splits/train.json`, `val.json`, `test.json`
- `src/data/dataset.py`
- `scripts/extract_frames.py`
- `scripts/audit_faceforensics.py`
- the current local `data/frames_ffpp_standard` directory
