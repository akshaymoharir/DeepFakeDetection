# DeepFakeDetection Technical Reference

## Scope

This document provides a detailed technical description of the repository as currently implemented in the tracked workspace. It covers repository goals, source layout, model architecture, dataset configuration, training flow, optimization setup, validation logic, evaluation metrics, container environment, and implementation caveats.

Where the code indicates intended behavior but the corresponding module is not present in the visible source tree, that is called out explicitly.

## 1. Repository Goal

The repository is intended to support deepfake detection experiments from end to end:

- dataset download and preparation,
- frame extraction from source videos,
- configurable training and validation,
- checkpointed experimentation,
- final test evaluation,
- compact ablation studies.

The project frames deepfake detection as a binary classification problem:

- `0` for real content
- `1` for fake content

The current model family is centered on `HSF-CVIT`, a hybrid architecture that combines a CNN-based spatial branch, an SRM-based forensic frequency branch, and an attention-based fusion head. This makes the system more sophisticated than a plain baseline CNN while remaining modular enough for beginner experimentation and comparison-driven research.

## 2. Source Layout

Tracked files relevant to the implemented training system include:

- [`train.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/train.py)
- [`configs/train_config.yaml`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/configs/train_config.yaml)
- [`configs/train_config_ablation.yaml`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/configs/train_config_ablation.yaml)
- [`configs/dataset_config.yaml`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/configs/dataset_config.yaml)
- [`configs/dataset_config_dummy.yaml`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/configs/dataset_config_dummy.yaml)
- [`src/models/hsf_cvit.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/models/hsf_cvit.py)
- [`src/models/efficientnet_branch.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/models/efficientnet_branch.py)
- [`src/models/srm_filter.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/models/srm_filter.py)
- [`src/models/cross_attention_vit.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/models/cross_attention_vit.py)
- [`src/training/trainer.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/training/trainer.py)
- [`src/training/losses.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/training/losses.py)
- [`src/training/metrics.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/training/metrics.py)
- [`src/utils/helpers.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/utils/helpers.py)
- [`scripts/extract_frames.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/scripts/extract_frames.py)
- [`scripts/download_FaceForensicsPP.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/scripts/download_FaceForensicsPP.py)
- [`scripts/audit_faceforensics.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/scripts/audit_faceforensics.py)
- [`scripts/create_dummy_datasets.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/scripts/create_dummy_datasets.py)

Logical areas:

- `configs/` holds experiment definitions.
- `scripts/` holds data preparation, auditing, and test-data generation utilities.
- `src/models/` holds the detection model and submodules.
- `src/training/` holds optimization, metrics, and checkpoint logic.
- `src/utils/` holds shared utilities such as seeding, logging, and frame extraction helpers.

## 3. Main Execution Flow

The central training entrypoint is [`train.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/train.py).

The flow implemented there is:

1. Parse CLI arguments.
2. Load training config YAML.
3. Apply CLI overrides.
4. Resolve and load the dataset config referenced from the training config.
5. Seed random generators.
6. Choose execution device.
7. Build train, validation, and test data loaders.
8. Build the model.
9. Build the trainer.
10. Optionally resume from checkpoint.
11. Either run evaluation only or perform full training.
12. Reload the best checkpoint if available.
13. Evaluate on the test set and print final metrics.

### 3.1 Supported CLI features

The entrypoint supports:

- `--config` to select a training config file
- `--dummy` to use the dummy dataset mode
- `--epochs`, `--batch-size`, `--lr`, `--workers` as CLI overrides
- `--no-amp` to disable mixed precision
- `--methods` to override FF++ manipulation methods
- `--real-dir` to override the real folder name
- `--resume` to continue from a saved checkpoint
- `--eval-only` to evaluate without training
- `--device` to force a specific compute device

### 3.2 Current implementation caveat

`train.py` imports:

- `from src.data.dataset import build_dataloaders`

However, no tracked `src/data` package is present in the visible repository tree I inspected. That means the training entrypoint assumes a data loader implementation that is not included in the tracked files visible here.

This affects interpretation of the codebase:

- the model, trainer, metrics, loss, configs, and dataset-preparation scripts are present,
- the repo clearly intends to support full training,
- but the visible workspace snapshot is incomplete with respect to data loading.

For documentation purposes, the training design can still be described accurately, but this missing module should be treated as an implementation gap unless it exists outside the tracked files.

## 4. Configuration System

The project uses YAML files for both training and dataset definitions.

### 4.1 Primary training config

[`configs/train_config.yaml`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/configs/train_config.yaml) defines:

- model settings
- data settings
- training settings
- checkpoint settings
- logging settings
- reproducibility seed

#### Model fields

- `name: hsf_cvit`
- `pretrained_spatial: true`
- `freeze_spatial_epochs: 2`
- `spatial_out_dim: 512`
- `freq_out_dim: 256`
- `fusion_heads: 4`
- `fusion_dim: 256`
- `dropout: 0.3`

#### Data fields

- `config: configs/dataset_config.yaml`
- `frames_per_clip: 8`
- `image_size: 224`
- `num_workers: 4`
- `pin_memory: true`
- selected methods:
  - `Deepfakes`
  - `Face2Face`
  - `FaceShifter`
  - `DeepFakeDetection`
- `real_dir: real`

#### Training fields

- `epochs: 50`
- `batch_size: 16`
- `optimizer: adamw`
- `lr: 3.0e-4`
- `weight_decay: 1.0e-4`
- `warmup_epochs: 5`
- `lr_schedule: cosine`
- `gradient_clip: 1.0`
- `label_smoothing: 0.1`
- `amp: true`

#### Checkpoint fields

- `dir: outputs/checkpoints`
- `save_every: 5`
- `early_stop_patience: 10`
- `keep_last: 3`

#### Logging fields

- `tensorboard_dir: outputs/runs`
- `log_interval: 10`
- `csv_log: outputs/training_log.csv`

#### Reproducibility

- `seed: 42`

### 4.2 Ablation config

[`configs/train_config_ablation.yaml`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/configs/train_config_ablation.yaml) defines a smaller experiment preset intended for quick comparison runs.

Differences from the main training config include:

- fewer frames per clip: `4`
- fewer fake methods: `Deepfakes`, `Face2Face`
- shorter training: `15` epochs
- smaller batch size: `8`
- shorter warmup: `2` epochs
- reduced patience and checkpoint retention
- explicit subset caps for small balanced studies

The ablation section includes:

- `max_real_videos_per_split`
- `max_fake_videos_per_method_per_split`
- `subset_seed`

This suggests the intended data loader supports deterministic subsetting for fast experiments.

### 4.3 Dataset config

[`configs/dataset_config.yaml`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/configs/dataset_config.yaml) defines dataset source paths and split metadata.

#### FaceForensics++ fields

- `root_dir: data`
- `compression: c23`
- manipulation methods:
  - `Deepfakes`
  - `Face2Face`
  - `FaceShifter`
  - `DeepFakeDetection`
- `original_dir: original_sequences/youtube`
- `original_dirs` can include:
  - `original_sequences/youtube`
  - `original_sequences/actors`
- `manipulated_dir: manipulated_sequences`
- frame extraction output: `data/frames`
- split ranges:
  - train: `[0, 720]`
  - val: `[720, 860]`
  - test: `[860, 1000]`

#### Celeb-DF fields

- `root_dir: data/Celeb-DF-v2`
- real dirs:
  - `Celeb-real`
  - `YouTube-real`
- fake dir: `Celeb-synthesis`
- test list: `List_of_testing_videos.txt`
- frame extraction output: `data/Celeb-DF-v2/frames`

#### Common fields

- `image_size: 224`
- `face_crop: true`
- `random_seed: 42`
- `num_workers: 4`

### 4.4 Dummy dataset config

[`configs/dataset_config_dummy.yaml`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/configs/dataset_config_dummy.yaml) points to synthetic data folders for smoke tests. It reduces split sizes and disables face cropping.

This config supports testing the training and analysis workflow without large real datasets.

## 5. Dataset and Data Preparation Workflow

The repository includes utilities that operate before model training.

### 5.1 FaceForensics++ download

[`scripts/download_FaceForensicsPP.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/scripts/download_FaceForensicsPP.py) is based on the official FaceForensics download logic.

Capabilities include:

- downloading original or manipulated sequences,
- choosing compression level,
- selecting videos, masks, or Deepfakes models,
- limiting the number of downloaded videos,
- selecting among multiple mirror servers.

The script requires user acknowledgment of the FaceForensics terms of use before continuing.

### 5.2 Frame extraction

[`scripts/extract_frames.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/scripts/extract_frames.py) converts source videos into frame folders organized by class and clip.

Key behavior:

- uniformly samples up to `max_frames_per_video`,
- resizes frames to square resolution, default `224`,
- saves JPEG images per clip,
- supports both FaceForensics++ and Celeb-DF workflows,
- allows method filtering and real-directory overrides.

For FaceForensics++, extracted frames are merged into a class-oriented output structure such as:

- `data/frames/real/<video>/frame_000.jpg`
- `data/frames/Deepfakes/<video>/frame_000.jpg`

This output is intended to be consumed by a data loader for per-frame or per-clip training.

### 5.3 Frame extraction backend logic

Frame decoding is handled by [`src/utils/helpers.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/utils/helpers.py), which tries multiple backends in order:

1. OpenCV
2. Decord
3. PyAV

Frames are sampled uniformly across the video and optionally resized. This multi-backend fallback design improves portability across environments with different codec support.

### 5.4 Audit workflow

[`scripts/audit_faceforensics.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/scripts/audit_faceforensics.py) validates extracted FF++ frame folders before ablation runs.

It reports:

- number of clip directories,
- non-empty versus empty clip folders,
- usable counts per split,
- unassigned clip folders,
- ablation subset limits from the train config.

The script uses clip naming conventions and split ranges to infer train, validation, and test assignment. This is valuable for catching incomplete extraction runs and invalid split layouts before training starts.

### 5.5 Dummy data generation

[`scripts/create_dummy_datasets.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/scripts/create_dummy_datasets.py) creates synthetic versions of FaceForensics++ and Celeb-DF-style frame layouts.

The generator produces:

- simple face-like real frames,
- fake frames with hand-crafted perturbations meant to mimic class differences,
- class- and clip-structured directories matching expected downstream usage.

This is especially useful for:

- quick code-path tests,
- debugging I/O and data structure assumptions,
- class demos or beginner experimentation before real data is available.

## 6. Model Architecture

The top-level model is implemented in [`src/models/hsf_cvit.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/models/hsf_cvit.py).

It composes three main parts:

1. `EfficientNetSpatialBranch`
2. `SRMFrequencyBranch`
3. `CrossAttentionViT`

Input shape assumed by the model:

- `(B, 3, H, W)`, typically `(B, 3, 224, 224)`

Output:

- raw logits of shape `(B, 1)`

The final sigmoid is intentionally not applied inside the model because training uses `BCEWithLogitsLoss` logic.

### 6.1 Spatial branch

Implemented in [`src/models/efficientnet_branch.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/models/efficientnet_branch.py).

Design:

- uses `timm.create_model("efficientnet_b4", pretrained=..., num_classes=0, global_pool="avg")`
- obtains a pooled backbone feature vector of size `1792`
- applies:
  - dropout
  - linear projection to `spatial_out_dim`
  - layer normalization

Default projected output dimension:

- `512`

Behavioral notes:

- pretrained ImageNet weights are enabled by default,
- the backbone can be frozen and unfrozen during warmup,
- the branch raises an import error if `timm` is unavailable.

This branch captures semantic and texture-level visual information from RGB input.

### 6.2 Frequency branch

Implemented in [`src/models/srm_filter.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/models/srm_filter.py).

This branch is designed for forensic signal extraction rather than natural-image semantics.

#### Fixed SRM front-end

The model builds three fixed 5x5 SRM high-pass kernels and applies them independently to each RGB channel, resulting in:

- input channels: `3`
- kernels per channel: `3`
- SRM output channels: `9`

The SRM filters are registered as buffers, not trainable parameters.

After filtering, the output is clamped through:

- `torch.tanh(noise * 10.0)`

This matches standard SRM-style preprocessing behavior used in deepfake forensic pipelines.

#### Trainable encoder

After SRM filtering, the branch applies:

- `Conv2d(9, 64, 3x3)`
- `BatchNorm2d`
- `ReLU`
- residual block `64 -> 128`, stride `2`
- residual block `128 -> 256`, stride `2`
- residual block `256 -> 256`, stride `2`
- `AdaptiveAvgPool2d(1)`
- flatten
- `Linear(256, out_dim)`
- `LayerNorm(out_dim)`

Default output dimension:

- `256`

This branch is intended to detect manipulation artifacts that survive in high-frequency residual space.

### 6.3 Fusion head

Implemented in [`src/models/cross_attention_vit.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/models/cross_attention_vit.py).

This module provides the transformer-like part of the architecture.

#### Inputs

- spatial feature vector `(B, spatial_dim)`
- frequency feature vector `(B, freq_dim)`

#### Internal flow

1. Project both features into `fusion_dim`.
2. Add learned positional embeddings to each token.
3. Use spatial token as query.
4. Use frequency token as key and value.
5. Apply multi-head cross-attention.
6. Apply residual connection and layer norm.
7. Apply MLP block with GELU and dropout.
8. Concatenate spatial and frequency tokens.
9. Apply multi-head self-attention over the two-token pair.
10. Use the first token as the final representation.
11. Apply dropout and a linear classifier to output one logit.

Default fusion settings:

- `fusion_dim: 256`
- `num_heads: 4`

This is why the repo can be described as transformer-based in a hybrid sense: the fusion mechanism uses attention blocks rather than pure MLP concatenation, but the system still relies on CNN and SRM feature extractors upstream.

### 6.4 Parameter management helpers

The top-level model exposes:

- `freeze_spatial()`
- `unfreeze_spatial()`
- `count_parameters()`

These helper methods support trainer-controlled warmup and printed parameter summaries.

## 7. Loss Function

The loss implementation is in [`src/training/losses.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/training/losses.py).

### 7.1 Base loss

The project uses binary cross-entropy with logits via:

- `F.binary_cross_entropy_with_logits`

The model therefore returns logits, not probabilities.

### 7.2 Label smoothing

The repo implements a custom `SmoothedBCELoss`.

With smoothing coefficient `epsilon`, target mapping is:

- real `0` becomes `epsilon / 2`
- fake `1` becomes `1 - epsilon / 2`

Default smoothing:

- `0.1`

This acts as a regularizer and can reduce overconfident predictions on noisy or imperfectly labeled deepfake datasets.

### 7.3 Class weighting

The loss class supports `pos_weight`, but the current `build_criterion` helper does not provide one from config. As implemented, the default training path uses only label smoothing and does not explicitly rebalance for class imbalance.

## 8. Training Engine

The training logic is implemented in [`src/training/trainer.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/training/trainer.py).

### 8.1 Optimizers

Supported optimizers:

- `adamw`
- `adam`
- `sgd`

Default:

- `AdamW`

SGD uses:

- momentum `0.9`

Common optimizer hyperparameters:

- `lr`
- `weight_decay`

### 8.2 Learning-rate schedules

Supported schedules:

- `cosine`
- `step`
- `plateau`

Behavior:

- `cosine` uses `CosineAnnealingLR`
- `step` uses `StepLR(step_size=10, gamma=0.5)`
- `plateau` uses `ReduceLROnPlateau(mode="max", patience=5, factor=0.5)`

When warmup is enabled and the schedule is not plateau, the trainer composes:

- `LinearLR(start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)`
- followed by the main scheduler using `SequentialLR`

Default warmup:

- `5` epochs in the main config
- `2` epochs in the ablation config

### 8.3 Mixed precision

Mixed precision is supported through:

- `torch.amp.autocast`
- `torch.amp.GradScaler`

AMP is enabled only if:

- config enables it, and
- the device type is CUDA

If CUDA is unavailable, `train.py` disables AMP automatically and prints a warning.

### 8.4 Gradient clipping

After unscaling gradients, the trainer applies:

- `nn.utils.clip_grad_norm_(..., self.grad_clip)`

Default:

- `1.0`

### 8.5 Backbone freezing warmup

The trainer freezes the EfficientNet backbone for the first `freeze_spatial_epochs` epochs, then unfreezes it.

Default:

- `2` epochs

Rationale:

- let the frequency branch and fusion head adapt first,
- reduce instability when jointly fine-tuning a pretrained spatial backbone from the first batch.

### 8.6 Epoch flow

For each epoch:

1. Freeze or unfreeze spatial backbone depending on epoch index.
2. Run `_train_epoch`.
3. Run `_val_epoch`.
4. Step the scheduler.
5. Print summary.
6. Log to TensorBoard.
7. Log to CSV.
8. Save checkpoints.
9. Update early stopping state.

### 8.7 Checkpointing

Checkpoint contents:

- epoch number
- model state dict
- optimizer state dict
- scheduler state dict
- scaler state dict
- best validation AUC
- global step

Checkpoint policy:

- save `best.pt` whenever validation ROC-AUC improves
- save periodic checkpoints every `save_every` epochs
- prune old periodic checkpoints to keep only the latest `keep_last`

Default main-config policy:

- periodic save every `5` epochs
- keep `3` periodic checkpoints

### 8.8 Resume behavior

`load_checkpoint()` restores:

- model
- optimizer
- scheduler
- scaler
- best validation AUC
- global step

and returns the next epoch index to resume from.

### 8.9 Early stopping

Early stopping is based on:

- validation ROC-AUC

The trainer increments a no-improvement counter whenever validation AUC does not beat the best value so far. Training stops when the counter reaches `early_stop_patience`.

Default patience:

- `10` in the main config
- `5` in the ablation config

## 9. Validation and Evaluation Metrics

Metric logic is in [`src/training/metrics.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/training/metrics.py).

### 9.1 Probability conversion

At epoch end, logits are:

- detached to CPU,
- clamped to `[-50, 50]`,
- passed through sigmoid,
- accumulated into arrays.

The clamping reduces the chance of metric instability from extreme early-epoch logits.

### 9.2 Metrics computed

The repository computes:

- `roc_auc`
- `f1`
- `precision`
- `recall`
- `accuracy`

Threshold-dependent metrics use:

- threshold `0.5`

### 9.3 Primary model-selection metric

The primary metric for:

- best checkpoint saving,
- plateau schedule stepping,
- early stopping,

is:

- `ROC-AUC`

This is appropriate for deepfake detection because it is less sensitive to threshold choice than raw accuracy and usually more informative when operating thresholds may vary.

### 9.4 Test evaluation

`Trainer.evaluate()`:

- runs the model in eval mode,
- computes mean loss across test batches,
- computes the same classification metrics,
- prints the results,
- returns them as a dictionary.

At the end of a training run, `train.py` attempts to load:

- `outputs/checkpoints/best.pt`

before final test evaluation. If not found, it evaluates using the most recent in-memory weights.

## 10. Logging and Reproducibility

### 10.1 TensorBoard

The trainer logs:

- train loss
- validation loss
- learning rate
- all train metrics
- all validation metrics
- batch loss at configured intervals

Default TensorBoard directory:

- `outputs/runs`

### 10.2 CSV logging

Each epoch appends a row containing:

- epoch
- learning rate
- train loss
- val loss
- train metrics
- val metrics

Default CSV path:

- `outputs/training_log.csv`

### 10.3 Console logging

The trainer prints per-epoch summaries with:

- epoch index
- wall-clock time
- learning rate
- train loss and AUC
- validation loss, AUC, and F1

The utility function `setup_script_logging()` in [`src/utils/helpers.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/src/utils/helpers.py) also mirrors stdout and stderr to timestamped log files under `outputs/logs`.

### 10.4 Seeding

`seed_everything()` sets:

- Python random seed
- NumPy seed
- `PYTHONHASHSEED`
- PyTorch CPU and CUDA seeds

It also sets:

- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

This favors reproducibility over maximum performance.

## 11. Environment and Dependencies

### 11.1 Python dependencies

[`requirements.txt`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/requirements.txt) lists key packages used by the repo:

- `timm`
- `pandas`
- `opencv-python-headless`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `decord`
- `av`
- `tqdm`
- `PyYAML`
- `scipy`
- `python-dotenv`
- `numpy<2`

The file notes that `torch`, `torchvision`, `numpy`, `Pillow`, and `tensorboard` are expected from the NVIDIA PyTorch base image rather than reinstalled from pip.

### 11.2 Docker environment

[`Dockerfile`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/Dockerfile) uses:

- `nvcr.io/nvidia/pytorch:24.12-py3`

It installs system packages such as:

- `ffmpeg`
- `libgl1`
- `git`
- `tmux`
- `htop`

It also sets environment flags for Ampere GPUs, including TF32 and allocator tuning.

### 11.3 Compose setup

[`docker-compose.yml`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/docker-compose.yml) defines a `dev` service with:

- NVIDIA GPU reservation
- project bind mount
- separate dataset mount
- persisted outputs directory
- exposed ports for TensorBoard and Jupyter
- enlarged shared memory for multi-worker PyTorch loaders

This is consistent with the repo’s intended usage as a GPU-backed research workspace.

## 12. Strengths of the Current Design

From a repository-design perspective, notable strengths include:

- clear separation between configuration, model, training, and scripts,
- support for both real and dummy data workflows,
- use of a hybrid model suited to deepfake forensics,
- good experiment tooling through logging, checkpointing, and early stopping,
- beginner-friendly command-line entrypoints,
- architecture that is configurable enough for comparative studies.

The project is especially well positioned for:

- course projects,
- lab experimentation,
- baseline-plus-ablation workflows,
- studying the value of spatial versus frequency information in deepfake detection.

## 13. Current Gaps and Risks

The most important technical caveats visible in the tracked workspace are:

### 13.1 Missing data-loader implementation

The main missing component is `src.data.dataset.build_dataloaders`, which is required by `train.py` but not present in the tracked files I reviewed.

Without that module, the codebase as currently visible cannot complete a full training run even though the rest of the training stack is implemented.

### 13.2 Documentation versus implementation mismatch

The README describes a complete pipeline, but the visible source tree does not fully match that claim because of the missing data module. A technical reader should therefore distinguish:

- implemented model and trainer components,
- intended but not fully visible end-to-end training path.

### 13.3 External dependency on pretrained weights

If `pretrained_spatial: true`, the first run of `timm` may need internet access to fetch EfficientNet weights unless they are already cached in the environment.

### 13.4 Class imbalance handling not fully wired

Although the loss class can accept `pos_weight`, the standard config path does not currently expose or compute it.

## 14. Recommended Interpretation of the Repo

The most accurate technical summary is:

This repository is a configurable deepfake detection experimentation framework built around a hybrid spatial-frequency attention model. It already includes a strong model implementation, a mature training engine, and practical data-preparation utilities. It is designed to make model study and comparison easier, especially for beginner-to-intermediate experimentation. However, in the currently visible tracked snapshot, the data-loading layer needed to connect extracted frames to the trainer is not present, so the code should be treated as nearly complete rather than fully self-contained.

## 15. Concise Technical Summary

Problem formulation:

- binary deepfake detection on extracted face frames

Primary datasets:

- FaceForensics++
- Celeb-DF v2

Model family:

- `HSF-CVIT`

Architecture:

- EfficientNet-B4 spatial branch
- SRM high-pass plus residual frequency branch
- transformer-style cross-attention plus self-attention fusion head

Training defaults:

- AdamW
- learning rate `3e-4`
- weight decay `1e-4`
- batch size `16`
- `50` epochs
- warmup `5`
- cosine scheduling
- label smoothing `0.1`
- gradient clip `1.0`
- mixed precision on CUDA

Validation and selection:

- ROC-AUC as the primary metric

Evaluation metrics:

- ROC-AUC
- F1
- precision
- recall
- accuracy

Experiment tooling:

- TensorBoard
- CSV logs
- periodic and best checkpoints
- early stopping

Main implementation caution:

- missing tracked `src.data.dataset` module in the current workspace snapshot
