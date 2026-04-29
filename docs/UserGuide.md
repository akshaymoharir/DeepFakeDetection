# DeepFakeDetection User Guide

This guide is for users who want to run the trained deepfake detector, evaluate a checkpoint, or understand the repository at a practical level. It focuses on using the code and model rather than implementation internals.

## Project Purpose

DeepFakeDetection implements an HSF-CVIT style deepfake detector. The model predicts whether an image frame or video is real or fake, with its output interpreted as `P(fake)`.

The detector combines:

- a spatial RGB branch for visual appearance cues,
- a frequency branch for forensic artifacts,
- a cross-attention fusion module,
- a binary classifier for real/fake prediction.

Primary user entrypoints:

- `predict.py`: run inference on one image or video.
- `evaluate_celeb_df.py`: evaluate a checkpoint on Celeb-DF v2.
- `train.py`: train the model or run test-only evaluation.
- `scripts/extract_frames.py`: extract frames from raw videos.
- `start_container.sh`: launch the recommended Docker runtime.

## Repository Structure

```text
DeepFakeDetection/
├── configs/
│   ├── train_config.yaml              # Default model/training/evaluation config
│   ├── train_config_extended.yaml     # Extended training config
│   ├── dataset_config.yaml            # Dataset path and frame extraction config
│   ├── dataset_config_extended.yaml   # Extended dataset config
│   └── ffpp_splits/                   # FaceForensics++ official split files
├── docs/                              # Project documentation
├── scripts/
│   ├── download_FaceForensicsPP.py    # FaceForensics++ downloader
│   ├── download_celeb_df_test.py      # Celeb-DF v2 test-list/download helper
│   ├── extract_frames.py              # Video frame extraction
│   ├── explore_dataset.py             # Dataset inspection
│   ├── audit_faceforensics.py         # FaceForensics++ audit utility
│   ├── analyse_quality.py             # Dataset/video quality analysis
│   ├── plot_samples.py                # Visualization utility
│   └── create_dummy_datasets.py       # Synthetic smoke-test data generator
├── src/
│   ├── data/                          # Dataset classes, dataloaders, transforms
│   ├── inference/                     # DeepFakeDetector inference API
│   ├── models/                        # HSF-CVIT model and component modules
│   ├── training/                      # Trainer, losses, metrics
│   └── utils/                         # Shared helpers
├── outputs/
│   ├── checkpoints/                   # Trained checkpoint files
│   ├── evaluation/                    # Evaluation reports
│   └── celeb_df_eval/                 # Celeb-DF evaluation outputs
├── data/                              # Local datasets and extracted frames
├── predict.py                         # Image/video inference CLI
├── evaluate_celeb_df.py               # Celeb-DF v2 evaluation CLI
├── train.py                           # Training and test-evaluation CLI
├── start_container.sh                 # Docker GPU environment launcher
├── Dockerfile                         # Runtime/training image definition
├── docker-compose.yml                 # Docker Compose workflow
└── requirements.txt                   # Python dependencies
```

## Runtime Environment

The recommended runtime is the Docker environment provided by `start_container.sh`. This avoids host Python dependency issues and gives the project the expected PyTorch/CUDA stack.

Start an interactive shell:

```bash
./start_container.sh
```

Rebuild after dependency changes:

```bash
./start_container.sh --build
```

Start TensorBoard:

```bash
./start_container.sh --tensorboard
```

Then open:

```text
http://localhost:6006
```

Start Jupyter Lab:

```bash
./start_container.sh --jupyter
```

Then open:

```text
http://localhost:8888
```

Inside the container, the project is mounted at:

```text
/workspace
```

## Checkpoints and Configs

Model checkpoints are expected under:

```text
outputs/checkpoints/
```

Common checkpoint paths:

```text
outputs/checkpoints/best.pt
outputs/checkpoints/best_b4.pt
outputs/checkpoints/best_iteration_swt_b7.pt
```

The default model config is:

```text
configs/train_config.yaml
```

Use the config that matches the checkpoint architecture. A checkpoint trained with one backbone or model shape may not load with a different config.

## Dataset Layout

The default dataset config expects local data under:

```text
data/
```

### Celeb-DF v2

For Celeb-DF v2 evaluation, place files like this:

```text
data/Celeb-DF-v2/
├── Celeb-real/
│   └── *.mp4
├── YouTube-real/
│   └── *.mp4
├── Celeb-synthesis/
│   └── *.mp4
└── List_of_testing_videos.txt
```

Celeb-DF's test list uses `1` for real and `0` for fake. The evaluator converts this internally to the project convention where `1` means fake.

### FaceForensics++

For FaceForensics++ training or evaluation through `train.py`, the default extracted frame path is:

```text
data/frames_ffpp_standard/
```

Raw FaceForensics++ videos are expected under the standard `data/original_sequences/` and `data/manipulated_sequences/` layout.

## Run Inference

Use `predict.py` for single image or video inference.

### Image Inference

```bash
python predict.py \
  --input path/to/image.jpg \
  --checkpoint outputs/checkpoints/best.pt \
  --config configs/train_config.yaml \
  --device cuda
```

### Video Inference

```bash
python predict.py \
  --input path/to/video.mp4 \
  --checkpoint outputs/checkpoints/best.pt \
  --config configs/train_config.yaml \
  --frames 16 \
  --aggregation mean \
  --device cuda
```

### JSON Output

```bash
python predict.py \
  --input path/to/video.mp4 \
  --checkpoint outputs/checkpoints/best.pt \
  --json
```

Example output shape:

```json
{
  "label": "fake",
  "probability": 0.8732,
  "threshold": 0.5,
  "frame_probs": [0.82, 0.91, 0.88],
  "num_frames_used": 16
}
```

Useful options:

- `--input`: image or video path.
- `--checkpoint`: trained `.pt` checkpoint.
- `--config`: training config defining the model architecture.
- `--frames`: number of frames sampled from a video.
- `--aggregation mean`: average frame probabilities. This is the default and is usually stable.
- `--aggregation max`: use the highest fake probability from sampled frames.
- `--no-face-detect`: skip face detection when inputs are already cropped face images.
- `--device cuda`: run on GPU.
- `--device cpu`: run on CPU.

## Evaluate on Celeb-DF v2

After downloading Celeb-DF v2 offline, place it under:

```text
data/Celeb-DF-v2/
```

Verify the test list:

```bash
python scripts/download_celeb_df_test.py \
  --out-dir data/Celeb-DF-v2 \
  --list-only
```

Run full evaluation:

```bash
python evaluate_celeb_df.py \
  --checkpoint outputs/checkpoints/best_iteration_swt_b7.pt \
  --config configs/train_config.yaml \
  --dataset-root data/Celeb-DF-v2 \
  --frames 16 \
  --aggregation mean \
  --device cuda
```

Run a small smoke test first:

```bash
python evaluate_celeb_df.py \
  --checkpoint outputs/checkpoints/best_iteration_swt_b7.pt \
  --dataset-root data/Celeb-DF-v2 \
  --frames 8 \
  --limit 25 \
  --device cuda
```

Evaluation outputs:

```text
outputs/celeb_df_eval/
├── per_video.csv
└── metrics.json
```

`per_video.csv` contains one row per video with the relative path, ground-truth label, predicted fake probability, frames used, and status. `metrics.json` contains summary metrics such as ROC-AUC, average precision, accuracy, balanced accuracy, precision, recall, and F1.

## Prepare FaceForensics++ Data

Download FaceForensics++ videos:

```bash
python scripts/download_FaceForensicsPP.py \
  data \
  --dataset all \
  --compression c23 \
  --type videos
```

Download a smaller subset:

```bash
python scripts/download_FaceForensicsPP.py \
  data \
  --dataset Deepfakes \
  --compression c23 \
  --type videos \
  --num_videos 50
```

Extract frames:

```bash
python scripts/extract_frames.py \
  --config configs/dataset_config.yaml \
  --dataset ff++ \
  --resize 380
```

Extract selected methods only:

```bash
python scripts/extract_frames.py \
  --dataset ff++ \
  --ff-methods Deepfakes,Face2Face \
  --max-frames 32 \
  --resize 380
```

The default output directory is:

```text
data/frames_ffpp_standard/
```

## Training Note

Training is handled by `train.py`.

Standard training:

```bash
python train.py \
  --config configs/train_config.yaml \
  --device cuda
```

Lightweight smoke run:

```bash
python train.py \
  --config configs/train_config.yaml \
  --epochs 1 \
  --batch-size 4 \
  --workers 2 \
  --items-per-clip 2 \
  --device cuda
```

Resume training:

```bash
python train.py \
  --config configs/train_config.yaml \
  --resume outputs/checkpoints/best.pt \
  --device cuda
```

Run test-only evaluation from a checkpoint:

```bash
python train.py \
  --config configs/train_config.yaml \
  --resume outputs/checkpoints/best.pt \
  --eval-only \
  --device cuda
```

Training outputs are usually written to:

```text
outputs/
├── checkpoints/
├── evaluation/
├── logs/
├── runs/
└── training_log.csv
```

## Python API Usage

For programmatic inference, use `DeepFakeDetector`:

```python
import yaml

from src.inference.detector import DeepFakeDetector

with open("configs/train_config.yaml", "r") as f:
    train_cfg = yaml.safe_load(f)

detector = DeepFakeDetector(
    checkpoint_path="outputs/checkpoints/best.pt",
    train_cfg=train_cfg,
    device="cuda",
)

result = detector.predict_video(
    "path/to/video.mp4",
    num_frames=16,
    detect_face=True,
    aggregation="mean",
)

print(result["label"], result["probability"])
```

Single image:

```python
result = detector.predict_image(
    "path/to/image.jpg",
    detect_face=True,
)
```

## Interpreting Results

The model output is interpreted as `P(fake)`.

```text
probability >= threshold  -> fake
probability < threshold   -> real
```

The threshold is loaded from the checkpoint when available. If a checkpoint does not include a saved threshold, inference defaults to `0.5`.

For videos:

- `mean` averages frame probabilities.
- `max` uses the highest fake probability among sampled frames.

`mean` is usually more stable. `max` is more sensitive to a single suspicious frame, but can increase false positives.

## Common Commands

Container:

```bash
./start_container.sh
./start_container.sh --build
./start_container.sh --tensorboard
```

Inference:

```bash
python predict.py --input path/to/file.mp4 --checkpoint outputs/checkpoints/best.pt
python predict.py --input path/to/file.jpg --json
```

Celeb-DF:

```bash
python scripts/download_celeb_df_test.py --out-dir data/Celeb-DF-v2 --list-only
python evaluate_celeb_df.py --dataset-root data/Celeb-DF-v2 --checkpoint outputs/checkpoints/best_iteration_swt_b7.pt
```

Training:

```bash
python train.py --config configs/train_config.yaml --device cuda
python train.py --config configs/train_config.yaml --resume outputs/checkpoints/best.pt --eval-only
```

## Troubleshooting

### ModuleNotFoundError on the Host

Use the Docker container:

```bash
./start_container.sh
```

The project depends on PyTorch, OpenCV, timm, scikit-learn, video decoders, and other ML packages. The container is the preferred runtime.

### CUDA Is Not Available

Check CUDA inside the container:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

For small tests, force CPU:

```bash
python predict.py --input path/to/file.mp4 --device cpu
```

### Checkpoint and Config Mismatch

If checkpoint loading fails with missing or unexpected keys, confirm that the checkpoint and config belong to the same model variant:

```text
--checkpoint outputs/checkpoints/<checkpoint>.pt
--config configs/train_config.yaml
```

### Celeb-DF Videos Report as Missing

Confirm these paths:

```text
data/Celeb-DF-v2/List_of_testing_videos.txt
data/Celeb-DF-v2/Celeb-real/
data/Celeb-DF-v2/YouTube-real/
data/Celeb-DF-v2/Celeb-synthesis/
```

The relative paths in `List_of_testing_videos.txt` must match the folder names exactly.

## Additional Documentation

More detailed project notes are available in:

- `docs/repo_overview.md`
- `docs/repo_technical_reference.md`
- `docs/model_architecture_design.md`
- `docs/dataset_configuration_report.md`
- `docs/training_run_summary.md`
- `docs/improvement_notes.md`

Future documentation can add citations, benchmark references, experiment reports, and model-card style usage notes.
