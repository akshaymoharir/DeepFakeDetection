# HSF-CVIT Improvement Notes

## Purpose

This document records practical next steps for improving the implemented project. It reflects the current codebase state:

- active model: EfficientNet-B7 spatial branch + SWT frequency branch + 3-token transformer fusion;
- default training data: FaceForensics++ standard four-method extracted frames;
- current best visible FaceForensics++ result: roughly `0.906` ROC-AUC in `outputs/logs/final_20260427_0322_clean.log`;
- Celeb-DF v2 cross-dataset evaluation completed: `0.788` ROC-AUC, `0.850` average precision, `0.799` F1 at the saved checkpoint threshold (`outputs/celeb_df_eval/metrics.json`).

## Current Strengths

The project already has a strong engineering baseline:

- modular model branches and fusion head;
- config-driven model/training settings;
- official FaceForensics++ split support;
- weighted sampling for class imbalance;
- deterministic validation/test sampling;
- threshold optimization and saved checkpoint thresholds;
- per-method reports;
- video-level test evaluation option;
- image/video inference API and CLI;
- Celeb-DF v2 evaluation script.

The move from the older SRM/B4-style baseline to the current SWT/B7-style implementation appears to have produced a meaningful improvement in FaceForensics++ test ROC-AUC.

## Current Weaknesses

### 1. NeuralTextures is still the hardest default method

Available reports consistently show `NeuralTextures` as the weakest method. In the strongest visible final log:

```text
Deepfakes      AUC 0.9474
Face2Face      AUC 0.9253
FaceSwap       AUC 0.9152
NeuralTextures AUC 0.8368
```

This suggests the detector handles identity-swap/blending artifacts better than texture-rendering artifacts.

### 2. Train/validation gap remains visible

The tail of `outputs/training_log.csv` shows high training ROC-AUC and lower validation ROC-AUC. That is expected for this problem, but it still signals overfitting pressure.

### 3. Report artifacts can be overwritten

`outputs/evaluation/test_metrics.json` is a mutable output path. Later eval-only runs may overwrite a stronger report. Final experiments should write to uniquely named report directories.

### 4. Celeb-DF generalization gap is sizeable

Celeb-DF v2 was evaluated with `best_iteration_swt_b7.pt` over 518 test videos at 16 frames per video with mean aggregation. The detector reaches `0.788` ROC-AUC, down from the `~0.906` ROC-AUC seen on the FaceForensics++ test split — an ~12-point drop that is consistent with the published FF++ → Celeb-DF generalization gap, but still leaves clear headroom. Specificity (`0.697`) is the weakest summary metric, indicating the FF++-trained detector is biased toward predicting "fake" on Celeb-DF reals. See `outputs/celeb_df_eval/metrics.json` for the full breakdown.

## Highest-Value Next Steps

### 1. Run and archive a final clean evaluation

Before changing model code, pin the current baseline.

Recommended command shape:

```bash
python train.py \
  --config configs/train_config.yaml \
  --resume outputs/checkpoints/best_iteration_swt_b7.pt \
  --eval-only \
  --report-dir outputs/evaluation_final_swt_b7 \
  --device cuda
```

Then preserve:

```text
outputs/evaluation_final_swt_b7/test_metrics.json
outputs/evaluation_final_swt_b7/test_per_method.csv
outputs/evaluation_final_swt_b7/test_predictions.csv
outputs/evaluation_final_swt_b7/test_confusion_matrix.csv
```

Why this matters:

- avoids confusion between best historical log and latest report file;
- makes the final checkpoint/config/report tuple explicit;
- gives future docs a stable source of truth.

### 2. Close the Celeb-DF v2 generalization gap

Celeb-DF v2 evaluation has been completed (see `outputs/celeb_df_eval/metrics.json` and the Weaknesses section above). The reproduction command is:

```bash
python evaluate_celeb_df.py \
  --checkpoint outputs/checkpoints/best_iteration_swt_b7.pt \
  --config configs/train_config.yaml \
  --dataset-root data/Celeb-DF-v2 \
  --frames 16 \
  --aggregation mean \
  --device cuda
```

The headline result (`0.788` ROC-AUC, `0.799` F1, `0.697` specificity at the saved threshold of `0.64`) shows the detector ranks Celeb-DF clips well above chance but has a real-vs-fake calibration bias on this dataset — far more reals are predicted "fake" than vice versa. Promising follow-ups, in roughly increasing cost:

- recalibrate the decision threshold on a Celeb-DF held-out slice (the dataset's optimal-balanced-accuracy threshold is `0.61`, slightly below the FF++-saved `0.64`, but the gain is small — the issue is ranking, not just threshold);
- evaluate at higher frame counts (`--frames 32`) to see whether per-clip variance is hurting the real class disproportionately;
- add Celeb-DF-style augmentations (compression, blur, identity-swap-only artifacts) to FF++ training without mixing in Celeb-DF data;
- as a last resort, mix a small Celeb-DF v2 train slice into training and re-measure FF++ performance to ensure no regression.

### 3. Improve regularization before adding architectural complexity

The current model already has a large B7 backbone and strong training-set fit. Reasonable next experiments:

- tune `dropout` around `0.4`;
- compare `weight_decay` values around `5e-4`;
- compare `label_smoothing` values around `0.05`, `0.10`, and `0.15`;
- try smaller `train_items_per_clip` values if epochs are seeing too many near-duplicate frame variants;
- consider early stopping/report selection by a metric that reflects the final objective.

These are lower-risk than changing model interfaces.

### 4. Add stronger occlusion or region-level augmentation

Current train transforms include crop, flip, color jitter, rotation, blur, and JPEG compression. A useful next augmentation experiment would be:

- random erasing/cutout on face crops;
- region dropout;
- CutMix-style mixing if implemented carefully for binary labels.

Goal:

- reduce reliance on a small artifact region;
- improve robustness to local occlusion and compression differences;
- narrow the train/validation gap.

### 5. Investigate NeuralTextures-specific failure cases

Recommended analysis:

- sort `test_predictions.csv` by false negatives for `NeuralTextures`;
- inspect high-confidence misses;
- compare frame quality and compression against better-performing methods;
- check whether misses cluster by identity, source video, or frame region.

This should happen before adding large model features. The failure mode may be data distribution, not architecture capacity.

## Architecture Ideas for Later

These are larger changes and should wait until the baseline is archived.

### 1. Patch-level fusion

Current fusion uses three tokens:

```text
[CLS], spatial, frequency
```

A richer design could preserve spatial feature maps or frequency feature maps and build patch tokens before fusion. This may help localization-sensitive artifacts but requires changing branch interfaces.

### 2. Learnable or hybrid frequency filters

The current SWT front-end uses fixed Haar filters. Later experiments could try:

- learnable high-pass filters initialized from forensic kernels;
- learnable wavelet-like filters;
- concatenating SWT maps with RGB-derived residual maps.

This may help NeuralTextures if fixed Haar subbands are not expressive enough.

### 3. Temporal modeling

The current model is frame-level. Video inference averages sampled frame probabilities. A temporal model could use:

- per-frame embedding aggregation;
- temporal transformer over frame embeddings;
- simple statistics over frame probabilities and embeddings;
- 3D CNN blocks for short clips.

This is more invasive and should be considered only after clean frame-level and video-aggregation baselines are documented.

## Documentation Improvements Still Needed

Future docs should add:

- a stable experiment table tying checkpoint, config, report directory, and date;
- a model card with intended use, limitations, and misuse warnings;
- citation/reference section for FaceForensics++, Celeb-DF, EfficientNet, SWT/wavelet detection, and deepfake detection baselines;
- a reproducibility checklist for final experiments.

## Summary

The current SWT+B7 checkpoint reaches `~0.906` ROC-AUC on the FaceForensics++ test split and `0.788` ROC-AUC on the Celeb-DF v2 test split. The most promising improvement work is now regularization, NeuralTextures-focused error analysis, and closing the Celeb-DF v2 calibration gap (`0.697` specificity).
