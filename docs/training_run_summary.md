# Training Run Summary

## Scope

This document summarizes the training/evaluation artifacts currently present in the repository and explains how to interpret them. It is not a benchmark paper; it is a project-state report tied to local files under `outputs/`.

Relevant artifacts:

```text
outputs/training_log.csv
outputs/checkpoints/
outputs/evaluation/
outputs/evaluation_multiframe_8/
outputs/evaluation_video_8/
outputs/evaluation_video_16/
outputs/evaluation_video_16_dropout04/
outputs/celeb_df_eval/
outputs/logs/final_20260427_0322_clean.log
```

## Current Best Observed Run

The strongest visible run artifact is the final test block in:

```text
outputs/logs/final_20260427_0322_clean.log
```

That log reports:

| Metric | Value |
|---|---:|
| Test ROC-AUC | 0.9062 |
| Test average precision | 0.9731 |
| Test F1 at saved threshold | 0.8810 |
| Test precision at saved threshold | 0.9523 |
| Test recall at saved threshold | 0.8196 |
| Test specificity at saved threshold | 0.8357 |
| Test balanced accuracy at saved threshold | 0.8277 |
| Test accuracy at saved threshold | 0.8229 |
| Test threshold | 0.6300 |

At the default threshold, the same log reports:

| Default-threshold metric | Value |
|---|---:|
| F1 | 0.8922 |
| precision | 0.9478 |
| recall | 0.8429 |
| specificity | 0.8143 |
| balanced accuracy | 0.8286 |
| accuracy | 0.8371 |

This suggests the checkpoint had strong ranking performance and that the default threshold was competitive for F1 on that test run.

## Per-Method Results from the Best Observed Log

The same final log reports per-method ROC-AUC:

| Method | ROC-AUC | Average Precision | Balanced Accuracy |
|---|---:|---:|---:|
| Deepfakes | 0.9474 | 0.9480 | 0.8750 |
| Face2Face | 0.9253 | 0.9217 | 0.8464 |
| FaceSwap | 0.9152 | 0.9182 | 0.8429 |
| NeuralTextures | 0.8368 | 0.8441 | 0.7464 |

The persistent pattern is that `NeuralTextures` is the hardest of the four default manipulation methods.

## Latest Evaluation Report File

The file:

```text
outputs/evaluation/test_metrics.json
```

currently reports:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8840 |
| Average precision | 0.9651 |
| F1 at saved threshold | 0.8071 |
| Precision at saved threshold | 0.9629 |
| Recall at saved threshold | 0.6946 |
| Specificity at saved threshold | 0.8929 |
| Balanced accuracy at saved threshold | 0.7938 |
| Accuracy at saved threshold | 0.7343 |
| Threshold | 0.8600 |

At the default threshold in the same report:

| Default-threshold metric | Value |
|---|---:|
| F1 | 0.8503 |
| precision | 0.9555 |
| recall | 0.7661 |
| specificity | 0.8571 |
| balanced accuracy | 0.8116 |
| accuracy | 0.7843 |

This report appears to be a later evaluation artifact than the best observed final log. Treat `outputs/evaluation/test_metrics.json` as the most recent report file, not necessarily the best historical result.

## Latest Per-Method Report File

`outputs/evaluation/test_per_method.csv` currently reports:

| Method | ROC-AUC | AP | Balanced Accuracy |
|---|---:|---:|---:|
| Deepfakes | 0.9271 | 0.9270 | 0.8714 |
| Face2Face | 0.9086 | 0.9010 | 0.7929 |
| FaceSwap | 0.8764 | 0.8797 | 0.8071 |
| NeuralTextures | 0.8239 | 0.8268 | 0.7036 |

The method ordering is consistent with the best log: `Deepfakes` is easiest, `NeuralTextures` is hardest.

## Training Log Status

`outputs/training_log.csv` contains epoch-level training and validation metrics. The visible tail of the file reaches epoch 34.

At epoch 34:

| Metric | Value |
|---|---:|
| Train ROC-AUC | 0.9886 |
| Train average precision | 0.9888 |
| Validation ROC-AUC | 0.8762 |
| Validation average precision | 0.9646 |
| Validation F1 at default threshold | 0.8874 |
| Validation balanced accuracy at default threshold | 0.8045 |
| Validation optimal threshold | 0.81 |
| Validation optimized balanced accuracy | 0.8205 |

The trend shows strong training-set fit and useful validation performance, with a noticeable train/validation AUC gap. That gap should be treated as evidence of overfitting or dataset-specific learning pressure.

## Checkpoints Present

The repository currently contains:

```text
outputs/checkpoints/best.pt
outputs/checkpoints/best_b4.pt
outputs/checkpoints/best_iteration_swt_b7.pt
outputs/checkpoints/epoch_025.pt
outputs/checkpoints/epoch_030.pt
outputs/checkpoints/epoch_035.pt
```

Interpretation:

- `best.pt` is the default checkpoint name produced by `Trainer`.
- `best_b4.pt` appears to preserve an older EfficientNet-B4 style run.
- `best_iteration_swt_b7.pt` appears to preserve the SWT + EfficientNet-B7 iteration used by Celeb-DF evaluation defaults.
- `epoch_*.pt` are periodic checkpoints.

Because checkpoint files do not encode the full experiment name in the standard trainer output, pair each checkpoint with the matching config before inference or evaluation.

## Multi-Frame / Video Evaluation Artifacts

Several evaluation directories compare frame aggregation settings:

```text
outputs/evaluation_multiframe_8/
outputs/evaluation_video_8/
outputs/evaluation_video_16/
outputs/evaluation_video_16_dropout04/
```

Observed ROC-AUC values in the available reports:

| Directory | ROC-AUC | Notes |
|---|---:|---|
| `outputs/evaluation_multiframe_8` | 0.6475 | Multi-frame setting performed poorly in this artifact |
| `outputs/evaluation_video_8` | 0.7718 | Video-level averaging with 8 frames |
| `outputs/evaluation_video_16` | 0.8135 | Video-level averaging with 16 frames |
| `outputs/evaluation_video_16_dropout04` | 0.7845 | 16-frame video eval from a dropout-0.4 run/artifact |

These numbers show that video-level aggregation can help, but results are checkpoint- and configuration-dependent. The best observed single-frame final log still reports stronger ROC-AUC than the later video-eval artifacts listed above.

## Celeb-DF v2 Cross-Dataset Evaluation

`outputs/celeb_df_eval/metrics.json` records the cross-dataset run produced by `evaluate_celeb_df.py` against `best_iteration_swt_b7.pt`, with 16 frames per video and mean aggregation. This is a generalization test: the model is trained on FaceForensics++ only.

Run configuration:

| Setting | Value |
|---|---|
| Checkpoint | `outputs/checkpoints/best_iteration_swt_b7.pt` |
| Frames per video | 16 |
| Aggregation | mean |
| Face detection | Haar cascade with 0.30 margin |
| Image size | 380 |
| Videos evaluated | 518 / 518 (no decode failures) |
| Class balance | 178 real, 340 fake |

Metrics at the saved checkpoint threshold (`0.64`):

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7878 |
| Average precision | 0.8497 |
| F1 | 0.7988 |
| Precision (fake) | 0.8291 |
| Recall (fake) | 0.7706 |
| Specificity | 0.6966 |
| Balanced accuracy | 0.7336 |
| Accuracy | 0.7452 |
| MCC | 0.4549 |

Confusion matrix at the same threshold:

| | Predicted real | Predicted fake |
|---|---:|---:|
| Actual real | 124 | 54 |
| Actual fake | 78 | 262 |

Re-optimizing the threshold on the Celeb-DF v2 test split itself moves it from `0.64` down to `0.61`, raising balanced accuracy to `0.7371` and recall to `0.8000` while specificity drops to `0.6742`. The ROC-AUC and average precision are unchanged because they are threshold-free.

Interpretation:

- The detector ranks Celeb-DF clips meaningfully above chance, but ROC-AUC drops roughly 12 points relative to the FaceForensics++ test split. This is the expected direction for a detector trained on a single source dataset.
- Specificity is the weakest summary metric. The FF++-trained checkpoint is biased toward predicting "fake" on Celeb-DF reals.
- A small threshold recalibration on Celeb-DF data closes part of the gap but does not change the underlying ranking quality.

## What Is Solid

The implemented training/evaluation stack includes:

- official FaceForensics++ split support in the default config;
- deterministic validation/test frame selection;
- weighted sampling for training imbalance;
- label smoothing;
- AMP and gradient clipping;
- checkpoint threshold saving;
- per-method reports;
- confusion matrix and prediction CSV export;
- single-file inference through `predict.py`;
- Celeb-DF v2 evaluation through `evaluate_celeb_df.py`.

## Open Interpretation Risks

### Report-file drift

`outputs/evaluation/test_metrics.json` can be overwritten by later evaluation commands. The strongest log and the current report file do not show identical numbers. Documentation should therefore distinguish "best observed log" from "current report file."

### Validation/test imbalance

The FaceForensics++ test report uses 140 real samples and 560 fake samples in the current per-method report. Accuracy and F1 should be read with that imbalance in mind. ROC-AUC, average precision, balanced accuracy, recall, and specificity are more informative than accuracy alone.

### NeuralTextures remains hardest

Across available reports, `NeuralTextures` is consistently the lowest-performing method. This is the clearest method-specific weakness to target in future work.

### Celeb-DF v2 calibration gap

The Celeb-DF v2 result is reported at the threshold saved with the FF++-trained checkpoint. That threshold is not optimal for Celeb-DF reals, and balanced accuracy / specificity should be read with that mismatch in mind. Re-optimized thresholds on the Celeb-DF test split itself improve balanced accuracy modestly but do not change ROC-AUC or average precision.

## Recommended Next Checks

1. Run a clean eval-only pass for the intended final checkpoint and archive the report directory under a unique name.
2. Confirm whether `best_iteration_swt_b7.pt` or `best.pt` is the intended final submitted checkpoint.
3. Compare default-threshold metrics against saved-threshold metrics before reporting a single F1/accuracy number.
4. Continue tracking per-method results, especially `NeuralTextures`.
5. Investigate the Celeb-DF v2 specificity gap (`0.697` at the saved threshold). Candidate fixes include threshold recalibration on a Celeb-DF held-out slice, higher frame counts at inference, or compression/blur augmentation during FF++ training.

## Summary

The current codebase has moved beyond the earlier sub-0.80 baseline. The strongest visible FaceForensics++ artifact reports approximately `0.906` ROC-AUC with the SWT+B7 style model, while the latest report file currently shows approximately `0.884` ROC-AUC. Cross-dataset evaluation on Celeb-DF v2 with the same checkpoint reaches `0.788` ROC-AUC over 518 test videos, with specificity (`0.697`) the weakest summary metric. The model is usable, but final reporting should pin a specific checkpoint, config, and evaluation output directory to avoid ambiguity.
