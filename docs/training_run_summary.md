# Training Run Summary

Run reviewed: `outputs/logs/log_data_20260330_201546.log`

## Executive Summary

This run appears healthy overall and was manually interrupted, not crashed by a training error. The model completed **16 full epochs out of the planned 50** and then stopped during epoch 17 with a `KeyboardInterrupt`.

Based on the completed epochs, the run was still improving through the mid-teens, but the rate of improvement had already started to slow. My read is that this run was likely moving toward a useful baseline rather than obviously diverging or collapsing.

## What Happened

- Device: `cuda`
- Dataset split: `train=4,703`, `val=820`, `test=904`
- Mixed precision: enabled (`AMP: True`)
- Gradient clipping: `1.0`
- Backbone warm-up: EfficientNet frozen for the first `2` epochs
- Model size: `21,923,849` trainable parameters
- Stop reason: manual interruption during epoch 17

The traceback at the end does **not** indicate a training bug in the model itself. It ends in the PyTorch dataloader loop and resolves to `KeyboardInterrupt`, which is consistent with stopping the run manually.

## Observed Training Behavior

### Metric trend

The run shows a steady rise in validation AUC:

- Epoch 1: `Val AUC=0.5378`, `F1=0.8203`
- Epoch 8: `Val AUC=0.7348`, `F1=0.8724`
- Epoch 13: `Val AUC=0.7863`, `F1=0.8801`
- Epoch 16: `Val AUC=0.7857`, `F1=0.8691`

Training loss also improved consistently:

- Epoch 1: `Train loss=0.7061`
- Epoch 16: `Train loss=0.5010`

Validation loss generally improved as well:

- Epoch 1: `Val loss=0.6538`
- Best observed validation loss: `0.4922` at epoch 14
- Epoch 16: `Val loss=0.5024`

### Best completed checkpoints by metric

- Best observed validation AUC: **epoch 13**, `0.7863`
- Best observed validation F1: **epoch 13**, `0.8801`
- Best observed validation loss: **epoch 14**, `0.4922`

This suggests the run was near a local sweet spot around epochs 13 to 16.

### Stability notes

- I do **not** see evidence of catastrophic overfitting by epoch 16.
- Training AUC (`0.7281`) is still below validation AUC (`0.7857`) at epoch 16, which is unusual but not impossible when augmentation is heavy or train-time conditions are harder than validation.
- One anomaly appears at **epoch 11**, where `Val loss=nan` but `Val AUC=0.7715` and `F1=0.8674` were still computed. That points more toward a numerical issue in validation loss aggregation than a total validation failure.

## Estimate Toward 50 Epochs

This section is an estimate, not an observed result.

If the same trend had continued without major instability, I would expect:

- Validation AUC to finish roughly in the **0.79 to 0.82** range
- Validation F1 to remain roughly in the **0.87 to 0.89** range
- Improvements after epoch 16 to be **incremental**, not dramatic

Why I think that:

- The first 10 to 13 epochs delivered the biggest gains.
- Epochs 13 to 16 already look close to a plateau.
- The model was still improving, but the marginal gain per epoch had become small.

So my judgment is that stopping at epoch 16 likely left some performance on the table, but probably **not a huge amount** unless later scheduler phases are especially important in this configuration.

## Dataset / Figure Observations

### 1. Strong class imbalance

![Class distribution](figures/ff++_class_distribution.png)

The class distribution figure shows:

- Real: `1000`
- Fake: `5507`

This is a major skew toward the fake class. Because of that:

- F1 can look fairly strong even when calibration is imperfect
- AUC is the more trustworthy headline metric here
- Threshold tuning and per-class metrics would be worth checking before treating this as deployment-ready

### 2. Temporal data looks meaningful

![Temporal frame sequence](figures/ff++_frame_sequence.png)

The temporal sequence figure shows a mostly coherent real-video clip with pose and expression variation across frames. That is encouraging for a hybrid spatial-frequency-temporal setup because it means the model likely sees meaningful motion and identity continuity rather than only static face crops.

One later frame is visibly degraded/noisy, which also hints that the pipeline may naturally expose the model to compression or corruption artifacts. That can help robustness, but it also means the model may partially learn source-quality cues if the dataset is not balanced carefully.

### 3. Broad visual diversity, but likely dataset bias remains

![Real vs fake grid](figures/ff++_real_vs_fake_grid.png)

The sample grid shows decent variation in:

- Subject identity
- Lighting
- Background
- camera framing

That is good for generalization. At the same time, many samples still look like internet/social/video-news style footage, so there is still a risk that the model learns dataset-specific compression, editing, or source-domain shortcuts instead of only manipulation artifacts.

## Overall Assessment

My overall take is:

- The run was **productive**
- The model was **learning**
- The run was **not yet fully converged**
- The strongest observed checkpoint is likely around **epoch 13 or 14**

If you already save best checkpoints during training, I would treat the best model from around that window as the current candidate baseline.

## Recommended Next Checks

- Inspect why validation loss became `nan` at epoch 11 even though AUC/F1 remained valid.
- Confirm which metric is used for checkpoint selection. I would lean toward **validation AUC** as the primary selector here.
- Add per-class precision/recall or a confusion matrix because of the `1000` vs `5507` class imbalance.
- If you rerun, let training continue past epoch 16 and compare whether epochs 20 to 30 provide real gains or just noise around the current plateau.
- If not already enabled, save a compact epoch-history CSV or JSON so future run reviews do not depend on parsing terminal progress output.

## Included Figures

The report includes copies of the original figures in:

- `outputs/reports/run_20260330_201546/figures/ff++_class_distribution.png`
- `outputs/reports/run_20260330_201546/figures/ff++_frame_sequence.png`
- `outputs/reports/run_20260330_201546/figures/ff++_real_vs_fake_grid.png`
