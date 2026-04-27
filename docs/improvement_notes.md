> **Temporary working doc.** Captures the assessment of the 2026-04-26 training
> run and a prioritized roadmap for pushing past AUC ~0.80. Delete or merge into
> the technical reference once the items below are landed.

# HSF-CVIT Improvement Roadmap

## Current State (2026-04-26)

Run: 32 epochs (early-stopped, patience 15), best epoch 17.

| Metric                     | Value  |
|----------------------------|--------|
| Test ROC-AUC               | 0.795  |
| Test Average Precision     | 0.937  |
| Test F1 @ thr=0.68         | 0.729  |
| Test F1 @ thr=0.50         | 0.754  |
| Test Recall @ thr=0.50     | 0.636  |
| Test Precision @ thr=0.50  | 0.927  |
| Best val AUC               | 0.793  |
| Best val threshold (T*)    | 0.68   |

Per-method test AUC: Deepfakes 0.876 > Face2Face 0.811 > FaceSwap 0.778 > NeuralTextures 0.714.

## Diagnosis

1. **Heavy overfitting.** Train AUC reaches 0.967 while val plateaus at 0.79
   (gap ≈ 0.18). The model memorizes training-frame artifacts past epoch ~17.
2. **Calibration drift in cosine tail.** Val loss grows from 0.78 → 1.55 while
   AUC stays roughly flat — overconfident wrong predictions.
3. **Threshold optimizer is noisy.** T* swings between 0.10 and 0.92 across
   epochs; val=700 is small for a 1% step grid. Default 0.50 actually beats
   the optimized threshold on F1 (0.754 vs 0.729).
4. **NeuralTextures is the bottleneck.** Without it, average AUC ≈ 0.82.
   Fixed SRM kernels target blending/resampling artifacts; texture-synthesis
   manipulations live in a different signal regime.
5. **Below FF++ literature.** Xception, EfficientNet-B7, F3-Net all hit
   ~0.95 AUC on FF++ c23. We are ~15 pp behind state-of-the-art.

## What is solid

- Engineering: official splits, deterministic eval, threshold optimization,
  per-method reports, video-level eval, inference API, AMP, gradient clip,
  weighted sampler, pos_weight calibration, reproducibility seeded.
- Architecture is principled and well-documented.
- The training loop is not the bottleneck.

## Improvement Roadmap (highest leverage first)

### 1. Learnable SRM front-end  *(in progress)*

**Problem.** `SRMConv2d` registers the three classical kernels as buffers, so
they never receive gradients. NeuralTextures artifacts sit in a different
frequency band than the high-pass priors capture.

**Plan.** Initialize the 9-channel filter bank with the classical SRM kernels
(2nd-order residual, 2D Laplacian, 5×5 average residual) but expose them as
`nn.Parameter` so the optimizer can adapt them. Add a config flag
`model.srm_learnable` so this can be toggled for ablation.

**Cost.** Half a day. Adds 9 × 5 × 5 = 225 trainable params (negligible).

**Expected impact.** +1 to +3 pp AUC, mostly on NeuralTextures.

### 2. Stronger augmentation: CutMix / face-region cutout

**Problem.** Train AUC 0.97, val 0.79 — the model is overfitting. Current
augmentation is colour jitter, rotation, blur, JPEG compression — all
appearance-level.

**Plan.** Add CutMix (or face-region cutout, à la FaceShifter regularization)
between mini-batches. This forces the model to use multiple regions of the
face rather than over-relying on a single artifact pocket.

**Cost.** Half a day. Just a `BatchCollator` or training-loop augmentation
hook.

**Expected impact.** +1 to +2 pp AUC, narrows the train/val gap.

### 3. Heavier backbone: EfficientNet-B7 or Xception

**Problem.** EfficientNet-B4 (18.5M params) is the lightest backbone the FF++
literature uses. Xception is the canonical FF++ baseline.

**Plan.** Swap `efficientnet_b4` → `xception` (28M params) or
`tf_efficientnet_b7` (66M params) via `timm`.  The branch wrapper interface
already isolates the backbone — only `EfficientNetSpatialBranch` needs
touching.

**Cost.** Half a day for the swap. Training time grows ~2-3×.

**Expected impact.** +3 to +5 pp AUC.

### 4. Patch-token ViT fusion

**Problem.** Current fusion compresses each branch to one vector before
attention. Two tokens cannot represent spatial localization, so the
"transformer" portion is barely doing transformer-style work.

**Plan.** Replace single-token fusion with patch tokens — keep the spatial
feature map from EfficientNet (B, 1792, 7, 7) → 49 patch tokens, plus the
frequency branch as a parallel sequence. Run a 2-layer cross-attention ViT
across both sequences.

**Cost.** 1-2 days. Touches `cross_attention_vit.py` and the spatial branch
output interface.

**Expected impact.** +2 to +4 pp AUC, plus better localization signal.

### 5. Stronger regularization

**Problem.** Same as #2 — train AUC saturates near 1.0 while val stagnates.

**Plan.** Drop the LR to 1e-4 once warmup ends, add `drop_path=0.2` to the
EfficientNet stem, raise dropout 0.3 → 0.4 in the fusion head. Already
have `weight_decay=1e-4`; no change needed there.

**Cost.** A few config tweaks.

**Expected impact.** +0.5 to +1 pp AUC, mostly via reduced overfitting.

## Recommendation for course deliverable

1. Land item #1 (learnable SRM) — small, low-risk, targets the weakest
   per-method result.
2. Run one training cycle to measure impact.
3. If time permits, also land #2 (CutMix). #3 and #4 are larger commitments.

After #1 + #2 we should expect AUC in the 0.82-0.85 range and a meaningfully
narrower train/val gap.
