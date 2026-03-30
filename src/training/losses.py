"""
src/training/losses.py — Loss Functions
========================================

BCEWithLogitsLoss with optional label smoothing.

Label smoothing prevents over-confident predictions by converting
hard 0/1 labels into:
    real (0) → ε/2
    fake (1) → 1 - ε/2

This regularizes against annotation noise in manipulation datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothedBCELoss(nn.Module):
    """Binary cross-entropy with optional label smoothing.

    Parameters
    ----------
    smoothing : float
        Label smoothing coefficient ε ∈ [0, 1).
        0.0 → standard BCEWithLogitsLoss.
    pos_weight : float or None
        Class imbalance weight for the positive (fake) class.
        Pass ``n_real / n_fake`` if classes are imbalanced.
    """

    def __init__(self, smoothing: float = 0.1, pos_weight: float = None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = (
            torch.tensor([pos_weight]) if pos_weight is not None else None
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B,) or (B, 1) — raw model outputs
        targets : (B,)           — 0.0 for real, 1.0 for fake

        Returns
        -------
        scalar loss
        """
        logits  = logits.view(-1)
        targets = targets.view(-1).float()

        if self.smoothing > 0.0:
            # Smooth: 0 → ε/2,  1 → 1 - ε/2
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing

        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


def build_criterion(train_cfg: dict) -> SmoothedBCELoss:
    """Build the loss function from a training config dict."""
    smoothing = train_cfg["training"].get("label_smoothing", 0.1)
    return SmoothedBCELoss(smoothing=smoothing)
