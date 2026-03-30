"""
src/models/efficientnet_branch.py — Spatial Branch (EfficientNet-B4)
=====================================================================

Wraps ``timm``'s EfficientNet-B4 pretrained on ImageNet,
replacing the classification head with a learnable linear projection.

Architecture
------------
Input  (B, 3, 224, 224)
  → EfficientNet-B4 (ImageNet pretrained, features only)
  → AdaptiveAvgPool2d → flatten   (B, 1792)
  → Linear 1792 → out_dim         (B, out_dim)
  → LayerNorm

The full backbone can optionally be frozen for a warm-up period
(``freeze()`` / ``unfreeze()`` helpers are provided for use by the trainer).
"""

import torch
import torch.nn as nn

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


class EfficientNetSpatialBranch(nn.Module):
    """EfficientNet-B4 backbone with a projection head.

    Parameters
    ----------
    out_dim : int
        Output feature dimension.
    pretrained : bool
        Load ImageNet weights from ``timm`` (requires internet on first run).
    dropout : float
        Dropout probability before the projection layer.
    """

    def __init__(
        self,
        out_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        if not _TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for EfficientNet-B4. "
                "Install with: pip install timm>=0.9.0"
            )

        # Load backbone without classifier head
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,          # removes the head → returns feature maps
            global_pool="avg",      # global average pool → (B, 1792)
        )
        backbone_out_dim = self.backbone.num_features  # 1792 for EfficientNet-B4

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    # ------------------------------------------------------------------
    #  Freeze / unfreeze helpers (called by Trainer during warm-up)
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze all backbone parameters (train head only)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    @property
    def is_frozen(self) -> bool:
        return not next(self.backbone.parameters()).requires_grad

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W) — input RGB tensors, ImageNet-normalised

        Returns
        -------
        (B, out_dim)
        """
        features = self.backbone(x)     # (B, 1792) — global avg-pooled
        return self.head(features)      # (B, out_dim)
