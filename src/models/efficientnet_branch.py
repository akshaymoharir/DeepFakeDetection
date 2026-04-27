"""
src/models/efficientnet_branch.py — Spatial Branch (EfficientNet)
=================================================================

Wraps a ``timm`` EfficientNet variant pretrained on ImageNet, replacing the
classification head with a learnable linear projection. The specific backbone
is selected via the ``backbone`` argument (e.g. ``efficientnet_b4``,
``tf_efficientnet_b7``).

Architecture
------------
Input  (B, 3, H, W)
  → EfficientNet-{variant} (ImageNet pretrained, features only)
  → AdaptiveAvgPool2d → flatten   (B, num_features)
  → Linear num_features → out_dim (B, out_dim)
  → LayerNorm

Reference feature widths: B4=1792, B7=2560.

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
    """EfficientNet backbone with a projection head.

    Parameters
    ----------
    out_dim : int
        Output feature dimension.
    pretrained : bool
        Load ImageNet weights from ``timm`` (requires internet on first run).
    dropout : float
        Dropout probability before the projection layer.
    backbone : str
        ``timm`` model name. Common choices: ``efficientnet_b4`` (18.5M params,
        native 380px), ``tf_efficientnet_b7`` (66M params, native 600px).
    """

    def __init__(
        self,
        out_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.3,
        backbone: str = "tf_efficientnet_b7",
    ):
        super().__init__()

        if not _TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for the spatial branch. "
                "Install with: pip install timm>=0.9.0"
            )

        self.backbone_name = backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        backbone_out_dim = self.backbone.num_features

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
        features = self.backbone(x)     # (B, num_features) — global avg-pooled
        return self.head(features)      # (B, out_dim)
