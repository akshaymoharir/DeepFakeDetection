"""
src/models/hsf_cvit.py — HSF-CVIT Top-Level Model
==================================================

Composes the two branches and the fusion head into a single
``nn.Module`` with a clean forward interface.

    HSF-CVIT (Hybrid Spatial-Frequency Cross-Attention ViT)
    ────────────────────────────────────────────────────────
    Input image (B, 3, 224, 224)
        │
        ├─► EfficientNetSpatialBranch  →  (B, spatial_dim)
        │
        └─► SRMFrequencyBranch        →  (B, freq_dim)
                │
                ▼
        CrossAttentionViT fusion
                │
                ▼
        logit (B, 1)   — raw output, pass through sigmoid for probability

Usage
-----
    from src.models.hsf_cvit import build_model

    model = build_model(train_cfg)
    logits = model(images)          # (B, 1)
    probs  = torch.sigmoid(logits)  # (B, 1)  — P(fake)
"""

import torch
import torch.nn as nn

from src.models.efficientnet_branch import EfficientNetSpatialBranch
from src.models.swt_filter import SWTFrequencyBranch
from src.models.cross_attention_vit import CrossAttentionViT


class HSF_CVIT(nn.Module):
    """Hybrid Spatial-Frequency Cross-Attention ViT for deepfake detection.

    Parameters
    ----------
    spatial_out_dim : int
        EfficientNet projection output dimension.
    freq_out_dim : int
        SRM encoder output dimension.
    fusion_dim : int
        Internal cross-attention transformer width.
    fusion_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate applied throughout.
    pretrained_spatial : bool
        Whether to load ImageNet weights for the EfficientNet backbone.
    srm_learnable : bool
        Whether the SRM front-end kernels are trained end-to-end.
    """

    def __init__(
        self,
        spatial_out_dim: int = 512,
        freq_out_dim: int = 256,
        fusion_dim: int = 256,
        fusion_heads: int = 4,
        dropout: float = 0.3,
        pretrained_spatial: bool = True,
        srm_learnable: bool = True,
        spatial_backbone: str = "tf_efficientnet_b7",
    ):
        super().__init__()

        self.spatial_branch = EfficientNetSpatialBranch(
            out_dim=spatial_out_dim,
            pretrained=pretrained_spatial,
            dropout=dropout,
            backbone=spatial_backbone,
        )

        self.freq_branch = SWTFrequencyBranch(
            out_dim=freq_out_dim,
        )

        self.fusion_head = CrossAttentionViT(
            spatial_dim=spatial_out_dim,
            freq_dim=freq_out_dim,
            fusion_dim=fusion_dim,
            num_heads=fusion_heads,
            dropout=dropout,
        )

        # EfficientNet consumes ImageNet-normalized tensors.
        # Before the SRM branch, map them back to image-space RGB in [0, 1].
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W)

        Returns
        -------
        logits : (B, 1)  — unscaled; apply sigmoid for probabilities
        """
        s = self.spatial_branch(x)          # (B, spatial_out_dim)
        x_rgb = (x * self.imagenet_std + self.imagenet_mean).clamp(0.0, 1.0)
        f = self.freq_branch(x_rgb)         # (B, freq_out_dim)
        return self.fusion_head(s, f)       # (B, 1)

    # ------------------------------------------------------------------
    #  Convenience wrappers (delegated to spatial branch)
    # ------------------------------------------------------------------

    def freeze_spatial(self) -> None:
        """Freeze EfficientNet backbone (called during warm-up)."""
        self.spatial_branch.freeze()

    def unfreeze_spatial(self) -> None:
        """Unfreeze EfficientNet backbone (called after warm-up)."""
        self.spatial_branch.unfreeze()

    def count_parameters(self) -> dict:
        """Return parameter counts per component."""
        def _count(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        s_total, s_train = _count(self.spatial_branch)
        f_total, f_train = _count(self.freq_branch)
        h_total, h_train = _count(self.fusion_head)
        total  = s_total + f_total + h_total
        train  = s_train + f_train + h_train

        return {
            "spatial_branch":  {"total": s_total, "trainable": s_train},
            "freq_branch":     {"total": f_total, "trainable": f_train},
            "fusion_head":     {"total": h_total, "trainable": h_train},
            "grand_total":     {"total": total,   "trainable": train},
        }


# ------------------------------------------------------------------ #
#  Factory
# ------------------------------------------------------------------ #

def build_model(train_cfg: dict) -> HSF_CVIT:
    """Instantiate ``HSF_CVIT`` from a training config dict.

    Parameters
    ----------
    train_cfg : dict
        Loaded ``train_config.yaml``.

    Returns
    -------
    HSF_CVIT
    """
    m = train_cfg["model"]
    return HSF_CVIT(
        spatial_out_dim=m.get("spatial_out_dim", 512),
        freq_out_dim=m.get("freq_out_dim", 256),
        fusion_dim=m.get("fusion_dim", 256),
        fusion_heads=m.get("fusion_heads", 4),
        dropout=m.get("dropout", 0.3),
        pretrained_spatial=m.get("pretrained_spatial", True),
        srm_learnable=m.get("srm_learnable", True),
        spatial_backbone=m.get("spatial_backbone", "tf_efficientnet_b7"),
    )
