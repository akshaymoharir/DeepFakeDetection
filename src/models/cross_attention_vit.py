"""
src/models/cross_attention_vit.py — Token Fusion Transformer Head
=================================================================

Fuses the spatial token from EfficientNet-B4 and the frequency token
from the SRM branch with a small transformer over a 3-token sequence:

    [CLS] + spatial + frequency

This replaces the earlier single-query/single-key cross-attention block,
which had effectively trivial attention weights. The public interface stays
the same so the rest of the training pipeline does not change.
"""

import torch
import torch.nn as nn


class CrossAttentionViT(nn.Module):
    """Transformer-style fusion of spatial and frequency tokens.

    Parameters
    ----------
    spatial_dim : int
        Dimension of the spatial token from EfficientNet.
    freq_dim : int
        Dimension of the frequency token from the SRM encoder.
    fusion_dim : int
        Shared embedding width used inside the fusion transformer.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate used throughout the fusion head.
    """

    def __init__(
        self,
        spatial_dim: int = 512,
        freq_dim: int = 256,
        fusion_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        if fusion_dim % num_heads != 0:
            raise ValueError(
                f"fusion_dim ({fusion_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.proj_spatial = nn.Linear(spatial_dim, fusion_dim)
        self.proj_freq = nn.Linear(freq_dim, fusion_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 3, fusion_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            norm=nn.LayerNorm(fusion_dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj_spatial.weight)
        nn.init.zeros_(self.proj_spatial.bias)
        nn.init.xavier_uniform_(self.proj_freq.weight)
        nn.init.zeros_(self.proj_freq.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        spatial_feat: torch.Tensor,
        freq_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        spatial_feat : (B, spatial_dim)
        freq_feat    : (B, freq_dim)

        Returns
        -------
        logits : (B, 1) — raw logits
        """
        batch_size = spatial_feat.size(0)

        spatial_token = self.proj_spatial(spatial_feat).unsqueeze(1)  # (B, 1, D)
        freq_token = self.proj_freq(freq_feat).unsqueeze(1)            # (B, 1, D)
        cls_token = self.cls_token.expand(batch_size, -1, -1)          # (B, 1, D)

        tokens = torch.cat([cls_token, spatial_token, freq_token], dim=1)
        tokens = tokens + self.pos_embed
        encoded = self.encoder(tokens)

        cls_rep = encoded[:, 0, :]
        return self.classifier(self.dropout(cls_rep))
