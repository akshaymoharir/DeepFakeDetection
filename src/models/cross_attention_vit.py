"""
src/models/cross_attention_vit.py — Cross-Attention Fusion Head
===============================================================

Performs cross-attention between the spatial token (from EfficientNet-B4)
and the frequency token (from the SRM branch), then classifies the fused
representation.

Architecture
------------
Spatial token s  (B, spatial_dim) → project → (B, 1, fusion_dim)  [query Q]
Freq token f     (B, freq_dim)    → project → (B, 1, fusion_dim)  [key K, value V]

MultiheadAttention(Q, K, V) → (B, 1, fusion_dim)
LayerNorm → MLP (fusion_dim → fusion_dim*2 → fusion_dim) → LayerNorm
→ squeeze → (B, fusion_dim)
→ Dropout → Linear(fusion_dim, 1)
→ raw logit (no sigmoid — BCEWithLogitsLoss in trainer)

Cross-attention direction: spatial queries "attend" to frequency features,
letting the model selectively weight which frequency artifacts matter for
each spatial pattern.
"""

import torch
import torch.nn as nn
import math


class CrossAttentionViT(nn.Module):
    """Cross-attention fusion of spatial and frequency tokens.

    Parameters
    ----------
    spatial_dim : int
        Dimension of the spatial token from EfficientNet.
    freq_dim : int
        Dimension of the frequency token from SRM encoder.
    fusion_dim : int
        Internal transformer hidden size (queries/keys/values).
    num_heads : int
        Number of attention heads (``fusion_dim`` must be divisible by this).
    dropout : float
        Dropout on attention weights and MLP.
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

        assert fusion_dim % num_heads == 0, (
            f"fusion_dim ({fusion_dim}) must be divisible by num_heads ({num_heads})"
        )

        # Project branches into shared fusion_dim space
        self.proj_spatial = nn.Linear(spatial_dim, fusion_dim)
        self.proj_freq    = nn.Linear(freq_dim, fusion_dim)

        # Learnable [CLS]-like position embeddings for the two tokens
        self.pos_spatial = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)
        self.pos_freq    = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)

        # Cross-attention: spatial queries attend to frequency key/value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,       # (B, seq, dim) convention
        )
        self.norm1 = nn.LayerNorm(fusion_dim)

        # Feed-forward MLP
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )
        self.norm2 = nn.LayerNorm(fusion_dim)

        # Self-attention over the concatenated pair (optional refinement)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm3 = nn.LayerNorm(fusion_dim)

        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj_spatial.weight)
        nn.init.xavier_uniform_(self.proj_freq.weight)
        nn.init.zeros_(self.proj_spatial.bias)
        nn.init.zeros_(self.proj_freq.bias)
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
        logits : (B, 1)  — raw logits (apply sigmoid externally for probabilities)
        """
        B = spatial_feat.size(0)

        # Project to fusion_dim and add position embeddings
        q = self.proj_spatial(spatial_feat).unsqueeze(1) + self.pos_spatial  # (B, 1, D)
        kv = self.proj_freq(freq_feat).unsqueeze(1)      + self.pos_freq      # (B, 1, D)

        # Cross-attention: spatial queries attend to frequency context
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)             # (B, 1, D)
        q = self.norm1(q + attn_out)                                          # residual

        # MLP block
        q = self.norm2(q + self.mlp(q))                                       # (B, 1, D)

        # Self-attention over [spatial, freq] pair (captures joint context)
        pair = torch.cat([q, kv], dim=1)                                      # (B, 2, D)
        pair_out, _ = self.self_attn(pair, pair, pair)
        pair = self.norm3(pair + pair_out)

        # Use the spatial (first) token as the final representation
        cls_token = pair[:, 0, :]                                             # (B, D)

        return self.classifier(self.dropout(cls_token))                       # (B, 1)
