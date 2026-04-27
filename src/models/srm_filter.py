"""
src/models/srm_filter.py — SRM Frequency Branch
================================================

Implements the Steganalysis Rich Model (SRM) high-pass filter bank
followed by a lightweight convolutional encoder.

The SRM kernels are initialized with the three classical SRM high-pass
filters and can either be frozen (fixed forensic prior) or trained
end-to-end (learnable specialization).  Learnable mode lets the front-end
adapt to manipulation families (e.g. NeuralTextures) whose artifacts sit
outside the original SRM frequency band.

Architecture
------------
Input  (B, 3, H, W)
  → SRMConv2d         (B, 9, H, W)   — 3 kernels × 3 channels
  → TanH clamp ×10                   — matches original SRM implementation
  → ResidualBlock × 3 (B, 64 → 128 → 256, stride 2)
  → AdaptiveAvgPool2d (B, 256, 1, 1)
  → Flatten + Linear  (B, out_dim)

References
----------
Chen, H. et al. (2021). "SRMNet: Steganalysis Rich Model for deep fake detection."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ------------------------------------------------------------------ #
#  Fixed SRM kernels
# ------------------------------------------------------------------ #

def _build_srm_kernels() -> torch.Tensor:
    """Return the three classical SRM high-pass kernels as a (3, 1, 5, 5) tensor."""
    # Kernel 1 — 2nd-order residual (center difference)
    k1 = np.array([
        [0,  0,  0,  0, 0],
        [0,  0,  0,  0, 0],
        [0,  0,  1, -2, 1],   # ← 1-D 2nd derivative in row
        [0,  0,  0,  0, 0],
        [0,  0,  0,  0, 0],
    ], dtype=np.float32) / 2.0

    # Kernel 2 — 2D Laplacian
    k2 = np.array([
        [ 0,  0,  0,  0,  0],
        [ 0, -1,  2, -1,  0],
        [ 0,  2, -4,  2,  0],
        [ 0, -1,  2, -1,  0],
        [ 0,  0,  0,  0,  0],
    ], dtype=np.float32) / 4.0

    # Kernel 3 — 3×3 average residual
    k3 = np.array([
        [-1, 2, -2,  2, -1],
        [ 2, -6, 8, -6,  2],
        [-2,  8, -12, 8, -2],
        [ 2, -6, 8, -6,  2],
        [-1, 2, -2,  2, -1],
    ], dtype=np.float32) / 12.0

    kernels = np.stack([k1, k2, k3], axis=0)           # (3, 5, 5)
    return torch.from_numpy(kernels.copy()).unsqueeze(1)  # (3, 1, 5, 5)


# ------------------------------------------------------------------ #
#  SRM Convolution — learnable by default, fixed when disabled
# ------------------------------------------------------------------ #

class SRMConv2d(nn.Module):
    """Apply 3 SRM-initialized kernels to each input channel independently.

    Each of the 3 kernels is applied to R, G, B separately, yielding
    9 feature maps total.

    Parameters
    ----------
    learnable : bool
        When True, the kernels are stored as ``nn.Parameter`` and trained
        end-to-end (initialized with the classical SRM high-pass filters).
        When False, they are stored as a buffer and stay fixed.
    """

    def __init__(self, learnable: bool = True):
        super().__init__()
        kernel = _build_srm_kernels()                   # (3, 1, 5, 5)
        # Repeat for each RGB channel via grouped convolution
        # weight shape: (out_ch, in_ch/groups, kH, kW)
        # We want: out_ch=9, groups=3 → each group has 3 filters, 1 in_ch
        # Expand kernels to 3 sets of 3: (9, 1, 5, 5)
        weight = kernel.repeat(3, 1, 1, 1).contiguous()  # (9, 1, 5, 5)
        if learnable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        # Apply depthwise grouped conv so each SRM kernel sees one channel
        # groups=3 → channel 0 → filters 0..2, ch1 → 3..5, ch2 → 6..8
        return F.conv2d(x, self.weight, padding=2, groups=3)


# ------------------------------------------------------------------ #
#  Lightweight residual block
# ------------------------------------------------------------------ #

class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


# ------------------------------------------------------------------ #
#  Full Frequency Branch
# ------------------------------------------------------------------ #

class SRMFrequencyBranch(nn.Module):
    """SRM-initialized filter bank + trainable residual encoder.

    Parameters
    ----------
    out_dim : int
        Dimension of the output feature vector.
    srm_learnable : bool
        When True (default), the SRM filter bank is initialized with the
        classical kernels but trained end-to-end. When False, the kernels
        stay fixed (legacy behavior).
    """

    def __init__(self, out_dim: int = 256, srm_learnable: bool = True):
        super().__init__()

        self.srm = SRMConv2d(learnable=srm_learnable)   # (B, 9, H, W)

        # Trainable encoder: 9 → 64 → 128 → 256
        self.encoder = nn.Sequential(
            nn.Conv2d(9, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            _ResBlock(64,  128, stride=2),
            _ResBlock(128, 256, stride=2),
            _ResBlock(256, 256, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W)  — input RGB images, values in [0, 1]

        Returns
        -------
        (B, out_dim)
        """
        noise = self.srm(x)                             # (B, 9, H, W)
        noise = torch.tanh(noise * 10.0)                # clamp as in original SRM
        feat  = self.encoder(noise)                     # (B, 256, 1, 1)
        return self.proj(feat)                          # (B, out_dim)
