"""
src/models/swt_filter.py — Selective Multi-Scale SWT Frequency Branch
======================================================================

Replaces SRMFrequencyBranch with Stationary Wavelet Transform (SWT).

Why SWT over SRM:
- SRM: 3 fixed noise patterns, single scale, shift-variant
- SWT: full frequency decomposition, multi-scale, shift-invariant

Design choices:
- 2 levels (Level 1 = pixel artifacts, Level 2 = texture artifacts)
- Skip LL subbands (EfficientNet spatial branch already handles low-freq)
- Grayscale input (6 maps: LH1, HL1, HH1, LH2, HL2, HH2)
- Same encoder structure as SRM — only input channels change (9 → 6)
- Pure PyTorch (Haar wavelet via fixed 2D conv) — no external dependencies

Architecture
------------
Input  (B, 3, H, W)
  → Grayscale          (B, 1, H, W)
  → SWT Level 1        LH1, HL1, HH1  (B, 3, H, W)
  → SWT Level 2        LH2, HL2, HH2  (B, 3, H, W)
  → Concat             (B, 6, H, W)
  → Conv(6→64) + ResBlocks             (B, 256, H/8, W/8)
  → AdaptiveAvgPool2d                  (B, 256, 1, 1)
  → Flatten + Linear                   (B, out_dim)

References
----------
Baru et al. (WACV 2025). "Wavelet-Driven Generalizable Framework for
    Deepfake Face Forgery Detection."
Peng et al. (ACM MM 2025). "WMamba: Wavelet-based Mamba for Face Forgery
    Detection."
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Lightweight residual block (same as srm_filter.py)
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
        return F.relu(out + self.shortcut(x), inplace=True)


# ------------------------------------------------------------------ #
#  Full SWT Frequency Branch
# ------------------------------------------------------------------ #

class SWTFrequencyBranch(nn.Module):
    """Selective multi-scale Haar SWT + trainable residual encoder.

    Parameters
    ----------
    out_dim : int
        Dimension of the output feature vector. (same as SRM default: 256)
    levels : int
        Number of SWT decomposition levels. Default 2.
    """

    def __init__(self, out_dim: int = 256, levels: int = 2):
        super().__init__()
        self.levels = levels
        n_maps = levels * 3      # 3 HF subbands × 2 levels = 6 (LL skipped)

        # ── Register Haar wavelet kernels as fixed non-trainable buffers ──
        # Haar 1D filters
        s  = 1.0 / math.sqrt(2)
        lo = torch.tensor([s,  s])   # low-pass  (smoothing)
        hi = torch.tensor([s, -s])   # high-pass (edge/noise)

        # Build 2D separable kernels via outer product for each level
        #   LH = hi_row × lo_col  → horizontal edges
        #   HL = lo_row × hi_col  → vertical edges
        #   HH = hi_row × hi_col  → diagonal / fine noise
        for lvl in range(1, levels + 1):
            for sname, rf, cf in [('LH', hi, lo), ('HL', lo, hi), ('HH', hi, hi)]:
                k2d = torch.outer(rf, cf).unsqueeze(0).unsqueeze(0)  # (1,1,2,2)
                self.register_buffer(f'k_{sname}_{lvl}', k2d)

        # ── Encoder: same structure as SRM, input channels 9→6 ──
        self.encoder = nn.Sequential(
            nn.Conv2d(n_maps, 64, 3, padding=1, bias=False),
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

    # ── Private helper ─────────────────────────────────────────────────

    def _swt_level(self, gray: torch.Tensor, lvl: int):
        """
        Apply one level of 2D Haar SWT to grayscale image.

        Padding formula: pad = dilation on top-left, 0 on bottom-right.
        This gives exact same-size output for a 2-tap kernel at any dilation.

        Parameters
        ----------
        gray : (B, 1, H, W)
        lvl  : decomposition level (1-indexed)

        Returns
        -------
        List of 3 tensors [LH, HL, HH], each (B, 1, H, W)
        """
        dil = 2 ** (lvl - 1)       # dilation: 1 at level1, 2 at level2
        pad = dil                  # asymmetric pad preserves spatial size

        subbands = []
        for sname in ['LH', 'HL', 'HH']:
            ker = getattr(self, f'k_{sname}_{lvl}')          # (1, 1, 2, 2)
            x_pad = F.pad(gray, (pad, 0, pad, 0))            # left, right, top, bottom
            out   = F.conv2d(x_pad, ker, dilation=dil)       # (B, 1, H, W)
            subbands.append(out)

        return subbands

    # ── Forward ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W) — input RGB images, values in [0, 1]

        Returns
        -------
        (B, out_dim)
        """
        # RGB → grayscale using standard luminance weights
        gray = (0.299 * x[:, 0:1]
              + 0.587 * x[:, 1:2]
              + 0.114 * x[:, 2:3])                           # (B, 1, H, W)

        # Collect HF subbands from all levels (skip LL)
        all_subbands = []
        for lvl in range(1, self.levels + 1):
            all_subbands.extend(self._swt_level(gray, lvl))

        freq_maps = torch.cat(all_subbands, dim=1)           # (B, 6, H, W)

        feat = self.encoder(freq_maps)                        # (B, 256, 1, 1)
        return self.proj(feat)                                # (B, out_dim)
