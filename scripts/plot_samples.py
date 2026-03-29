#!/usr/bin/env python3
"""
plot_samples.py — Visualise sample frames from the datasets
===========================================================

Generates a grid of sample frames comparing real vs. fake videos,
with optional frequency-domain (FFT / DCT) visualisation to motivate
the spatial-frequency dual-branch architecture.

Usage
-----
    python scripts/plot_samples.py --config configs/dataset_config.yaml
    python scripts/plot_samples.py --dataset ff++ --n-samples 8
    python scripts/plot_samples.py --show-frequency           # FFT overlay
    python scripts/plot_samples.py --output outputs/figures   # custom dir
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.fft import dctn

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.helpers import (
    load_config,
    get_video_paths,
    extract_frames,
    seed_everything,
)


# ------------------------------------------------------------------ #
#  Frequency domain helpers
# ------------------------------------------------------------------ #

def compute_fft_magnitude(image: np.ndarray) -> np.ndarray:
    """Return log-scaled magnitude spectrum of a grayscale image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    return magnitude


def compute_dct_energy(image: np.ndarray) -> np.ndarray:
    """Return log-scaled 2-D DCT of a grayscale image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
    dct = dctn(gray, type=2, norm="ortho")
    return np.log1p(np.abs(dct))


# ------------------------------------------------------------------ #
#  Sampling helpers
# ------------------------------------------------------------------ #

def _sample_videos(video_list: list, n: int) -> list:
    """Return up to *n* randomly chosen paths from *video_list*."""
    if len(video_list) <= n:
        return video_list
    return list(np.random.choice(video_list, size=n, replace=False))


def _get_middle_frame(video_path: str, size: int = 224) -> np.ndarray:
    """Extract a single frame from the middle of a video."""
    frames = extract_frames(video_path, max_frames=1, resize=(size, size))
    if len(frames) == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return frames[0]


# ------------------------------------------------------------------ #
#  Plot: Real vs Fake grid
# ------------------------------------------------------------------ #

def plot_real_vs_fake_grid(
    real_paths: list,
    fake_paths: list,
    n_samples: int = 6,
    title: str = "Real vs Fake Samples",
    save_path: str = None,
):
    """Side-by-side grid of real (top row) and fake (bottom row) frames."""
    real_sample = _sample_videos(real_paths, n_samples)
    fake_sample = _sample_videos(fake_paths, n_samples)

    n_cols = max(len(real_sample), len(fake_sample))
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6.5))

    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    for col, vpath in enumerate(real_sample):
        frame = _get_middle_frame(vpath)
        axes[0, col].imshow(frame)
        axes[0, col].set_title(Path(vpath).stem, fontsize=7)
        axes[0, col].axis("off")

    for col, vpath in enumerate(fake_sample):
        frame = _get_middle_frame(vpath)
        axes[1, col].imshow(frame)
        axes[1, col].set_title(Path(vpath).stem, fontsize=7)
        axes[1, col].axis("off")

    # Hide unused columns
    for col in range(len(real_sample), n_cols):
        axes[0, col].axis("off")
    for col in range(len(fake_sample), n_cols):
        axes[1, col].axis("off")

    # Row labels
    axes[0, 0].set_ylabel("REAL", fontsize=13, fontweight="bold",
                           color="green", rotation=0, labelpad=50)
    axes[1, 0].set_ylabel("FAKE", fontsize=13, fontweight="bold",
                           color="red", rotation=0, labelpad=50)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved → {save_path}")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Plot: Frequency analysis comparison
# ------------------------------------------------------------------ #

def plot_frequency_comparison(
    real_paths: list,
    fake_paths: list,
    n_samples: int = 4,
    title: str = "Spatial vs Frequency Domain",
    save_path: str = None,
):
    """For each sample: show RGB frame, FFT magnitude, DCT energy.

    Layout:
        Row 0-1: Real  (RGB | FFT | DCT) × n_samples
        Row 2-3: Fake  (RGB | FFT | DCT) × n_samples
    """
    real_sample = _sample_videos(real_paths, n_samples)
    fake_sample = _sample_videos(fake_paths, n_samples)

    n = max(len(real_sample), len(fake_sample))
    fig, axes = plt.subplots(2 * n, 3, figsize=(9, 3 * n * 2 + 1))

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

    def _fill_row(ax_row, frame, label_prefix):
        ax_row[0].imshow(frame)
        ax_row[0].set_title(f"{label_prefix} — RGB", fontsize=9)
        ax_row[0].axis("off")

        fft_mag = compute_fft_magnitude(frame)
        ax_row[1].imshow(fft_mag, cmap="inferno")
        ax_row[1].set_title(f"{label_prefix} — FFT", fontsize=9)
        ax_row[1].axis("off")

        dct_e = compute_dct_energy(frame)
        ax_row[2].imshow(dct_e, cmap="viridis")
        ax_row[2].set_title(f"{label_prefix} — DCT", fontsize=9)
        ax_row[2].axis("off")

    row = 0
    for vpath in real_sample:
        frame = _get_middle_frame(vpath)
        _fill_row(axes[row], frame, "Real")
        row += 1

    for vpath in fake_sample:
        frame = _get_middle_frame(vpath)
        _fill_row(axes[row], frame, "Fake")
        row += 1

    # Hide unused rows
    for r in range(row, axes.shape[0]):
        for c in range(3):
            axes[r, c].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved → {save_path}")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Plot: Frame sequence strip
# ------------------------------------------------------------------ #

def plot_frame_sequence(
    video_path: str,
    n_frames: int = 8,
    title: str = None,
    save_path: str = None,
):
    """Plot a horizontal strip of *n_frames* uniformly sampled from one video."""
    frames = extract_frames(video_path, max_frames=n_frames, resize=(224, 224))
    if len(frames) == 0:
        print(f"  ⚠  Could not extract frames from {video_path}")
        return

    fig, axes = plt.subplots(1, len(frames), figsize=(2.5 * len(frames), 3))
    if len(frames) == 1:
        axes = [axes]

    for i, (ax, frame) in enumerate(zip(axes, frames)):
        ax.imshow(frame)
        ax.set_title(f"Frame {i}", fontsize=8)
        ax.axis("off")

    fig.suptitle(title or Path(video_path).stem, fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved → {save_path}")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Plot: Dataset class balance
# ------------------------------------------------------------------ #

def plot_class_distribution(
    real_count: int,
    fake_count: int,
    dataset_name: str = "Dataset",
    save_path: str = None,
):
    """Bar chart showing real vs fake video counts."""
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Real", "Fake"],
        [real_count, fake_count],
        color=["#2ecc71", "#e74c3c"],
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_ylabel("Number of videos")
    ax.set_title(f"{dataset_name} — Class Distribution", fontweight="bold")

    # Value labels on bars
    for bar, val in zip(bars, [real_count, fake_count]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(real_count, fake_count) * 0.02,
            str(val),
            ha="center",
            fontweight="bold",
        )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved → {save_path}")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Orchestrator: FaceForensics++
# ------------------------------------------------------------------ #

def visualise_faceforensics(cfg: dict, args):
    """Generate all plots for FaceForensics++."""
    ff_cfg = cfg["faceforensics"]
    root = ff_cfg["root_dir"]
    compression = ff_cfg["compression"]
    out_dir = args.output

    real_dir = os.path.join(root, ff_cfg["original_dir"], compression)
    real_videos = get_video_paths(real_dir) if os.path.isdir(real_dir) else []

    fake_videos = []
    for method in ff_cfg["manipulation_methods"]:
        manip_dir = os.path.join(root, ff_cfg["manipulated_dir"], method, compression)
        if os.path.isdir(manip_dir):
            fake_videos.extend(get_video_paths(manip_dir))

    if not real_videos and not fake_videos:
        print("  ⚠  No FF++ videos found. Skipping visualisation.")
        return

    # 1. Grid
    plot_real_vs_fake_grid(
        real_videos, fake_videos,
        n_samples=args.n_samples,
        title="FaceForensics++ — Real vs Fake",
        save_path=os.path.join(out_dir, "ff++_real_vs_fake_grid.png"),
    )

    # 2. Class distribution
    plot_class_distribution(
        len(real_videos), len(fake_videos),
        dataset_name="FaceForensics++",
        save_path=os.path.join(out_dir, "ff++_class_distribution.png"),
    )

    # 3. Frequency analysis
    if args.show_frequency:
        plot_frequency_comparison(
            real_videos, fake_videos,
            n_samples=min(args.n_samples, 4),
            title="FaceForensics++ — Frequency Analysis",
            save_path=os.path.join(out_dir, "ff++_frequency_analysis.png"),
        )

    # 4. Frame strip (first real video)
    if real_videos:
        plot_frame_sequence(
            real_videos[0],
            n_frames=8,
            title="FF++ — Temporal Frame Sequence (Real)",
            save_path=os.path.join(out_dir, "ff++_frame_sequence.png"),
        )


# ------------------------------------------------------------------ #
#  Orchestrator: Celeb-DF
# ------------------------------------------------------------------ #

def visualise_celeb_df(cfg: dict, args):
    """Generate all plots for Celeb-DF v2."""
    cdf_cfg = cfg["celeb_df"]
    root = cdf_cfg["root_dir"]
    out_dir = args.output

    real_videos = []
    for rdir in cdf_cfg["real_dirs"]:
        full = os.path.join(root, rdir)
        if os.path.isdir(full):
            real_videos.extend(get_video_paths(full))

    fake_dir = os.path.join(root, cdf_cfg["fake_dir"])
    fake_videos = get_video_paths(fake_dir) if os.path.isdir(fake_dir) else []

    if not real_videos and not fake_videos:
        print("  ⚠  No Celeb-DF videos found. Skipping visualisation.")
        return

    # 1. Grid
    plot_real_vs_fake_grid(
        real_videos, fake_videos,
        n_samples=args.n_samples,
        title="Celeb-DF v2 — Real vs Fake",
        save_path=os.path.join(out_dir, "celeb_df_real_vs_fake_grid.png"),
    )

    # 2. Class distribution
    plot_class_distribution(
        len(real_videos), len(fake_videos),
        dataset_name="Celeb-DF v2",
        save_path=os.path.join(out_dir, "celeb_df_class_distribution.png"),
    )

    # 3. Frequency analysis
    if args.show_frequency:
        plot_frequency_comparison(
            real_videos, fake_videos,
            n_samples=min(args.n_samples, 4),
            title="Celeb-DF v2 — Frequency Analysis",
            save_path=os.path.join(out_dir, "celeb_df_frequency_analysis.png"),
        )

    # 4. Frame strip
    if real_videos:
        plot_frame_sequence(
            real_videos[0],
            n_frames=8,
            title="Celeb-DF — Temporal Frame Sequence (Real)",
            save_path=os.path.join(out_dir, "celeb_df_frame_sequence.png"),
        )


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Visualise deepfake dataset samples")
    p.add_argument("--config", default="configs/dataset_config.yaml")
    p.add_argument("--dataset", choices=["ff++", "celeb-df", "all"], default="all")
    p.add_argument("--n-samples", type=int, default=6, help="Samples per class")
    p.add_argument("--show-frequency", action="store_true",
                   help="Include FFT/DCT frequency-domain plots")
    p.add_argument("--output", default="outputs/figures",
                   help="Directory for saved figures")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(42)
    cfg = load_config(args.config)

    print("=" * 60)
    print("  DeepFake Detection — Sample Visualiser")
    print("=" * 60)

    if args.dataset in ("ff++", "all"):
        print("\n▶ FaceForensics++")
        visualise_faceforensics(cfg, args)

    if args.dataset in ("celeb-df", "all"):
        print("\n▶ Celeb-DF (v2)")
        visualise_celeb_df(cfg, args)

    print("\nDone. Check outputs in:", args.output)


if __name__ == "__main__":
    main()
