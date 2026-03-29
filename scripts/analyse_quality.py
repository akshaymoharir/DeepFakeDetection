#!/usr/bin/env python3
"""
analyse_quality.py — Image / video quality analysis
====================================================

Computes per-frame quality metrics across the datasets to understand
the distribution of image quality and guide preprocessing decisions.

Metrics computed:
  • BRISQUE-like sharpness (Laplacian variance)
  • Mean brightness (luminance channel)
  • Colour saturation (mean S in HSV)
  • Mean SSIM between consecutive frames (temporal consistency)

Results are plotted as histograms and box-plots comparing real vs fake.

Usage
-----
    python scripts/analyse_quality.py --config configs/dataset_config.yaml
    python scripts/analyse_quality.py --dataset ff++ --n-videos 100
"""

import argparse
import os
import sys
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.helpers import (
    load_config,
    get_video_paths,
    extract_frames,
    seed_everything,
)


# ------------------------------------------------------------------ #
#  Per-frame quality metrics
# ------------------------------------------------------------------ #

def laplacian_variance(frame_rgb: np.ndarray) -> float:
    """Sharpness proxy: variance of Laplacian on grayscale image."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def mean_brightness(frame_rgb: np.ndarray) -> float:
    """Average luminance (Y channel of YCrCb)."""
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    return float(ycrcb[:, :, 0].mean())


def mean_saturation(frame_rgb: np.ndarray) -> float:
    """Average saturation (S channel of HSV)."""
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean())


# ------------------------------------------------------------------ #
#  Collect metrics
# ------------------------------------------------------------------ #

def analyse_videos(
    video_paths: list,
    label: str,
    max_frames: int = 8,
    resize: int = 224,
) -> list:
    """Compute quality metrics for a list of videos.

    Returns a list of dicts: {label, sharpness, brightness, saturation}.
    """
    records = []
    for vpath in tqdm(video_paths, desc=f"  {label}", unit="vid"):
        try:
            frames = extract_frames(vpath, max_frames=max_frames,
                                    resize=(resize, resize))
        except Exception:
            continue

        for frame in frames:
            records.append({
                "label": label,
                "sharpness": laplacian_variance(frame),
                "brightness": mean_brightness(frame),
                "saturation": mean_saturation(frame),
            })
    return records


# ------------------------------------------------------------------ #
#  Plots
# ------------------------------------------------------------------ #

def _plot_metric_comparison(
    real_vals: list,
    fake_vals: list,
    metric_name: str,
    xlabel: str,
    save_path: str,
):
    """Overlapping histograms for one metric, real vs fake."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(real_vals, bins=50, alpha=0.6, color="#2ecc71", label="Real", density=True)
    axes[0].hist(fake_vals, bins=50, alpha=0.6, color="#e74c3c", label="Fake", density=True)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"{metric_name} — Histogram")
    axes[0].legend()

    # Box plot
    bp = axes[1].boxplot(
        [real_vals, fake_vals],
        labels=["Real", "Fake"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#2ecc71")
    bp["boxes"][1].set_facecolor("#e74c3c")
    axes[1].set_ylabel(xlabel)
    axes[1].set_title(f"{metric_name} — Box Plot")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved → {save_path}")


def plot_all_metrics(records: list, dataset_name: str, out_dir: str):
    """Generate comparison plots for every metric."""
    real_recs = [r for r in records if r["label"] == "real"]
    fake_recs = [r for r in records if r["label"] == "fake"]

    if not real_recs or not fake_recs:
        print("  ⚠  Need both real and fake data for comparison plots.")
        return

    metrics = [
        ("sharpness", "Sharpness (Laplacian Var)", "Laplacian Variance"),
        ("brightness", "Brightness (Mean Y)", "Luminance"),
        ("saturation", "Saturation (Mean S)", "Saturation"),
    ]

    for key, title, xlabel in metrics:
        _plot_metric_comparison(
            [r[key] for r in real_recs],
            [r[key] for r in fake_recs],
            f"{dataset_name} — {title}",
            xlabel,
            os.path.join(out_dir, f"{dataset_name.lower().replace(' ', '_')}_{key}.png"),
        )


# ------------------------------------------------------------------ #
#  Orchestrators
# ------------------------------------------------------------------ #

def run_faceforensics(cfg: dict, args):
    ff_cfg = cfg["faceforensics"]
    root = ff_cfg["root_dir"]
    compression = ff_cfg["compression"]

    real_dir = os.path.join(root, ff_cfg["original_dir"], compression)
    real_vids = get_video_paths(real_dir) if os.path.isdir(real_dir) else []

    fake_vids = []
    for method in ff_cfg["manipulation_methods"]:
        d = os.path.join(root, ff_cfg["manipulated_dir"], method, compression)
        if os.path.isdir(d):
            fake_vids.extend(get_video_paths(d))

    # Limit for speed
    if args.n_videos:
        real_vids = real_vids[: args.n_videos]
        fake_vids = fake_vids[: args.n_videos]

    records = []
    if real_vids:
        records.extend(analyse_videos(real_vids, "real"))
    if fake_vids:
        records.extend(analyse_videos(fake_vids, "fake"))

    plot_all_metrics(records, "FaceForensics++", args.output)


def run_celeb_df(cfg: dict, args):
    cdf_cfg = cfg["celeb_df"]
    root = cdf_cfg["root_dir"]

    real_vids = []
    for rdir in cdf_cfg["real_dirs"]:
        d = os.path.join(root, rdir)
        if os.path.isdir(d):
            real_vids.extend(get_video_paths(d))

    fake_dir = os.path.join(root, cdf_cfg["fake_dir"])
    fake_vids = get_video_paths(fake_dir) if os.path.isdir(fake_dir) else []

    if args.n_videos:
        real_vids = real_vids[: args.n_videos]
        fake_vids = fake_vids[: args.n_videos]

    records = []
    if real_vids:
        records.extend(analyse_videos(real_vids, "real"))
    if fake_vids:
        records.extend(analyse_videos(fake_vids, "fake"))

    plot_all_metrics(records, "Celeb-DF v2", args.output)


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Quality analysis for deepfake datasets")
    p.add_argument("--config", default="configs/dataset_config.yaml")
    p.add_argument("--dataset", choices=["ff++", "celeb-df", "all"], default="all")
    p.add_argument("--n-videos", type=int, default=None,
                   help="Limit number of videos per class (for quick testing)")
    p.add_argument("--output", default="outputs/figures")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(42)
    cfg = load_config(args.config)

    print("=" * 60)
    print("  DeepFake Detection — Quality Analysis")
    print("=" * 60)

    if args.dataset in ("ff++", "all"):
        print("\n▶ FaceForensics++")
        run_faceforensics(cfg, args)

    if args.dataset in ("celeb-df", "all"):
        print("\n▶ Celeb-DF (v2)")
        run_celeb_df(cfg, args)

    print("\n✓ Analysis complete.")


if __name__ == "__main__":
    main()
