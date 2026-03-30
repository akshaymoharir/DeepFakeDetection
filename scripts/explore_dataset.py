#!/usr/bin/env python3
"""
explore_dataset.py — Dataset structure exploration & statistics
================================================================

Walks FaceForensics++ and/or Celeb-DF directory trees, collects
metadata (video counts, frame counts, resolution, duration), and
prints a structured summary.  Optionally saves the raw metadata to
a CSV for downstream analysis.

Usage
-----
    python scripts/explore_dataset.py --config configs/dataset_config.yaml
    python scripts/explore_dataset.py --dataset ff++          # FF++ only
    python scripts/explore_dataset.py --dataset celeb-df      # Celeb-DF only
    python scripts/explore_dataset.py --save-csv              # persist stats
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.helpers import load_config, get_video_paths, setup_script_logging


# ------------------------------------------------------------------ #
#  Per-video metadata
# ------------------------------------------------------------------ #

def video_metadata(video_path: str) -> dict:
    """Return frame count, FPS, width, height, and duration for a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "path": video_path,
            "frames": 0,
            "fps": 0,
            "width": 0,
            "height": 0,
            "duration_sec": 0,
        }

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = n_frames / fps if fps > 0 else 0
    return {
        "path": video_path,
        "frames": n_frames,
        "fps": round(fps, 2),
        "width": w,
        "height": h,
        "duration_sec": round(duration, 2),
    }


# ------------------------------------------------------------------ #
#  FaceForensics++
# ------------------------------------------------------------------ #

def explore_faceforensics(cfg: dict) -> list:
    """Walk the FF++ directory tree and return per-video metadata dicts."""
    ff_cfg = cfg["faceforensics"]
    root = ff_cfg["root_dir"]
    compression = ff_cfg["compression"]

    if not os.path.isdir(root):
        print(f"[WARN] FF++ root not found: {root}")
        return []

    categories = {}

    # Real videos
    real_dir = os.path.join(root, ff_cfg["original_dir"], compression, "videos")
    if os.path.isdir(real_dir):
        categories["original"] = real_dir
    else:
        # Try without /videos suffix (some downloads just have frames)
        alt = os.path.join(root, ff_cfg["original_dir"], compression)
        if os.path.isdir(alt):
            categories["original"] = alt

    # Manipulated videos
    for method in ff_cfg["manipulation_methods"]:
        manip_dir = os.path.join(root, ff_cfg["manipulated_dir"], method, compression, "videos")
        if os.path.isdir(manip_dir):
            categories[method] = manip_dir
        else:
            alt = os.path.join(root, ff_cfg["manipulated_dir"], method, compression)
            if os.path.isdir(alt):
                categories[method] = alt

    all_meta = []
    for category, path in categories.items():
        vids = get_video_paths(path)
        print(f"\n  [{category.upper()}]  {len(vids)} videos in {path}")
        for v in vids:
            meta = video_metadata(v)
            meta["dataset"] = "FaceForensics++"
            meta["category"] = category
            meta["label"] = "real" if category == "original" else "fake"
            all_meta.append(meta)

    return all_meta


# ------------------------------------------------------------------ #
#  Celeb-DF
# ------------------------------------------------------------------ #

def explore_celeb_df(cfg: dict) -> list:
    """Walk the Celeb-DF directory tree and return per-video metadata dicts."""
    cdf_cfg = cfg["celeb_df"]
    root = cdf_cfg["root_dir"]

    if not os.path.isdir(root):
        print(f"[WARN] Celeb-DF root not found: {root}")
        return []

    categories = {}

    for rdir in cdf_cfg["real_dirs"]:
        full = os.path.join(root, rdir)
        if os.path.isdir(full):
            categories[rdir] = ("real", full)

    fake_dir = os.path.join(root, cdf_cfg["fake_dir"])
    if os.path.isdir(fake_dir):
        categories[cdf_cfg["fake_dir"]] = ("fake", fake_dir)

    all_meta = []
    for cat_name, (label, path) in categories.items():
        vids = get_video_paths(path)
        print(f"\n  [{cat_name.upper()}]  {len(vids)} videos in {path}")
        for v in vids:
            meta = video_metadata(v)
            meta["dataset"] = "Celeb-DF-v2"
            meta["category"] = cat_name
            meta["label"] = label
            all_meta.append(meta)

    return all_meta


# ------------------------------------------------------------------ #
#  Summary printer
# ------------------------------------------------------------------ #

def print_summary(metadata: list) -> None:
    """Print an aggregated summary table to stdout."""
    if not metadata:
        print("\n  ⚠  No video metadata collected.  Check dataset paths.\n")
        return

    # Group by (dataset, category)
    groups = defaultdict(list)
    for m in metadata:
        groups[(m["dataset"], m["category"], m["label"])].append(m)

    header = f"{'Dataset':<20} {'Category':<18} {'Label':<6} {'Videos':>7} {'Avg Frames':>11} {'Avg Dur (s)':>12} {'Resolution':>14}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    total_videos = 0
    for (ds, cat, label), items in sorted(groups.items()):
        n = len(items)
        total_videos += n
        avg_frames = np.mean([i["frames"] for i in items])
        avg_dur = np.mean([i["duration_sec"] for i in items])
        # Most common resolution
        resolutions = [f'{i["width"]}x{i["height"]}' for i in items]
        common_res = max(set(resolutions), key=resolutions.count) if resolutions else "N/A"
        print(f"{ds:<20} {cat:<18} {label:<6} {n:>7} {avg_frames:>11.1f} {avg_dur:>12.1f} {common_res:>14}")

    print("=" * len(header))
    print(f"{'TOTAL':<46} {total_videos:>7}")
    print()


# ------------------------------------------------------------------ #
#  CSV export
# ------------------------------------------------------------------ #

def save_csv(metadata: list, out_path: str = "outputs/dataset_metadata.csv") -> None:
    """Persist all per-video metadata to CSV."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not metadata:
        return
    fieldnames = list(metadata[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)
    print(f"  ✓ Metadata saved → {out_path}")


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Explore deepfake dataset structure & statistics")
    p.add_argument("--config", default="configs/dataset_config.yaml", help="YAML config path")
    p.add_argument("--dataset", choices=["ff++", "celeb-df", "all"], default="all",
                   help="Which dataset(s) to explore")
    p.add_argument("--save-csv", action="store_true", help="Save per-video metadata to CSV")
    return p.parse_args()


def main():
    setup_script_logging("explore_dataset")
    args = parse_args()
    cfg = load_config(args.config)

    print("=" * 60)
    print("  DeepFake Detection — Dataset Explorer")
    print("=" * 60)

    metadata = []

    if args.dataset in ("ff++", "all"):
        print("\n▶ FaceForensics++")
        metadata.extend(explore_faceforensics(cfg))

    if args.dataset in ("celeb-df", "all"):
        print("\n▶ Celeb-DF (v2)")
        metadata.extend(explore_celeb_df(cfg))

    print_summary(metadata)

    if args.save_csv:
        save_csv(metadata)


if __name__ == "__main__":
    main()
