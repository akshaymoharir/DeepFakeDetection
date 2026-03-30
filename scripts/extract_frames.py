#!/usr/bin/env python3
"""
extract_frames.py — Batch frame extraction from video datasets
==============================================================

Reads videos from FaceForensics++ and/or Celeb-DF, uniformly samples
N frames per video, and saves them as JPEG images in a structured
output directory ready for PyTorch's ImageFolder or custom Dataset
classes.

Output layout
-------------
    <output_dir>/
        FaceForensics++/
            real/
                <video_stem>/
                    frame_000.jpg
                    frame_001.jpg
                    ...
            Deepfakes/
                <video_stem>/
                    frame_000.jpg
                    ...
        Celeb-DF-v2/
            real/
                ...
            fake/
                ...

Usage
-----
    python scripts/extract_frames.py --config configs/dataset_config.yaml
    python scripts/extract_frames.py --dataset ff++ --max-frames 16
    python scripts/extract_frames.py --resize 256
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.helpers import (
    load_config,
    get_video_paths,
    extract_frames,
    setup_script_logging,
)


# ------------------------------------------------------------------ #
#  Core extraction routine
# ------------------------------------------------------------------ #

def save_frames_for_video(
    video_path: str,
    output_dir: str,
    max_frames: int = 32,
    resize: int = 224,
    quality: int = 95,
) -> int:
    """Extract and save frames from a single video.

    Returns the number of frames successfully saved.
    """
    stem = Path(video_path).stem
    vid_out = os.path.join(output_dir, stem)

    try:
        frames = extract_frames(video_path, max_frames=max_frames,
                                resize=(resize, resize))
    except (IOError, ValueError) as e:
        print(f"  ⚠  Skipping {video_path}: {e}")
        return 0

    if len(frames) == 0:
        print(f"  ⚠  Skipping {video_path}: no frames decoded by any backend.")
        return 0

    os.makedirs(vid_out, exist_ok=True)

    saved = 0
    for i, frame in enumerate(frames):
        fname = os.path.join(vid_out, f"frame_{i:03d}.jpg")
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fname, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        saved += 1

    return saved


# ------------------------------------------------------------------ #
#  FaceForensics++
# ------------------------------------------------------------------ #

def extract_faceforensics(cfg: dict, args):
    """Extract frames from all FF++ categories."""
    ff_cfg = cfg["faceforensics"]
    if not ff_cfg["frame_extraction"]["enabled"]:
        print("  Frame extraction disabled in config. Skipping FF++.")
        return

    root = ff_cfg["root_dir"]
    compression = ff_cfg["compression"]
    out_base = ff_cfg["frame_extraction"]["output_dir"]
    max_frames = args.max_frames or ff_cfg["frame_extraction"]["max_frames_per_video"]
    resize = args.resize
    methods = [m.strip() for m in args.ff_methods.split(",")] if args.ff_methods else ff_cfg["manipulation_methods"]
    real_dirs = (
        [d.strip() for d in args.ff_real_dirs.split(",")]
        if args.ff_real_dirs else
        ff_cfg.get("original_dirs", [ff_cfg["original_dir"]])
    )

    # Real (possibly merged from multiple sources)
    out_dir = os.path.join(out_base, "real")
    for real_subdir in real_dirs:
        real_dir = os.path.join(root, real_subdir, compression)
        if os.path.isdir(real_dir):
            videos = get_video_paths(real_dir)
            print(f"  Extracting {len(videos)} real videos from {real_subdir} → {out_dir}")
            for vpath in tqdm(videos, desc="  FF++ Real", unit="vid"):
                save_frames_for_video(vpath, out_dir, max_frames, resize)

    # Manipulated
    for method in methods:
        manip_dir = os.path.join(root, ff_cfg["manipulated_dir"], method, compression)
        if os.path.isdir(manip_dir):
            videos = get_video_paths(manip_dir)
            out_dir = os.path.join(out_base, method)
            print(f"  Extracting {len(videos)} {method} videos → {out_dir}")
            for vpath in tqdm(videos, desc=f"  FF++ {method}", unit="vid"):
                save_frames_for_video(vpath, out_dir, max_frames, resize)


# ------------------------------------------------------------------ #
#  Celeb-DF
# ------------------------------------------------------------------ #

def extract_celeb_df(cfg: dict, args):
    """Extract frames from all Celeb-DF categories."""
    cdf_cfg = cfg["celeb_df"]
    if not cdf_cfg["frame_extraction"]["enabled"]:
        print("  Frame extraction disabled in config. Skipping Celeb-DF.")
        return

    root = cdf_cfg["root_dir"]
    out_base = cdf_cfg["frame_extraction"]["output_dir"]
    max_frames = args.max_frames or cdf_cfg["frame_extraction"]["max_frames_per_video"]
    resize = args.resize

    # Real directories
    for rdir in cdf_cfg["real_dirs"]:
        full = os.path.join(root, rdir)
        if os.path.isdir(full):
            videos = get_video_paths(full)
            out_dir = os.path.join(out_base, "real")
            print(f"  Extracting {len(videos)} videos from {rdir} → {out_dir}")
            for vpath in tqdm(videos, desc=f"  {rdir}", unit="vid"):
                save_frames_for_video(vpath, out_dir, max_frames, resize)

    # Fake
    fake_dir = os.path.join(root, cdf_cfg["fake_dir"])
    if os.path.isdir(fake_dir):
        videos = get_video_paths(fake_dir)
        out_dir = os.path.join(out_base, "fake")
        print(f"  Extracting {len(videos)} fake videos → {out_dir}")
        for vpath in tqdm(videos, desc="  Celeb-synthesis", unit="vid"):
            save_frames_for_video(vpath, out_dir, max_frames, resize)


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Extract frames from deepfake video datasets")
    p.add_argument("--config", default="configs/dataset_config.yaml")
    p.add_argument("--dataset", choices=["ff++", "celeb-df", "all"], default="all")
    p.add_argument("--ff-methods", type=str, default=None,
                   help="Comma-separated FF++ methods to extract (default: config list)")
    p.add_argument("--ff-real-dirs", type=str, default=None,
                   help="Comma-separated FF++ real dirs relative to root_dir (default: config list)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Override max frames per video (default: from config)")
    p.add_argument("--resize", type=int, default=224,
                   help="Resize frames to NxN pixels")
    return p.parse_args()


def main():
    setup_script_logging("extract_frames")
    args = parse_args()
    cfg = load_config(args.config)

    print("=" * 60)
    print("  DeepFake Detection — Frame Extractor")
    print("=" * 60)

    if args.dataset in ("ff++", "all"):
        print("\n▶ FaceForensics++")
        extract_faceforensics(cfg, args)

    if args.dataset in ("celeb-df", "all"):
        print("\n▶ Celeb-DF (v2)")
        extract_celeb_df(cfg, args)

    print("\n✓ Frame extraction complete.")


if __name__ == "__main__":
    main()
