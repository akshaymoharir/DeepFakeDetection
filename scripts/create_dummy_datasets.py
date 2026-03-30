#!/usr/bin/env python3
"""
create_dummy_datasets.py — Generate synthetic dummy datasets
============================================================

Replicates the directory structure of FaceForensics++ and Celeb-DF v2
using synthetic frames saved as JPEG images organized in per-video
subdirectories.

This avoids video codec dependencies and produces data in the exact
format the training pipeline consumes (extracted frames).

Usage
-----
    python scripts/create_dummy_datasets.py
    python scripts/create_dummy_datasets.py --n-videos 10 --n-frames 20
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.helpers import seed_everything, setup_script_logging


# ------------------------------------------------------------------ #
#  Synthetic face-like frame generators
# ------------------------------------------------------------------ #

def _draw_face(frame, colour_base):
    """Draw a simple elliptical face with eyes, nose, mouth."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    skin = tuple(min(255, c + v) for c, v in zip(colour_base, (140, 120, 100)))
    cv2.ellipse(frame, (cx, cy), (w // 4, h // 3), 0, 0, 360, skin, -1)

    eye_y = cy - h // 10
    for dx in (-w // 8, w // 8):
        cv2.circle(frame, (cx + dx, eye_y), w // 20, (255, 255, 255), -1)
        cv2.circle(frame, (cx + dx, eye_y), w // 35, (40, 30, 20), -1)

    cv2.ellipse(frame, (cx, cy + h // 6), (w // 8, h // 25), 0, 0, 180, (150, 80, 80), 2)
    return frame


def generate_real_frame(w, h, frame_idx, vid_id):
    """Smooth gradient + face, subtle temporal motion."""
    bg = np.full((h, w, 3), (100 + vid_id % 20, 80 + vid_id % 30, 60 + vid_id % 40), dtype=np.uint8)
    grad = np.linspace(0, 30, h, dtype=np.uint8).reshape(-1, 1)
    bg[:, :, 0] = np.clip(bg[:, :, 0].astype(int) + grad, 0, 255).astype(np.uint8)
    bg = _draw_face(bg, (60 + vid_id % 40, 80, 100))

    sx = int(2 * np.sin(frame_idx * 0.3))
    M = np.float32([[1, 0, sx], [0, 1, 0]])
    return cv2.warpAffine(bg, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def generate_fake_frame(w, h, frame_idx, vid_id, method="Deepfakes"):
    """Real base + artefacts specific to the manipulation method."""
    frame = generate_real_frame(w, h, frame_idx, vid_id + 500)
    cx, cy = w // 2, h // 2
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (w // 4, h // 3), 0, 0, 360, 255, -1)

    if method == "Deepfakes":
        edge_mask = np.zeros_like(mask)
        cv2.ellipse(edge_mask, (cx, cy), (w // 4, h // 3), 0, 0, 360, 255, 3)
        frame[edge_mask > 0] = np.clip(frame[edge_mask > 0].astype(int) + 30, 0, 255).astype(np.uint8)
        frame[:, :, 2] = np.where(mask > 0, np.clip(frame[:, :, 2].astype(int) + 15, 0, 255), frame[:, :, 2]).astype(np.uint8)
    elif method == "Face2Face":
        region = frame[cy:cy + h // 6, cx - w // 6:cx + w // 6]
        if region.size > 0:
            noise = np.random.randint(-20, 20, region.shape, dtype=np.int16)
            region[:] = np.clip(region.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif method == "FaceSwap":
        frame[:, :, 2] = np.where(mask > 0, np.clip(frame[:, :, 2].astype(int) + 25, 0, 255), frame[:, :, 2]).astype(np.uint8)
        frame[:, :, 0] = np.where(mask > 0, np.clip(frame[:, :, 0].astype(int) - 15, 0, 255), frame[:, :, 0]).astype(np.uint8)
    elif method == "NeuralTextures":
        noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
        frame = np.where(mask[:, :, None] > 0, np.clip(frame.astype(np.int16) + noise, 0, 255), frame).astype(np.uint8)

    jitter = int(3 * np.sin(frame_idx * 0.8 + vid_id))
    M = np.float32([[1, 0, jitter], [0, 1, -jitter]])
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)


# ------------------------------------------------------------------ #
#  Save frames helper
# ------------------------------------------------------------------ #

def save_frames(out_dir, vid_name, frames):
    """Save list of BGR frames as numbered JPEGs in out_dir/vid_name/."""
    vid_dir = os.path.join(out_dir, vid_name)
    os.makedirs(vid_dir, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(vid_dir, f"frame_{i:03d}.jpg"), f)


# ------------------------------------------------------------------ #
#  FaceForensics++ dummy
# ------------------------------------------------------------------ #

def create_ff(root, n_vid, n_fr, res):
    """
    dummy_FaceForensics/
    ├── original_sequences/youtube/c23/videos/
    │   └── 000/ 001/ ...   (each has frame_000.jpg ...)
    └── manipulated_sequences/{method}/c23/videos/
        └── 000_001/ ...
    """
    print("\n▶ Creating dummy FaceForensics++")

    real_dir = os.path.join(root, "original_sequences", "youtube", "c23", "videos")
    for i in range(n_vid):
        frames = [generate_real_frame(res, res, f, i) for f in range(n_fr)]
        save_frames(real_dir, f"{i:03d}", frames)
    print(f"  ✓ {n_vid} real → {real_dir}")

    for method in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        m_dir = os.path.join(root, "manipulated_sequences", method, "c23", "videos")
        for i in range(n_vid):
            src = f"{(i + 1) % n_vid:03d}"
            frames = [generate_fake_frame(res, res, f, i, method) for f in range(n_fr)]
            save_frames(m_dir, f"{i:03d}_{src}", frames)
        print(f"  ✓ {n_vid} {method} → {m_dir}")


# ------------------------------------------------------------------ #
#  Celeb-DF v2 dummy
# ------------------------------------------------------------------ #

def create_cdf(root, n_vid, n_fr, res):
    """
    dummy_Celeb_DF_v2/
    ├── Celeb-real/      id0_0000/ id0_0001/ ...
    ├── YouTube-real/    00000/ 00001/ ...
    ├── Celeb-synthesis/ id0_id1_0000/ ...
    └── List_of_testing_videos.txt
    """
    print("\n▶ Creating dummy Celeb-DF v2")
    test_lines = []

    # Celeb-real
    cr_dir = os.path.join(root, "Celeb-real")
    n_cr = max(6, n_vid // 2)
    for i in range(n_cr):
        name = f"id{i // 3}_{i % 3:04d}"
        frames = [generate_real_frame(res, res, f, i + 100) for f in range(n_fr)]
        save_frames(cr_dir, name, frames)
        if i >= n_cr - 2:
            test_lines.append(f"1 Celeb-real/{name}")
    print(f"  ✓ {n_cr} Celeb-real → {cr_dir}")

    # YouTube-real
    yt_dir = os.path.join(root, "YouTube-real")
    n_yt = max(4, n_vid // 4)
    for i in range(n_yt):
        name = f"{i:05d}"
        frames = [generate_real_frame(res, res, f, i + 200) for f in range(n_fr)]
        save_frames(yt_dir, name, frames)
        if i >= n_yt - 2:
            test_lines.append(f"1 YouTube-real/{name}")
    print(f"  ✓ {n_yt} YouTube-real → {yt_dir}")

    # Celeb-synthesis
    sy_dir = os.path.join(root, "Celeb-synthesis")
    n_sy = max(10, n_vid)
    for i in range(n_sy):
        name = f"id{i % 4}_id{(i + 1) % 4}_{i // 4:04d}"
        frames = [generate_fake_frame(res, res, f, i + 300, "Deepfakes") for f in range(n_fr)]
        save_frames(sy_dir, name, frames)
        if i >= n_sy - 3:
            test_lines.append(f"0 Celeb-synthesis/{name}")
    print(f"  ✓ {n_sy} Celeb-synthesis → {sy_dir}")

    # Test list
    test_path = os.path.join(root, "List_of_testing_videos.txt")
    with open(test_path, "w") as f:
        f.write("\n".join(test_lines) + "\n")
    print(f"  ✓ Test list ({len(test_lines)} entries) → {test_path}")


# ------------------------------------------------------------------ #
#  Summary
# ------------------------------------------------------------------ #

def print_tree(root):
    total_files = 0
    total_bytes = 0
    print("\n" + "=" * 55)
    print("  Generated Structure")
    print("=" * 55)
    for dirpath, dirs, files in sorted(os.walk(root)):
        depth = dirpath.replace(root, "").count(os.sep)
        indent = "  " * depth
        n_jpg = len([f for f in files if f.endswith(".jpg")])
        basename = os.path.basename(dirpath)
        extra = f"  ({n_jpg} frames)" if n_jpg else ""
        if depth <= 5:
            print(f"  {indent}📁 {basename}/{extra}")
        total_files += n_jpg
        total_bytes += sum(os.path.getsize(os.path.join(dirpath, f)) for f in files)
    print(f"\n  Total: {total_files} frames, {total_bytes / 1e6:.1f} MB")


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def main():
    setup_script_logging("create_dummy_datasets")
    p = argparse.ArgumentParser(description="Generate dummy deepfake datasets")
    p.add_argument("--output", default="data")
    p.add_argument("--n-videos", type=int, default=10)
    p.add_argument("--n-frames", type=int, default=15)
    p.add_argument("--resolution", type=int, default=224)
    args = p.parse_args()

    seed_everything(42)

    print("=" * 55)
    print("  Dummy Dataset Generator")
    print("=" * 55)

    ff_root = os.path.join(args.output, "dummy_FaceForensics")
    create_ff(ff_root, args.n_videos, args.n_frames, args.resolution)

    cdf_root = os.path.join(args.output, "dummy_Celeb_DF_v2")
    create_cdf(cdf_root, args.n_videos, args.n_frames, args.resolution)

    print_tree(args.output)

    print(f"\n✓ Done! Update your config to use these paths:")
    print(f'  faceforensics.root_dir: "{ff_root}"')
    print(f'  celeb_df.root_dir:      "{cdf_root}"')
    print(f"\n  Or use the ready-made config:")
    print(f"  --config configs/dataset_config_dummy.yaml")


if __name__ == "__main__":
    main()
