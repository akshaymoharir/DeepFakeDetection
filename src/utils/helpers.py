"""
Shared helpers used by all exploration / preprocessing scripts.

Functions
---------
load_config      – Load a YAML config file and return a dict.
get_video_paths  – Recursively find video files under a root directory.
extract_frames   – Sample N frames from a video and return as numpy arrays.
seed_everything  – Set random seeds for reproducibility.
"""

import os
import random
import glob
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml


# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #

def load_config(config_path: str = "configs/dataset_config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ------------------------------------------------------------------ #
#  File discovery
# ------------------------------------------------------------------ #

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def get_video_paths(root_dir: str, extensions: set = None) -> List[str]:
    """Recursively collect video file paths from *root_dir*.

    Parameters
    ----------
    root_dir : str
        Directory tree to walk.
    extensions : set, optional
        Allowed file suffixes (default VIDEO_EXTENSIONS).

    Returns
    -------
    list[str]
        Sorted list of absolute paths.
    """
    extensions = extensions or VIDEO_EXTENSIONS
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if Path(fname).suffix.lower() in extensions:
                paths.append(os.path.join(dirpath, fname))
    return sorted(paths)


def get_image_paths(root_dir: str) -> List[str]:
    """Recursively collect image file paths (jpg / png)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if Path(fname).suffix.lower() in exts:
                paths.append(os.path.join(dirpath, fname))
    return sorted(paths)


# ------------------------------------------------------------------ #
#  Frame extraction
# ------------------------------------------------------------------ #

def extract_frames(
    video_path: str,
    max_frames: int = 32,
    resize: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Uniformly sample *max_frames* from a video file.

    Parameters
    ----------
    video_path : str
        Path to video file.
    max_frames : int
        Number of frames to sample (uniform spacing).
    resize : tuple(int, int) or None
        If given, resize each frame to (width, height).

    Returns
    -------
    np.ndarray
        Array of shape (N, H, W, 3) in RGB colour space.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has 0 frames: {video_path}")

    indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()
    return np.array(frames)


# ------------------------------------------------------------------ #
#  Reproducibility
# ------------------------------------------------------------------ #

def seed_everything(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and (if available) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
