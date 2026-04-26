#!/usr/bin/env python3
"""
predict.py — HSF-CVIT DeepFake Detection CLI
=============================================

Run deepfake detection on a single image or video file.

Usage
-----
    # Auto-detect file type
    python predict.py --input face.jpg
    python predict.py --input clip.mp4

    # Video options
    python predict.py --input clip.mp4 --frames 16 --aggregation max

    # Skip face detection (input is already face-cropped)
    python predict.py --input face_crop.jpg --no-face-detect

    # Custom checkpoint or config
    python predict.py --input face.jpg --checkpoint outputs/checkpoints/best.pt

    # Machine-readable output
    python predict.py --input face.jpg --json
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference.detector import DeepFakeDetector

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HSF-CVIT deepfake detector — image or video inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to image or video file")
    p.add_argument(
        "--checkpoint", default="outputs/checkpoints/best.pt",
        help="Trained model checkpoint (.pt)",
    )
    p.add_argument(
        "--config", default="configs/train_config.yaml",
        help="Training config YAML (defines model architecture)",
    )
    p.add_argument(
        "--frames", type=int, default=16,
        help="Number of frames to sample from video input",
    )
    p.add_argument(
        "--aggregation", choices=["mean", "max"], default="mean",
        help="Video clip aggregation: mean (robust) or max (most sensitive frame)",
    )
    p.add_argument(
        "--no-face-detect", action="store_true",
        help="Skip face detection — use when input is already a face crop",
    )
    p.add_argument(
        "--face-margin", type=float, default=0.3,
        help="Fractional padding added around detected face box",
    )
    p.add_argument("--device", default=None, help="Force device: 'cuda' or 'cpu'")
    p.add_argument(
        "--json", action="store_true",
        help="Print result as JSON (useful for scripting)",
    )
    return p.parse_args()


def _infer_file_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _IMAGE_EXTS:
        return "image"
    # Unknown extension — try PIL open as a probe
    try:
        from PIL import Image
        Image.open(path).verify()
        return "image"
    except Exception:
        return "video"


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.config):
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(args.config, "r") as f:
        train_cfg = yaml.safe_load(f)

    detector = DeepFakeDetector(
        checkpoint_path=args.checkpoint,
        train_cfg=train_cfg,
        device=args.device,
        face_margin=args.face_margin,
    )

    detect_face = not args.no_face_detect
    file_type = _infer_file_type(args.input)

    if file_type == "video":
        result = detector.predict_video(
            args.input,
            num_frames=args.frames,
            detect_face=detect_face,
            aggregation=args.aggregation,
        )
    else:
        result = detector.predict_image(args.input, detect_face=detect_face)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    label = result.get("label", "?").upper()
    prob  = result.get("probability", float("nan"))
    thr   = result.get("threshold", 0.5)

    print(f"\n{'='*52}")
    print(f"  File       : {args.input}")
    print(f"  Type       : {file_type}")
    print(f"  Prediction : {label}")
    print(f"  P(fake)    : {prob:.4f}  (threshold {thr:.2f})")
    if "face_detected" in result:
        fd = result["face_detected"]
        print(f"  Face found : {'yes' if fd else 'no (used full frame)'}")
    if "num_frames_used" in result:
        print(f"  Frames used: {result['num_frames_used']}")
    if result.get("frame_probs"):
        fp = result["frame_probs"]
        print(
            f"  Frame probs: min={min(fp):.3f}  "
            f"max={max(fp):.3f}  "
            f"std={float(np.std(fp)):.3f}"
        )
    if "error" in result:
        print(f"  Error      : {result['error']}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
