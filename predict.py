#!/usr/bin/env python3
"""
predict.py — HSF-CVIT DeepFake Detection CLI
=============================================

Run deepfake detection on a single image or video file, or on a directory
of images.

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

    # Directory of images — writes per_image.csv + summary.json
    python predict.py --input data/_custom_test/ \
      --checkpoint outputs/checkpoints/best_iteration_swt_b7.pt \
      --gt-label real \
      --output-dir outputs/custom_eval
"""

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

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
        "--checkpoint", default="outputs/checkpoints/best_iteration_swt_b7.pt",
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
    p.add_argument(
        "--gt-label", choices=["real", "fake"], default=None,
        help="Directory mode only: ground-truth label applied to every image "
             "(used to compute accuracy/specificity for an all-real or all-fake batch).",
    )
    p.add_argument(
        "--output-dir", default="outputs/custom_eval",
        help="Directory mode only: where per_image.csv + summary.json are written.",
    )
    p.add_argument(
        "--recursive", action="store_true",
        help="Directory mode only: recurse into subdirectories.",
    )
    return p.parse_args()


def _list_images(root: str, recursive: bool) -> list[str]:
    """Return a sorted list of image paths under *root*, filtered by extension."""
    base = Path(root)
    walker = base.rglob("*") if recursive else base.iterdir()
    paths: list[str] = []
    for p in walker:
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            paths.append(str(p))
    paths.sort()
    return paths


def _predict_directory(detector, args) -> None:
    """Run predict_image over every image in args.input and emit CSV + summary JSON."""
    image_paths = _list_images(args.input, args.recursive)
    if not image_paths:
        print(f"Error: no images found in {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"  Found {len(image_paths)} images under {args.input}")
    if args.gt_label:
        print(f"  Ground truth: all images labeled '{args.gt_label}'")

    detect_face = not args.no_face_detect
    rows: list[dict] = []
    from PIL import Image
    for path in image_paths:
        try:
            with Image.open(path) as img:
                width, height = img.size
        except Exception:
            width, height = -1, -1

        try:
            result = detector.predict_image(path, detect_face=detect_face)
            row = {
                "filename": os.path.relpath(path, args.input),
                "width": width,
                "height": height,
                "probability": float(result["probability"]),
                "predicted_label": result["label"],
                "threshold": float(result["threshold"]),
                "face_detected": bool(result.get("face_detected", False)),
                "status": "ok",
            }
        except Exception as exc:
            row = {
                "filename": os.path.relpath(path, args.input),
                "width": width,
                "height": height,
                "probability": float("nan"),
                "predicted_label": "error",
                "threshold": detector.threshold,
                "face_detected": False,
                "status": str(exc),
            }
        if args.gt_label is not None:
            row["ground_truth"] = args.gt_label
            row["correct"] = (
                row["predicted_label"] == args.gt_label and row["status"] == "ok"
            )
        rows.append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "per_image.csv")
    fieldnames = [
        "filename", "width", "height", "probability", "predicted_label",
        "threshold", "face_detected", "status",
    ]
    if args.gt_label is not None:
        fieldnames += ["ground_truth", "correct"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    probs = [r["probability"] for r in ok_rows]
    pred_real = sum(1 for r in ok_rows if r["predicted_label"] == "real")
    pred_fake = sum(1 for r in ok_rows if r["predicted_label"] == "fake")
    face_hits = sum(1 for r in ok_rows if r["face_detected"])

    summary = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "threshold": float(detector.threshold),
        "input_dir": args.input,
        "recursive": args.recursive,
        "face_detection": detect_face,
        "face_margin": args.face_margin,
        "total_images": len(rows),
        "evaluated": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "predicted_real": pred_real,
        "predicted_fake": pred_fake,
        "face_detection_rate": (face_hits / len(ok_rows)) if ok_rows else 0.0,
        "probability_stats": _prob_stats(probs),
        "input_size_stats": _size_stats(ok_rows),
    }
    if args.gt_label is not None and ok_rows:
        correct = sum(1 for r in ok_rows if r["correct"])
        summary["ground_truth"] = args.gt_label
        summary["accuracy"] = correct / len(ok_rows)
        if args.gt_label == "real":
            summary["specificity"] = pred_real / len(ok_rows)
        else:
            summary["recall_fake"] = pred_fake / len(ok_rows)

    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    _print_directory_summary(summary, csv_path, json_path)


def _prob_stats(probs: list[float]) -> dict:
    if not probs:
        return {"n": 0}
    return {
        "n": len(probs),
        "mean": float(np.mean(probs)),
        "median": float(np.median(probs)),
        "std": float(np.std(probs)),
        "min": float(min(probs)),
        "max": float(max(probs)),
        "p25": float(np.percentile(probs, 25)),
        "p75": float(np.percentile(probs, 75)),
    }


def _size_stats(rows: list[dict]) -> dict:
    widths = [r["width"] for r in rows if r["width"] > 0]
    heights = [r["height"] for r in rows if r["height"] > 0]
    if not widths:
        return {}
    return {
        "width_min": min(widths),
        "width_max": max(widths),
        "width_median": int(statistics.median(widths)),
        "height_min": min(heights),
        "height_max": max(heights),
        "height_median": int(statistics.median(heights)),
    }


def _print_directory_summary(summary: dict, csv_path: str, json_path: str) -> None:
    print(f"\n{'='*52}")
    print(f"  Custom directory eval — {summary['evaluated']}/{summary['total_images']} images")
    print(f"  Checkpoint           : {summary['checkpoint']}")
    print(f"  Threshold (saved)    : {summary['threshold']:.2f}")
    print(f"  Predicted real       : {summary['predicted_real']}")
    print(f"  Predicted fake       : {summary['predicted_fake']}")
    print(f"  Face detection rate  : {summary['face_detection_rate']:.2%}")
    if summary["probability_stats"].get("n"):
        ps = summary["probability_stats"]
        print(f"  P(fake) mean / med   : {ps['mean']:.3f} / {ps['median']:.3f}")
        print(f"  P(fake) min / max    : {ps['min']:.3f} / {ps['max']:.3f}")
    if summary.get("input_size_stats"):
        ss = summary["input_size_stats"]
        print(
            f"  Input size (WxH)     : "
            f"{ss['width_min']}-{ss['width_max']} x {ss['height_min']}-{ss['height_max']} "
            f"(median {ss['width_median']}x{ss['height_median']})"
        )
    if "accuracy" in summary:
        print(f"  Ground truth         : {summary['ground_truth']}")
        print(f"  Accuracy             : {summary['accuracy']:.4f}")
        if "specificity" in summary:
            print(f"  Specificity          : {summary['specificity']:.4f}")
        if "recall_fake" in summary:
            print(f"  Recall (fake)        : {summary['recall_fake']:.4f}")
    print(f"  Per-image CSV        -> {csv_path}")
    print(f"  Summary JSON         -> {json_path}")
    print(f"{'='*52}\n")


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

    is_dir = os.path.isdir(args.input)
    if not is_dir and not os.path.isfile(args.input):
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

    if is_dir:
        _predict_directory(detector, args)
        return

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
