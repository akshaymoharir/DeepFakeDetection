#!/usr/bin/env python3
"""
evaluate_celeb_df.py — Cross-dataset evaluation on Celeb-DF v2
==============================================================

Runs a trained HSF-CVIT checkpoint on the Celeb-DF v2 test split and
reports ROC-AUC plus thresholded metrics. The model was trained on
FaceForensics++ at 380x380, so this is a generalisation test — expect
lower AUC than on FF++.

Label convention
----------------
``List_of_testing_videos.txt`` uses ``1`` for REAL and ``0`` for FAKE.
This script inverts to the model convention (``1`` = FAKE) before
computing metrics.

Inputs
------
- ``--checkpoint``   trained model (default: outputs/checkpoints/best_iteration_swt_b7.pt)
- ``--config``       train config (default: configs/train_config.yaml — image_size 380)
- ``--dataset-root`` directory containing ``List_of_testing_videos.txt`` and
                     the ``Celeb-real / YouTube-real / Celeb-synthesis`` subfolders
                     (default: data/Celeb-DF-v2)
- ``--frames``       frames sampled per video (default: 16)
- ``--limit``        cap number of test videos for a smoke run

Outputs
-------
- ``outputs/celeb_df_eval/per_video.csv`` — per-video predictions
- ``outputs/celeb_df_eval/metrics.json``  — summary metrics
- printout of the most useful numbers

Usage
-----
    python evaluate_celeb_df.py
    python evaluate_celeb_df.py --frames 32 --limit 50
    python evaluate_celeb_df.py --checkpoint outputs/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ------------------------------------------------------------------ #
#  Test list parsing
# ------------------------------------------------------------------ #

def load_test_list(list_path: str) -> List[Tuple[int, str]]:
    """Return ``[(model_label, relative_video_path), ...]``.

    Celeb-DF lists ``1`` for real and ``0`` for fake. We invert to the
    model's convention where ``1`` means fake.
    """
    samples: List[Tuple[int, str]] = []
    malformed = 0
    seen = set()
    with open(list_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                malformed += 1
                continue
            celeb_label, rel = parts
            try:
                celeb_label_int = int(celeb_label)
            except ValueError:
                malformed += 1
                continue
            if celeb_label_int not in (0, 1) or "/" not in rel:
                malformed += 1
                continue
            if rel in seen:
                malformed += 1
                continue
            model_label = 1 - celeb_label_int
            samples.append((model_label, rel))
            seen.add(rel)
    if malformed:
        print(f"  WARNING: skipped {malformed} malformed/duplicate test-list entries")
    return samples


# ------------------------------------------------------------------ #
#  Evaluation loop
# ------------------------------------------------------------------ #

def evaluate(
    detector: DeepFakeDetector,
    samples: List[Tuple[int, str]],
    dataset_root: str,
    num_frames: int,
    aggregation: str,
    detect_face: bool,
) -> List[dict]:
    """Run inference on every sample, returning per-video result rows."""
    rows: List[dict] = []
    for label, rel in tqdm(samples, desc="Celeb-DF test", unit="vid"):
        video_path = os.path.join(dataset_root, rel)
        if not os.path.isfile(video_path):
            rows.append({
                "video": rel, "label": label, "prob": float("nan"),
                "frames_used": 0, "status": "missing",
            })
            continue

        result = detector.predict_video(
            video_path,
            num_frames=num_frames,
            detect_face=detect_face,
            aggregation=aggregation,
        )

        if result.get("label") == "error":
            rows.append({
                "video": rel, "label": label, "prob": float("nan"),
                "frames_used": 0, "status": result.get("error", "decode_error"),
            })
            continue
        if int(result.get("num_frames_used", 0)) == 0:
            rows.append({
                "video": rel, "label": label, "prob": float("nan"),
                "frames_used": 0, "status": "no_decoded_frames",
            })
            continue

        rows.append({
            "video": rel,
            "label": label,
            "prob": float(result["probability"]),
            "frames_used": int(result.get("num_frames_used", 0)),
            "status": "ok",
        })
    return rows


def write_per_video_csv(rows: List[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video", "label", "prob", "frames_used", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: List[dict], default_threshold: float) -> dict:
    """Compute summary metrics across all successfully-evaluated rows."""
    valid = [r for r in rows if r["status"] == "ok" and not np.isnan(r["prob"])]
    labels = np.array([r["label"] for r in valid], dtype=int)
    probs  = np.array([r["prob"]  for r in valid], dtype=float)

    if len(valid) == 0 or len(np.unique(labels)) < 2:
        return {
            "evaluated": len(valid),
            "total": len(rows),
            "failed": len(rows) - len(valid),
            "note": "Insufficient samples or single-class set; metrics undefined.",
        }

    base = compute_metrics(labels, probs, threshold=default_threshold)
    best = find_best_threshold(labels, probs, metric="balanced_accuracy")
    return {
        "evaluated": len(valid),
        "total": len(rows),
        "failed": len(rows) - len(valid),
        "default_threshold": default_threshold,
        "metrics_at_default": base,
        "best_threshold_balacc": best["threshold"],
        "metrics_at_best": best["metrics"],
    }


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate HSF-CVIT on the Celeb-DF v2 test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",
                   default="outputs/checkpoints/best_iteration_swt_b7.pt")
    p.add_argument("--config", default="configs/train_config.yaml")
    p.add_argument("--dataset-root", default="data/Celeb-DF-v2")
    p.add_argument("--test-list", default=None,
                   help="Defaults to <dataset-root>/List_of_testing_videos.txt")
    p.add_argument("--frames", type=int, default=16,
                   help="Frames per video for video-level aggregation.")
    p.add_argument("--aggregation", choices=["mean", "max"], default="mean")
    p.add_argument("--no-face-detect", action="store_true",
                   help="Skip Haar face detection and classify whole frames.")
    p.add_argument("--face-margin", type=float, default=0.3,
                   help="Fractional padding added around detected face boxes.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N videos (smoke test).")
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default="outputs/celeb_df_eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global np, yaml, tqdm, DeepFakeDetector, compute_metrics, find_best_threshold
    try:
        import numpy as np
        import yaml
        from tqdm import tqdm

        from src.inference.detector import DeepFakeDetector
        from src.training.metrics import compute_metrics, find_best_threshold
    except ImportError as exc:
        print(
            f"ERROR: missing Python dependency while preparing evaluation: {exc}\n"
            "       Install the project dependencies, for example:\n"
            "       pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    test_list = args.test_list or os.path.join(
        args.dataset_root, "List_of_testing_videos.txt"
    )
    for required in (args.checkpoint, args.config, args.dataset_root, test_list):
        if not os.path.exists(required):
            print(f"ERROR: missing required path: {required}", file=sys.stderr)
            sys.exit(1)

    with open(args.config, "r") as f:
        train_cfg = yaml.safe_load(f)

    image_size = train_cfg["data"].get("image_size", 224)
    print(f"  Model input size: {image_size}x{image_size}")

    detector = DeepFakeDetector(
        checkpoint_path=args.checkpoint,
        train_cfg=train_cfg,
        device=args.device,
        face_margin=args.face_margin,
    )

    samples = load_test_list(test_list)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        print("ERROR: test list contains no usable samples.", file=sys.stderr)
        sys.exit(1)
    print(f"  Test list: {len(samples)} videos")
    real_n = sum(1 for lbl, _ in samples if lbl == 0)
    fake_n = len(samples) - real_n
    print(f"  Class balance: real={real_n}  fake={fake_n}")

    rows = evaluate(
        detector,
        samples,
        dataset_root=args.dataset_root,
        num_frames=args.frames,
        aggregation=args.aggregation,
        detect_face=not args.no_face_detect,
    )

    csv_path = os.path.join(args.output_dir, "per_video.csv")
    write_per_video_csv(rows, csv_path)
    print(f"  Per-video predictions -> {csv_path}")

    summary = summarise(rows, default_threshold=detector.threshold)
    summary["checkpoint"] = args.checkpoint
    summary["frames_per_video"] = args.frames
    summary["aggregation"] = args.aggregation
    summary["face_detection"] = not args.no_face_detect
    summary["face_margin"] = args.face_margin
    summary["image_size"] = image_size

    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary metrics       -> {json_path}")

    metrics = summary.get("metrics_at_default")
    if metrics:
        thr = summary["default_threshold"]
        print(f"\n{'='*52}")
        print(f"  Celeb-DF v2 — {summary['evaluated']}/{summary['total']} videos scored")
        print(f"  threshold (from checkpoint): {thr:.2f}")
        print(f"  ROC-AUC          : {metrics['roc_auc']:.4f}")
        print(f"  Avg precision    : {metrics['average_precision']:.4f}")
        print(f"  Accuracy         : {metrics['accuracy']:.4f}")
        print(f"  Balanced acc.    : {metrics['balanced_accuracy']:.4f}")
        print(f"  Recall (fake)    : {metrics['recall']:.4f}")
        print(f"  Precision (fake) : {metrics['precision']:.4f}")
        print(f"  F1               : {metrics['f1']:.4f}")
        best_thr = summary["best_threshold_balacc"]
        best = summary["metrics_at_best"]
        print(f"\n  Best balanced acc threshold: {best_thr:.2f}")
        print(f"    balanced_acc   : {best['balanced_accuracy']:.4f}")
        print(f"    roc_auc        : {best['roc_auc']:.4f}")
        print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
