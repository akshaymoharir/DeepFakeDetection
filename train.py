#!/usr/bin/env python3
"""
train.py — HSF-CVIT Training Entry Point
=========================================

Usage
-----
# Smoke-test with dummy data (no FaceForensics++ needed)
python train.py --dummy

# Train on real FF++ (after extract_frames.py has run)
python train.py

# Override config path
python train.py --config configs/train_config.yaml

# Resume from checkpoint
python train.py --resume outputs/checkpoints/best.pt

# Quick smoke-test: 2 epochs, batch-size 4
python train.py --dummy --epochs 2 --batch-size 4

# Evaluate on test set only (no training)
python train.py --eval-only --resume outputs/checkpoints/best.pt

# Multi-frame eval-only test
python train.py --eval-only --resume outputs/checkpoints/best.pt --eval-frames 8 --eval-frame-strategy uniform

# True video-level eval: run each frame separately, average probabilities per clip
python train.py --eval-only --video-eval --resume outputs/checkpoints/best.pt --eval-frames 8 --eval-frame-strategy uniform
"""

import argparse
import os
import sys

import torch
import yaml

# ── Project root on path ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.helpers import seed_everything, setup_script_logging
from src.data.dataset  import build_dataloaders
from src.models.hsf_cvit import build_model
from src.training.trainer import Trainer


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train HSF-CVIT deepfake detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    p.add_argument(
        "--config", default="configs/train_config.yaml",
        help="Path to train_config.yaml",
    )

    # Data mode
    p.add_argument(
        "--dummy", action="store_true",
        help="Use in-memory dummy dataset (no real data needed)",
    )

    # Overrides (take precedence over YAML)
    p.add_argument("--epochs",     type=int,   default=None, help="Override training.epochs")
    p.add_argument("--batch-size", type=int,   default=None, help="Override training.batch_size")
    p.add_argument("--lr",         type=float, default=None, help="Override training.lr")
    p.add_argument("--no-amp",     action="store_true",      help="Disable mixed-precision")
    p.add_argument("--workers",    type=int,   default=None, help="Override data.num_workers")
    p.add_argument(
        "--methods", type=str, default=None,
        help="Comma-separated FF++ fake methods (e.g. Deepfakes,Face2Face,FaceShifter)",
    )
    p.add_argument(
        "--real-dir", type=str, default=None,
        help="Name of real folder under extracted frames (default: real)",
    )
    p.add_argument(
        "--items-per-clip", type=int, default=None,
        help="Override data.train_items_per_clip (lower = faster epochs, less per-epoch frame diversity)",
    )
    p.add_argument(
        "--eval-frames", type=int, default=None,
        help="Override evaluation.eval_frames_per_clip for val/test loaders",
    )
    p.add_argument(
        "--eval-frame-strategy", choices=["center", "uniform", "random"], default=None,
        help="Override evaluation.eval_frame_strategy",
    )
    p.add_argument(
        "--video-eval", action="store_true",
        help="Evaluate clips by averaging per-frame model probabilities",
    )
    p.add_argument(
        "--report-dir", type=str, default=None,
        help="Override evaluation.report_dir for saved eval reports",
    )

    # Checkpoint
    p.add_argument("--resume",    default=None, help="Path to checkpoint to resume from")

    # Eval-only mode
    p.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, only run test-set evaluation (requires --resume)",
    )

    # Device
    p.add_argument("--device", default=None, help="Force device: 'cuda', 'cpu', 'cuda:1' …")

    return p.parse_args()


# ------------------------------------------------------------------ #
#  Config loader + overrides
# ------------------------------------------------------------------ #

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Patch cfg in-place with any command-line overrides."""
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.no_amp:
        cfg["training"]["amp"] = False
    if args.workers is not None:
        cfg["data"]["num_workers"] = args.workers
    if args.methods:
        cfg["data"]["methods"] = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.real_dir:
        cfg["data"]["real_dir"] = args.real_dir
    if args.items_per_clip is not None:
        cfg["data"]["train_items_per_clip"] = args.items_per_clip
    if args.eval_frames is not None:
        cfg.setdefault("evaluation", {})["eval_frames_per_clip"] = args.eval_frames
    if args.eval_frame_strategy is not None:
        cfg.setdefault("evaluation", {})["eval_frame_strategy"] = args.eval_frame_strategy
    if args.video_eval:
        cfg.setdefault("evaluation", {})["video_eval"] = True
    if args.report_dir is not None:
        cfg.setdefault("evaluation", {})["report_dir"] = args.report_dir
    return cfg


# ------------------------------------------------------------------ #
#  Model summary printer
# ------------------------------------------------------------------ #

def print_model_summary(model) -> None:
    if not hasattr(model, "count_parameters"):
        return
    counts = model.count_parameters()
    print("\n  ── Model parameter counts ────────────────────────────")
    for name, c in counts.items():
        total, train = c["total"], c["trainable"]
        print(f"    {name:20s}  total={total:>10,}  trainable={train:>10,}")
    print("  ─────────────────────────────────────────────────────\n")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main() -> None:
    setup_script_logging("train")
    args = parse_args()
    if args.video_eval and not args.eval_only:
        raise ValueError("--video-eval is only supported together with --eval-only.")

    # --- Config ---
    train_cfg = load_config(args.config)
    train_cfg = apply_cli_overrides(train_cfg, args)

    # May need to load the dataset config referenced inside train_config
    dataset_cfg_path = train_cfg["data"].get("config", "configs/dataset_config_dummy.yaml")
    dataset_cfg = load_config(dataset_cfg_path)

    # --- Reproducibility ---
    seed = train_cfg.get("seed", 42)
    seed_everything(seed)

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("  ⚠  CUDA not available — running on CPU (slow).")
        if train_cfg["training"].get("amp", True):
            train_cfg["training"]["amp"] = False
            print("  ⚠  AMP disabled (CPU-only).")

    print(f"\n{'='*60}")
    print(f"  HSF-CVIT DeepFake Detector")
    print(f"  Device : {device}")
    print(f"  Config : {args.config}")
    print(f"  Dummy  : {args.dummy}")
    if args.resume:
        print(f"  Resume : {args.resume}")
    print(f"{'='*60}")

    # --- Build dataloaders ---
    print("\n▶ Building dataloaders …")
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_cfg=dataset_cfg,
        train_cfg=train_cfg,
        dummy=args.dummy,
    )
    items_per_clip = train_cfg["data"].get("train_items_per_clip", 1)
    print(
        f"  train={len(train_loader.dataset):,}  "
        f"val={len(val_loader.dataset):,}  "
        f"test={len(test_loader.dataset):,} samples"
        + (f"  (×{items_per_clip} items/clip)" if items_per_clip > 1 and not args.dummy else "")
    )

    # --- Build model ---
    print("\n▶ Building HSF-CVIT model …")
    model = build_model(train_cfg)
    print_model_summary(model)

    # --- Build trainer ---
    trainer = Trainer(model, train_cfg, device)

    # --- Resume ---
    start_epoch = 0
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)

    # --- Eval-only mode ---
    if args.eval_only:
        if not args.resume:
            print("  ⚠  --eval-only with no --resume: evaluating randomly initialised model.")
        trainer.evaluate(test_loader)
        return

    # --- Train ---
    trainer.fit(train_loader, val_loader, start_epoch=start_epoch)

    # --- Final test evaluation ---
    print("\n▶ Loading best checkpoint for test evaluation …")
    best_ckpt = os.path.join(train_cfg["checkpoints"]["dir"], "best.pt")
    if os.path.isfile(best_ckpt):
        trainer.load_checkpoint(best_ckpt)
    else:
        print("  ⚠  best.pt not found — using last epoch weights.")

    test_results = trainer.evaluate(test_loader)

    # Print final summary
    print("Final test results:")
    for k, v in test_results.items():
        print(f"  {k:12s}: {v:.4f}")


if __name__ == "__main__":
    main()
