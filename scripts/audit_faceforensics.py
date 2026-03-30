#!/usr/bin/env python3
"""
audit_faceforensics.py — Validate FF++ extracted frames and study readiness.

Reports, per class and per split:
- total clip directories
- non-empty clip directories
- usable clip directories after split assignment

This is meant to be the first step before any ablation run so we do not
accidentally train on empty placeholder folders or invalid split layouts.
"""

import argparse
import hashlib
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def collect_video_dirs(root: str):
    if not os.path.isdir(root):
        return []
    return sorted(
        os.path.join(root, name)
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name)) and not name.startswith(".")
    )


def collect_frames(video_dir: str):
    if not os.path.isdir(video_dir):
        return []
    return sorted(
        os.path.join(video_dir, name)
        for name in os.listdir(video_dir)
        if Path(name).suffix.lower() in IMAGE_EXTS and not name.startswith(".")
    )


def extract_numeric_tokens(video_name: str):
    stem = Path(video_name).stem
    if stem.startswith("."):
        return []
    prefix = stem.split("__")[0]
    tokens = []
    for part in prefix.split("_"):
        if re.fullmatch(r"\d+", part):
            tokens.append(int(part))
        else:
            break
    return tokens


def infer_split(video_name: str, split_ranges: dict, split_ratios: dict):
    ids = extract_numeric_tokens(video_name)
    if not ids:
        return None

    if max(ids) >= 100:
        matched = []
        for clip_id in ids:
            found = None
            for split_name, (lo, hi) in split_ranges.items():
                if lo <= clip_id < hi:
                    found = split_name
                    break
            if found is None:
                return None
            matched.append(found)
        return matched[0] if len(set(matched)) == 1 else None

    digest = hashlib.sha1(video_name.encode("utf-8")).hexdigest()
    bucket = (int(digest[:8], 16) % 10_000) / 10_000.0
    train_cut = split_ratios["train"]
    val_cut = train_cut + split_ratios["val"]
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def parse_args():
    p = argparse.ArgumentParser(description="Audit local FaceForensics++ extracted frames")
    p.add_argument("--dataset-config", default="configs/dataset_config.yaml")
    p.add_argument("--train-config", default="configs/train_config_ablation.yaml")
    return p.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    args = parse_args()

    dataset_cfg = load_config(args.dataset_config)
    train_cfg = load_config(args.train_config)

    ff_cfg = dataset_cfg["faceforensics"]
    data_cfg = train_cfg["data"]
    frames_dir = Path(ff_cfg["frame_extraction"]["output_dir"])
    split_ranges = {
        "train": ff_cfg["splits"]["train"],
        "val": ff_cfg["splits"]["val"],
        "test": ff_cfg["splits"]["test"],
    }
    split_ratios = dataset_cfg.get("common", {}).get(
        "split_ratios",
        {"train": 0.72, "val": 0.14, "test": 0.14},
    )

    class_names = [data_cfg.get("real_dir", "real"), *data_cfg.get("methods", [])]

    print("=" * 72)
    print("  FaceForensics++ Frame Audit")
    print("=" * 72)
    print(f"  frames_dir   : {frames_dir}")
    print(f"  real_dir     : {data_cfg.get('real_dir', 'real')}")
    print(f"  fake_methods : {', '.join(data_cfg.get('methods', []))}")
    print(f"  train config : {args.train_config}")
    print("=" * 72)

    summary = defaultdict(lambda: defaultdict(int))
    any_usable = False

    for class_name in class_names:
        class_dir = frames_dir / class_name
        video_dirs = collect_video_dirs(str(class_dir))
        print(f"\n[{class_name}]")
        print(f"  clip_dirs          : {len(video_dirs)}")

        for video_dir in video_dirs:
            clip_name = Path(video_dir).name
            split = infer_split(clip_name, split_ranges, split_ratios)
            has_frames = bool(collect_frames(video_dir))

            summary[class_name]["all"] += 1
            if has_frames:
                summary[class_name]["nonempty"] += 1
            else:
                summary[class_name]["empty"] += 1

            if split is None:
                summary[class_name]["unassigned"] += 1
                continue

            summary[class_name][f"{split}_all"] += 1
            if has_frames:
                summary[class_name][f"{split}_usable"] += 1
                any_usable = True

        print(
            "  nonempty/empty     : "
            f"{summary[class_name]['nonempty']} / {summary[class_name]['empty']}"
        )
        print(
            "  assigned usable    : "
            f"train={summary[class_name]['train_usable']}  "
            f"val={summary[class_name]['val_usable']}  "
            f"test={summary[class_name]['test_usable']}"
        )
        if summary[class_name]["unassigned"]:
            print(f"  unassigned clips   : {summary[class_name]['unassigned']}")

    ablation_cfg = data_cfg.get("ablation", {})
    print("\n" + "-" * 72)
    print("  Ablation Config Limits")
    print("-" * 72)
    print(f"  max_real_videos_per_split           : {ablation_cfg.get('max_real_videos_per_split')}")
    print(f"  max_fake_videos_per_method_per_split: {ablation_cfg.get('max_fake_videos_per_method_per_split')}")
    print(f"  subset_seed                         : {ablation_cfg.get('subset_seed')}")

    if not any_usable:
        print("\n  Result: no non-empty extracted clips were found for the selected setup.")
        print("  Next step: re-run frame extraction before starting ablation training.")
        print("  Suggested command:")
        print("    python3 scripts/extract_frames.py --dataset ff++ --resize 224")
        return

    print("\n  Result: usable extracted clips were found.")
    print("  Suggested next checks:")
    print("    1. Confirm train/val/test counts are not too skewed per method.")
    print("    2. Run the ablation config with a short pilot training job.")


if __name__ == "__main__":
    main()
