"""
src/data/dataset.py — Dataset Classes for DeepFake Detection
=============================================================

Classes
-------
FaceForensicsDataset
    Loads extracted frames from the real FaceForensics++ c23 directory
    structure produced by scripts/extract_frames.py.

    Expected layout (produced by extract_frames.py):
        <frames_dir>/
            real/
                <video_id>/
                    frame_000.jpg
                    frame_001.jpg
                    ...
            Deepfakes/
                <video_id>/
                    frame_000.jpg
                    ...
            Face2Face/  FaceSwap/  NeuralTextures/

DummyFaceForensicsDataset
    Returns randomly generated tensors (no I/O). Use this to validate
    the full training pipeline while the real dataset is downloading.

Usage
-----
    from src.data.dataset import FaceForensicsDataset, DummyFaceForensicsDataset, build_dataloaders

    # Real data
    train_dl, val_dl, test_dl = build_dataloaders(cfg, train_cfg)

    # Dummy (--dummy flag in train.py)
    train_dl, val_dl, test_dl = build_dataloaders(cfg, train_cfg, dummy=True)
"""

import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

from src.data.transforms import get_transforms


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _collect_video_dirs(root: str) -> List[str]:
    """Return sorted list of sub-directory paths (one per video clip)."""
    if not os.path.isdir(root):
        return []
    entries = sorted([
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    ])
    return entries


def _collect_frames(video_dir: str) -> List[str]:
    """Return sorted list of image paths inside *video_dir*."""
    if not os.path.isdir(video_dir):
        return []
    return sorted([
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if Path(f).suffix.lower() in IMAGE_EXTS and not f.startswith(".")
    ])


def _extract_numeric_tokens(video_name: str) -> List[int]:
    """Extract leading numeric identifiers from a FF++ clip name.

    Examples
    --------
    000 -> [0]
    014_790 -> [14, 790]
    02_15__meeting_serious__TSRK8QS4 -> [2, 15]
    """
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


def _format_ffpp_id(clip_id: int) -> str:
    """Format an FF++ clip identifier as a zero-padded 3-digit string."""
    return f"{int(clip_id):03d}"


def _load_official_ffpp_splits(split_files: Dict[str, str]) -> dict:
    """Load official FF++ split json files into real-id and fake-pair lookup tables."""
    required = ("train", "val", "test")
    missing = [name for name in required if not split_files.get(name)]
    if missing:
        raise ValueError(
            "Official FF++ split mode requires split_files for train/val/test. "
            f"Missing: {missing}"
        )

    split_pairs = {}
    split_real_ids = {}

    for split_name in required:
        path = split_files[split_name]
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Official FF++ split file not found for split={split_name}: {path}"
            )

        with open(path, "r", encoding="utf-8") as handle:
            entries = json.load(handle)

        if not isinstance(entries, list):
            raise ValueError(
                f"Official FF++ split file must contain a list of pairs: {path}"
            )

        pair_set = set()
        real_id_set = set()
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(
                    f"Invalid FF++ split entry in {path}: expected a 2-item list, got {entry!r}"
                )

            a = _format_ffpp_id(entry[0])
            b = _format_ffpp_id(entry[1])
            pair_set.add(tuple(sorted((a, b))))
            real_id_set.update((a, b))

        split_pairs[split_name] = pair_set
        split_real_ids[split_name] = real_id_set

    return {
        "pairs": split_pairs,
        "real_ids": split_real_ids,
    }


def _infer_split(
    video_name: str,
    split_ranges: dict,
    split_ratios: Optional[dict] = None,
    split_mode: str = "numeric",
    official_splits: Optional[dict] = None,
) -> Optional[str]:
    """Infer train / val / test membership from a FF++ clip directory name.

    Numeric youtube-style names follow explicit index ranges.
    Actor-pair DeepFakeDetection names use a deterministic hash split because
    their IDs do not map onto the 0..999 FF++ youtube index ranges.
    """
    split_mode = (split_mode or "numeric").lower()
    split_ratios = split_ratios or {"train": 0.72, "val": 0.14, "test": 0.14}
    ids = _extract_numeric_tokens(video_name)
    if not ids:
        return None

    if split_mode == "official":
        if official_splits is None:
            raise ValueError("official split mode requires loaded official_splits data")

        if len(ids) >= 2:
            pair = tuple(sorted(_format_ffpp_id(clip_id) for clip_id in ids[:2]))
            matches = [
                split_name
                for split_name, pair_set in official_splits["pairs"].items()
                if pair in pair_set
            ]
        else:
            real_id = _format_ffpp_id(ids[0])
            matches = [
                split_name
                for split_name, real_ids in official_splits["real_ids"].items()
                if real_id in real_ids
            ]
        return matches[0] if len(matches) == 1 else None

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

    # DeepFakeDetection actor IDs are 1..28 rather than youtube indices.
    digest = hashlib.sha1(video_name.encode("utf-8")).hexdigest()
    bucket = (int(digest[:8], 16) % 10_000) / 10_000.0
    train_cut = split_ratios["train"]
    val_cut = train_cut + split_ratios["val"]
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def _sample_dirs(video_dirs: List[str], limit: Optional[int], seed: int) -> List[str]:
    """Deterministically subsample a sorted list of clip directories."""
    if limit is None or limit <= 0 or len(video_dirs) <= limit:
        return sorted(video_dirs)
    rng = random.Random(seed)
    sampled = rng.sample(sorted(video_dirs), limit)
    return sorted(sampled)


def _uniform_indices(length: int, count: int) -> List[int]:
    """Return ``count`` approximately-uniform indices across ``length`` items."""
    if length <= 0:
        return []
    if count <= 1:
        return [length // 2]
    if count >= length:
        return list(range(length))

    last = length - 1
    return [round(i * last / (count - 1)) for i in range(count)]


# ------------------------------------------------------------------ #
#  FaceForensicsDataset
# ------------------------------------------------------------------ #

class FaceForensicsDataset(Dataset):
    """Frame-level dataset for FaceForensics++ (c23 extracted frames).

    Each call to ``__getitem__`` returns ONE randomly sampled frame from
    ONE randomly sampled video clip. This gives natural data augmentation
    across epochs without pre-computing a fixed frame list.

    Parameters
    ----------
    frames_dir : str
        Root of the extracted frames (e.g. ``data/FaceForensics++/frames``).
    split : str
        ``"train"``, ``"val"``, or ``"test"``.
    methods : list[str]
        Manipulation method sub-directories to treat as *fake*.
    split_ranges : dict
        Keys ``train``, ``val``, ``test`` mapping to ``[start, end]``
        video index ranges (integer video IDs).
    frames_per_clip : int
        Number of frames to randomly sample per video *per item*.
        ``1`` is the standard frame-level setting. Values ``>1`` average
        multiple sampled frames into one tensor before the model sees them,
        unless ``return_clip=True``.
    deterministic_sampling : bool
        When True, select a fixed set of frames for each clip instead of
        random sampling. Intended for stable validation/test evaluation.
    frame_selection_strategy : str
        ``"random"``, ``"center"``, or ``"uniform"``.
    image_size : int
        Spatial resize target.
    return_clip : bool
        When True, return a fixed stack of sampled frames shaped
        ``(frames_per_clip, C, H, W)`` for video-level evaluation.
    """

    def __init__(
        self,
        frames_dir: str,
        split: str = "train",
        methods: Optional[List[str]] = None,
        real_dir_name: str = "real",
        split_ranges: Optional[dict] = None,
        frames_per_clip: int = 8,
        image_size: int = 224,
        split_ratios: Optional[dict] = None,
        split_mode: str = "numeric",
        official_splits: Optional[dict] = None,
        max_real_videos: Optional[int] = None,
        max_fake_videos_per_method: Optional[int] = None,
        subset_seed: int = 42,
        deterministic_sampling: bool = False,
        frame_selection_strategy: str = "random",
        return_clip: bool = False,
        items_per_clip: int = 1,
    ):
        self.split = split
        self.frames_per_clip = frames_per_clip
        self.return_clip = bool(return_clip)
        self.transform = get_transforms(split, image_size)
        self.split_ratios = split_ratios or {"train": 0.72, "val": 0.14, "test": 0.14}
        self.split_mode = split_mode
        self.official_splits = official_splits
        self.deterministic_sampling = bool(deterministic_sampling)
        self.frame_selection_strategy = frame_selection_strategy.lower()

        valid_strategies = {"random", "center", "uniform"}
        if self.frame_selection_strategy not in valid_strategies:
            raise ValueError(
                "frame_selection_strategy must be one of "
                f"{sorted(valid_strategies)}, got {frame_selection_strategy!r}"
            )

        methods = methods or ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        split_ranges = split_ranges or {
            "train": [0, 720],
            "val":   [720, 860],
            "test":  [860, 1000],
        }

        # ------- Collect real clips -------
        real_root = os.path.join(frames_dir, real_dir_name)
        real_dirs = _collect_video_dirs(real_root)
        real_dirs = self._filter_dirs_for_split(real_dirs, split, split_ranges)
        self._frame_paths_by_video: Dict[str, List[str]] = {}
        real_dirs = self._filter_nonempty_dirs(real_dirs)
        real_dirs = _sample_dirs(real_dirs, max_real_videos, subset_seed + 11)

        # ------- Collect fake clips -------
        fake_samples = []
        self.fake_dirs_by_method = {}
        for method_idx, method in enumerate(methods):
            method_root = os.path.join(frames_dir, method)
            vids = _collect_video_dirs(method_root)
            vids = self._filter_dirs_for_split(vids, split, split_ranges)
            vids = self._filter_nonempty_dirs(vids)
            vids = _sample_dirs(vids, max_fake_videos_per_method, subset_seed + 101 + method_idx)
            self.fake_dirs_by_method[method] = len(vids)
            fake_samples.extend(
                (d, 1, method, Path(d).name)
                for d in vids
            )

        # Build flat sample list: (video_dir, label, method, video_id)
        self.samples: List[Tuple[str, int, str, str]] = (
            [(d, 0, real_dir_name, Path(d).name) for d in real_dirs] +
            fake_samples
        )

        # Repeat each clip entry so that one epoch sees items_per_clip
        # independently-sampled frames per video.  Only applied during
        # training (deterministic_sampling=False); val/test pass 1.
        if items_per_clip > 1 and not self.deterministic_sampling:
            self.samples = self.samples * items_per_clip

        if len(real_dirs) == 0 or len(fake_samples) == 0:
            raise RuntimeError(
                f"Insufficient class coverage for split={split} in {frames_dir}: "
                f"real={len(real_dirs)}, fake={len(fake_samples)}. "
                "Check selected methods, split configuration, and whether extracted "
                "frame folders contain image files."
            )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found in {frames_dir} for split={split}. "
                "Run scripts/extract_frames.py first, or use --dummy flag."
            )

    def _filter_dirs_for_split(
        self,
        video_dirs: List[str],
        split: str,
        split_ranges: dict,
    ) -> List[str]:
        """Keep only clip directories assigned to the requested split."""
        filtered = []
        for d in video_dirs:
            assigned = _infer_split(
                Path(d).name,
                split_ranges,
                self.split_ratios,
                split_mode=self.split_mode,
                official_splits=self.official_splits,
            )
            if assigned == split:
                filtered.append(d)
        return filtered

    def _filter_nonempty_dirs(self, video_dirs: List[str]) -> List[str]:
        """Cache frame paths once and keep only directories with image files."""
        filtered = []
        for video_dir in video_dirs:
            frame_paths = _collect_frames(video_dir)
            if frame_paths:
                self._frame_paths_by_video[video_dir] = frame_paths
                filtered.append(video_dir)
        return filtered

    def __len__(self) -> int:
        return len(self.samples)

    def _select_frame_paths(self, frame_paths: List[str]) -> List[str]:
        """Choose frame paths according to the dataset sampling policy."""
        target_frames = max(1, int(self.frames_per_clip))
        num_frames = min(target_frames, len(frame_paths))
        if num_frames <= 0:
            return []

        if not self.deterministic_sampling or self.frame_selection_strategy == "random":
            return random.sample(frame_paths, num_frames)

        if self.frame_selection_strategy == "center":
            center = len(frame_paths) // 2
            if num_frames == 1:
                return [frame_paths[center]]

            half = num_frames // 2
            start = max(0, center - half)
            end = start + num_frames
            if end > len(frame_paths):
                end = len(frame_paths)
                start = end - num_frames
            return frame_paths[start:end]

        indices = _uniform_indices(len(frame_paths), num_frames)
        return [frame_paths[i] for i in indices]

    def _pad_frame_paths(self, chosen: List[str]) -> List[str]:
        """Repeat the last selected frame so clip batches have fixed length."""
        target_frames = max(1, int(self.frames_per_clip))
        if not chosen or len(chosen) >= target_frames:
            return chosen
        return chosen + [chosen[-1]] * (target_frames - len(chosen))

    def __getitem__(self, idx: int):
        video_dir, label, method, video_id = self.samples[idx]
        frame_paths = self._frame_paths_by_video.get(video_dir)
        if frame_paths is None:
            frame_paths = _collect_frames(video_dir)
            if frame_paths:
                self._frame_paths_by_video[video_dir] = frame_paths

        if not frame_paths:
            raise RuntimeError(
                f"Clip directory has no extracted frames: {video_dir}. "
                "Re-run scripts/extract_frames.py or remove empty placeholders."
            )

        # Training stays stochastic; validation/test can use fixed frame choices.
        chosen = self._select_frame_paths(frame_paths)
        if self.return_clip:
            chosen = self._pad_frame_paths(chosen)
        frames = [self.transform(Image.open(p).convert("RGB")) for p in chosen]

        frame_tensor = torch.stack(frames)
        image = frame_tensor if self.return_clip else frame_tensor.mean(dim=0)
        meta = {
            "method": method,
            "video_id": video_id,
            "video_dir": video_dir,
            "split": self.split,
            "num_frames_used": len(chosen),
            "video_eval": self.return_clip,
        }
        return image, torch.tensor(label, dtype=torch.float32), meta


# ------------------------------------------------------------------ #
#  DummyFaceForensicsDataset — in-memory, no I/O
# ------------------------------------------------------------------ #

class DummyFaceForensicsDataset(Dataset):
    """In-memory dummy dataset for pipeline smoke-testing.

    Generates random image tensors with balanced real/fake labels.
    No disk I/O — safe to use while the real dataset is downloading.

    Parameters
    ----------
    size : int
        Number of samples (default 256 — enough for a few batches).
    image_size : int
        Spatial dimension of each generated tensor.
    split : str
        Affects augmentation pipeline (train vs val).
    """

    def __init__(self, size: int = 256, image_size: int = 224, split: str = "train"):
        self.size = size
        self.image_size = image_size
        self.split = split
        # Fixed seed for reproducibility within each split
        gen = torch.Generator()
        gen.manual_seed({"train": 0, "val": 1, "test": 2}.get(split, 0))
        self._images = torch.randn(size, 3, image_size, image_size, generator=gen)
        self._labels = torch.randint(0, 2, (size,), dtype=torch.float32, generator=gen)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        label = int(self._labels[idx].item())
        meta = {
            "method": "dummy_fake" if label == 1 else "dummy_real",
            "video_id": f"{self.split}_{idx:05d}",
            "video_dir": "",
            "split": self.split,
            "num_frames_used": 1,
            "video_eval": False,
        }
        return self._images[idx], self._labels[idx], meta


# ------------------------------------------------------------------ #
#  build_dataloaders — factory
# ------------------------------------------------------------------ #

def build_dataloaders(
    dataset_cfg: dict,
    train_cfg: dict,
    dummy: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders.

    Parameters
    ----------
    dataset_cfg : dict
        Loaded ``dataset_config.yaml`` or ``dataset_config_dummy.yaml``.
    train_cfg : dict
        Loaded ``train_config.yaml``.
    dummy : bool
        If True, use ``DummyFaceForensicsDataset`` regardless of config.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    data_cfg = train_cfg["data"]
    batch_size = train_cfg["training"]["batch_size"]
    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", True)
    image_size = data_cfg.get("image_size", 224)
    frames_per_clip = data_cfg.get("frames_per_clip", 8)
    methods = data_cfg.get("methods", None)
    real_dir_name = data_cfg.get("real_dir", "real")
    ablation_cfg = data_cfg.get("ablation", {})
    split_ratios = dataset_cfg.get("common", {}).get(
        "split_ratios",
        {"train": 0.72, "val": 0.14, "test": 0.14},
    )
    split_mode = dataset_cfg.get("faceforensics", {}).get("split_mode", "numeric")
    balance_strategy = data_cfg.get("balance_strategy", "none").lower()
    train_items_per_clip = int(data_cfg.get("train_items_per_clip", 1))
    eval_cfg = train_cfg.get("evaluation", {})
    deterministic_eval = bool(eval_cfg.get("deterministic_eval", True))
    eval_frames_per_clip = int(eval_cfg.get("eval_frames_per_clip", 1))
    video_eval = bool(eval_cfg.get("video_eval", False))
    eval_frame_strategy = eval_cfg.get("eval_frame_strategy")
    if eval_frame_strategy is None:
        eval_frame_strategy = "center" if eval_frames_per_clip == 1 else "uniform"
    eval_frame_strategy = eval_frame_strategy.lower()

    def _split_limit(key: str, split: str) -> Optional[int]:
        value = ablation_cfg.get(key)
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, dict):
            split_value = value.get(split)
            return int(split_value) if split_value is not None else None
        raise TypeError(f"{key} must be an int or split->int mapping.")

    if dummy:
        # Sizes: train=512, val=128, test=128 — small but enough for smoke test
        train_ds = DummyFaceForensicsDataset(size=512, image_size=image_size, split="train")
        val_ds   = DummyFaceForensicsDataset(size=128, image_size=image_size, split="val")
        test_ds  = DummyFaceForensicsDataset(size=128, image_size=image_size, split="test")
    else:
        ff_cfg = dataset_cfg["faceforensics"]
        frames_dir = ff_cfg["frame_extraction"]["output_dir"]
        if methods is None:
            methods = ff_cfg.get(
                "manipulation_methods",
                ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
            )
        methods = list(methods)

        existing_methods = [m for m in methods if os.path.isdir(os.path.join(frames_dir, m))]
        missing_methods = [m for m in methods if m not in existing_methods]
        if missing_methods:
            print(f"  ⚠  Skipping missing frame folders for methods: {missing_methods}")
        if not existing_methods:
            available = sorted(
                d for d in os.listdir(frames_dir)
                if os.path.isdir(os.path.join(frames_dir, d)) and d != real_dir_name
            ) if os.path.isdir(frames_dir) else []
            raise RuntimeError(
                "No selected fake method folders were found in extracted frames.\n"
                f"  frames_dir={frames_dir}\n"
                f"  requested={methods}\n"
                f"  available={available}"
            )

        split_ranges = {
            "train": ff_cfg["splits"]["train"],
            "val":   ff_cfg["splits"]["val"],
            "test":  ff_cfg["splits"]["test"],
        }
        official_splits = None
        if split_mode == "official":
            official_splits = _load_official_ffpp_splits(ff_cfg.get("split_files", {}))

        train_ds = FaceForensicsDataset(
            frames_dir=frames_dir, split="train", methods=existing_methods,
            real_dir_name=real_dir_name,
            split_ranges=split_ranges, frames_per_clip=frames_per_clip,
            image_size=image_size,
            split_ratios=split_ratios,
            split_mode=split_mode,
            official_splits=official_splits,
            max_real_videos=_split_limit("max_real_videos_per_split", "train"),
            max_fake_videos_per_method=_split_limit("max_fake_videos_per_method_per_split", "train"),
            subset_seed=int(ablation_cfg.get("subset_seed", 42)),
            deterministic_sampling=False,
            frame_selection_strategy="random",
            items_per_clip=train_items_per_clip,
        )
        val_ds = FaceForensicsDataset(
            frames_dir=frames_dir, split="val", methods=existing_methods,
            real_dir_name=real_dir_name,
            split_ranges=split_ranges, frames_per_clip=eval_frames_per_clip,
            image_size=image_size,
            split_ratios=split_ratios,
            split_mode=split_mode,
            official_splits=official_splits,
            max_real_videos=_split_limit("max_real_videos_per_split", "val"),
            max_fake_videos_per_method=_split_limit("max_fake_videos_per_method_per_split", "val"),
            subset_seed=int(ablation_cfg.get("subset_seed", 42)),
            deterministic_sampling=deterministic_eval,
            frame_selection_strategy=eval_frame_strategy,
            return_clip=video_eval,
        )
        test_ds = FaceForensicsDataset(
            frames_dir=frames_dir, split="test", methods=existing_methods,
            real_dir_name=real_dir_name,
            split_ranges=split_ranges, frames_per_clip=eval_frames_per_clip,
            image_size=image_size,
            split_ratios=split_ratios,
            split_mode=split_mode,
            official_splits=official_splits,
            max_real_videos=_split_limit("max_real_videos_per_split", "test"),
            max_fake_videos_per_method=_split_limit("max_fake_videos_per_method_per_split", "test"),
            subset_seed=int(ablation_cfg.get("subset_seed", 42)),
            deterministic_sampling=deterministic_eval,
            frame_selection_strategy=eval_frame_strategy,
            return_clip=video_eval,
        )

    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        common_kwargs["persistent_workers"] = True
        common_kwargs["prefetch_factor"] = data_cfg.get("prefetch_factor", 4)

    train_sampler = None
    train_shuffle = True
    if not dummy and balance_strategy == "weighted_sampler":
        label_counts = Counter(int(sample[1]) for sample in train_ds.samples)
        sample_weights = [
            1.0 / label_counts[int(sample[1])]
            for sample in train_ds.samples
        ]
        generator = torch.Generator()
        generator.manual_seed(int(train_cfg.get("seed", 42)))
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_ds,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=True,
        **common_kwargs,
    )
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **common_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **common_kwargs)

    return train_loader, val_loader, test_loader
