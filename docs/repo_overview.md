# DeepFakeDetection Repository Overview

## Purpose

This repository is code for a deepfake detection algorithm, including data preparation, model training, validation, evaluation, and wrapper scripts around the core model workflow. The overall goal is to provide a practical experimentation environment for studying deepfake detection with a model that is still approachable for beginners.

At its core, the repository is organized around a hybrid deepfake detector named `HSF-CVIT`, short for Hybrid Spatial-Frequency Cross-Attention Vision Transformer. The design combines:

- a spatial branch that learns appearance cues from RGB frames,
- a frequency branch that highlights forensic artifacts and manipulation noise,
- a transformer-style attention fusion block that combines both signals for final prediction.

This makes the project a good fit for introductory research and course work: it is more advanced than a plain CNN baseline, but still modular enough that individual components can be replaced and compared in a controlled way.

## Project Objective

The main objective of the repository is to support deepfake detection experiments in a configurable and reproducible way. Rather than hard-coding one fixed pipeline, the repo is structured so different settings can be studied with minimal code changes.

In practical terms, the repository aims to support:

- training a deepfake detector from prepared frame data,
- validating the model during training using standard binary classification metrics,
- evaluating a final checkpoint on a held-out test split,
- preparing and auditing datasets before experiments begin,
- swapping or tuning model and training settings through YAML configuration files.

This makes the repository useful both as:

- a beginner-friendly starting point for deepfake detection research, and
- a framework for small ablation studies on architecture and training choices.

## High-Level Pipeline

The intended workflow of the repository is:

1. Download or place raw datasets locally.
2. Extract a fixed number of frames from each video.
3. Organize frames by class and clip.
4. Build train, validation, and test data loaders.
5. Train the hybrid detector.
6. Track validation performance and save checkpoints.
7. Evaluate the best model on a held-out test set.

At a high level, the repository breaks down into four layers:

- data acquisition and frame extraction,
- configuration-driven experiment setup,
- model definition and training engine,
- evaluation and experiment logging.

## Dataset Support

The repository is oriented around two commonly used deepfake datasets:

- FaceForensics++
- Celeb-DF v2

### FaceForensics++

The FaceForensics++ configuration supports:

- real videos from YouTube originals,
- optionally actor sequences as additional real data,
- multiple manipulation methods, including `Deepfakes`, `Face2Face`, `FaceShifter`, and `DeepFakeDetection`,
- configurable compression settings such as `c23`.

The frame extraction flow saves frames into class-specific folders such as:

- `data/frames/real/...`
- `data/frames/Deepfakes/...`
- `data/frames/Face2Face/...`

This layout is intended to make downstream dataset loading straightforward.

### Celeb-DF v2

The config also includes paths for:

- `Celeb-real`
- `YouTube-real`
- `Celeb-synthesis`

and a provided test list file for test-set handling.

### Dummy Data Workflow

For development and smoke testing, the repository includes a synthetic data generator. This is useful when the real datasets are unavailable or when the training pipeline needs to be checked without downloading large video collections.

## Model Architecture

The implemented model is a hybrid spatial-frequency detector with transformer-style fusion.

### 1. Spatial Branch

The spatial branch uses `EfficientNet-B4` through `timm`. It extracts high-level appearance features from RGB face frames and projects them into a configurable embedding dimension.

This branch represents the more conventional visual understanding part of the detector:

- facial texture,
- blending artifacts,
- expression inconsistencies,
- identity and appearance-level cues.

### 2. Frequency Branch

The frequency branch applies fixed SRM high-pass filters, then sends the result through a lightweight residual encoder.

This branch is meant to capture forensic traces that may be less obvious in ordinary RGB space, such as:

- resampling artifacts,
- blending noise,
- local high-frequency inconsistencies,
- manipulation fingerprints.

### 3. Transformer-Style Fusion

The fusion head uses cross-attention and self-attention to combine the spatial and frequency embeddings. The spatial token attends to the frequency token, then a small transformer-like refinement block produces the final representation used for classification.

Because of this attention-based fusion stage, the repository can reasonably be described as transformer-based, but not as a pure Vision Transformer pipeline from raw image patches. It is better described as a hybrid model:

- CNN backbone for spatial learning,
- SRM residual encoder for forensic features,
- transformer-style attention for multimodal fusion.

## Training and Evaluation Flow

The training entrypoint is [`train.py`](/home/akshay/CSI-6550-advanced-visual-computing/DeepFakeDetection/train.py). It is designed to:

- load YAML configuration,
- apply command-line overrides,
- build data loaders,
- instantiate the model,
- run training and validation,
- save best and periodic checkpoints,
- evaluate the best model on the test set.

The training engine includes:

- mixed precision when CUDA is available,
- optimizer selection from config,
- cosine, step, or plateau learning-rate schedules,
- warm-up epochs,
- gradient clipping,
- early stopping based on validation ROC-AUC,
- TensorBoard and CSV logging.

Evaluation uses standard binary classification metrics:

- ROC-AUC
- F1
- precision
- recall
- accuracy

## Repository Structure

The repository is broadly divided as follows:

- `configs/`: experiment and dataset configuration
- `scripts/`: dataset download, extraction, audit, and synthetic data utilities
- `src/models/`: model components
- `src/training/`: loss, metrics, and trainer
- `src/utils/`: shared helpers
- `train.py`: main training entrypoint

## Configurability

One of the strongest aspects of this repo is that most important settings are externalized into YAML files. That supports easier experimentation without editing the training code directly.

Examples of configurable behavior include:

- model dimensions,
- number of fusion heads,
- dropout,
- fake methods used for training,
- frames sampled per clip,
- optimizer type,
- learning rate,
- label smoothing,
- checkpoint locations,
- logging destinations.

This is important for deepfake detection research, where comparing models and training settings is often more valuable than building a single rigid pipeline.

## Current State and Practical Notes

The repository already contains a substantial training and modeling stack, plus scripts for dataset preparation and audit. It is clearly intended to function as an experimental deepfake detection framework.

At the same time, the current tracked files suggest the data-loading module expected by `train.py` is not present in the visible `src` tree. Specifically, `train.py` imports `src.data.dataset.build_dataloaders`, but there is no tracked `src/data` package in the repository snapshot I reviewed. That means the design and interfaces for full training are present, but the active workspace may still be incomplete unless that module exists outside the tracked files or is added later.

## Summary

This repository should be understood as a configurable deepfake detection experimentation framework centered on a hybrid beginner-friendly model. It combines:

- dataset preparation utilities,
- a modular training and evaluation pipeline,
- a hybrid CNN plus frequency plus attention architecture,
- configuration-driven experimentation support.

The repo is especially well suited for:

- studying deepfake detection methods,
- running controlled training experiments,
- comparing architecture or training variants,
- learning how modern deepfake detectors are organized end to end.
