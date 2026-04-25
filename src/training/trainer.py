"""
src/training/trainer.py — Training & Validation Engine
=======================================================

Features
--------
- Mixed-precision training (torch.amp) with GradScaler
- Cosine / step / plateau LR schedules + linear warm-up
- Gradient clipping
- EfficientNet warm-up (spatial branch frozen for first N epochs)
- Best-checkpoint saving on val ROC-AUC
- Periodic checkpoint saving + auto-pruning of old saves
- Early stopping on val ROC-AUC
- TensorBoard logging (scalar loss, all metrics, LR)
- CSV row-per-epoch log
- Progress bars via tqdm

Usage (called from train.py)
----------------------------
    from src.training.trainer import Trainer
    trainer = Trainer(model, train_cfg, device)
    trainer.fit(train_loader, val_loader, start_epoch=0)
    results = trainer.evaluate(test_loader)
"""

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    LinearLR,
    SequentialLR,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.training.losses import build_criterion
from src.training.metrics import MetricAccumulator, compute_metrics


# ------------------------------------------------------------------ #
#  Trainer
# ------------------------------------------------------------------ #

class Trainer:
    """Full training + validation engine for HSF-CVIT.

    Parameters
    ----------
    model : nn.Module
        The HSF-CVIT model (or any nn.Module).
    train_cfg : dict
        Loaded ``configs/train_config.yaml``.
    device : torch.device
        Target compute device.
    """

    def __init__(self, model: nn.Module, train_cfg: dict, device: torch.device):
        self.model     = model.to(device)
        self.device    = device
        self.train_cfg = train_cfg

        t_cfg  = train_cfg["training"]
        ck_cfg = train_cfg["checkpoints"]
        lg_cfg = train_cfg["logging"]
        ev_cfg = train_cfg.get("evaluation", {})

        # ---- Optimizer ----
        self.optimizer = self._build_optimizer(t_cfg)

        # ---- LR Schedule (with optional warm-up) ----
        self.scheduler, self.warmup_scheduler = self._build_scheduler(t_cfg)

        # ---- Loss ----
        self.criterion = build_criterion(train_cfg).to(device)

        # ---- AMP ----
        self.amp_enabled = t_cfg.get("amp", True) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.amp_enabled)

        # ---- Gradient clipping ----
        self.grad_clip = t_cfg.get("gradient_clip", 1.0)

        # ---- Warm-up epochs for spatial branch ----
        self.freeze_spatial_epochs = train_cfg["model"].get("freeze_spatial_epochs", 2)

        # ---- Evaluation / thresholding ----
        self.default_threshold = float(ev_cfg.get("decision_threshold", 0.5))
        self.optimize_threshold = bool(ev_cfg.get("optimize_threshold", True))
        self.threshold_metric = ev_cfg.get("threshold_metric", "f1")
        self.threshold_min = float(ev_cfg.get("threshold_min", 0.05))
        self.threshold_max = float(ev_cfg.get("threshold_max", 0.95))
        self.threshold_step = float(ev_cfg.get("threshold_step", 0.01))
        self.report_dir = ev_cfg.get("report_dir", "outputs/evaluation")
        self.save_reports = bool(ev_cfg.get("save_reports", True))
        os.makedirs(self.report_dir, exist_ok=True)

        # ---- Checkpointing ----
        self.ckpt_dir      = ck_cfg["dir"]
        self.save_every    = ck_cfg.get("save_every", 5)
        self.keep_last     = ck_cfg.get("keep_last", 3)
        self.patience      = ck_cfg.get("early_stop_patience", 10)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # ---- Logging ----
        self.log_interval = lg_cfg.get("log_interval", 10)
        self.tb_writer    = SummaryWriter(log_dir=lg_cfg.get("tensorboard_dir", "outputs/runs"))
        self.csv_path     = lg_cfg.get("csv_log", "outputs/training_log.csv")
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self._csv_initialized = False

        # ---- State ----
        self.global_step   = 0
        self.best_val_auc  = 0.0
        self.best_threshold = self.default_threshold
        self.epochs_no_imp = 0           # early-stop counter

    # ------------------------------------------------------------------
    #  Optimizer & LR schedule builders
    # ------------------------------------------------------------------

    def _build_optimizer(self, t_cfg: dict):
        opt_name = t_cfg.get("optimizer", "adamw").lower()
        lr       = float(t_cfg.get("lr", 3e-4))
        wd       = float(t_cfg.get("weight_decay", 1e-4))

        if opt_name == "adamw":
            return AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adam":
            return Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_scheduler(self, t_cfg: dict):
        schedule     = t_cfg.get("lr_schedule", "cosine").lower()
        epochs       = t_cfg.get("epochs", 50)
        warmup_ep    = t_cfg.get("warmup_epochs", 5)
        lr           = float(t_cfg.get("lr", 3e-4))

        # Main scheduler (post-warm-up)
        if schedule == "cosine":
            main_sched = CosineAnnealingLR(
                self.optimizer, T_max=max(1, epochs - warmup_ep), eta_min=lr * 0.01
            )
        elif schedule == "step":
            main_sched = StepLR(self.optimizer, step_size=10, gamma=0.5)
        elif schedule == "plateau":
            main_sched = ReduceLROnPlateau(self.optimizer, mode="max", patience=5, factor=0.5)
            return main_sched, None  # plateau doesn't compose with SequentialLR
        else:
            raise ValueError(f"Unknown lr_schedule: {schedule}")

        if warmup_ep > 0:
            warmup_sched = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_ep,
            )
            combined = SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[warmup_ep],
            )
            return combined, None
        return main_sched, None

    # ------------------------------------------------------------------
    #  Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader:   torch.utils.data.DataLoader,
        start_epoch:  int = 0,
    ) -> None:
        """Run the full training loop.

        Parameters
        ----------
        train_loader, val_loader : DataLoader
        start_epoch : int
            Resume from this epoch (0-indexed).
        """
        epochs = self.train_cfg["training"]["epochs"]
        print(f"\n{'='*60}")
        print(f"  Training for {epochs} epochs on {self.device}")
        print(f"  AMP: {self.amp_enabled}  |  Grad clip: {self.grad_clip}")
        print(f"{'='*60}\n")

        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()

            # --- Spatial branch freeze control ---
            if epoch < self.freeze_spatial_epochs:
                if hasattr(self.model, "freeze_spatial"):
                    self.model.freeze_spatial()
                    if epoch == 0:
                        print(f"  [Warm-up] EfficientNet backbone frozen for {self.freeze_spatial_epochs} epoch(s).")
            else:
                if hasattr(self.model, "unfreeze_spatial"):
                    self.model.unfreeze_spatial()
                    if epoch == self.freeze_spatial_epochs:
                        print(f"  [Epoch {epoch+1}] EfficientNet backbone unfrozen.")

            # --- Train ---
            train_loss, train_metrics = self._train_epoch(train_loader, epoch)

            # --- Validate ---
            val_loss, val_metrics, threshold_info = self._val_epoch(val_loader, epoch)

            # --- LR step ---
            sched = self.scheduler
            if isinstance(sched, ReduceLROnPlateau):
                sched.step(val_metrics["roc_auc"])
            else:
                sched.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            # --- Console summary ---
            self._print_epoch_summary(
                epoch, epochs, epoch_time, current_lr,
                train_loss, train_metrics,
                val_loss,   val_metrics,
            )

            # --- TensorBoard ---
            self._log_epoch_tb(epoch, train_loss, train_metrics, val_loss, val_metrics, current_lr)

            # --- CSV ---
            self._log_epoch_csv(epoch, train_loss, train_metrics, val_loss, val_metrics, current_lr)

            # --- Checkpoint ---
            val_auc = val_metrics.get("roc_auc", 0.0)
            if not math.isnan(val_auc) and val_auc > self.best_val_auc:
                self.best_threshold = (
                    threshold_info["threshold"]
                    if threshold_info is not None
                    else self.default_threshold
                )
            self._maybe_save_checkpoint(epoch, val_auc)

            # --- Early stopping ---
            if self._early_stop():
                print(f"\n  ⏹  Early stopping triggered at epoch {epoch+1} "
                      f"(no improvement for {self.patience} epochs).\n")
                break

        self.tb_writer.close()
        print(f"\n✓ Training complete. Best val AUC: {self.best_val_auc:.4f}")

    # ------------------------------------------------------------------
    #  Single epoch: train
    # ------------------------------------------------------------------

    def _train_epoch(self, loader, epoch: int):
        self.model.train()
        accumulator = MetricAccumulator()
        total_loss  = 0.0
        n_batches   = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1:03d} [train]", leave=False, ncols=100)
        for batch_idx, batch in enumerate(pbar):
            images, labels, _ = self._unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits = self.model(images)             # (B, 1)
                loss   = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first)
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_loss = loss.item()
            total_loss += batch_loss
            n_batches  += 1
            self.global_step += 1

            accumulator.update(logits, labels)

            # Batch-level TensorBoard log
            if self.global_step % self.log_interval == 0:
                self.tb_writer.add_scalar("train/batch_loss", batch_loss, self.global_step)

            pbar.set_postfix(loss=f"{batch_loss:.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        metrics  = accumulator.compute()
        return avg_loss, metrics

    # ------------------------------------------------------------------
    #  Single epoch: validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _val_epoch(self, loader, epoch: int):
        self.model.eval()
        accumulator = MetricAccumulator()
        total_loss  = 0.0
        n_batches   = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1:03d} [val]  ", leave=False, ncols=100)
        for batch in pbar:
            images, labels, _ = self._unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            total_loss += loss.item()
            n_batches  += 1
            accumulator.update(logits, labels)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        metrics  = accumulator.compute(threshold=self.default_threshold)
        threshold_info = None
        if self.optimize_threshold:
            threshold_info = accumulator.best_threshold(
                metric=self.threshold_metric,
                threshold_min=self.threshold_min,
                threshold_max=self.threshold_max,
                threshold_step=self.threshold_step,
            )
            metrics["optimal_threshold"] = threshold_info["threshold"]
            metrics[f"optimal_{self.threshold_metric}"] = threshold_info["score"]
            for key, value in threshold_info["metrics"].items():
                metrics[f"thresholded_{key}"] = value
        return avg_loss, metrics, threshold_info

    # ------------------------------------------------------------------
    #  Test / evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader, threshold: Optional[float] = None) -> dict:
        """Evaluate on a held-out test set and return metrics dict."""
        self.model.eval()
        accumulator = MetricAccumulator()
        prediction_rows = []
        total_loss  = 0.0
        n_batches   = 0
        eval_threshold = self.best_threshold if threshold is None else float(threshold)

        print("\n▶ Evaluating on test set …")
        for batch in tqdm(loader, desc="  Test", ncols=100):
            images, labels, metadata = self._unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            total_loss += loss.item()
            n_batches  += 1
            accumulator.update(logits, labels)
            prediction_rows.extend(self._prediction_rows(logits, labels, metadata))

        metrics = accumulator.compute(threshold=eval_threshold)
        metrics["loss"] = total_loss / max(n_batches, 1)
        metrics["threshold"] = eval_threshold
        if abs(eval_threshold - self.default_threshold) > 1e-8:
            fixed_metrics = accumulator.compute(threshold=self.default_threshold)
            for key, value in fixed_metrics.items():
                metrics[f"default_{key}"] = value

        per_method_rows = self._per_method_metrics(prediction_rows, eval_threshold)
        if self.save_reports:
            self._write_eval_reports(metrics, per_method_rows, prediction_rows)

        print("\n  ── Test Results ──────────────────────")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"    {k:20s}: {v:.4f}")
        print("  ──────────────────────────────────────\n")
        if per_method_rows:
            print("  Per-method ROC-AUC:")
            for row in per_method_rows:
                if row["method"] == "all":
                    continue
                print(
                    f"    {row['method']:16s} "
                    f"AUC={row['roc_auc']:.4f}  "
                    f"AP={row['average_precision']:.4f}  "
                    f"BalAcc={row['balanced_accuracy']:.4f}"
                )
            print()
        return metrics

    # ------------------------------------------------------------------
    #  Checkpoint helpers
    # ------------------------------------------------------------------

    def _maybe_save_checkpoint(self, epoch: int, val_auc: float) -> None:
        is_best = not math.isnan(val_auc) and val_auc > self.best_val_auc

        if is_best:
            self.best_val_auc  = val_auc
            self.epochs_no_imp = 0
        else:
            self.epochs_no_imp += 1

        state = {
            "epoch":          epoch + 1,
            "model":          self.model.state_dict(),
            "optimizer":      self.optimizer.state_dict(),
            "scheduler":      self.scheduler.state_dict(),
            "scaler":         self.scaler.state_dict(),
            "best_val_auc":   self.best_val_auc,
            "best_threshold": self.best_threshold,
            "global_step":    self.global_step,
        }

        # Best model
        if is_best:
            best_path = os.path.join(self.ckpt_dir, "best.pt")
            torch.save(state, best_path)
            print(f"  💾 New best val AUC: {val_auc:.4f} → {best_path}")

        # Periodic checkpoint
        if (epoch + 1) % self.save_every == 0:
            pt_path = os.path.join(self.ckpt_dir, f"epoch_{epoch+1:03d}.pt")
            torch.save(state, pt_path)
            self._prune_old_checkpoints()

    def _prune_old_checkpoints(self) -> None:
        """Keep only the last ``keep_last`` periodic checkpoints."""
        pts = sorted(Path(self.ckpt_dir).glob("epoch_*.pt"))
        for old in pts[: max(0, len(pts) - self.keep_last)]:
            old.unlink()

    def load_checkpoint(self, path: str) -> int:
        """Load a checkpoint and return the next epoch to resume from."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.best_val_auc = ckpt.get("best_val_auc", 0.0)
        self.best_threshold = float(ckpt.get("best_threshold", self.default_threshold))
        self.global_step  = ckpt.get("global_step", 0)
        start_epoch = ckpt["epoch"]
        print(
            f"  ▶ Resumed from {path}  (epoch {start_epoch}, "
            f"best AUC {self.best_val_auc:.4f}, threshold {self.best_threshold:.2f})"
        )
        return start_epoch

    # ------------------------------------------------------------------
    #  Early stopping
    # ------------------------------------------------------------------

    def _early_stop(self) -> bool:
        return self.epochs_no_imp >= self.patience

    # ------------------------------------------------------------------
    #  Batch / report helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 2:
            images, labels = batch
            return images, labels, None
        if len(batch) == 3:
            images, labels, metadata = batch
            return images, labels, metadata
        raise ValueError(f"Expected batch with 2 or 3 items, got {len(batch)}.")

    @staticmethod
    def _meta_at(metadata, key: str, index: int, default=None):
        if metadata is None or key not in metadata:
            return default
        value = metadata[key]
        if torch.is_tensor(value):
            item = value[index]
            return item.item() if item.ndim == 0 else item.tolist()
        if isinstance(value, (list, tuple)):
            return value[index]
        return value

    def _prediction_rows(self, logits, labels, metadata) -> list:
        probs = torch.sigmoid(logits.detach().cpu().float().clamp(-50.0, 50.0)).view(-1)
        labels_cpu = labels.detach().cpu().view(-1)
        rows = []
        for i, (label, prob) in enumerate(zip(labels_cpu.tolist(), probs.tolist())):
            rows.append({
                "label": int(label),
                "prob": float(prob),
                "method": self._meta_at(metadata, "method", i, "unknown"),
                "video_id": self._meta_at(metadata, "video_id", i, ""),
                "video_dir": self._meta_at(metadata, "video_dir", i, ""),
                "split": self._meta_at(metadata, "split", i, ""),
                "num_frames_used": int(self._meta_at(metadata, "num_frames_used", i, 1)),
            })
        return rows

    @staticmethod
    def _per_method_metrics(prediction_rows: list, threshold: float) -> list:
        if not prediction_rows:
            return []

        labels = [row["label"] for row in prediction_rows]
        probs = [row["prob"] for row in prediction_rows]
        all_metrics = compute_metrics(labels, probs, threshold=threshold)
        rows = [{
            "method": "all",
            "n_real": int(sum(label == 0 for label in labels)),
            "n_fake": int(sum(label == 1 for label in labels)),
            **all_metrics,
        }]

        fake_methods = sorted({
            row["method"]
            for row in prediction_rows
            if row["label"] == 1
        })
        for method in fake_methods:
            subset = [
                row for row in prediction_rows
                if row["label"] == 0 or row["method"] == method
            ]
            method_labels = [row["label"] for row in subset]
            method_probs = [row["prob"] for row in subset]
            method_metrics = compute_metrics(method_labels, method_probs, threshold=threshold)
            rows.append({
                "method": method,
                "n_real": int(sum(label == 0 for label in method_labels)),
                "n_fake": int(sum(label == 1 for label in method_labels)),
                **method_metrics,
            })
        return rows

    @staticmethod
    def _json_ready(value):
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, dict):
            return {k: Trainer._json_ready(v) for k, v in value.items()}
        if isinstance(value, list):
            return [Trainer._json_ready(v) for v in value]
        return value

    def _write_eval_reports(self, metrics: dict, per_method_rows: list, prediction_rows: list) -> None:
        os.makedirs(self.report_dir, exist_ok=True)

        metrics_path = os.path.join(self.report_dir, "test_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self._json_ready(metrics), f, indent=2, sort_keys=True)

        confusion_path = os.path.join(self.report_dir, "test_confusion_matrix.csv")
        confusion_fields = ["threshold", "tn", "fp", "fn", "tp"]
        with open(confusion_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=confusion_fields)
            writer.writeheader()
            writer.writerow({
                "threshold": metrics.get("threshold", self.default_threshold),
                "tn": metrics.get("tn", 0),
                "fp": metrics.get("fp", 0),
                "fn": metrics.get("fn", 0),
                "tp": metrics.get("tp", 0),
            })

        if per_method_rows:
            per_method_path = os.path.join(self.report_dir, "test_per_method.csv")
            fields = list(per_method_rows[0].keys())
            with open(per_method_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self._json_ready(per_method_rows))

        if prediction_rows:
            predictions_path = os.path.join(self.report_dir, "test_predictions.csv")
            fields = list(prediction_rows[0].keys())
            with open(predictions_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(prediction_rows)

        print(f"  Evaluation reports saved to {self.report_dir}")

    # ------------------------------------------------------------------
    #  Logging helpers
    # ------------------------------------------------------------------

    def _print_epoch_summary(
        self, epoch, epochs, t, lr,
        tr_loss, tr_m, vl_loss, vl_m,
    ) -> None:
        star = "★" if not math.isnan(vl_m.get("roc_auc", float("nan"))) and \
               vl_m.get("roc_auc", 0) >= self.best_val_auc else " "
        print(
            f"Epoch {epoch+1:03d}/{epochs}  [{t:.0f}s]  "
            f"LR={lr:.2e}  "
            f"Train loss={tr_loss:.4f}  AUC={tr_m.get('roc_auc',float('nan')):.4f}  "
            f"| Val loss={vl_loss:.4f}  AUC={vl_m.get('roc_auc',float('nan')):.4f}  "
            f"BalAcc@0.50={vl_m.get('balanced_accuracy',0):.4f}  "
            f"F1@0.50={vl_m.get('f1',0):.4f}  "
            f"T*={vl_m.get('optimal_threshold', self.default_threshold):.2f}  "
            f"{self.threshold_metric.upper()}*={vl_m.get(f'optimal_{self.threshold_metric}', vl_m.get(self.threshold_metric, 0)):.4f}  "
            f"{star}"
        )

    def _log_epoch_tb(self, epoch, tr_loss, tr_m, vl_loss, vl_m, lr) -> None:
        w = self.tb_writer
        w.add_scalar("train/loss",    tr_loss, epoch)
        w.add_scalar("val/loss",      vl_loss, epoch)
        w.add_scalar("lr",            lr,      epoch)
        for k, v in tr_m.items():
            if not math.isnan(v):
                w.add_scalar(f"train/{k}", v, epoch)
        for k, v in vl_m.items():
            if not math.isnan(v):
                w.add_scalar(f"val/{k}",   v, epoch)

    def _log_epoch_csv(self, epoch, tr_loss, tr_m, vl_loss, vl_m, lr) -> None:
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": tr_loss,
            "val_loss":   vl_loss,
            **{f"train_{k}": v for k, v in tr_m.items()},
            **{f"val_{k}":   v for k, v in vl_m.items()},
        }
        mode = "w" if not self._csv_initialized else "a"
        with open(self.csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(row)
