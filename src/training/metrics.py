"""
src/training/metrics.py — Evaluation Metrics
=============================================

Wraps sklearn to compute frame-level binary classification metrics
at the end of each train / val / test epoch.

Metrics
-------
roc_auc      Primary checkpointing metric (area under ROC curve).
f1           F1 score at threshold 0.5.
precision    Precision at threshold 0.5.
recall       Recall at threshold 0.5.
accuracy     Accuracy at threshold 0.5.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    accuracy_score,
)


def compute_metrics(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute binary classification metrics.

    Parameters
    ----------
    all_labels : (N,) int/float array — ground truth (0=real, 1=fake)
    all_probs  : (N,) float array    — predicted probabilities P(fake)
    threshold  : float               — decision boundary

    Returns
    -------
    dict with ranking metrics, thresholded metrics, and confusion counts.
    """
    labels = np.asarray(all_labels, dtype=int)
    probs  = np.asarray(all_probs,  dtype=float)
    # Guard against any residual NaN / Inf
    probs  = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    preds  = (probs >= threshold).astype(int)

    # Ranking metrics require at least one sample of each class.
    n_classes = len(np.unique(labels))
    if n_classes < 2:
        roc_auc = float("nan")
        average_precision = float("nan")
    else:
        roc_auc = float(roc_auc_score(labels, probs))
        average_precision = float(average_precision_score(labels, probs))

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    recall = recall_score(labels, preds, zero_division=0)
    balanced_accuracy = balanced_accuracy_score(labels, preds)
    youden_j = recall + specificity - 1.0

    return {
        "roc_auc":             roc_auc,
        "average_precision":   average_precision,
        "f1":                  float(f1_score(labels, preds, zero_division=0)),
        "precision":           float(precision_score(labels, preds, zero_division=0)),
        "recall":              float(recall),
        "specificity":         float(specificity),
        "balanced_accuracy":   float(balanced_accuracy),
        "accuracy":            float(accuracy_score(labels, preds)),
        "mcc":                 float(matthews_corrcoef(labels, preds)),
        "youden_j":            float(youden_j),
        "predicted_fake_rate": float(np.mean(preds)),
        "true_fake_rate":      float(np.mean(labels)),
        "tn":                  int(tn),
        "fp":                  int(fp),
        "fn":                  int(fn),
        "tp":                  int(tp),
    }


def find_best_threshold(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    metric: str = "f1",
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01,
) -> dict:
    """Sweep thresholds and return the best one for the requested metric."""
    labels = np.asarray(all_labels, dtype=int)
    probs  = np.asarray(all_probs, dtype=float)
    probs  = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)

    if len(labels) == 0:
        raise ValueError("Cannot optimize threshold with zero samples.")

    metric = metric.lower()
    valid_metrics = {
        "f1",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "accuracy",
        "mcc",
        "youden_j",
    }
    if metric not in valid_metrics:
        raise ValueError(f"Unsupported threshold metric: {metric}")

    best = None
    thresholds = np.arange(threshold_min, threshold_max + threshold_step * 0.5, threshold_step)
    for threshold in thresholds:
        metrics = compute_metrics(labels, probs, threshold=float(threshold))
        score = metrics[metric]
        candidate = {
            "threshold": float(threshold),
            "score": float(score),
            "metrics": metrics,
        }
        if best is None:
            best = candidate
            continue
        if candidate["score"] > best["score"]:
            best = candidate
            continue
        if candidate["score"] == best["score"] and abs(candidate["threshold"] - 0.5) < abs(best["threshold"] - 0.5):
            best = candidate

    return best


class MetricAccumulator:
    """Accumulate logits and labels across mini-batches, compute at epoch end.

    Usage
    -----
        acc = MetricAccumulator()
        for logits, labels in loader:
            acc.update(logits, labels)
        results = acc.compute()
    """

    def __init__(self):
        self._labels: list = []
        self._probs:  list = []

    def update(self, logits: "torch.Tensor", labels: "torch.Tensor") -> None:
        import torch
        with torch.no_grad():
            # Clamp logits to prevent NaN from very large values in early epochs
            logits_clamped = logits.detach().cpu().float().clamp(-50.0, 50.0)
            probs = torch.sigmoid(logits_clamped).view(-1)
            self._probs.extend(probs.numpy().tolist())
            self._labels.extend(labels.detach().cpu().view(-1).numpy().tolist())

    def update_probs(self, probs: "torch.Tensor", labels: "torch.Tensor") -> None:
        import torch
        with torch.no_grad():
            probs = probs.detach().cpu().float().view(-1).clamp(0.0, 1.0)
            self._probs.extend(probs.numpy().tolist())
            self._labels.extend(labels.detach().cpu().view(-1).numpy().tolist())

    def compute(self, threshold: float = 0.5) -> dict:
        return compute_metrics(
            np.array(self._labels),
            np.array(self._probs),
            threshold=threshold,
        )

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array(self._labels, dtype=int),
            np.array(self._probs, dtype=float),
        )

    def best_threshold(
        self,
        metric: str = "f1",
        threshold_min: float = 0.05,
        threshold_max: float = 0.95,
        threshold_step: float = 0.01,
    ) -> dict:
        labels, probs = self.arrays()
        return find_best_threshold(
            labels,
            probs,
            metric=metric,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_step=threshold_step,
        )

    def reset(self) -> None:
        self._labels.clear()
        self._probs.clear()

    def __len__(self) -> int:
        return len(self._labels)
