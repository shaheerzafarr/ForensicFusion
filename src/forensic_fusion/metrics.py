from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(targets: np.ndarray, probabilities: np.ndarray, objective: str = "f1") -> float:
    if objective != "f1":
        raise ValueError("evaluation.threshold_metric currently supports only 'f1'.")
    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(f1 == np.nanmax(f1))
    # On ties, use the highest threshold to prefer precision.
    return float(thresholds[int(best[-1])])


def binary_metrics(targets: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    try:
        roc_auc = float(roc_auc_score(targets, probabilities))
        pr_auc = float(average_precision_score(targets, probabilities))
    except ValueError:
        roc_auc = float("nan")
        pr_auc = float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
