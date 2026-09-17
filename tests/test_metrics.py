from __future__ import annotations

import numpy as np

from forensic_fusion.metrics import binary_metrics, choose_threshold


def test_threshold_and_metrics_on_separable_predictions() -> None:
    targets = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.01, 0.10, 0.20, 0.80, 0.90, 0.99])
    threshold = choose_threshold(targets, probabilities)
    metrics = binary_metrics(targets, probabilities, threshold)
    assert 0.2 < threshold <= 0.8
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["pr_auc"] == 1.0
