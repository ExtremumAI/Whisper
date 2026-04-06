# utils/dec_utils.py
# Utility functions for the DEC (Deep Embedded Clustering) module.

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution P from the soft assignment Q.

    P sharpens Q by squaring and re-normalizing:
        p_ij = (q_ij^2 / sum_i q_ij) / sum_j (q_ij^2 / sum_i q_ij)

    Args:
        q: (N, K) soft assignment probabilities from DEC.

    Returns:
        p: (N, K) sharpened target distribution.
    """
    weight = (q ** 2) / q.sum(dim=0, keepdim=True)
    p = weight / weight.sum(dim=1, keepdim=True)
    return p


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute clustering accuracy using the Hungarian algorithm to find the
    best one-to-one mapping between predicted cluster IDs and true labels.

    Args:
        y_true: (N,) ground-truth class labels.
        y_pred: (N,) predicted cluster assignments.

    Returns:
        Accuracy in [0, 1].
    """
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    assert y_true.shape == y_pred.shape, (
        f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
    )

    n_classes = max(y_true.max(), y_pred.max()) + 1
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)

    for true, pred in zip(y_true, y_pred):
        confusion[true, pred] += 1

    # Hungarian matching: maximize total assignment
    row_ind, col_ind = linear_sum_assignment(-confusion)
    accuracy = confusion[row_ind, col_ind].sum() / max(len(y_true), 1)
    return float(accuracy)
