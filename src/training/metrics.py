"""
Evaluation metrics computed from a NumPy confusion matrix.

No scikit-learn dependency: precision / recall / F1 per class,
macro & weighted F1, macro recall and accuracy are derived directly.
"""

from typing import Dict

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Row=true class, column=predicted class. Shape [num_classes, num_classes]."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)
    return cm


def metrics_from_confusion(cm: np.ndarray) -> Dict:
    """Compute accuracy + per-class and aggregate metrics."""
    tp = np.diag(cm).astype(np.float64)
    pred_sums = cm.sum(axis=0).astype(np.float64)   # column sums
    true_sums = cm.sum(axis=1).astype(np.float64)   # row sums
    total = cm.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_sums > 0, tp / pred_sums, 0.0)
        recall = np.where(true_sums > 0, tp / true_sums, 0.0)
        f1_denom = precision + recall
        f1 = np.where(f1_denom > 0, 2 * precision * recall / f1_denom, 0.0)

    support = true_sums
    present = support > 0

    return {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f1 * support).sum() / support.sum()) if total else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.astype(int).tolist(),
    }
