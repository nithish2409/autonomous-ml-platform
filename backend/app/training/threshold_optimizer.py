"""
Threshold Optimization Module — finds the decision threshold that maximises
a chosen metric (default: F1) for binary/multi-class classifiers.
"""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

logger = logging.getLogger("threshold_optimizer")

# Supported metric functions (binary / weighted-average)
_METRIC_FNS = {
    "f1": lambda y, yp: f1_score(y, yp, average="weighted", zero_division=0),
    "accuracy": accuracy_score,
    "precision": lambda y, yp: precision_score(y, yp, average="weighted", zero_division=0),
    "recall": lambda y, yp: recall_score(y, yp, average="weighted", zero_division=0),
}


def find_best_threshold(
    y_true: np.ndarray | Any,
    y_probs: np.ndarray | Any,
    metric: str = "f1",
) -> dict[str, Any]:
    """
    Sweep thresholds from 0.1 to 0.9 and return the one that maximises *metric*.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_probs : array-like
        Predicted probabilities for the **positive class** (1-D) or the
        full ``predict_proba`` output (2-D — column 1 is used).
    metric : str
        One of ``"f1"``, ``"accuracy"``, ``"precision"``, ``"recall"``.

    Returns
    -------
    dict
        ``{"best_threshold": float, "best_score": float, "metric": str}``
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    # If 2-D probability matrix, take probabilities of the positive class
    if y_probs.ndim == 2:
        y_probs = y_probs[:, 1]

    metric_fn = _METRIC_FNS.get(metric)
    if metric_fn is None:
        logger.warning("Unknown metric '%s' — falling back to f1", metric)
        metric = "f1"
        metric_fn = _METRIC_FNS["f1"]

    thresholds = np.arange(0.1, 0.91, 0.05)
    best_threshold = 0.5
    best_score = -1.0

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        score = float(metric_fn(y_true, y_pred))
        if score > best_score:
            best_score = score
            best_threshold = float(round(t, 2))

    logger.info(
        "Threshold optimization (%s): best_threshold=%.2f  best_score=%.4f",
        metric, best_threshold, best_score,
    )

    return {
        "best_threshold": best_threshold,
        "best_score": round(best_score, 4),
        "metric": metric,
    }
