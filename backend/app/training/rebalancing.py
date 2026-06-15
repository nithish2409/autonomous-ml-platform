"""
Data Rebalancing Module — handles class imbalance before training.

Detects imbalance ratio and applies SMOTE (if imblearn available)
or falls back to returning class-weight metadata for models to use.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("rebalancing")

# ── Safe defaults ────────────────────────────────────────────────────
_DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "imbalance_threshold": 0.4,  # minority/majority ratio below this triggers rebalancing
}


def _compute_imbalance_ratio(y: np.ndarray) -> float:
    """Return minority_count / majority_count (0–1). Lower = more imbalanced."""
    unique, counts = np.unique(y, return_counts=True)
    if len(counts) < 2:
        return 1.0
    return float(min(counts) / max(counts))


def rebalance_data(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """
    Apply class imbalance handling.

    Returns
    -------
    X_resampled, y_resampled, metadata
        metadata includes ``imbalance_ratio``, ``method_used``, ``rebalanced``
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    y_arr = np.asarray(y)

    imbalance_ratio = _compute_imbalance_ratio(y_arr)
    meta: dict[str, Any] = {
        "imbalance_ratio": round(imbalance_ratio, 4),
        "method_used": "none",
        "rebalanced": False,
    }

    if not cfg["enabled"]:
        logger.debug("Rebalancing disabled — returning original data")
        return X, y, meta

    threshold = float(cfg.get("imbalance_threshold", 0.4))

    if imbalance_ratio >= threshold:
        logger.info(
            "Imbalance ratio %.4f >= threshold %.4f — no rebalancing needed",
            imbalance_ratio, threshold,
        )
        return X, y, meta

    # ── Try SMOTE first ──
    try:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        meta["method_used"] = "smote"
        meta["rebalanced"] = True
        meta["original_size"] = len(y_arr)
        meta["resampled_size"] = len(y_res)

        logger.info(
            "SMOTE applied: %d → %d samples (ratio %.4f → balanced)",
            len(y_arr), len(y_res), imbalance_ratio,
        )

        # Preserve DataFrame structure if input was DataFrame
        if isinstance(X, pd.DataFrame):
            X_res = pd.DataFrame(X_res, columns=X.columns)
            y_res = pd.Series(y_res, name=getattr(y, "name", "target"))

        return X_res, y_res, meta

    except ImportError:
        logger.info("imblearn not available — falling back to class_weight strategy")

    # ── Fallback: signal that class weights should be used ──
    meta["method_used"] = "class_weight"
    meta["rebalanced"] = False
    logger.info(
        "Class-weight strategy flagged (imbalance ratio %.4f). "
        "Models should use class_weight='balanced'.",
        imbalance_ratio,
    )
    return X, y, meta
