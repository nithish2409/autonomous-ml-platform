"""
Feature Selection Module — selects the most informative features before training.

Supports two methods:
  • variance   — drops low-variance features via sklearn VarianceThreshold
  • model_importance — ranks features by RandomForest importance, keeps top_k
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger("feature_selection")

# ── Safe defaults ────────────────────────────────────────────────────
_DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "method": "model_importance",
    "top_k": 20,
    "variance_threshold": 0.0,     # for VarianceThreshold (drop constant cols)
}


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Select features from *X* based on *config*.

    Returns
    -------
    selected_X : pd.DataFrame
        DataFrame with only the selected columns.
    selected_feature_names : list[str]
        Column names that were kept.
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}

    if not cfg["enabled"]:
        logger.debug("Feature selection disabled — returning all %d features", X.shape[1])
        return X, list(X.columns)

    method = cfg["method"]

    if method == "variance":
        return _variance_selection(X, cfg)
    elif method == "model_importance":
        return _model_importance_selection(X, y, cfg)
    else:
        logger.warning("Unknown feature selection method '%s' — returning all features", method)
        return X, list(X.columns)


# ── Private helpers ──────────────────────────────────────────────────

def _variance_selection(
    X: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """Drop features whose variance is below the configured threshold."""
    threshold = float(cfg.get("variance_threshold", 0.0))
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)

    mask = selector.get_support()
    selected_cols = X.columns[mask].tolist()

    logger.info(
        "Variance selection (threshold=%.4f): %d → %d features",
        threshold, X.shape[1], len(selected_cols),
    )
    return X[selected_cols], selected_cols


def _model_importance_selection(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """Train a lightweight RandomForest to rank features by importance, keep top_k."""
    top_k = int(cfg.get("top_k", 20))
    top_k = min(top_k, X.shape[1])  # can't keep more than available

    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]
    selected_cols = X.columns[indices].tolist()

    logger.info(
        "Model importance selection (top_k=%d): %d → %d features",
        top_k, X.shape[1], len(selected_cols),
    )
    return X[selected_cols], selected_cols
