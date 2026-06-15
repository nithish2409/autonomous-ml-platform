"""
Automated Model Selection Module — trains multiple algorithms and picks the best.

Supported classifiers: LogisticRegression, RandomForest, GradientBoosting,
DecisionTree, SVC, KNN, AdaBoost, ExtraTrees, XGBoost, LightGBM.

Each is evaluated on a validation set; the best model is chosen by a configurable
metric (default: f1_score).
"""

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger("model_selector")

# ── Safe defaults ────────────────────────────────────────────────────
_DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "metric": "f1_score",
    "random_seed": 42,
}

# ── Algorithm catalogue ─────────────────────────────────────────────
_ALGORITHMS: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "decision_tree": DecisionTreeClassifier,
    "svc": SVC,
    "knn": KNeighborsClassifier,
    "adaboost": AdaBoostClassifier,
    "extra_trees": ExtraTreesClassifier,
}

_DEFAULT_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"max_iter": 500},
    "random_forest": {"n_estimators": 100, "max_depth": 10},
    "gradient_boosting": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
    "decision_tree": {"max_depth": 10},
    "svc": {"probability": True},  # probability=True needed for predict_proba
    "knn": {"n_neighbors": 5},
    "adaboost": {"n_estimators": 100, "learning_rate": 1.0},
    "extra_trees": {"n_estimators": 100, "max_depth": 10},
}

# ── Optional external frameworks (graceful fallback) ────────────────
try:
    from xgboost import XGBClassifier
    _ALGORITHMS["xgboost"] = XGBClassifier
    _DEFAULT_HYPERPARAMS["xgboost"] = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
except ImportError:
    logger.debug("xgboost not installed — skipping in model selection")

try:
    from lightgbm import LGBMClassifier
    _ALGORITHMS["lightgbm"] = LGBMClassifier
    _DEFAULT_HYPERPARAMS["lightgbm"] = {
        "n_estimators": 100,
        "max_depth": -1,
        "learning_rate": 0.1,
        "verbosity": -1,
    }
except ImportError:
    logger.debug("lightgbm not installed — skipping in model selection")


def _evaluate(model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute classification metrics for a fitted model."""
    y_pred = model.predict(X)
    return {
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y, y_pred, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y, y_pred, average="weighted", zero_division=0)), 4),
    }


def train_candidate_models(
    X_train: np.ndarray | Any,
    y_train: np.ndarray | Any,
    X_val: np.ndarray | Any,
    y_val: np.ndarray | Any,
    config: dict[str, Any] | None = None,
) -> tuple[Any, str, dict[str, float], list[dict[str, Any]]]:
    """
    Train multiple algorithms and evaluate on validation set.

    Parameters
    ----------
    X_train, y_train : Training data.
    X_val, y_val : Validation data (used for leaderboard scoring).
    config : dict with ``enabled``, ``metric``, ``random_seed``.

    Returns
    -------
    best_model : fitted sklearn estimator
    best_model_name : str
    best_metrics : dict of metric scores
    leaderboard : list of dicts sorted by chosen metric
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    seed = int(cfg.get("random_seed", 42))
    ranking_metric = cfg.get("metric", "f1_score")

    leaderboard: list[dict[str, Any]] = []
    best_model = None
    best_model_name = ""
    best_score = -1.0
    best_metrics: dict[str, float] = {}

    for name, cls in _ALGORITHMS.items():
        try:
            params = dict(_DEFAULT_HYPERPARAMS.get(name, {}))
            # Inject random_state if the estimator accepts it
            import inspect
            sig = inspect.signature(cls)
            if "random_state" in sig.parameters:
                params["random_state"] = seed

            model = cls(**params)
            model.fit(X_train, y_train)

            metrics = _evaluate(model, X_val, y_val)
            score = metrics.get(ranking_metric, 0.0)

            leaderboard.append({"model": name, **metrics})

            logger.info("Model %s — %s=%.4f", name, ranking_metric, score)

            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name
                best_metrics = metrics

        except Exception as exc:
            logger.warning("Failed to train %s: %s", name, exc)
            leaderboard.append({"model": name, "error": str(exc)})

    # Sort leaderboard by ranking metric (descending)
    leaderboard.sort(key=lambda x: x.get(ranking_metric, 0.0), reverse=True)

    logger.info(
        "Model selection complete — best: %s (%s=%.4f)",
        best_model_name, ranking_metric, best_score,
    )

    return best_model, best_model_name, best_metrics, leaderboard
