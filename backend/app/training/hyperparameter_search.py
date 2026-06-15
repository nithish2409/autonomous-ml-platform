"""
Hyperparameter Search Module — Optuna-based Bayesian optimisation.

Supports model types: RandomForest, LogisticRegression, XGBoost, LightGBM,
DecisionTree, SVC, KNN, AdaBoost, ExtraTrees, GradientBoosting.
Each type has a dedicated search space.  If disabled, returns sensible defaults.
"""

import logging
from typing import Any

import numpy as np
import optuna
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
from sklearn.model_selection import cross_val_score

logger = logging.getLogger("hyperparameter_search")

# Silence Optuna's verbose default logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Safe defaults ────────────────────────────────────────────────────
_DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "trials": 20,
}

_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "RandomForest": {"n_estimators": 100, "max_depth": 8},
    "LogisticRegression": {"C": 1.0},
    "XGBoost": {"learning_rate": 0.1, "max_depth": 6},
    "LightGBM": {"learning_rate": 0.1, "max_depth": -1, "num_leaves": 31},
    "GradientBoosting": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
    "DecisionTree": {"max_depth": 8},
    "SVC": {"C": 1.0, "kernel": "rbf"},
    "KNN": {"n_neighbors": 5},
    "AdaBoost": {"n_estimators": 100, "learning_rate": 1.0},
    "ExtraTrees": {"n_estimators": 100, "max_depth": 8},
}


def run_hyperparameter_search(
    model_type: str,
    X_train: np.ndarray | Any,
    y_train: np.ndarray | Any,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run Optuna hyperparameter search for the given *model_type*.

    Parameters
    ----------
    model_type : str
        One of the supported algorithm keys (e.g. ``"RandomForest"``,
        ``"LogisticRegression"``, ``"XGBoost"``, ``"LightGBM"``, etc.).
    X_train, y_train :
        Training data (array-like).
    config : dict, optional
        Must contain ``enabled`` (bool) and ``trials`` (int).

    Returns
    -------
    dict
        Best hyperparameters found (or defaults if disabled / unknown type).
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}

    if not cfg["enabled"]:
        defaults = _DEFAULT_PARAMS.get(model_type, {})
        logger.debug("Hyperparameter search disabled — using defaults: %s", defaults)
        return defaults

    n_trials = int(cfg.get("trials", 20))

    objective = _OBJECTIVES.get(model_type)
    if objective is None:
        logger.warning(
            "No search space defined for model type '%s' — returning defaults", model_type,
        )
        return _DEFAULT_PARAMS.get(model_type, {})

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_params
    logger.info(
        "Optuna search complete for %s (%d trials): best_params=%s  best_score=%.4f",
        model_type, n_trials, best, study.best_value,
    )
    return best


# ── Objective functions per model type ───────────────────────────────

def _rf_objective(trial: optuna.Trial, X: Any, y: Any) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
    }
    clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=3, scoring="f1_weighted")
    return float(scores.mean())


def _lr_objective(trial: optuna.Trial, X: Any, y: Any) -> float:
    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    clf = LogisticRegression(C=C, max_iter=500, random_state=42)
    scores = cross_val_score(clf, X, y, cv=3, scoring="f1_weighted")
    return float(scores.mean())


def _xgb_objective(trial: optuna.Trial, X: Any, y: Any) -> float:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.error("xgboost not installed — cannot run XGBoost search")
        return 0.0

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    clf = XGBClassifier(**params, random_state=42)
    scores = cross_val_score(clf, X, y, cv=3, scoring="f1_weighted")
    return float(scores.mean())


def _lgbm_objective(trial: optuna.Trial, X: Any, y: Any) -> float:
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        logger.error("lightgbm not installed — cannot run LightGBM search")
        return 0.0

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "verbosity": -1,
    }
    clf = LGBMClassifier(**params, random_state=42)
    scores = cross_val_score(clf, X, y, cv=3, scoring="f1_weighted")
    return float(scores.mean())


def _gb_objective(trial: optuna.Trial, X: Any, y: Any) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    }
    clf = GradientBoostingClassifier(**params, random_state=42)
    scores = cross_val_score(clf, X, y, cv=3, scoring="f1_weighted")
    return float(scores.mean())


def _dt_objective(trial: optuna.Trial, X: Any, y: Any) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 20),
    }
    clf = DecisionTreeClassifier(**params, random_state=42)
    scores = cross_val_score(clf, X, y, cv=3, scoring="f1_weighted")
    return float(scores.mean())


def _svc_objective(trial: optuna.Trial, X: Any, y: Any) -> float:
    params = {
        "C": trial.suggest_float("C", 0.01, 100.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
    }
    clf = SVC(**params, probability=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=3, scoring="f1_weighted")
    return float(scores.mean())


def _knn_objective(trial: optuna.Trial, X: Any, y: Any) -> float:
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
    }
    clf = KNeighborsClassifier(**params)
    scores = cross_val_score(clf, X, y, cv=3, scoring="f1_weighted")
    return float(scores.mean())


def _adaboost_objective(trial: optuna.Trial, X: Any, y: Any) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 30, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
    }
    clf = AdaBoostClassifier(**params, random_state=42)
    scores = cross_val_score(clf, X, y, cv=3, scoring="f1_weighted")
    return float(scores.mean())


def _et_objective(trial: optuna.Trial, X: Any, y: Any) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
    }
    clf = ExtraTreesClassifier(**params, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=3, scoring="f1_weighted")
    return float(scores.mean())


_OBJECTIVES = {
    "RandomForest": _rf_objective,
    "LogisticRegression": _lr_objective,
    "XGBoost": _xgb_objective,
    "LightGBM": _lgbm_objective,
    "GradientBoosting": _gb_objective,
    "DecisionTree": _dt_objective,
    "SVC": _svc_objective,
    "KNN": _knn_objective,
    "AdaBoost": _adaboost_objective,
    "ExtraTrees": _et_objective,
}
