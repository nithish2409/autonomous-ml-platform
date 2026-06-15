"""
Config Generator — converts a training strategy into a concrete training_config
that drives the Phase A + Phase B pipeline modules.
"""

import logging
from typing import Any

logger = logging.getLogger("config_generator")

# ── Strategy → training_config templates ─────────────────────────────

_STRATEGY_CONFIGS: dict[str, dict[str, Any]] = {
    "full_search": {
        "feature_selection": {"enabled": True, "method": "model_importance", "top_k": 20},
        "rebalancing": {"enabled": True, "imbalance_threshold": 0.4},
        "hyperparameter_search": {"enabled": True, "trials": 25},
        "model_selection": {"enabled": True, "metric": "f1_score"},
        "threshold_optimization": {"enabled": True, "metric": "f1"},
        "ensemble": {"enabled": True, "weight_new": 0.7},
    },
    "quick_tune": {
        "feature_selection": {"enabled": False},
        "rebalancing": {"enabled": True, "imbalance_threshold": 0.4},
        "hyperparameter_search": {"enabled": True, "trials": 10},
        "model_selection": {"enabled": False},
        "threshold_optimization": {"enabled": True, "metric": "f1"},
        "ensemble": {"enabled": False},
    },
    "incremental": {
        "feature_selection": {"enabled": False},
        "rebalancing": {"enabled": False},
        "hyperparameter_search": {"enabled": False},
        "model_selection": {"enabled": False},
        "threshold_optimization": {"enabled": False},
        "ensemble": {"enabled": False},
    },
}


def generate_training_config(strategy: str) -> dict[str, Any] | None:
    """
    Convert a strategy name into a full ``training_config`` dict.

    Parameters
    ----------
    strategy : str
        One of ``"full_search"``, ``"quick_tune"``, ``"incremental"``, ``"no_action"``.

    Returns
    -------
    dict or None
        ``None`` means "do not retrain". Otherwise a config dict
        consumable by ``TrainingRunner.train_candidate()``.
    """
    if strategy == "no_action":
        logger.info("Strategy 'no_action' → returning None (skip retraining)")
        return None

    config = _STRATEGY_CONFIGS.get(strategy)
    if config is None:
        logger.warning("Unknown strategy '%s' → returning None", strategy)
        return None

    logger.info("Generated training config for strategy '%s'", strategy)
    return config
