"""
Strategy Engine — maps LLM action + drift info to a training strategy.

Strategies determine how aggressively the training pipeline will search
for improvements (full_search > quick_tune > incremental > no_action).
"""

import logging
from typing import Any

logger = logging.getLogger("strategy_engine")

# ── Strategy mapping ─────────────────────────────────────────────────
_RETRAIN_STRATEGY_MAP = {
    "concept_drift": "full_search",
    "feature_drift": "quick_tune",
    "no_drift": "incremental",
}


def choose_strategy(
    llm_action: str,
    drift_info: dict[str, Any],
) -> str:
    """
    Choose a training strategy based on the LLM decision and drift analysis.

    Parameters
    ----------
    llm_action : str
        Action chosen by the LLM (e.g. ``"retrain"``, ``"rollback"``).
    drift_info : dict
        Output of ``analyze_drift()`` — must contain ``drift_type``.

    Returns
    -------
    str
        One of ``"full_search"``, ``"quick_tune"``, ``"incremental"``, ``"no_action"``.
    """
    if llm_action.lower() != "retrain":
        logger.info("LLM action '%s' is not retrain → strategy=no_action", llm_action)
        return "no_action"

    drift_type = drift_info.get("drift_type", "no_drift")
    strategy = _RETRAIN_STRATEGY_MAP.get(drift_type, "incremental")

    logger.info(
        "Strategy chosen: drift_type=%s → strategy=%s",
        drift_type, strategy,
    )
    return strategy
