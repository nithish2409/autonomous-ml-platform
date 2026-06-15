"""
Drift Analyzer — classifies drift type from monitoring metrics.

Translates numerical drift scores and performance deltas into
categorized drift types with severity levels.
"""

import logging
from typing import Any
from app.core.monitoring_config import DRIFT_THRESHOLD, DRIFT_CRITICAL

logger = logging.getLogger("drift_analyzer")


def analyze_drift(
    drift_score: float,
    performance_delta: float = 0.0,
) -> dict[str, Any]:
    """
    Classify drift from monitoring signals.

    Parameters
    ----------
    drift_score : float
        Data drift score (0–1). Higher = more drift.
    performance_delta : float
        Change in primary metric (negative = degradation).

    Returns
    -------
    dict with ``drift_type`` and ``severity``.
    """
    if drift_score < DRIFT_THRESHOLD:
        drift_type = "no_drift"
        severity = "low"
    elif drift_score <= DRIFT_CRITICAL:
        drift_type = "feature_drift"
        severity = "medium"
    else:
        # High drift — check for concept drift (performance also dropped)
        if performance_delta < 0:
            drift_type = "concept_drift"
            severity = "high"
        else:
            drift_type = "feature_drift"
            severity = "medium"

    result = {
        "drift_type": drift_type,
        "severity": severity,
        "drift_score": round(drift_score, 4),
        "performance_delta": round(performance_delta, 4),
    }

    logger.info(
        "Drift analysis: score=%.4f, perf_delta=%.4f → type=%s, severity=%s",
        drift_score, performance_delta, drift_type, severity,
    )
    return result
