"""
Early Stopping — prevents unnecessary retraining when metrics plateau.

Checks whether recent training runs show meaningful improvement.
If improvement stays below a threshold for ``patience`` consecutive runs,
signals that retraining should be skipped.
"""

import logging
from typing import Any

logger = logging.getLogger("early_stopping")


def should_stop_early(
    history: list[dict[str, Any]],
    metric: str = "f1_score",
    min_improvement: float = 0.01,
    patience: int = 2,
) -> bool:
    """
    Determine whether retraining should be skipped.

    Parameters
    ----------
    history : list of dict
        Recent training metrics (newest last). Each dict should contain
        the ``metric`` key.
    metric : str
        Metric name to track (default ``"f1_score"``).
    min_improvement : float
        Minimum relative improvement to count as "progressing".
    patience : int
        Number of consecutive non-improving runs before stopping.

    Returns
    -------
    bool
        ``True`` if no meaningful improvement in the last *patience* runs.
    """
    if len(history) < patience + 1:
        logger.debug("Not enough history (%d runs) for early stopping", len(history))
        return False

    recent = history[-(patience + 1):]
    scores = [run.get(metric) for run in recent if run.get(metric) is not None]

    if len(scores) < patience + 1:
        logger.debug("Insufficient '%s' values in history for early stopping", metric)
        return False

    # Check consecutive improvements
    non_improving = 0
    for i in range(1, len(scores)):
        improvement = scores[i] - scores[i - 1]
        if improvement < min_improvement:
            non_improving += 1
        else:
            non_improving = 0  # Reset streak

    stop = non_improving >= patience

    if stop:
        logger.info(
            "Early stopping triggered: %d consecutive runs with < %.2f%% improvement in '%s'",
            non_improving, min_improvement * 100, metric,
        )
    else:
        logger.debug(
            "Early stopping not triggered (non_improving=%d, patience=%d)",
            non_improving, patience,
        )

    return stop
