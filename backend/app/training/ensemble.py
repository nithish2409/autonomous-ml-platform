"""
Ensemble Module — combines old and new model predictions with configurable weights.

Provides a ``WeightedEnsemble`` wrapper that blends ``predict_proba`` outputs
and a ``build_ensemble`` helper for conditional construction.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("ensemble")


class WeightedEnsemble:
    """Blends predictions from an old and a new model using weighted averaging."""

    def __init__(self, old_model: Any, new_model: Any, weight_new: float = 0.7):
        self.old_model = old_model
        self.new_model = new_model
        self.weight_new = weight_new

    def predict_proba(self, X: Any) -> np.ndarray:
        new_probs = self.new_model.predict_proba(X)
        old_probs = self.old_model.predict_proba(X)
        return self.weight_new * new_probs + (1 - self.weight_new) * old_probs

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

    def __repr__(self) -> str:
        return (
            f"WeightedEnsemble(weight_new={self.weight_new}, "
            f"old={type(self.old_model).__name__}, new={type(self.new_model).__name__})"
        )


def build_ensemble(
    old_model: Any | None,
    new_model: Any,
    config: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Optionally wrap *new_model* in a ``WeightedEnsemble`` with *old_model*.

    Parameters
    ----------
    old_model : previously trained model (may be None).
    new_model : newly trained model to potentially combine.
    config : dict with ``enabled`` (bool) and ``weight_new`` (float, 0–1).

    Returns
    -------
    model : either the ensemble or the plain new_model
    metadata : dict with ``ensemble_used`` flag
    """
    cfg = config or {}
    meta: dict[str, Any] = {"ensemble_used": False}

    if not cfg.get("enabled", False):
        return new_model, meta

    if old_model is None:
        logger.info("Ensemble enabled but no old model available — using new model only")
        return new_model, meta

    # Verify both models support predict_proba
    if not (hasattr(old_model, "predict_proba") and hasattr(new_model, "predict_proba")):
        logger.warning(
            "Ensemble requires predict_proba — one model lacks it. Skipping ensemble."
        )
        return new_model, meta

    weight_new = float(cfg.get("weight_new", 0.7))
    ensemble = WeightedEnsemble(old_model, new_model, weight_new=weight_new)

    meta["ensemble_used"] = True
    meta["weight_new"] = weight_new
    logger.info("Built WeightedEnsemble (weight_new=%.2f)", weight_new)

    return ensemble, meta
