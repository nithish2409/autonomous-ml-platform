"""Base adapter interface for framework-agnostic model inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseAdapter(ABC):
    """Every framework adapter must implement predict() and optionally predict_proba()."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> list[Any]:
        """Return predictions for the given input DataFrame."""
        ...

    def predict_proba(self, df: pd.DataFrame) -> list[list[float]] | None:
        """Return class probabilities if supported, else None."""
        return None

    def get_feature_names(self) -> list[str] | None:
        """Return expected feature names if the model exposes them."""
        if hasattr(self.model, "feature_names_in_"):
            return list(self.model.feature_names_in_)
        return None
