"""Adapter for scikit-learn / joblib models."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.adapters.base_adapter import BaseAdapter


class SklearnAdapter(BaseAdapter):
    """Handles sklearn-style models that expose .predict() and optionally .predict_proba()."""

    def predict(self, df: pd.DataFrame) -> list[Any]:
        return self.model.predict(df).tolist()

    def predict_proba(self, df: pd.DataFrame) -> list[list[float]] | None:
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(df)
                return [[round(float(p), 4) for p in row] for row in proba]
            except Exception:
                return None
        return None
