"""Adapter for XGBoost models (Booster and sklearn-API wrappers)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.adapters.base_adapter import BaseAdapter


class XGBoostAdapter(BaseAdapter):
    """Handles both xgb.Booster and sklearn-style XGB wrappers."""

    def predict(self, df: pd.DataFrame) -> list[Any]:
        try:
            import xgboost as xgb

            if isinstance(self.model, xgb.Booster):
                dmatrix = xgb.DMatrix(df)
                return self.model.predict(dmatrix).tolist()
        except ImportError:
            pass

        # sklearn-style fallback
        return self.model.predict(df).tolist()

    def predict_proba(self, df: pd.DataFrame) -> list[list[float]] | None:
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(df)
                return [[round(float(p), 4) for p in row] for row in proba]
            except Exception:
                return None
        return None
