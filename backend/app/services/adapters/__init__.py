"""Framework adapter package."""

from app.services.adapters.base_adapter import BaseAdapter
from app.services.adapters.sklearn_adapter import SklearnAdapter
from app.services.adapters.xgboost_adapter import XGBoostAdapter
from app.services.adapters.pytorch_adapter import PytorchAdapter


def get_adapter(model, framework: str) -> BaseAdapter:
    """Return the appropriate adapter for the given framework."""
    fw = framework.lower()
    if fw in ("sklearn", "joblib", "lightgbm"):
        return SklearnAdapter(model)
    if fw == "xgboost":
        return XGBoostAdapter(model)
    if fw in ("pytorch", "torch"):
        return PytorchAdapter(model)
    # Default: sklearn-style
    return SklearnAdapter(model)
