"""Adapter for PyTorch models."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.adapters.base_adapter import BaseAdapter


class PytorchAdapter(BaseAdapter):
    """Handles PyTorch nn.Module models."""

    def predict(self, df: pd.DataFrame) -> list[Any]:
        import torch

        tensor = torch.tensor(df.values, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(tensor)

        if output.dim() > 1 and output.shape[1] > 1:
            # Multi-class: return argmax
            return output.argmax(dim=1).cpu().numpy().tolist()
        return output.squeeze().cpu().numpy().tolist()

    def predict_proba(self, df: pd.DataFrame) -> list[list[float]] | None:
        try:
            import torch
            import torch.nn.functional as F

            tensor = torch.tensor(df.values, dtype=torch.float32)
            with torch.no_grad():
                output = self.model(tensor)
                proba = F.softmax(output, dim=1)
            return [[round(float(p), 4) for p in row] for row in proba]
        except Exception:
            return None
