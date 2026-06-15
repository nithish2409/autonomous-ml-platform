"""Pydantic request / response schemas for the model serving endpoints."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Requests ─────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Batch-capable prediction request.

    ``input_data`` is a list of dicts, each dict representing one sample.
    A single-sample request is simply a list with one element.
    """

    input_data: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of input samples (each a dict of feature→value)",
    )


class BatchCSVRequest(BaseModel):
    """Batch CSV prediction request."""

    csv_content: str = Field(
        ...,
        description="Raw CSV string (with header row) containing input samples",
    )


# ── Responses ────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    model_id: str
    version: str
    framework: str
    predictions: list[Any]
    confidence: Optional[list[float]] = None
    latency_ms: float


class BatchPredictResponse(BaseModel):
    model_id: str
    version: str
    framework: str
    n_samples: int
    predictions: list[Any]
    latency_ms: float


class ModelStatusResponse(BaseModel):
    model_id: str
    loaded: bool
    version: Optional[str] = None
    framework: Optional[str] = None
    cached: bool = False


class ActiveModelResponse(BaseModel):
    model_id: str
    dataset_id: str
    framework: str
    model_class: str
    version: Optional[str] = None
    status: str


class InputSchemaResponse(BaseModel):
    model_id: str
    version: Optional[str] = None
    framework: Optional[str] = None
    model_class: Optional[str] = None
    feature_names: Optional[list[str]] = None
    n_features: Optional[int] = None
    target_column: Optional[str] = None
    task_type: Optional[str] = None
