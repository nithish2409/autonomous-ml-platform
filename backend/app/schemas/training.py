"""Pydantic request / response schemas for training orchestration."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class StartTrainingRequest(BaseModel):
    """Launch a training job with full configuration."""

    model_id: str = Field(..., description="ID of registered model to train")
    target_column: Optional[str] = Field(None, description="Target column name (auto-detected if omitted)")
    hyperparameters: Optional[dict[str, Any]] = Field(None, description="Model hyperparameters")
    split_ratio: float = Field(0.2, ge=0.05, le=0.5, description="Test split ratio")
    random_seed: int = Field(42, description="Random seed for reproducibility")


class TrainingJobResponse(BaseModel):
    """Response after starting or querying a training job."""

    job_id: str
    model_id: str
    version: Optional[str] = None
    status: str
    framework: Optional[str] = None
    model_class: Optional[str] = None
    artifact_path: Optional[str] = None
    metrics: Optional[dict[str, Any]] = None
    config: Optional[dict[str, Any]] = None
    created_at: Optional[str] = None
    message: Optional[str] = None


class TrainingLogsResponse(BaseModel):
    """Container logs for a training job."""

    job_id: str
    logs: list[str] = []
