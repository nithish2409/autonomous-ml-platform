"""Pydantic request / response schemas for model lifecycle management."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Requests ─────────────────────────────────────────────────────

class RegisterModelRequest(BaseModel):
    """Register a new model for a dataset."""

    dataset_id: str
    framework: str = Field(..., description="sklearn, xgboost, pytorch, etc.")
    model_class: str = Field(..., description="e.g. RandomForestClassifier")


class AddVersionRequest(BaseModel):
    """Add a new version to an existing model."""

    version_number: str = Field(..., description="e.g. v2, v3")
    artifact_path: str = Field(..., description="MinIO object key for model artifact")
    metrics: Optional[dict[str, Any]] = Field(
        None,
        description="Standardised metrics: accuracy, precision, recall, f1_score, roc_auc, training_time",
    )
    training_job_id: Optional[str] = None
    parent_version: Optional[str] = None


class PromoteRequest(BaseModel):
    """Promote a specific version to production."""

    version: str = Field(..., description="Version number to promote, e.g. v2")


class RollbackRequest(BaseModel):
    """Rollback to a specific previous version."""

    target_version: str = Field(..., description="Version number to rollback to")


# ── Responses ────────────────────────────────────────────────────

class RegisterModelResponse(BaseModel):
    model_id: str
    dataset_id: str
    framework: str
    model_class: str
    status: str
    message: str


class VersionResponse(BaseModel):
    version_id: str
    model_id: str
    version_number: str
    artifact_path: str
    metrics: Optional[dict[str, Any]] = None
    hyperparameters: Optional[dict[str, Any]] = None
    training_job_id: Optional[str] = None
    parent_version: Optional[str] = None
    created_at: Optional[str] = None


class PromoteResponse(BaseModel):
    model_id: str
    promoted_version: str
    previous_version: Optional[str] = None
    status: str
    message: str


class RollbackResponse(BaseModel):
    model_id: str
    rolled_back_to: str
    previous_version: str
    status: str
    message: str


class ModelDetailResponse(BaseModel):
    model_id: str
    dataset_id: str
    framework: str
    model_class: str
    current_version: Optional[str] = None
    status: str
    created_at: Optional[str] = None
    versions: list[VersionResponse] = []


class ModelListResponse(BaseModel):
    models: list[ModelDetailResponse]
    total: int
