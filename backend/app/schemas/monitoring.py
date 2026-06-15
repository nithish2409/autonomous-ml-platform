"""Pydantic schemas for monitoring API responses."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel


class MonitoringSignalResponse(BaseModel):
    """A single monitoring measurement."""

    signal_id: str
    model_id: str
    drift_score: float
    performance_delta: float
    request_count: Optional[int] = None
    latency_avg: Optional[float] = None
    severity: Optional[str] = None
    created_at: Optional[str] = None


class ModelMonitoringResponse(BaseModel):
    """Per-model monitoring summary with latest metrics and trend."""

    model_id: str
    version: Optional[str] = None
    status: Optional[str] = None
    latest_drift_score: Optional[float] = None
    latest_performance_delta: Optional[float] = None
    latest_request_count: Optional[int] = None
    latest_latency_avg: Optional[float] = None
    latest_severity: Optional[str] = None
    trend: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
