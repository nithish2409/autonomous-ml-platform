"""
Signal Service — converts raw monitoring metrics into structured signals
with status classification for the LLM decision engine.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_

from app.models.model_registry import ModelRegistry
from app.models.monitoring_metric import MonitoringMetric
from app.models.inference_log import InferenceLog
from app.core.monitoring_config import (
    DRIFT_THRESHOLD,
    PERFORMANCE_DROP_THRESHOLD,
    INFERENCE_WINDOW_MINUTES,
)

logger = logging.getLogger("signal_service")


class SignalService:
    """Builds structured signals from monitoring metrics."""

    async def build_signal(self, model_id: str, db: AsyncSession) -> dict | None:
        """
        Build a structured signal for the given model from the latest
        monitoring metrics and recent inference logs.

        Returns None if no metrics are available.
        """
        # Get model info
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalars().first()
        if model is None:
            return None

        # Get latest monitoring signal
        result = await db.execute(
            select(MonitoringMetric).where(
                and_(
                    MonitoringMetric.model_id == model_id,
                    MonitoringMetric.details["type"].as_string() == "monitoring_signal",
                )
            ).order_by(MonitoringMetric.created_at.desc()).limit(1)
        )
        latest_metric = result.scalars().first()

        # Get recent inference count
        window = datetime.now(timezone.utc) - timedelta(minutes=INFERENCE_WINDOW_MINUTES)
        result = await db.execute(
            select(InferenceLog).where(
                and_(
                    InferenceLog.model_id == model_id,
                    InferenceLog.created_at >= window,
                )
            )
        )
        recent_logs = result.scalars().all()

        # Extract values
        drift_score = latest_metric.drift_score if latest_metric else 0.0
        performance_delta = latest_metric.performance_delta if latest_metric else 0.0
        request_count = latest_metric.request_count if latest_metric else len(recent_logs)
        latency_avg = latest_metric.latency_avg if latest_metric else 0.0

        # Classify status
        status = self._classify_status(drift_score, performance_delta)

        signal = {
            "model_id": str(model_id),
            "model_name": model.model_class or "unknown",
            "framework": model.framework,
            "version": model.current_version,
            "drift_score": round(drift_score, 4),
            "performance_delta": round(performance_delta, 4),
            "request_count": request_count,
            "latency_avg": round(latency_avg, 2) if latency_avg else 0.0,
            "status": status,
            "n_recent_inferences": len(recent_logs),
            "severity": (latest_metric.details or {}).get("severity", "unknown") if latest_metric else "unknown",
        }

        logger.info(
            "Signal built for model %s: status=%s, drift=%.4f, delta=%.4f",
            model_id, status, drift_score, performance_delta,
        )
        return signal

    async def build_all_signals(self, db: AsyncSession) -> list[dict]:
        """Build signals for all active models."""
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.status == "active")
        )
        active_models = result.scalars().all()

        signals = []
        for model in active_models:
            if not model.current_version:
                continue
            try:
                signal = await self.build_signal(str(model.id), db)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error("Signal build failed for model %s: %s", model.id, e)
                signals.append({"model_id": str(model.id), "error": str(e)})

        return signals

    @staticmethod
    def _classify_status(drift_score: float, performance_delta: float) -> str:
        """
        Classify model status:
        - healthy: low drift, no performance degradation
        - warning: moderate drift or slight degradation
        - degrading: high drift or significant degradation
        """
        high_drift = drift_score > DRIFT_THRESHOLD * 2  # > 0.4
        moderate_drift = drift_score > DRIFT_THRESHOLD   # > 0.2
        degraded = performance_delta < -PERFORMANCE_DROP_THRESHOLD

        if high_drift or degraded:
            return "degrading"
        elif moderate_drift:
            return "warning"
        else:
            return "healthy"
