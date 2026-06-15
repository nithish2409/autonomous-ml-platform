"""
Monitoring service — analyzes recent predictions against baseline stats
to compute drift scores, performance deltas, system metrics, and alerts.
"""

import uuid
import logging
import math
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_

from app.models.model_registry import ModelRegistry
from app.models.model_version import ModelVersion
from app.models.monitoring_metric import MonitoringMetric
from app.models.inference_log import InferenceLog
from app.models.automation_log import AutomationLog
from app.models.dataset import Dataset
from app.core.metrics import drift_score_gauge
from app.core.monitoring_config import (
    DRIFT_THRESHOLD,
    DRIFT_CRITICAL,
    PERFORMANCE_DROP_THRESHOLD,
    LATENCY_THRESHOLD_MS,
    INFERENCE_WINDOW_MINUTES,
)

logger = logging.getLogger("monitoring_service")


class MonitoringService:
    """Computes drift, performance, system metrics and generates alerts."""

    async def check_all_active_models(self, db: AsyncSession) -> list[dict]:
        """Fetch all active models and run analysis on each."""
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.status == "active")
        )
        active_models = result.scalars().all()

        signals = []
        for model in active_models:
            if not model.current_version:
                continue
            try:
                signal = await self._analyze_model(model, db)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error("Analysis failed for model %s: %s", model.id, e)
                signals.append({
                    "model_id": str(model.id),
                    "error": str(e),
                })
        return signals

    async def _analyze_model(self, model: ModelRegistry, db: AsyncSession) -> dict | None:
        """
        Analyze a single model:
        1. Get baseline stats from the dataset
        2. Get recent inference logs
        3. Compute drift score (PSI + feature coverage)
        4. Compute performance delta
        5. Compute system metrics (request count, latency avg)
        6. Generate alerts if thresholds exceeded
        7. Store signal in monitoring_metrics
        """
        # Get dataset baseline stats
        result = await db.execute(
            select(Dataset).where(Dataset.id == model.dataset_id)
        )
        dataset = result.scalars().first()
        if dataset is None or not dataset.baseline_stats:
            return None

        baseline = dataset.baseline_stats

        from app.models.model_version import ModelVersion
        version_result = await db.execute(
            select(ModelVersion).where(
                and_(
                    ModelVersion.model_id == model.id,
                    ModelVersion.version_number == model.current_version,
                )
            )
        )
        current_version_entry = version_result.scalars().first()
        version_timestamp = current_version_entry.created_at if current_version_entry else None

        # Get recent inference logs (rolling window = 100), strictly AFTER the current version
        query = select(InferenceLog).where(InferenceLog.model_id == str(model.id))
        
        if version_timestamp:
            query = query.where(InferenceLog.created_at >= version_timestamp)
            
        result = await db.execute(
            query.order_by(InferenceLog.created_at.desc()).limit(100)
        )
        recent_logs = result.scalars().all()

        if not recent_logs:
            return None  # No recent predictions — nothing to analyze

        # Compute drift score
        drift_score = self._compute_drift_score(baseline, recent_logs)
        drift_score_gauge.labels(model_id=str(model.id)).set(drift_score)

        # Compute performance delta
        performance_delta = await self._compute_performance_delta(model, db)

        # Compute system metrics
        request_count = len(recent_logs)
        latency_avg = 0.0
        if recent_logs:
            latencies = [log.latency_ms for log in recent_logs if log.latency_ms]
            latency_avg = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

        # Determine severity
        severity = self._compute_severity(drift_score, performance_delta)

        # Store monitoring signal
        signal_id = str(uuid.uuid4())
        metric = MonitoringMetric(
            id=signal_id,
            model_id=str(model.id),
            drift_score=drift_score,
            performance_delta=performance_delta,
            request_count=request_count,
            latency_avg=latency_avg,
            details={
                "type": "monitoring_signal",
                "version": model.current_version,
                "framework": model.framework,
                "model_class": model.model_class,
                "severity": severity,
                "n_recent_inferences": len(recent_logs),
                "drift_score": drift_score,
                "performance_delta": performance_delta,
                "request_count": request_count,
                "latency_avg": latency_avg,
            },
        )
        db.add(metric)

        # Generate alerts if thresholds exceeded
        alerts_generated = await self._check_thresholds_and_alert(
            model_id=str(model.id),
            drift_score=drift_score,
            performance_delta=performance_delta,
            latency_avg=latency_avg,
            severity=severity,
            db=db,
        )

        await db.commit()

        return {
            "signal_id": signal_id,
            "model_id": str(model.id),
            "drift_score": drift_score,
            "performance_delta": performance_delta,
            "request_count": request_count,
            "latency_avg": latency_avg,
            "severity": severity,
            "alerts_generated": alerts_generated,
            "n_recent_inferences": len(recent_logs),
        }

    # ── Per-model monitoring ─────────────────────────────────────

    async def get_model_metrics(self, model_id: str, db: AsyncSession) -> dict:
        """Get latest monitoring metrics and trend for a specific model."""

        # Get model info
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalars().first()
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        # Get latest monitoring signal
        result = await db.execute(
            select(MonitoringMetric).where(
                and_(
                    MonitoringMetric.model_id == model_id,
                    MonitoringMetric.details["type"].as_string() == "monitoring_signal",
                )
            ).order_by(MonitoringMetric.created_at.desc()).limit(1)
        )
        latest = result.scalars().first()

        # Get trend (last 20 signals)
        result = await db.execute(
            select(MonitoringMetric).where(
                and_(
                    MonitoringMetric.model_id == model_id,
                    MonitoringMetric.details["type"].as_string() == "monitoring_signal",
                )
            ).order_by(MonitoringMetric.created_at.desc()).limit(20)
        )
        trend_entries = result.scalars().all()

        trend = [
            {
                "drift_score": e.drift_score,
                "performance_delta": e.performance_delta,
                "request_count": e.request_count,
                "latency_avg": e.latency_avg,
                "severity": (e.details or {}).get("severity"),
                "created_at": str(e.created_at) if e.created_at else None,
            }
            for e in trend_entries
        ]

        # Get recent alerts
        result = await db.execute(
            select(AutomationLog).where(
                and_(
                    AutomationLog.model_id == model_id,
                    AutomationLog.action.in_(["DRIFT_ALERT", "PERFORMANCE_ALERT", "LATENCY_ALERT"]),
                )
            ).order_by(AutomationLog.created_at.desc()).limit(10)
        )
        alert_entries = result.scalars().all()
        alerts = [
            {
                "action": a.action,
                "reason": a.reason,
                "created_at": str(a.created_at) if a.created_at else None,
            }
            for a in alert_entries
        ]

        return {
            "model_id": model_id,
            "version": model.current_version,
            "status": model.status,
            "latest_drift_score": latest.drift_score if latest else None,
            "latest_performance_delta": latest.performance_delta if latest else None,
            "latest_request_count": latest.request_count if latest else None,
            "latest_latency_avg": latest.latency_avg if latest else None,
            "latest_severity": (latest.details or {}).get("severity") if latest else None,
            "trend": trend,
            "alerts": alerts,
        }

    # ── Drift computation ────────────────────────────────────────

    @staticmethod
    def _compute_drift_score(baseline: dict, recent_logs: list) -> float:
        """
        Compute drift score based strictly on normalized mean shift
        of recent inference inputs vs baseline stats.
        """
        import numpy as np

        feature_values = {}
        for log in recent_logs:
            input_summary = log.input_summary or {}
            for k, v in input_summary.items():
                if isinstance(v, (int, float)):
                    if k not in feature_values:
                        feature_values[k] = []
                    feature_values[k].append(v)
                    
        if not feature_values:
            return 0.0

        feature_drifts = []
        for feat, values in feature_values.items():
            if feat.lower() in ("target", "label", "class", "risk_category", "at_risk"):
                continue
                
            stats = baseline.get(feat)
            if not stats or not isinstance(stats, dict) or "mean" not in stats:
                continue
                
            baseline_mean = stats["mean"]
            baseline_std = stats["std"] or 1e-6
            current_mean = np.mean(values)
            
            drift = abs(current_mean - baseline_mean) / baseline_std
            drift = min(drift / 3.0, 1.0)
            feature_drifts.append(drift)
            
        if not feature_drifts:
            return 0.0
            
        global_drift = sum(feature_drifts) / len(feature_drifts)
        return round(global_drift, 4)

    @staticmethod
    async def _compute_performance_delta(model: ModelRegistry, db: AsyncSession) -> float:
        """Compare training score of current version vs previous version."""
        result = await db.execute(
            select(ModelVersion)
            .where(ModelVersion.model_id == model.id)
            .order_by(ModelVersion.created_at.desc())
            .limit(2)
        )
        versions = result.scalars().all()

        if len(versions) < 2:
            return 0.0

        current_metrics = versions[0].metrics or {}
        previous_metrics = versions[1].metrics or {}

        # Try multiple metric keys
        for key in ("accuracy", "f1_score", "train_score"):
            current_val = current_metrics.get(key)
            previous_val = previous_metrics.get(key)
            if current_val is not None and previous_val is not None:
                return round(current_val - previous_val, 4)

        return 0.0

    # ── Severity classification ──────────────────────────────────

    @staticmethod
    def _compute_severity(drift_score: float, performance_delta: float) -> str:
        """Classify severity: low, medium, high, critical."""
        high_drift = drift_score > DRIFT_CRITICAL
        medium_drift = drift_score > DRIFT_THRESHOLD
        degraded = performance_delta < -PERFORMANCE_DROP_THRESHOLD

        if high_drift and degraded:
            return "critical"
        elif high_drift or degraded:
            return "high"
        elif medium_drift:
            return "medium"
        else:
            return "low"

    # ── Alert generation ─────────────────────────────────────────

    async def _check_thresholds_and_alert(
        self,
        model_id: str,
        drift_score: float,
        performance_delta: float,
        latency_avg: float,
        severity: str,
        db: AsyncSession,
    ) -> int:
        """Generate alerts when thresholds are exceeded. Returns count of alerts."""
        alerts = 0

        if drift_score > DRIFT_THRESHOLD:
            level = "CRITICAL" if drift_score > DRIFT_CRITICAL else "WARNING"
            log = AutomationLog(
                id=uuid.uuid4(),
                model_id=model_id,
                action="DRIFT_ALERT",
                reason=f"[{level}] Drift score {drift_score:.4f} exceeds threshold {DRIFT_THRESHOLD}",
                log_metadata={"drift_score": drift_score, "threshold": DRIFT_THRESHOLD, "severity": severity},
                status="alert",
            )
            db.add(log)
            alerts += 1
            logger.warning("DRIFT_ALERT for model %s: score=%.4f", model_id, drift_score)

        if performance_delta < -PERFORMANCE_DROP_THRESHOLD:
            log = AutomationLog(
                id=uuid.uuid4(),
                model_id=model_id,
                action="PERFORMANCE_ALERT",
                reason=f"Performance drop {performance_delta:.4f} exceeds threshold {PERFORMANCE_DROP_THRESHOLD}",
                log_metadata={"performance_delta": performance_delta, "threshold": PERFORMANCE_DROP_THRESHOLD},
                status="alert",
            )
            db.add(log)
            alerts += 1
            logger.warning("PERFORMANCE_ALERT for model %s: delta=%.4f", model_id, performance_delta)

        if latency_avg > LATENCY_THRESHOLD_MS:
            log = AutomationLog(
                id=uuid.uuid4(),
                model_id=model_id,
                action="LATENCY_ALERT",
                reason=f"Average latency {latency_avg:.1f}ms exceeds threshold {LATENCY_THRESHOLD_MS}ms",
                log_metadata={"latency_avg": latency_avg, "threshold": LATENCY_THRESHOLD_MS},
                status="alert",
            )
            db.add(log)
            alerts += 1
            logger.warning("LATENCY_ALERT for model %s: avg=%.1fms", model_id, latency_avg)

        return alerts
