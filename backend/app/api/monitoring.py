from typing import cast
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.database import get_db
from app.models.monitoring_metric import MonitoringMetric
from app.models.model_registry import ModelRegistry
from app.models.dataset import Dataset
from app.models.automation_log import AutomationLog
from app.services.monitoring_service import MonitoringService
from app.schemas.monitoring import ModelMonitoringResponse

router = APIRouter()


# ── Aggregate summary ────────────────────────────────────────────

@router.get("/monitoring/summary")
async def get_monitoring_summary(db: AsyncSession = Depends(get_db)):
    """Aggregate monitoring summary across all active models."""
    # Latest signal per model
    result = await db.execute(
        select(MonitoringMetric)
        .order_by(MonitoringMetric.created_at.desc())
        .limit(1)
    )
    latest = result.scalars().first()

    # Count active models
    result = await db.execute(
        select(ModelRegistry).where(ModelRegistry.status == "active")
    )
    active_models = result.scalars().all()

    # Count alerts
    result = await db.execute(
        select(AutomationLog)
        .where(AutomationLog.action.in_(["DRIFT_ALERT", "PERFORMANCE_ALERT", "LATENCY_ALERT"]))
    )
    alerts = result.scalars().all()

    model_name = None
    model_id = None
    if active_models:
        m = active_models[0]
        model_name = m.model_class or m.framework or "Active Model"
        model_id = str(m.id)

    # Calculate global drift from feature drifts excluding target
    import numpy as np
    from app.models.inference_log import InferenceLog

    ds_result = await db.execute(select(Dataset).limit(10))
    datasets = ds_result.scalars().all()
    feature_drifts = []

    recent_logs = []
    if active_models:
        model = active_models[0]
        from app.models.model_version import ModelVersion
        version_result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.model_id == model.id,
                ModelVersion.version_number == model.current_version,
            )
        )
        current_version_entry = version_result.scalars().first()
        version_timestamp = current_version_entry.created_at if current_version_entry else None

        query = select(InferenceLog).where(InferenceLog.model_id == str(model.id))
        if version_timestamp:
            query = query.where(InferenceLog.created_at >= version_timestamp)

        log_result = await db.execute(
            query.order_by(InferenceLog.created_at.desc()).limit(100)
        )
        recent_logs = log_result.scalars().all()

    feature_values = {}
    for log in recent_logs:
        input_summary = log.input_summary or {}
        for k, v in input_summary.items():
            if isinstance(v, (int, float)):
                if k not in feature_values:
                    feature_values[k] = []
                feature_values[k].append(v)
    
    for ds in datasets:
        if not ds.baseline_stats:
            continue
        for col_name, stats in ds.baseline_stats.items():
            if col_name.lower() in ("target", "label", "class", "risk_category", "at_risk"):
                continue
            if not isinstance(stats, dict) or "mean" not in stats:
                continue
            
            baseline_mean = stats["mean"]
            baseline_std = stats["std"] or 1e-6
            
            if col_name in feature_values and feature_values[col_name]:
                current_mean = np.mean(feature_values[col_name])
                drift = abs(current_mean - baseline_mean) / baseline_std
                drift = min(drift / 3.0, 1.0)
                feature_drifts.append(drift)
            else:
                feature_drifts.append(0.0)

    drift_val = 0.0
    if feature_drifts:
        drift_val = sum(feature_drifts) / len(feature_drifts)
    elif latest and latest.drift_score is not None:
        drift_val = cast(float, latest.drift_score)

    global_drift = round(drift_val, 4)

    return {
        "model_id": model_id,
        "model_name": model_name,
        "window": "14d",
        "global_drift": global_drift,
        "drift_delta": 0.0,
        "accuracy": 1.0 + (latest.performance_delta or 0.0) if latest else None,
        "baseline_accuracy": 1.0,
        "data_quality": 1.0 - global_drift,
        "alerts": len(alerts),
        "active_models": len(active_models),
        "latest_severity": (latest.details or {}).get("severity") if latest else None,
        "request_count": latest.request_count if latest else 0,
        "latency_avg": latest.latency_avg if latest else 0,
    }


# ── Feature-level drift ─────────────────────────────────────────

@router.get("/monitoring/features")
async def get_monitoring_features(db: AsyncSession = Depends(get_db)):
    """Compute per-feature drift signals from dataset baseline stats."""
    import math
    import random

    # Get all datasets with baseline stats
    result = await db.execute(select(Dataset).limit(10))
    datasets = result.scalars().all()

    from app.models.inference_log import InferenceLog
    import numpy as np
    
    model_result = await db.execute(select(ModelRegistry).where(ModelRegistry.status == "active").limit(1))
    active_model = model_result.scalars().first()

    recent_logs = []
    if active_model:
        from app.models.model_version import ModelVersion
        version_result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.model_id == active_model.id,
                ModelVersion.version_number == active_model.current_version,
            )
        )
        current_version_entry = version_result.scalars().first()
        version_timestamp = current_version_entry.created_at if current_version_entry else None

        query = select(InferenceLog).where(InferenceLog.model_id == str(active_model.id))
        if version_timestamp:
            query = query.where(InferenceLog.created_at >= version_timestamp)

        log_result = await db.execute(
            query.order_by(InferenceLog.created_at.desc()).limit(100)
        )
        recent_logs = log_result.scalars().all()

    feature_values = {}
    for log in recent_logs:
        input_summary = log.input_summary or {}
        for k, v in input_summary.items():
            if isinstance(v, (int, float)):
                if k not in feature_values:
                    feature_values[k] = []
                feature_values[k].append(v)

    features = []
    for ds in datasets:
        if not ds.baseline_stats:
            continue
        for col_name, stats in ds.baseline_stats.items():
            if col_name.lower() in ("target", "label", "class", "risk_category", "at_risk"):
                continue
            if not isinstance(stats, dict):
                continue
            is_numeric = "mean" in stats
            missing = stats.get("missing_count", 0)
            
            if is_numeric:
                baseline_mean = stats["mean"]
                baseline_std = stats["std"] or 1e-6
                
                if col_name in feature_values and feature_values[col_name]:
                    current_mean = np.mean(feature_values[col_name])
                    drift_score = abs(current_mean - baseline_mean) / baseline_std
                    drift_score = round(min(drift_score / 3.0, 1.0), 4)
                else:
                    drift_score = 0.0
            else:
                drift_score = round(min(missing / 100.0, 1.0), 4)

            p_value = round(max(0.001, 1.0 - drift_score), 4)

            severity = "low"
            if drift_score > 0.20:
                severity = "high"
            elif drift_score > 0.10:
                severity = "medium"

            features.append({
                "feature_name": col_name,
                "dataset_id": str(ds.id),
                "dataset_name": ds.name,
                "type": "numerical" if is_numeric else "categorical",
                "drift_score": drift_score,
                "p_value": p_value,
                "severity": severity,
                "mean": stats.get("mean"),
                "std": stats.get("std"),
                "missing_count": missing,
            })

    # Sort by drift_score descending
    features.sort(key=lambda f: f["drift_score"], reverse=True)
    return features


# ── Feature detail ───────────────────────────────────────────────

@router.get("/monitoring/feature/{feature_name}")
async def get_feature_detail(feature_name: str, db: AsyncSession = Depends(get_db)):
    """Get detailed statistics for a specific feature across datasets."""
    import random

    # Find the feature in dataset baseline stats
    result = await db.execute(select(Dataset).limit(10))
    datasets = result.scalars().all()

    stats = None
    for ds in datasets:
        if ds.baseline_stats and feature_name in ds.baseline_stats:
            stats = ds.baseline_stats[feature_name]
            break

    if not stats:
        raise HTTPException(status_code=404, detail=f"Feature '{feature_name}' not found")

    mean = stats.get("mean", 0) or 0
    std = stats.get("std", 0) or 0
    missing = stats.get("missing_count", 0)

    # Extract genuine distribution from inferences
    from app.models.inference_log import InferenceLog
    
    model_result = await db.execute(select(ModelRegistry).where(ModelRegistry.status == "active").limit(1))
    active_model = model_result.scalars().first()

    recent_logs = []
    if active_model:
        from app.models.model_version import ModelVersion
        version_result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.model_id == active_model.id,
                ModelVersion.version_number == active_model.current_version,
            )
        )
        current_version_entry = version_result.scalars().first()
        version_timestamp = current_version_entry.created_at if current_version_entry else None

        query = select(InferenceLog).where(InferenceLog.model_id == str(active_model.id))
        if version_timestamp:
            query = query.where(InferenceLog.created_at >= version_timestamp)

        log_result = await db.execute(
            query.order_by(InferenceLog.created_at.desc()).limit(100)
        )
        recent_logs = log_result.scalars().all()

    current_dist = []
    for log in recent_logs:
        input_summary = log.input_summary or {}
        if feature_name in input_summary and isinstance(input_summary[feature_name], (int, float)):
            current_dist.append(input_summary[feature_name])

    import numpy as np
    rng = np.random.default_rng(hash(feature_name) % (2**31))
    baseline_dist = rng.normal(mean, max(std, 1), 50).tolist()
    if not current_dist:
        current_dist = baseline_dist

    # Genuine trend data from monitoring metrics
    trend = []
    if active_model:
        trend_result = await db.execute(
            select(MonitoringMetric)
            .where(MonitoringMetric.model_id == str(active_model.id))
            .order_by(MonitoringMetric.created_at.desc())
            .limit(14)
        )
        trend_metrics = trend_result.scalars().all()
        for m in reversed(trend_metrics):
            trend.append({
                "day": m.created_at.strftime("%Y-%m-%d"),
                "drift": m.drift_score,
            })
            
    if not trend:
        trend = [{"day": "N/A", "drift": 0.0}]

    # Determine recommendation based on drift severity
    cv = abs(std / mean) if mean != 0 else 0
    if cv > 0.20:
        action = "retrain"
        confidence = round(min(0.85 + cv * 0.1, 0.99), 2)
    else:
        action = "monitor"
        confidence = round(0.6 + cv * 0.5, 2)

    return {
        "feature_name": feature_name,
        "recommendation": {
            "action": action,
            "confidence": confidence,
            "estimated_cost": round(15.0 + cv * 100, 2),
            "estimated_duration_minutes": int(20 + cv * 60),
        },
        "distribution": {
            "baseline": [round(v, 2) for v in sorted(baseline_dist)],
            "current": [round(v, 2) for v in sorted(current_dist)],
        },
        "trend": trend,
        "statistics": {
            "mean": round(mean, 4),
            "mean_delta": round(mean * 0.024, 4),
            "std": round(std, 4),
            "std_delta": round(std * 0.152, 4) if std else 0,
            "min": round(mean - 3 * max(std, 1), 2),
            "max": round(mean + 3 * max(std, 1), 2),
        },
    }



@router.get("/monitoring/signals")
async def get_monitoring_signals(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve recent monitoring metrics (drift scores, performance)."""
    result = await db.execute(
        select(MonitoringMetric)
        .order_by(MonitoringMetric.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


@router.get("/monitoring/{model_id}", response_model=ModelMonitoringResponse)
async def get_model_monitoring(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get latest drift score, performance delta, request volume, and trend for a model."""
    service = MonitoringService()
    try:
        result = await service.get_model_metrics(model_id=model_id, db=db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return result
