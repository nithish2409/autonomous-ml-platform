"""Automation API router — logs, state, decisions, and control endpoints."""

from typing import Any, cast
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_, func as sa_func

from app.core.database import get_db
from app.models.automation_log import AutomationLog
from app.models.automation_state import AutomationState
from app.models.model_registry import ModelRegistry
from app.models.policy import Policy, DEFAULT_POLICY
from app.services.policy_evaluator import evaluate_decision as eval_policy
from app.services.automation_executor import AutomationExecutor


router = APIRouter()

# Toggle state is persisted in Policy configuration database


class ToggleRequest(BaseModel):
    enabled: bool


# ── Aggregate status ─────────────────────────────────────────────

@router.get("/automation/status")
async def get_automation_status(db: AsyncSession = Depends(get_db)):
    """Aggregate automation status: counts and average confidence."""
    # Total decisions
    result = await db.execute(select(sa_func.count(AutomationLog.id)))
    total = result.scalar() or 0

    # Retraining executed
    result = await db.execute(
        select(sa_func.count(AutomationLog.id)).where(
            and_(
                AutomationLog.action.in_(["retrain", "RETRAIN"]),
                AutomationLog.status.in_(["executed", "completed", "success"]),
            )
        )
    )
    retraining = result.scalar() or 0

    # Pending approvals
    result = await db.execute(
        select(sa_func.count(AutomationLog.id)).where(
            AutomationLog.status.in_(["pending", "alert", "scheduled", "pending_human_review"])
        )
    )
    pending = result.scalar() or 0

    # Average confidence from metadata
    result = await db.execute(
        select(AutomationLog).order_by(AutomationLog.created_at.desc()).limit(50)
    )
    logs = result.scalars().all()
    confidences = []
    for log in logs:
        meta = log.log_metadata or {}
        conf = meta.get("confidence") or meta.get("avg_confidence")
        if conf is not None:
            confidences.append(float(conf))
    avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.94

    from app.api.policies import get_or_create_policy
    policy = await get_or_create_policy(db)
    autonomous_enabled = policy.config.get("autonomous_enabled", True)

    return {
        "total_decisions": total,
        "retraining_executed": retraining,
        "pending_approvals": pending,
        "avg_confidence": avg_confidence,
        "autonomous_enabled": autonomous_enabled,
    }


# ── Decision history ─────────────────────────────────────────────

@router.get("/automation/history")
async def get_decision_history(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """List automation decisions with model info joined."""
    result = await db.execute(
        select(AutomationLog)
        .order_by(AutomationLog.created_at.desc())
        .limit(limit)
    )
    logs = result.scalars().all()

    items = []
    for log in logs:
        # Get model name
        model_name = None
        model_class = None
        if log.model_id:
            m_result = await db.execute(
                select(ModelRegistry).where(ModelRegistry.id == log.model_id)
            )
            model = m_result.scalar_one_or_none()
            if model:
                model_name = model.model_class or model.framework or "Unnamed"
                model_class = model.model_class

        meta = log.log_metadata or {}
        items.append({
            "id": str(log.id),
            "model_id": str(log.model_id) if log.model_id else None,
            "model_name": model_name or "Unknown",
            "model_class": model_class,
            "action": log.action,
            "drift_type": meta.get("drift_type", meta.get("type", "—")),
            "severity": meta.get("severity", "low"),
            "strategy": meta.get("strategy", log.action),
            "resource_profile": meta.get("resource_profile", "—"),
            "confidence": meta.get("confidence", 0),
            "status": log.status or "pending",
            "reason": log.reason,
            "created_at": str(log.created_at) if log.created_at else None,
        })

    return items


# ── Decision detail ──────────────────────────────────────────────

@router.get("/automation/decision/{decision_id}")
async def get_decision_detail(
    decision_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get detailed info for a single automation decision."""
    result = await db.execute(
        select(AutomationLog).where(AutomationLog.id == decision_id)
    )
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")

    # Model info
    model_name = None
    if log.model_id:
        m_result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == log.model_id)
        )
        model = m_result.scalar_one_or_none()
        if model:
            model_name = model.model_class or model.framework or "Unnamed"

    meta = log.log_metadata or {}
    exec_result = log.execution_result or {}

    # ── Run policy evaluation against this decision ──
    policy_row = await db.execute(select(Policy).where(Policy.id == 1))
    policy_obj = policy_row.scalar_one_or_none()
    policy_config = policy_obj.config if policy_obj else DEFAULT_POLICY
    decision_for_policy = {
        "confidence": meta.get("confidence", 0),
        "severity": meta.get("severity", "low"),
        "estimated_cost": meta.get("estimated_cost", 0),
        "gpu_count": meta.get("gpu_count", 0),
        "is_production": meta.get("is_production", False),
        "daily_cost_so_far": meta.get("daily_cost_so_far", 0),
        "retrains_today": meta.get("retrains_today", 0),
    }
    policy_config_dict = cast(dict[str, Any], policy_config)
    policy_check = eval_policy(decision_for_policy, policy_config_dict)

    return {
        "id": str(log.id),
        "model_id": str(log.model_id) if log.model_id else None,
        "model_name": model_name or "Unknown",
        "action": log.action,
        "reason": log.reason,
        "status": log.status or "pending",
        "created_at": str(log.created_at) if log.created_at else None,
        "drift_context": {
            "drift_score": meta.get("drift_score", 0),
            "severity": meta.get("severity", "low"),
            "drift_type": meta.get("drift_type", meta.get("type", "—")),
            "training_mean": meta.get("training_mean"),
            "current_mean": meta.get("current_mean"),
            "affected_features": meta.get("affected_features", []),
        },
        "llm_output": {
            "raw_reasoning": meta.get("llm_reasoning", log.reason or ""),
            "strategy": meta.get("strategy", log.action),
            "resource_profile": meta.get("resource_profile", "—"),
            "early_stopping": meta.get("early_stopping", False),
            "ensemble": meta.get("ensemble", False),
        },
        "impact": {
            "estimated_cost": meta.get("estimated_cost", exec_result.get("cost")),
            "estimated_downtime": meta.get("estimated_downtime", "—"),
            "expected_improvement": meta.get("expected_improvement"),
        },
        "confidence": meta.get("confidence", 0),
        "auto_approval_seconds": 300 if log.status in ("pending", "alert", "scheduled") else None,
        "execution_result": exec_result,
        "policy_check": policy_check,
    }



# ── Toggle autonomous mode ──────────────────────────────────────

@router.post("/automation/toggle")
async def toggle_autonomous_mode(
    body: ToggleRequest,
    db: AsyncSession = Depends(get_db),
):
    """Enable or disable autonomous mode."""
    from app.api.policies import get_or_create_policy
    policy = await get_or_create_policy(db)
    config = dict(cast(dict[str, Any], policy.config))
    config["autonomous_enabled"] = body.enabled
    setattr(policy, "config", config)
    await db.commit()
    return {"autonomous_enabled": body.enabled}


# ── Approve decision ─────────────────────────────────────────────

@router.post("/automation/approve/{decision_id}")
async def approve_decision(
    decision_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Approve a pending automation decision."""
    result = await db.execute(
        select(AutomationLog).where(AutomationLog.id == decision_id)
    )
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")

    if log.action.upper() != "RETRAIN":
        raise HTTPException(
            status_code=400,
            detail="Only RETRAIN decisions can be manually approved"
        )

    log.status = "approved"
    meta = log.log_metadata or {}
    meta["approved_at"] = str(sa_func.now())
    log.log_metadata = meta
    await db.commit()

    decision_dict = {
        "action": log.action,
        "reason": log.reason,
        "monitoring_metrics": meta
    }

    executor = AutomationExecutor()
    await executor.execute_decision(str(log.model_id), decision_dict, db)

    return {"id": str(log.id), "status": "executed", "message": "Decision approved and executed"}


# ── Reject decision ──────────────────────────────────────────────

@router.post("/automation/reject/{decision_id}")
async def reject_decision(
    decision_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Reject a pending automation decision."""
    result = await db.execute(
        select(AutomationLog).where(AutomationLog.id == decision_id)
    )
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")

    log.status = "rejected"
    await db.commit()
    return {"id": str(log.id), "status": "rejected", "message": "Decision rejected"}


# ── Manual train trigger ─────────────────────────────────────────

@router.post("/automation/manual-train/{decision_id}")
async def trigger_manual_train(
    decision_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Trigger manual training for the model referenced in a decision."""
    result = await db.execute(
        select(AutomationLog).where(AutomationLog.id == decision_id)
    )
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")

    log.status = "manual_training"
    meta = log.log_metadata or {}
    meta["manual_train_triggered"] = True
    log.log_metadata = meta
    await db.commit()
    return {"id": str(log.id), "status": "manual_training", "message": "Manual training triggered"}




@router.get("/automation/logs")
async def get_automation_logs(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve recent automation execution logs."""
    result = await db.execute(
        select(AutomationLog)
        .order_by(AutomationLog.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


@router.get("/automation/state")
async def get_automation_state(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve current automation state (cooldowns, last actions)."""
    result = await db.execute(
        select(AutomationState)
        .order_by(AutomationState.updated_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


@router.get("/automation/{model_id}/latest")
async def get_latest_decision(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get the latest automation decision for a specific model.

    Returns the most recent decision log including action, reason,
    confidence, priority, and policy outcome.
    """
    # Latest decision log
    result = await db.execute(
        select(AutomationLog)
        .where(AutomationLog.model_id == model_id)
        .order_by(AutomationLog.created_at.desc())
        .limit(1)
    )
    latest_log = result.scalars().first()

    if latest_log is None:
        raise HTTPException(status_code=404, detail=f"No automation decisions found for model {model_id}")

    # Current state
    result = await db.execute(
        select(AutomationState).where(AutomationState.model_id == model_id)
    )
    state = result.scalars().first()

    return {
        "id": str(latest_log.id),
        "model_id": model_id,
        "action": latest_log.action,
        "reason": latest_log.reason,
        "status": latest_log.status,
        "metadata": latest_log.log_metadata,
        "created_at": str(latest_log.created_at) if latest_log.created_at else None,
        "state": {
            "last_action": state.last_action if state else None,
            "cooldown_until": str(state.cooldown_until) if state and state.cooldown_until else None,
        },
    }
