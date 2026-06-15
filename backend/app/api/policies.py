"""Policies API — get, update, and simulate policy evaluation."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Any, cast, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.database import get_db
from app.models.policy import Policy, DEFAULT_POLICY
from app.models.automation_log import AutomationLog
from app.services.policy_evaluator import evaluate_decision

router = APIRouter()


# ── Pydantic schemas ─────────────────────────────────────────

class AutoApprovalConfig(BaseModel):
    min_confidence: int = Field(ge=0, le=100)
    max_cost: float = Field(gt=0)
    allowed_severity: list[str]
    block_production: bool

class GuardrailsConfig(BaseModel):
    max_gpu_per_job: int = Field(gt=0)
    max_daily_cost: float = Field(gt=0)
    max_retrains_24h: int = Field(gt=0)
    freeze_window: bool

class EscalationConfig(BaseModel):
    notify_on_critical: bool
    webhook_url: Optional[str] = None
    email_alerts: list[str] = []

class PolicyUpdate(BaseModel):
    auto_approval: AutoApprovalConfig
    guardrails: GuardrailsConfig
    escalation: EscalationConfig


# ── Helpers ──────────────────────────────────────────────────

async def get_or_create_policy(db: AsyncSession) -> Policy:
    """Get the singleton policy row, creating it with defaults if absent."""
    result = await db.execute(select(Policy).where(Policy.id == 1))
    policy = result.scalar_one_or_none()
    if not policy:
        policy = Policy(id=1, config=DEFAULT_POLICY)
        db.add(policy)
        await db.commit()
        await db.refresh(policy)
    return policy


# ── GET current policies ─────────────────────────────────────

@router.get("/policies")
async def get_policies(db: AsyncSession = Depends(get_db)):
    """Return the current policy configuration."""
    policy = await get_or_create_policy(db)
    return policy.config


# ── PUT update policies ──────────────────────────────────────

@router.put("/policies")
async def update_policies(
    body: PolicyUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update the policy configuration with validation."""
    policy = await get_or_create_policy(db)
    current_config = policy.config or {}
    new_config = body.model_dump()
    if "autonomous_enabled" in current_config:
        new_config["autonomous_enabled"] = current_config["autonomous_enabled"]
    else:
        new_config["autonomous_enabled"] = True
    setattr(policy, "config", new_config)
    await db.commit()
    await db.refresh(policy)
    return policy.config


# ── POST simulate policy against a decision ──────────────────

@router.post("/policies/simulate/{decision_id}")
async def simulate_policy(
    decision_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Simulate the current policy against an existing decision (read-only)."""
    # Fetch decision
    result = await db.execute(
        select(AutomationLog).where(AutomationLog.id == decision_id)
    )
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")

    # Fetch policy
    policy = await get_or_create_policy(db)

    # Build decision dict for evaluator
    meta = log.log_metadata or {}
    decision_dict = {
        "confidence": meta.get("confidence", 0),
        "severity": meta.get("severity", "low"),
        "estimated_cost": meta.get("estimated_cost", 0),
        "gpu_count": meta.get("gpu_count", 0),
        "is_production": meta.get("is_production", False),
        "daily_cost_so_far": meta.get("daily_cost_so_far", 0),
        "retrains_today": meta.get("retrains_today", 0),
    }

    sim_result = evaluate_decision(decision_dict, cast(dict[str, Any], policy.config))
    return {
        "decision_id": str(log.id),
        "model_name": meta.get("model_name", log.action),
        "simulation": sim_result,
    }
