"""
Background scheduler — runs monitoring checks every 5 minutes
using APScheduler's BackgroundScheduler.

Uses a dedicated SQLAlchemy engine per cycle so asyncpg connections
are bound to the background thread's event loop, not the main FastAPI loop.
"""

import asyncio
import logging
import os
from typing import Any, cast

from apscheduler.schedulers.background import BackgroundScheduler

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.services.monitoring_service import MonitoringService
from app.services.llm_decision_engine import LLMDecisionEngine
from app.services.automation_executor import AutomationExecutor

# Define logger and scheduler at module level so they are
# available inside _async_monitoring_cycle and _run_monitoring_cycle.
logger = logging.getLogger("monitoring_scheduler")
scheduler = BackgroundScheduler()


def _make_session_factory():
    """Create a fresh async engine + session factory for the current event loop."""
    db_url = os.getenv("DATABASE_URL")
    engine = create_async_engine(db_url, echo=False, future=True)
    factory = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    return engine, factory


async def _async_monitoring_cycle():
    """Run the monitoring service, LLM decision engine, policy evaluator, and executor."""
    from app.models.model_registry import ModelRegistry
    from app.models.training_job import TrainingJob
    from app.models.policy import Policy, DEFAULT_POLICY
    from app.services.policy_evaluator import evaluate_decision as eval_policy
    from app.core.metrics import policy_decisions_total, policy_violations_total
    from app.core.monitoring_config import DRIFT_THRESHOLD
    from sqlalchemy.future import select
    from sqlalchemy import func as sa_func
    from datetime import datetime, timezone

    # Create a dedicated engine for this background thread's event loop
    engine = None
    try:
        engine, SessionFactory = _make_session_factory()
        async with SessionFactory() as db:
            # 1. Run dynamic drift calculation checks
            service = MonitoringService()
            model_signals = await service.check_all_active_models(db)

            if not model_signals:
                logger.info("No active models or recent inferences to analyze.")
                return

            signal_info = model_signals[0]
            drift_score = signal_info["drift_score"]
            model_id = signal_info["model_id"]

            logger.info(f"[Automation] True Drift mathematically computed: {drift_score}")

            # Determine drift type and severity using the shared config threshold
            if drift_score > DRIFT_THRESHOLD:
                drift_type = "feature_drift"
                severity = "high"
            else:
                drift_type = "no_drift"
                severity = "low"

            # 2. Get the active model so we can construct the signal
            result = await db.execute(select(ModelRegistry).where(ModelRegistry.id == model_id).limit(1))
            active_model = result.scalars().first()
            if not active_model:
                logger.warning("No active models found to evaluate.")
                return

            # 3. Construct the signal expected by the LLM Decision Engine
            signal = {
                "model_id": str(active_model.id),
                "model_name": active_model.model_class or active_model.framework or "Active Model",
                "framework": active_model.framework,
                "version": active_model.current_version,
                "drift_score": drift_score,
                "performance_delta": signal_info.get("performance_delta", 0.0),
                "request_count": signal_info.get("request_count", 0),
                "latency_avg": signal_info.get("latency_avg", 0.0),
                "status": "drift_detected",
                "severity": severity,
                "drift_type": drift_type,
                "n_recent_inferences": signal_info.get("n_recent_inferences", 0)
            }

            # 4. Invoke LLM decision engine
            llm_engine = LLMDecisionEngine()
            executor = AutomationExecutor()

            logger.info(
                f"  Model {signal['model_id']}: "
                f"drift={signal['drift_score']}, "
                f"delta={signal['performance_delta']}, "
                f"severity={signal['severity']}"
            )

            try:
                decision = await llm_engine.evaluate_signal(signal, db)
                logger.info(
                    f"  → LLM decision: action={decision['action']}, "
                    f"status={decision['status']}, "
                    f"reason={decision['reason']}"
                )

                # ── Only proceed if the LLM layer approved the action ──
                if decision["status"] != "approved":
                    logger.info(f"  → LLM policy blocked: {decision.get('policy_reason')}")
                    return

                # ── Full Policy & Guardrails evaluation ──────────────────
                # Fetch current policy config (fall back to defaults if not set)
                policy_row = await db.execute(select(Policy).where(Policy.id == 1))
                policy_obj = policy_row.scalar_one_or_none()
                policy_config = policy_obj.config if policy_obj else DEFAULT_POLICY

                # Count how many retraining jobs have run today (for rate limit)
                today_start = datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                retrains_result = await db.execute(
                    select(sa_func.count(TrainingJob.id)).where(
                        TrainingJob.created_at >= today_start,
                        TrainingJob.status.in_(["completed", "running"]),
                    )
                )
                retrains_today = retrains_result.scalar() or 0

                # Determine whether the active model is in production
                is_production = active_model.status in ("active", "production")

                # Build the decision context for the policy evaluator
                decision_context = {
                    "confidence":        decision.get("confidence", 0.0),
                    "severity":          severity,
                    "estimated_cost":    0.0,   # in-process training, no cloud cost
                    "gpu_count":         0,      # in-process training, no GPU allocation
                    "is_production":     is_production,
                    "daily_cost_so_far": 0.0,
                    "retrains_today":    retrains_today,
                }

                policy_result = eval_policy(decision_context, cast(dict[str, Any], policy_config))

                # ── Emit Prometheus policy metrics ────────────────────────
                if policy_result["blocked"]:
                    outcome = "blocked"
                elif policy_result["requires_human"]:
                    outcome = "requires_human"
                else:
                    outcome = "approved"

                policy_decisions_total.labels(outcome=outcome).inc()

                # Identify which guardrail rule fired for the violation counter
                if not policy_result["approved"]:
                    reason_lower = policy_result["reason"].lower()
                    if "freeze" in reason_lower:
                        policy_violations_total.labels(rule="freeze_window").inc()
                    elif "gpu" in reason_lower:
                        policy_violations_total.labels(rule="gpu_limit").inc()
                    elif "daily cost" in reason_lower:
                        policy_violations_total.labels(rule="daily_cost_cap").inc()
                    elif "retrain count" in reason_lower:
                        policy_violations_total.labels(rule="retrain_rate_limit").inc()
                    elif "severity" in reason_lower:
                        policy_violations_total.labels(rule="severity_filter").inc()
                    elif "production" in reason_lower:
                        policy_violations_total.labels(rule="production_block").inc()
                    elif "confidence" in reason_lower or "cost" in reason_lower:
                        policy_violations_total.labels(rule="confidence_threshold").inc()

                logger.info(
                    f"  → Policy evaluation: outcome={outcome} — {policy_result['reason']}"
                )

                # ── Hard block: stop execution entirely ───────────────────
                if policy_result["blocked"]:
                    logger.warning(
                        f"  → POLICY BLOCKED execution for model {model_id}: "
                        f"{policy_result['reason']}"
                    )
                    return

                # ── Requires human review: mark log as pending ────────────
                if policy_result["requires_human"]:
                    logger.info(
                        f"  → POLICY ESCALATED to human review for model {model_id}: "
                        f"{policy_result['reason']}"
                    )
                    # Update the latest automation_log to pending_human_review so
                    # the Automation UI shows it for manual approval.
                    from app.models.automation_log import AutomationLog
                    log_result = await db.execute(
                        select(AutomationLog)
                        .where(AutomationLog.model_id == active_model.id)
                        .order_by(AutomationLog.created_at.desc())
                        .limit(1)
                    )
                    latest_log = log_result.scalars().first()
                    if latest_log:
                        latest_log.status = "pending_human_review"
                        meta = latest_log.log_metadata or {}
                        meta["policy_reason"] = policy_result["reason"]
                        meta["retrains_today"] = retrains_today
                        meta["is_production"] = is_production
                        latest_log.log_metadata = meta
                        await db.commit()
                    return

                # ── Fully approved: execute the action ────────────────────
                logger.info(f"  → Policy APPROVED — executing action: {decision['action']}")
                if not policy_config.get("autonomous_enabled", True):
                    logger.info(
                        f"  → Autonomous execution is disabled globally. "
                        f"Skipping auto-execution of action: {decision['action']}"
                    )
                    return

                async with SessionFactory() as exec_db:
                    decision["monitoring_metrics"] = signal
                    exec_result = await executor.execute_decision(
                        signal["model_id"], decision, exec_db,
                    )
                logger.info(
                    f"  → Execution result: {exec_result.get('status')} — "
                    f"{exec_result.get('result') or exec_result.get('reason')}"
                )

            except Exception as e:
                logger.error(f"  → Automation cycle failed for model {signal['model_id']}: {e}")
    finally:
        if engine is not None:
            await engine.dispose()



def _run_monitoring_cycle():
    """Synchronous wrapper that creates an event loop for the async monitoring."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_async_monitoring_cycle())
    except Exception as e:
        logger.error(f"Monitoring cycle failed: {e}")
    finally:
        try:
            loop.close()
        except Exception as e:
            logger.error(f"Error closing loop: {e}")


def start_scheduler():
    """Start the background monitoring scheduler (5-minute interval)."""
    scheduler.add_job(
        _run_monitoring_cycle,
        trigger="interval",
        minutes=5,
        id="monitoring_cycle",
        name="Model Monitoring Cycle",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Monitoring scheduler started (interval: 5 minutes)")


def stop_scheduler():
    """Shut down the scheduler gracefully."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Monitoring scheduler stopped")


_cycle_running = False
_cycle_requested = False
_cycle_lock = None

async def trigger_monitoring_loop():
    """Trigger the monitoring cycle safely in a coalesced background loop."""
    global _cycle_running, _cycle_requested, _cycle_lock
    if _cycle_lock is None:
        _cycle_lock = asyncio.Lock()

    async with _cycle_lock:
        if _cycle_running:
            _cycle_requested = True
            logger.info("Monitoring cycle already running, queueing next execution.")
            return
        _cycle_running = True

    try:
        while True:
            await _async_monitoring_cycle()
            async with _cycle_lock:
                if not _cycle_requested:
                    _cycle_running = False
                    break
                _cycle_requested = False
    except Exception as e:
        logger.error(f"Error in triggered monitoring loop: {e}")
        async with _cycle_lock:
            _cycle_running = False
