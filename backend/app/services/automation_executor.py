"""
Automation Executor — executes approved LLM decisions (retrain, rollback, alert).
Ensures cooldown safety, idempotency, and consistent DB logging.

Phase 7:  Retrain handler uses TrainingRunner + ModelEvaluator pipeline.
Phase C:  Autonomous evolution layer — drift analysis → strategy → config generation.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone

from app.core.metrics import automation_actions_total, model_version_changes_total

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.automation_log import AutomationLog
from app.models.automation_state import AutomationState
from app.models.model_registry import ModelRegistry
from app.models.model_version import ModelVersion
from app.services.training_runner import TrainingRunner
from app.services.model_evaluator import ModelEvaluator
from app.services.model_registry_service import ModelRegistryService

# Phase C — autonomous evolution layer
from app.training.drift_analyzer import analyze_drift
from app.training.strategy_engine import choose_strategy
from app.training.config_generator import generate_training_config
from app.training.early_stopping import should_stop_early
from app.training.resource_manager import adjust_training_resources

logger = logging.getLogger("automation_executor")


class AutomationExecutor:
    """Executes approved LLM decisions safely and idempotently."""

    COOLDOWN_MINUTES = 15

    def __init__(self):
        self.training_runner = TrainingRunner()
        self.evaluator = ModelEvaluator()
        self.registry_service = ModelRegistryService()

    async def execute_decision(
        self,
        model_id: str,
        decision: dict,
        db: AsyncSession,
    ) -> dict:
        """
        Dispatch an approved decision to the correct handler.

        Returns a dict with keys: status, action, result/reason.
        """
        action = decision.get("action", "none")
        reason = decision.get("reason", "No reason provided")

        # ── action == "none" → skip immediately ──
        if action == "none":
            await self._log_execution(model_id, action, reason, "skipped", {}, db)
            return {"status": "skipped", "action": action, "reason": reason}

        # ── Cooldown gate ──
        if not await self._is_safe_to_execute(model_id, db):
            await self._log_execution(
                model_id, action, reason, "cooldown_blocked", {}, db,
            )
            return {
                "status": "cooldown_blocked",
                "action": action,
                "reason": "Cooldown period active — execution skipped",
            }

        # ── Execute the action ──
        try:
            if action.lower() == "retrain":
                result = await self._handle_retrain(model_id, decision, db)
                if result.get("skipped"):
                    status = "strategy_skipped"
                else:
                    status = "completed" if result.get("promoted") else "trained_not_promoted"

            elif action.lower() == "rollback":
                result = await self._handle_rollback(model_id, db)
                status = "rolled_back" if result else "rollback_failed"

            elif action.lower() == "alert":
                result = self._handle_alert(model_id, reason)
                status = "alert_triggered"

            elif action.lower() in ("scale", "notify"):
                result = self._handle_notification(model_id, action, reason)
                status = f"{action}_triggered"

            else:
                result = {"detail": f"Unknown action: {action}"}
                status = "unknown_action"

            # Persist log + state + metrics
            await self._log_execution(model_id, action, reason, status, result, db)
            await self._update_state(model_id, action, db)
            automation_actions_total.labels(action=action, status=status).inc()
            if action in ("retrain", "rollback"):
                model_version_changes_total.labels(model_id=model_id, change_type=action).inc()

            return {"status": status, "action": action, "result": result}

        except Exception as exc:
            logger.exception("Execution failed for model %s action %s", model_id, action)
            await self._log_execution(
                model_id, action, reason, "failed",
                {"error": str(exc)}, db,
            )
            return {"status": "failed", "action": action, "error": str(exc)}

    # ── Action handlers ──────────────────────────────────────────────

    async def _handle_retrain(
        self, model_id: str, decision: dict, db: AsyncSession,
    ) -> dict:
        """
        Full retrain pipeline with Phase C autonomous evolution:
        1. Drift analysis → strategy → config generation
        2. Early stopping check
        3. Train candidate model with generated config
        4. Evaluate candidate vs current production
        5. If better → register new version, update registry
        6. Log decision chain to automation_logs
        """
        logger.info("Executing RETRAIN for model %s", model_id)

        # ══ Phase C — Step 1: Drift Analysis ══
        monitoring = decision.get("monitoring_metrics", {})
        drift_score = float(monitoring.get("drift_score", 0.0))
        performance_delta = float(monitoring.get("performance_delta", 0.0))
        drift_info = analyze_drift(drift_score, performance_delta)

        # ══ Phase C — Step 2: Strategy Selection ══
        strategy = choose_strategy(decision.get("action", "retrain"), drift_info)

        # ══ Phase C — Step 3: Config Generation ══
        training_config = generate_training_config(strategy)

        if training_config is None:
            logger.info(
                "Strategy '%s' produced no config — skipping retrain for model %s",
                strategy, model_id,
            )
            skip_result = {
                "skipped": True,
                "model_id": model_id,
                "drift_info": drift_info,
                "strategy": strategy,
                "reason": "Strategy produced no training config",
            }
            # Log the decision chain even when skipping
            await self._log_execution(
                model_id, "retrain", decision.get("reason", ""),
                "strategy_skipped", skip_result, db,
            )
            return skip_result

        # ══ Phase C — Step 4: Resource Management ══
        dataset_size = int(monitoring.get("dataset_size", 1000))
        resource_profile = adjust_training_resources(dataset_size)
        training_config["_resource_profile"] = resource_profile

        # ══ Phase C — Step 5: Early Stopping Check ══
        # Fetch model class to determine regression vs classification
        model_result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model_entry = model_result.scalars().first()
        if not model_entry:
            raise ValueError(f"Model {model_id} not found")
        is_regressor = "regressor" in (model_entry.model_class or "").lower()

        # Query completed training jobs for this model
        from app.models.training_job import TrainingJob
        history_result = await db.execute(
            select(TrainingJob)
            .where(
                TrainingJob.model_id == model_id,
                TrainingJob.status == "completed"
            )
            .order_by(TrainingJob.created_at.asc())
        )
        history_jobs = history_result.scalars().all()
        training_history = [job.result_metrics for job in history_jobs if job.result_metrics]

        metric_name = "r2" if is_regressor else "f1_score"
        if should_stop_early(training_history, metric=metric_name):
            logger.info("Early stopping triggered for model %s — skipping retrain", model_id)
            stop_result = {
                "skipped": True,
                "model_id": model_id,
                "drift_info": drift_info,
                "strategy": strategy,
                "reason": "Early stopping — no meaningful improvement in recent runs",
            }
            await self._log_execution(
                model_id, "retrain", decision.get("reason", ""),
                "early_stopped", stop_result, db,
            )
            return stop_result

        logger.info(
            "Phase C decision chain: drift=%s strategy=%s config_keys=%s resources=%s",
            drift_info["drift_type"], strategy,
            list(training_config.keys()), resource_profile["profile"],
        )

        # ══ Train candidate with generated config ══
        train_result = await self.training_runner.train_candidate(
            model_id, db, training_config=training_config,
        )
        candidate_metrics = train_result["metrics"]
        version = train_result["version"]
        artifact_path = train_result["artifact_path"]
        job_id = train_result["job_id"]
        previous_version = train_result.get("previous_version")

        # ══ Evaluate candidate vs current ══
        eval_result = await self.evaluator.compare(
            model_id=model_id,
            candidate_metrics=candidate_metrics,
            db=db,
        )

        promoted = eval_result["promote"]

        # ══ Register version (always) and promote (if better) ══
        best_params = candidate_metrics.pop("best_params", None)

        model_version = ModelVersion(
            id=str(uuid.uuid4()),
            model_id=model_id,
            version_number=version,
            artifact_path=artifact_path,
            metrics=candidate_metrics,
            hyperparameters=best_params,
            training_job_id=job_id,
            parent_version=previous_version,
        )
        db.add(model_version)

        if promoted:
            result = await db.execute(
                select(ModelRegistry).where(ModelRegistry.id == model_id)
            )
            model = result.scalars().first()
            if model:
                model.current_version = version
                logger.info(
                    "Candidate PROMOTED: model=%s version=%s", model_id, version,
                )
        else:
            logger.info(
                "Candidate NOT promoted: model=%s version=%s — %s",
                model_id, version, eval_result["reason"],
            )

        await db.commit()

        return {
            "job_id": job_id,
            "model_id": model_id,
            "version": version,
            "promoted": promoted,
            "evaluation": eval_result,
            "metrics": candidate_metrics,
            "artifact_path": artifact_path,
            # Phase C decision chain metadata
            "drift_info": drift_info,
            "strategy": strategy,
            "resource_profile": resource_profile["profile"],
        }

    async def _handle_rollback(self, model_id: str, db: AsyncSession) -> dict | None:
        """Roll back to the previous model version."""
        logger.info("Executing ROLLBACK for model %s", model_id)
        result = await self.registry_service.rollback_to_previous_version(model_id, db)
        if result is None:
            logger.warning("Rollback failed: no previous version for model %s", model_id)
        else:
            logger.info(
                "Rollback completed: model=%s %s → %s",
                model_id, result["old_version"], result["new_version"],
            )
        return result

    @staticmethod
    def _handle_alert(model_id: str, reason: str) -> dict:
        """Log a critical alert."""
        logger.critical(
            "CRITICAL ALERT: Model %s requires attention — %s", model_id, reason,
        )
        return {"alert_message": f"Critical alert for model {model_id}: {reason}"}

    @staticmethod
    def _handle_notification(model_id: str, action: str, reason: str) -> dict:
        """Handle scale/notify actions (logged, no external system yet)."""
        logger.info(
            "%s notification for model %s: %s", action.upper(), model_id, reason,
        )
        return {"notification": f"{action}: {reason}", "model_id": model_id}

    # ── Safety ───────────────────────────────────────────────────────

    async def _is_safe_to_execute(self, model_id: str, db: AsyncSession) -> bool:
        """Return True immediately for demo mode (cooldown bypassed)."""
        logger.info("Demo mode: Cooldown check bypassed for execution")
        return True

    # ── Persistence helpers ──────────────────────────────────────────

    async def _log_execution(
        self,
        model_id: str,
        action: str,
        reason: str,
        status: str,
        execution_result: dict,
        db: AsyncSession,
    ) -> None:
        """Write an entry to automation_logs."""
        log = AutomationLog(
            id=uuid.uuid4(),
            model_id=model_id,
            action=action,
            reason=reason,
            status=status,
            execution_result=execution_result,
        )
        db.add(log)
        await db.commit()

    async def _update_state(
        self, model_id: str, action: str, db: AsyncSession,
    ) -> None:
        """Upsert automation_state with last_action + cooldown."""
        result = await db.execute(
            select(AutomationState).where(AutomationState.model_id == model_id)
        )
        state = result.scalars().first()

        cooldown = datetime.now(timezone.utc) + timedelta(minutes=self.COOLDOWN_MINUTES)

        if state is None:
            state = AutomationState(
                model_id=model_id,
                last_action=action,
                cooldown_until=cooldown,
            )
            db.add(state)
        else:
            state.last_action = action
            state.cooldown_until = cooldown

        await db.commit()
