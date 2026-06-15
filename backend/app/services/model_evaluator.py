"""
Model Evaluator — compares candidate model metrics against current production model
to decide whether to promote the candidate.
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.model_registry import ModelRegistry
from app.models.model_version import ModelVersion

logger = logging.getLogger("model_evaluator")

# Minimum improvement required for promotion
ACCURACY_IMPROVEMENT_THRESHOLD = 0.01   # 1%
R2_IMPROVEMENT_THRESHOLD = 0.01        # 1%
MSE_IMPROVEMENT_THRESHOLD = 0.01       # candidate MSE must be this much lower


class ModelEvaluator:
    """Compares candidate model metrics against current production model."""

    async def compare(
        self,
        model_id: str,
        candidate_metrics: dict,
        db: AsyncSession,
    ) -> dict:
        """
        Compare candidate model metrics against current production model.

        Returns:
        {
            "promote": True/False,
            "current_metrics": {...},
            "candidate_metrics": {...},
            "reason": "...",
            "comparison": {...}
        }
        """
        # Get current production version metrics
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalars().first()
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        current_metrics = {}
        if model.current_version:
            result = await db.execute(
                select(ModelVersion).where(
                    ModelVersion.model_id == model_id,
                    ModelVersion.version_number == model.current_version,
                )
            )
            current_version = result.scalars().first()
            if current_version and current_version.metrics:
                current_metrics = current_version.metrics

        # If no current metrics, always promote (first real training)
        if not current_metrics:
            return {
                "promote": True,
                "current_metrics": current_metrics,
                "candidate_metrics": candidate_metrics,
                "reason": "No existing metrics — promoting candidate as first version",
                "comparison": {},
            }

        # ── Compare metrics ──
        comparison = {}
        promote = False
        reasons = []

        # Accuracy comparison (classification)
        if "accuracy" in candidate_metrics and "accuracy" in current_metrics:
            current_acc = current_metrics["accuracy"]
            candidate_acc = candidate_metrics["accuracy"]
            delta = candidate_acc - current_acc
            comparison["accuracy_delta"] = round(delta, 4)
            comparison["accuracy_improved"] = delta >= ACCURACY_IMPROVEMENT_THRESHOLD

            if delta >= ACCURACY_IMPROVEMENT_THRESHOLD:
                promote = True
                reasons.append(
                    f"Accuracy improved by {delta:.4f} "
                    f"({current_acc:.4f} → {candidate_acc:.4f})"
                )
            elif delta >= 0:
                reasons.append(
                    f"Accuracy marginal change ({delta:+.4f}), below threshold"
                )
            else:
                reasons.append(
                    f"Accuracy degraded by {abs(delta):.4f} "
                    f"({current_acc:.4f} → {candidate_acc:.4f})"
                )

        # F1 comparison (classification)
        if "f1_score" in candidate_metrics and "f1_score" in current_metrics:
            current_f1 = current_metrics["f1_score"]
            candidate_f1 = candidate_metrics["f1_score"]
            delta = candidate_f1 - current_f1
            comparison["f1_delta"] = round(delta, 4)
            comparison["f1_improved"] = delta > 0

        # R2 comparison (regression)
        if "r2" in candidate_metrics and "r2" in current_metrics:
            current_r2 = current_metrics["r2"]
            candidate_r2 = candidate_metrics["r2"]
            delta = candidate_r2 - current_r2
            comparison["r2_delta"] = round(delta, 4)
            comparison["r2_improved"] = delta >= R2_IMPROVEMENT_THRESHOLD

            if delta >= R2_IMPROVEMENT_THRESHOLD:
                promote = True
                reasons.append(
                    f"R² improved by {delta:.4f} "
                    f"({current_r2:.4f} → {candidate_r2:.4f})"
                )
            elif delta >= 0:
                reasons.append(
                    f"R² marginal change ({delta:+.4f}), below threshold"
                )
            else:
                reasons.append(
                    f"R² degraded by {abs(delta):.4f} "
                    f"({current_r2:.4f} → {candidate_r2:.4f})"
                )

        # MSE comparison (regression — lower is better)
        if "mse" in candidate_metrics and "mse" in current_metrics:
            current_mse = current_metrics["mse"]
            candidate_mse = candidate_metrics["mse"]
            delta = current_mse - candidate_mse  # positive = improvement
            comparison["mse_delta"] = round(delta, 4)
            comparison["mse_improved"] = delta >= MSE_IMPROVEMENT_THRESHOLD

        if not reasons:
            reasons.append("Insufficient metrics for comparison — not promoting")

        reason = "; ".join(reasons)
        logger.info(
            "Evaluation for model %s: promote=%s — %s",
            model_id, promote, reason,
        )

        return {
            "promote": promote,
            "current_metrics": current_metrics,
            "candidate_metrics": candidate_metrics,
            "reason": reason,
            "comparison": comparison,
        }

    async def get_latest_comparison(
        self,
        model_id: str,
        db: AsyncSession,
    ) -> dict:
        """
        Get the latest two versions and compare their metrics.
        Useful for the GET /models/{model_id}/compare endpoint.
        """
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalars().first()
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        # Get last two versions
        result = await db.execute(
            select(ModelVersion)
            .where(ModelVersion.model_id == model_id)
            .order_by(ModelVersion.created_at.desc())
            .limit(2)
        )
        versions = result.scalars().all()

        if len(versions) < 2:
            return {
                "model_id": model_id,
                "current_model": versions[0].metrics if versions else None,
                "candidate_model": None,
                "promoted": None,
                "reason": "Fewer than 2 versions available for comparison",
            }

        latest = versions[0]   # candidate (newest)
        previous = versions[1]  # current production

        eval_result = await self.compare(
            model_id=model_id,
            candidate_metrics=latest.metrics or {},
            db=db,
        )

        return {
            "model_id": model_id,
            "current_model": {
                "version": previous.version_number,
                "metrics": previous.metrics,
            },
            "candidate_model": {
                "version": latest.version_number,
                "metrics": latest.metrics,
            },
            "promoted": latest.version_number == model.current_version,
            "evaluation": eval_result,
        }
