"""Training API — training jobs, manual retrain with evaluation, and comparison."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.future import select

from app.core.database import get_db
from app.models.training_job import TrainingJob
from app.models.model_registry import ModelRegistry
from app.services.training_orchestrator import TrainingOrchestrator
from app.services.training_runner import TrainingRunner
from app.services.model_evaluator import ModelEvaluator
from app.services.automation_executor import AutomationExecutor
from app.schemas.training import (
    StartTrainingRequest,
    TrainingJobResponse,
    TrainingLogsResponse,
)

router = APIRouter()


# ── List all training jobs ────────────────────────────────────────

@router.get("/training-jobs")
async def list_training_jobs(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """List all training jobs ordered by creation date."""
    result = await db.execute(
        select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(limit)
    )
    jobs = result.scalars().all()

    items = []
    for j in jobs:
        # Fetch related model info
        model_name = None
        model_class = None
        framework = None
        if j.model_id:
            m_result = await db.execute(
                select(ModelRegistry).where(ModelRegistry.id == j.model_id)
            )
            model = m_result.scalar_one_or_none()
            if model:
                model_name = model.model_class or "Unnamed"
                model_class = model.model_class
                framework = model.framework

        items.append({
            "id": str(j.id),
            "model_id": str(j.model_id) if j.model_id else None,
            "model_name": model_name,
            "model_class": model_class,
            "framework": framework,
            "status": j.status,
            "config": j.config,
            "result_metrics": j.result_metrics,
            "created_at": str(j.created_at) if j.created_at else None,
        })

    return items


# ── Existing endpoint (backward compatible) ──────────────────────

@router.post("/models/{model_id}/train")
async def train_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Trigger training for a registered model in an isolated Docker container."""
    orchestrator = TrainingOrchestrator()
    try:
        result = await orchestrator.train(model_id=model_id, db=db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


# ── Phase 3: Enhanced training endpoints ─────────────────────────

@router.post("/training/start", response_model=TrainingJobResponse)
async def start_training(
    request: StartTrainingRequest,
    db: AsyncSession = Depends(get_db),
):
    """Launch a training job with full configuration."""
    orchestrator = TrainingOrchestrator()
    try:
        result = await orchestrator.start_training(
            model_id=request.model_id,
            db=db,
            target_column=request.target_column,
            hyperparameters=request.hyperparameters,
            split_ratio=request.split_ratio,
            random_seed=request.random_seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@router.get("/training/{job_id}", response_model=TrainingJobResponse)
async def get_training_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get training job status and details."""
    orchestrator = TrainingOrchestrator()
    try:
        result = await orchestrator.get_job_status(job_id=job_id, db=db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result


@router.get("/training/{job_id}/logs", response_model=TrainingLogsResponse)
async def get_training_logs(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get training container logs."""
    orchestrator = TrainingOrchestrator()
    try:
        result = await orchestrator.get_job_logs(job_id=job_id, db=db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result


@router.delete("/training/{job_id}")
async def delete_training_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a training job and associated logs."""
    orchestrator = TrainingOrchestrator()
    try:
        await orchestrator.delete_job(job_id=job_id, db=db)
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase 7: Manual retrain with evaluation ──────────────────────

@router.post("/models/{model_id}/retrain")
async def retrain_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Manual retrain trigger with full evaluation pipeline:
    1. Train candidate (in-process sklearn)
    2. Compare candidate vs current production
    3. Promote if better
    4. Return comparison result
    """
    executor = AutomationExecutor()
    try:
        result = await executor.execute_decision(
            model_id=model_id,
            decision={"action": "retrain", "reason": "Manual retrain triggered via API"},
            db=db,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@router.get("/models/{model_id}/compare")
async def compare_model_versions(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Compare latest candidate vs current production model.
    Returns metrics comparison and promotion status.
    """
    evaluator = ModelEvaluator()
    try:
        result = await evaluator.get_latest_comparison(model_id=model_id, db=db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result
