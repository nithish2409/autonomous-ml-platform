"""Model Lifecycle API router — register, version, promote, rollback, list."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.model_registry import (
    RegisterModelRequest,
    RegisterModelResponse,
    AddVersionRequest,
    VersionResponse,
    PromoteRequest,
    PromoteResponse,
    RollbackRequest,
    RollbackResponse,
    ModelDetailResponse,
    ModelListResponse,
)
from app.services.model_registry_service import ModelRegistryService

router = APIRouter(tags=["lifecycle"])


# ── Static routes first ──────────────────────────────────────────

@router.get("/models/all", response_model=ModelListResponse)
async def list_all_models(
    db: AsyncSession = Depends(get_db),
):
    """List all registered models with status info."""

    service = ModelRegistryService()
    models = await service.get_all_models(db)
    items = []
    for m in models:
        items.append({
            "model_id": str(m.id),
            "dataset_id": str(m.dataset_id),
            "framework": m.framework,
            "model_class": m.model_class,
            "current_version": m.current_version,
            "status": m.status,
            "created_at": str(m.created_at) if m.created_at else None,
            "versions": [],
        })
    return {"models": items, "total": len(items)}


@router.post("/models/lifecycle/register", response_model=RegisterModelResponse)
async def register_model(
    request: RegisterModelRequest,
    db: AsyncSession = Depends(get_db),
):
    """Register a new model (lifecycle-aware, no auto-training)."""

    service = ModelRegistryService()
    result = await service.register_model_v2(
        dataset_id=request.dataset_id,
        framework=request.framework,
        model_class=request.model_class,
        db=db,
    )
    return result


# ── Parameterised routes ─────────────────────────────────────────

@router.post("/models/{model_id}/versions", response_model=VersionResponse)
async def add_version(
    model_id: str,
    request: AddVersionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Add a new version with metrics and lineage."""

    service = ModelRegistryService()
    try:
        result = await service.add_version(
            model_id=model_id,
            version_number=request.version_number,
            artifact_path=request.artifact_path,
            db=db,
            metrics=request.metrics,
            training_job_id=request.training_job_id,
            parent_version=request.parent_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return result


@router.post("/models/{model_id}/promote", response_model=PromoteResponse)
async def promote_model(
    model_id: str,
    request: PromoteRequest,
    db: AsyncSession = Depends(get_db),
):
    """Promote a specific version to production."""

    service = ModelRegistryService()
    try:
        result = await service.promote_version(
            model_id=model_id,
            version=request.version,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return result


@router.post("/models/{model_id}/rollback", response_model=RollbackResponse)
async def rollback_model(
    model_id: str,
    request: RollbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """Rollback to a specific previous version."""

    service = ModelRegistryService()
    try:
        result = await service.rollback_to_version(
            model_id=model_id,
            target_version=request.target_version,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return result


@router.get("/models/{model_id}/detail", response_model=ModelDetailResponse)
async def get_model_detail(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get full model details including all versions."""

    service = ModelRegistryService()
    try:
        result = await service.get_model_detail(model_id=model_id, db=db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return result


@router.get("/models/{model_id}/versions", response_model=list[VersionResponse])
async def list_model_versions(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List all versions for a model."""

    service = ModelRegistryService()
    try:
        result = await service.get_model_versions(model_id=model_id, db=db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return result
