from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.model_registry_service import ModelRegistryService

router = APIRouter()


class RegisterModelRequest(BaseModel):
    dataset_id: str
    framework: str
    model_class: str


@router.post("/models/register")
async def register_model(
    request: RegisterModelRequest,
    db: AsyncSession = Depends(get_db),
):
    """Register a new model for a dataset."""
    service = ModelRegistryService()
    result = await service.register_model(
        dataset_id=request.dataset_id,
        framework=request.framework,
        model_class=request.model_class,
        db=db,
    )
    return result


@router.post("/models/{model_id}/activate")
async def activate_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Activate a model (deactivates any other active model for the same dataset)."""
    service = ModelRegistryService()
    try:
        result = await service.activate_model(model_id=model_id, db=db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result


@router.get("/models")
async def list_models(db: AsyncSession = Depends(get_db)):
    """List all registered models."""
    service = ModelRegistryService()
    models = await service.get_all_models(db)
    
    out = []
    for m in models:
        versions = await service.get_model_versions(str(m.id), db)
        v_list = [v["version_number"] for v in versions]
        m_dict = {
            "id": str(m.id),
            "name": getattr(m, "name", None),
            "dataset_id": str(m.dataset_id),
            "framework": m.framework,
            "model_class": m.model_class,
            "current_version": m.current_version,
            "status": m.status,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "all_versions": v_list
        }
        out.append(m_dict)
    return out


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a registered model and all associated artifacts/logs."""
    service = ModelRegistryService()
    try:
        result = await service.delete_model(model_id=model_id, db=db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result
