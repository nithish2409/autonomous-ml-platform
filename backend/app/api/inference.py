from fastapi import APIRouter, Depends, HTTPException, Body, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.core.database import get_db
from app.services.inference_service import InferenceService
from app.scheduler.monitoring_scheduler import trigger_monitoring_loop

router = APIRouter()

class SwitchVersionRequest(BaseModel):
    version: str


@router.post("/inference/predict")
async def predict(
    background_tasks: BackgroundTasks,
    payload: dict = Body(...),
    db: AsyncSession = Depends(get_db),
):
    """Run inference using the active model."""
    service = InferenceService()
    try:
        # Frontend sends pure dict, we pass it as input_data
        result = await service.predict(
            input_data=payload,
            db=db,
        )
        if background_tasks:
            background_tasks.add_task(trigger_monitoring_loop)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result

@router.get("/inference/status")
async def get_inference_status(db: AsyncSession = Depends(get_db)):
    """Get endpoint status and active version."""
    service = InferenceService()
    return await service.get_status(db=db)

@router.get("/inference/metrics")
async def get_inference_metrics(db: AsyncSession = Depends(get_db)):
    """Get live inference KPIs."""
    service = InferenceService()
    return await service.get_metrics(db=db)

@router.get("/inference/logs")
async def get_inference_logs(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Get recent inference logs."""
    service = InferenceService()
    return await service.get_logs(limit=limit, db=db)

@router.post("/inference/switch-version")
async def switch_version(
    request: SwitchVersionRequest,
    db: AsyncSession = Depends(get_db)
):
    """Switch the active version of the current model."""
    service = InferenceService()
    try:
        return await service.switch_version(version=request.version, db=db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
