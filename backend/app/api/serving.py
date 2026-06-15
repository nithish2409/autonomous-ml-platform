"""Serving API router — model inference, status, batch, schema, and active-model lookup."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.serving import (
    PredictRequest,
    PredictResponse,
    BatchCSVRequest,
    BatchPredictResponse,
    ModelStatusResponse,
    ActiveModelResponse,
    InputSchemaResponse,
)
from app.services.model_serving_service import ModelServingService

router = APIRouter(prefix="/serve", tags=["serving"])


@router.get("/active", response_model=ActiveModelResponse)
async def active_model(
    db: AsyncSession = Depends(get_db),
):
    """Return the currently active deployed model (if any)."""
    service = ModelServingService()
    result = await service.get_active_model(db=db)
    if result is None:
        raise HTTPException(status_code=404, detail="No active model found")
    return result


@router.post("/{model_id}/predict", response_model=PredictResponse)
async def predict(
    model_id: str,
    request: PredictRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run inference on the given model.

    Accepts a batch of input samples and returns predictions.
    The model must be **active** and have a trained version.
    """
    service = ModelServingService()
    try:
        result = await service.predict(
            model_id=model_id,
            input_data=request.input_data,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@router.post("/{model_id}/batch", response_model=BatchPredictResponse)
async def batch_predict(
    model_id: str,
    request: BatchCSVRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run batch inference from CSV data.

    Accepts raw CSV string (with header row) and returns predictions for all rows.
    """
    service = ModelServingService()
    try:
        result = await service.batch_predict(
            model_id=model_id,
            csv_content=request.csv_content,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@router.get("/{model_id}/schema", response_model=InputSchemaResponse)
async def input_schema(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return the expected input schema for the model."""
    service = ModelServingService()
    try:
        result = await service.get_input_schema(model_id=model_id, db=db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return result


@router.get("/{model_id}/status", response_model=ModelStatusResponse)
async def model_status(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Check whether a model is loaded in the serving cache."""
    service = ModelServingService()
    return await service.get_model_status(model_id=model_id, db=db)
