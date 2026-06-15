from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.services.dataset_service import DatasetService

router = APIRouter()


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """Upload a CSV dataset: validate, profile, store to MinIO, and save metadata."""
    service = DatasetService()
    try:
        result = await service.upload_dataset(file, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.get("/datasets")
async def list_datasets(db: AsyncSession = Depends(get_db)):
    """List all available datasets."""
    service = DatasetService()
    datasets = await service.get_all_datasets(db)
    return datasets


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a dataset specifically, only if no models rely on it."""
    service = DatasetService()
    try:
        result = await service.delete_dataset(dataset_id, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result
