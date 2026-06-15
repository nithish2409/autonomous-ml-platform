import uuid
import io
import json
import tempfile
import os

import pandas as pd
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.storage import MinioClient
from app.models.dataset import Dataset
from sqlalchemy.future import select


class DatasetService:
    """Service layer for dataset upload, profiling, and storage."""

    def __init__(self):
        self.storage = MinioClient()

    async def upload_dataset(self, file: UploadFile, db: AsyncSession) -> dict:
        """
        Orchestrates the full dataset upload flow:
        1. Validate CSV
        2. Generate UUID
        3. Profile the dataset
        4. Upload to MinIO
        5. Store metadata in PostgreSQL
        """
        contents = await file.read()

        # Validate CSV by attempting to parse it
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise ValueError(f"Invalid CSV file: {e}")

        dataset_id = str(uuid.uuid4())

        # Profile the dataset
        baseline_stats = self.profile_dataset(df)

        # Upload to MinIO
        minio_path = self.store_to_minio(dataset_id, contents)

        # Build schema as list of column names
        schema = list(df.columns)

        # Store metadata in PostgreSQL
        dataset = Dataset(
            id=dataset_id,
            name=file.filename,
            schema=json.dumps(schema),
            baseline_stats=baseline_stats,
            minio_path=minio_path,
        )

        db.add(dataset)
        await db.commit()

        return {
            "dataset_id": dataset_id,
            "name": file.filename,
            "rows": len(df),
            "columns": schema,
            "baseline_stats": baseline_stats,
            "minio_path": minio_path,
        }

    async def get_all_datasets(self, db: AsyncSession) -> list[Dataset]:
        """Return all datasets."""
        result = await db.execute(select(Dataset))
        return result.scalars().all()

    async def delete_dataset(self, dataset_id: str, db: AsyncSession) -> dict:
        """
        Delete a dataset and its raw CSV file in MinIO.
        Blocks deletion if any models in ModelRegistry depend on this dataset.
        """
        from sqlalchemy import delete
        from app.models.model_registry import ModelRegistry

        # 1. Verify dataset exists
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalars().first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        # 2. Check for attached models
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.dataset_id == dataset_id)
        )
        attached_models = result.scalars().all()
        if attached_models:
            raise ValueError(
                f"Cannot delete dataset {dataset_id} because {len(attached_models)} model(s) "
                "are still registered to it. Please delete the models first."
            )

        # 3. Delete from MinIO
        if dataset.minio_path:
            try:
                self.storage.client.remove_object(self.storage.bucket, dataset.minio_path)
            except Exception as e:
                # Log the error but proceed with DB deletion if MinIO fails
                print(f"Warning: Failed to delete minio artifact {dataset.minio_path}: {e}")

        # 4. Delete from Postgres
        await db.execute(delete(Dataset).where(Dataset.id == dataset_id))
        await db.commit()

        return {
            "dataset_id": dataset_id,
            "status": "deleted",
            "message": "Dataset deleted successfully",
        }

    @staticmethod
    def profile_dataset(df: pd.DataFrame) -> dict:
        """
        Compute per-column baseline statistics:
        - mean (numeric columns only)
        - std  (numeric columns only)
        - missing_count (all columns)
        """
        stats = {}
        for col in df.columns:
            col_stats = {
                "missing_count": int(df[col].isnull().sum()),
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats["mean"] = round(float(df[col].mean()), 4) if not df[col].isnull().all() else None
                col_stats["std"] = round(float(df[col].std()), 4) if not df[col].isnull().all() else None
            stats[col] = col_stats
        return stats

    def store_to_minio(self, dataset_id: str, file_bytes: bytes) -> str:
        """Upload raw file bytes to MinIO at datasets/{dataset_id}.csv."""
        object_name = f"datasets/{dataset_id}.csv"
        data = io.BytesIO(file_bytes)
        self.storage.upload_bytes(object_name, data, len(file_bytes))
        return object_name
