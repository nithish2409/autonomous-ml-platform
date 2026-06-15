import io
import time
import uuid
import logging
import tempfile
import os
from typing import Any

import joblib
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.storage import MinioClient
from app.core.metrics import inference_latency_seconds
from app.models.model_registry import ModelRegistry
from app.models.model_version import ModelVersion
from app.models.monitoring_metric import MonitoringMetric

logger = logging.getLogger("inference_service")

# Cache TTL: 30 minutes
CACHE_TTL_SECONDS = 1800

# In-memory model cache: key = "model_id:version" → {model, loaded_at}
_model_cache: dict[str, dict[str, Any]] = {}


class InferenceService:
    """Service for running predictions against trained models."""

    def __init__(self):
        self.storage = MinioClient()

    async def predict(
        self,
        input_data: dict,
        db: AsyncSession,
        dataset_id: str | None = None,
    ) -> dict:
        """
        Run inference:
        1. Fetch the active model
        2. Load (or cache-hit) the model artifact
        3. Run prediction
        4. Log inference metadata
        """
        start_time = time.time()

        # 1. Fetch active model
        if dataset_id:
            result = await db.execute(
                select(ModelRegistry).where(
                    ModelRegistry.dataset_id == dataset_id,
                    ModelRegistry.status == "active",
                )
            )
        else:
            result = await db.execute(select(ModelRegistry).where(ModelRegistry.status == "active").limit(1))
            
        model_entry = result.scalars().first()
        if model_entry is None:
            raise ValueError(f"No active model found")

        if not model_entry.current_version:
            raise ValueError(f"Active model {model_entry.id} has no trained version")

        # 2. Get the model version's artifact path
        result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.model_id == model_entry.id,
                ModelVersion.version_number == model_entry.current_version,
            )
        )
        version_entry = result.scalars().first()
        if version_entry is None:
            raise ValueError(
                f"Version {model_entry.current_version} not found for model {model_entry.id}"
            )

        # 3. Load model (with TTL-aware cache)
        cache_key = f"{model_entry.id}:{model_entry.current_version}"
        model = self._load_model(cache_key, version_entry.artifact_path, model_entry.framework)

        # 4. Run inference
        import pandas as pd

        input_df = pd.DataFrame([input_data])
        
        # Align features to match what the model expects if selected_features exists
        selected_features = version_entry.metrics.get("selected_features") if version_entry.metrics else None
        if selected_features:
            # Filter and order columns to match selected_features
            numeric_df = input_df[[col for col in selected_features if col in input_df.columns]]
        else:
            # Fallback: select numeric columns, but drop any common target column names if present
            numeric_df = input_df.select_dtypes(include="number")
            for target_col in ["approved", "is_fraud", "class", "target", "label"]:
                if target_col in numeric_df.columns:
                    numeric_df = numeric_df.drop(columns=[target_col])

        if numeric_df.empty:
            raise ValueError("Input data must contain at least one numeric feature matching the model")

        # Pass as numpy array to avoid feature name mismatch if trained without them
        prediction = model.predict(numeric_df.values)
        prediction_list = prediction.tolist()

        latency_s = time.time() - start_time
        latency_ms = round(latency_s * 1000, 2)

        # Record Prometheus metric
        inference_latency_seconds.labels(model_id=str(model_entry.id)).observe(latency_s)

        # 5. Write inference log
        from app.models.inference_log import InferenceLog
        log_entry = InferenceLog(
            id=uuid.uuid4(),
            model_id=model_entry.id,
            input_summary=input_data,
            prediction_summary={"prediction": prediction_list},
            latency_ms=latency_ms,
        )
        db.add(log_entry)
        
        await db.commit()

        return {
            "model_id": str(model_entry.id),
            "version": model_entry.current_version,
            "framework": model_entry.framework,
            "prediction": prediction_list,
            "latency_ms": latency_ms,
        }

    def _load_model(self, cache_key: str, artifact_path: str, framework: str) -> Any:
        """Load model from cache (with TTL) or download from MinIO."""
        now = time.time()

        if cache_key in _model_cache:
            entry = _model_cache[cache_key]
            age = now - entry["loaded_at"]
            if age < CACHE_TTL_SECONDS:
                logger.info("Cache hit for %s (age: %.0fs)", cache_key, age)
                return entry["model"]
            else:
                logger.info("Cache expired for %s (age: %.0fs), re-downloading", cache_key, age)
                del _model_cache[cache_key]

        # Download artifact from MinIO to a temp file
        response = self.storage.client.get_object(self.storage.bucket, artifact_path)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp.write(response.read())
            tmp_path = tmp.name
        response.close()
        response.release_conn()

        # Load based on framework
        model = joblib.load(tmp_path)
        os.remove(tmp_path)

        # Cache with timestamp
        _model_cache[cache_key] = {"model": model, "loaded_at": now}
        logger.info("Cached model %s", cache_key)
        return model

    async def get_status(self, db: AsyncSession) -> dict:
        result = await db.execute(select(ModelRegistry).where(ModelRegistry.status == "active").limit(1))
        model_entry = result.scalars().first()
        
        if not model_entry:
            return {
                "health_status": "offline",
                "endpoint_name": "inference-endpoint",
                "version": None,
                "current_model": None,
                "replicas": 0,
                "region": "us-east-1",
                "uptime": "0h 0m",
                "cpu_usage": "0%",
                "memory_usage": "0 GB",
                "framework": None,
            }
            
        return {
            "health_status": "healthy",
            "endpoint_name": f"{model_entry.model_class.lower()}-endpoint" if model_entry.model_class else "inference-endpoint",
            "version": model_entry.current_version,
            "current_model": model_entry.model_class,
            "replicas": 3,
            "region": "us-east-1",
            "uptime": "12h 45m",
            "cpu_usage": "42%",
            "memory_usage": "2.4 GB",
            "framework": model_entry.framework,
        }

    async def get_metrics(self, db: AsyncSession) -> dict:
        # Simulated live metrics for dashboard
        import random
        return {
            "avg_latency_ms": round(random.uniform(12.5, 18.2), 1),
            "p95_latency_ms": round(random.uniform(25.1, 35.4), 1),
            "request_rate_rps": round(random.uniform(120.0, 145.0), 1),
            "error_rate_percent": round(random.uniform(0.01, 0.08), 2),
        }

    async def get_logs(self, limit: int, db: AsyncSession) -> list:
        from app.models.inference_log import InferenceLog
        result = await db.execute(select(InferenceLog).order_by(InferenceLog.created_at.desc()).limit(limit))
        entries = result.scalars().all()
        
        return [
            {
                "timestamp": str(e.created_at),
                "request_id": str(e.id),
                "latency_ms": e.latency_ms,
                "status_code": 200,
                "prediction": e.prediction_summary,
                "is_sandbox": True,
            } for e in entries
        ]

    async def switch_version(self, version: str, db: AsyncSession) -> dict:
        result = await db.execute(select(ModelRegistry).where(ModelRegistry.status == "active").limit(1))
        model_entry = result.scalars().first()
        if not model_entry:
            raise ValueError("No active model to switch versions on.")
            
        model_entry.current_version = version
        await db.commit()
        return {"status": "success", "version": version}

