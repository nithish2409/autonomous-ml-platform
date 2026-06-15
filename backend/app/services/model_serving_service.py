"""
Model Serving Service — loads models from MinIO, caches them in memory,
runs inference via framework adapters, and logs every prediction.

Supports frameworks: sklearn/joblib, xgboost, lightgbm, pytorch.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
import uuid
from typing import Any

import joblib
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.storage import MinioClient
from app.core.metrics import (
    inference_latency_seconds,
    inference_requests_total,
    inference_errors_total,
)
from app.models.model_registry import ModelRegistry
from app.models.model_version import ModelVersion
from app.models.inference_log import InferenceLog
from app.services.adapters import get_adapter

logger = logging.getLogger("model_serving_service")

# ── In-memory model cache ────────────────────────────────────────
# key = model_id (str)  →  {model, version, framework, loaded_at, adapter}
CACHE_TTL_SECONDS = 1800  # 30 minutes

_model_cache: dict[str, dict[str, Any]] = {}


# ── Multi-framework loader ───────────────────────────────────────

def load_model(artifact_path: str, framework: str, storage: MinioClient) -> Any:
    """Download artifact from MinIO and deserialise based on *framework*."""

    response = storage.client.get_object(storage.bucket, artifact_path)
    suffix = _suffix_for_framework(framework)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(response.read())
        tmp_path = tmp.name
    response.close()
    response.release_conn()

    try:
        model = _deserialise(tmp_path, framework)
    finally:
        os.remove(tmp_path)

    return model


def _suffix_for_framework(framework: str) -> str:
    fw = framework.lower()
    if fw in ("sklearn", "joblib", "lightgbm"):
        return ".joblib"
    if fw == "xgboost":
        return ".json"
    if fw in ("pytorch", "torch"):
        return ".pt"
    return ".bin"


def _deserialise(path: str, framework: str) -> Any:
    fw = framework.lower()

    if fw in ("sklearn", "joblib", "lightgbm"):
        return joblib.load(path)

    if fw == "xgboost":
        try:
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(path)
            return booster
        except Exception:
            # Fallback: model may have been saved with joblib
            return joblib.load(path)

    if fw in ("pytorch", "torch"):
        try:
            import torch

            model = torch.load(path, map_location="cpu", weights_only=False)
            if hasattr(model, "eval"):
                model.eval()
            return model
        except ImportError:
            raise RuntimeError(
                "PyTorch is not installed. Add 'torch' to requirements.txt "
                "to enable PyTorch model serving."
            )

    # Generic fallback — try joblib
    logger.warning("Unknown framework '%s', attempting joblib.load()", framework)
    return joblib.load(path)


# ── Service class ────────────────────────────────────────────────

class ModelServingService:
    """High-level service consumed by the API router."""

    def __init__(self) -> None:
        self.storage = MinioClient()

    # ── predict ──────────────────────────────────────────────────

    async def predict(
        self,
        model_id: str,
        input_data: list[dict[str, Any]],
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Run inference for *model_id* on a batch of *input_data* rows."""

        start = time.time()
        inference_requests_total.labels(model_id=model_id).inc()

        try:
            # 1. Resolve model + version from DB
            model_entry, version_entry = await self._resolve_model(model_id, db)

            # 2. Load / cache (with hot reload if version changed)
            adapter = self._get_or_load_adapter(
                model_id, version_entry.artifact_path,
                model_entry.framework, model_entry.current_version,
            )

            # 3. Build input DataFrame
            input_df = pd.DataFrame(input_data)
            numeric_df = input_df.select_dtypes(include="number")
            if numeric_df.empty:
                raise ValueError("Input data must contain at least one numeric feature")

            # 4. Inference via adapter
            predictions = adapter.predict(numeric_df)

            latency_ms = round((time.time() - start) * 1000, 2)

            # 5. Prometheus metric
            inference_latency_seconds.labels(model_id=model_id).observe(latency_ms / 1000)

            # 6. Confidence (via adapter)
            confidence = None
            proba = adapter.predict_proba(numeric_df)
            if proba is not None:
                confidence = [round(float(max(row)), 4) for row in proba]

            # 7. Log to inference_logs
            # Calculate column means and convert values to standard Python floats for JSON serialisation
            batch_means = {k: float(v) for k, v in numeric_df.mean().to_dict().items()}
            log_entry = InferenceLog(
                id=uuid.uuid4(),
                model_id=model_id,
                input_summary=batch_means,
                prediction_summary={
                    "predictions": predictions,
                    "confidence": confidence,
                    "n_samples": len(input_data),
                },
                latency_ms=latency_ms,
            )
            db.add(log_entry)
            await db.commit()

            return {
                "model_id": model_id,
                "version": model_entry.current_version,
                "framework": model_entry.framework,
                "predictions": predictions,
                "confidence": confidence,
                "latency_ms": latency_ms,
            }

        except Exception:
            inference_errors_total.labels(model_id=model_id).inc()
            raise

    # ── batch predict (CSV) ──────────────────────────────────────

    async def batch_predict(
        self,
        model_id: str,
        csv_content: str,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Run batch inference on CSV data."""

        start = time.time()
        inference_requests_total.labels(model_id=model_id).inc()

        try:
            model_entry, version_entry = await self._resolve_model(model_id, db)

            adapter = self._get_or_load_adapter(
                model_id, version_entry.artifact_path,
                model_entry.framework, model_entry.current_version,
            )

            # Parse CSV
            df = pd.read_csv(io.StringIO(csv_content))
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.empty:
                raise ValueError("CSV must contain at least one numeric column")

            predictions = adapter.predict(numeric_df)

            latency_ms = round((time.time() - start) * 1000, 2)
            inference_latency_seconds.labels(model_id=model_id).observe(latency_ms / 1000)

            # Log
            log_entry = InferenceLog(
                id=uuid.uuid4(),
                model_id=model_id,
                input_summary={
                    "type": "batch_csv",
                    "n_samples": len(df),
                    "columns": list(df.columns),
                },
                prediction_summary={
                    "n_predictions": len(predictions),
                    "sample": predictions[:5],
                },
                latency_ms=latency_ms,
            )
            db.add(log_entry)
            await db.commit()

            return {
                "model_id": model_id,
                "version": model_entry.current_version,
                "framework": model_entry.framework,
                "n_samples": len(df),
                "predictions": predictions,
                "latency_ms": latency_ms,
            }

        except Exception:
            inference_errors_total.labels(model_id=model_id).inc()
            raise

    # ── input schema ─────────────────────────────────────────────

    async def get_input_schema(
        self,
        model_id: str,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Return expected input features for the model."""

        model_entry, version_entry = await self._resolve_model(model_id, db)

        # Try to get feature names from the cached model
        feature_names = None
        adapter = self._get_or_load_adapter(
            model_id, version_entry.artifact_path,
            model_entry.framework, model_entry.current_version,
        )
        feature_names = adapter.get_feature_names()

        # Also pull from training metrics if available
        metrics = version_entry.metrics or {}

        return {
            "model_id": model_id,
            "version": model_entry.current_version,
            "framework": model_entry.framework,
            "model_class": model_entry.model_class,
            "feature_names": feature_names,
            "n_features": metrics.get("n_features"),
            "target_column": metrics.get("target_column"),
            "task_type": metrics.get("task_type"),
        }

    # ── model status ─────────────────────────────────────────────

    async def get_model_status(
        self,
        model_id: str,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Return whether *model_id* is cached and basic metadata."""

        cached = model_id in _model_cache
        entry: dict[str, Any] = {"model_id": model_id, "loaded": cached, "cached": cached}

        if cached:
            info = _model_cache[model_id]
            entry["version"] = info.get("version")
            entry["framework"] = info.get("framework")
        else:
            # Fetch from DB so we can report version / framework even if not loaded
            result = await db.execute(
                select(ModelRegistry).where(ModelRegistry.id == model_id)
            )
            model = result.scalars().first()
            if model:
                entry["version"] = model.current_version
                entry["framework"] = model.framework

        return entry

    # ── active model ─────────────────────────────────────────────

    async def get_active_model(self, db: AsyncSession) -> dict[str, Any] | None:
        """Return metadata for the first model with status='active'."""

        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.status == "active")
        )
        model = result.scalars().first()
        if model is None:
            return None

        return {
            "model_id": str(model.id),
            "dataset_id": str(model.dataset_id),
            "framework": model.framework,
            "model_class": model.model_class,
            "version": model.current_version,
            "status": model.status,
        }

    # ── internal helpers ─────────────────────────────────────────

    async def _resolve_model(
        self,
        model_id: str,
        db: AsyncSession,
    ) -> tuple[ModelRegistry, ModelVersion]:
        """Fetch ModelRegistry + latest ModelVersion or raise."""

        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model_entry = result.scalars().first()
        if model_entry is None:
            raise ValueError(f"Model {model_id} not found")
        if model_entry.status != "active":
            raise ValueError(f"Model {model_id} is not active (status={model_entry.status})")
        if not model_entry.current_version:
            raise ValueError(f"Model {model_id} has no trained version")

        result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.model_id == model_entry.id,
                ModelVersion.version_number == model_entry.current_version,
            )
        )
        version_entry = result.scalars().first()
        if version_entry is None:
            raise ValueError(
                f"Artifact for version {model_entry.current_version} not found"
            )

        return model_entry, version_entry

    def _get_or_load_adapter(
        self, model_id: str, artifact_path: str, framework: str, version: str,
    ):
        """Return cached adapter or download model + create adapter. Supports hot reload."""

        now = time.time()

        if model_id in _model_cache:
            entry = _model_cache[model_id]
            age = now - entry["loaded_at"]

            # Hot reload: if DB version differs from cached version, force reload
            if entry.get("version") != version:
                logger.info(
                    "Hot reload: version changed %s → %s for %s",
                    entry.get("version"), version, model_id,
                )
                del _model_cache[model_id]
            elif age < CACHE_TTL_SECONDS:
                logger.info("Cache hit for %s (age %.0fs)", model_id, age)
                return entry["adapter"]
            else:
                logger.info("Cache expired for %s, reloading", model_id)
                del _model_cache[model_id]

        logger.info("Loading model %s from MinIO (%s)", model_id, artifact_path)
        model = load_model(artifact_path, framework, self.storage)
        adapter = get_adapter(model, framework)

        _model_cache[model_id] = {
            "model": model,
            "adapter": adapter,
            "version": version,
            "framework": framework,
            "loaded_at": now,
        }
        return adapter
