import io
import uuid
import logging
import tempfile
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.model_registry import ModelRegistry
from app.models.model_version import ModelVersion
from app.models.dataset import Dataset
from app.models.automation_log import AutomationLog
from app.core.storage import MinioClient
from app.utils.lifecycle_enums import LifecycleAction

logger = logging.getLogger("model_registry_service")


class ModelRegistryService:
    """Service layer for model registration, activation, and lifecycle management."""

    def __init__(self):
        self.storage = MinioClient()

    # ── Existing methods (unchanged) ─────────────────────────────

    async def register_model(
        self,
        dataset_id: str,
        framework: str,
        model_class: str,
        db: AsyncSession,
    ) -> dict:
        """
        Register a new model entry for a given dataset.
        
        DEMO ENHANCEMENT:
        Automatically trains a quick "v1" baseline model on the dataset
        so that the model is immediately usable for inference.
        """
        model_id = str(uuid.uuid4())
        version = "v1"

        # 1. Create Model Registry Entry
        model = ModelRegistry(
            id=model_id,
            dataset_id=dataset_id,
            framework=framework,
            model_class=model_class,
            current_version=version,  # Set to v1 immediately
            status="active",
        )
        db.add(model)
        
        # 2. Train Baseline (Demo Logic)
        try:
            artifact_path, metrics = await self._train_baseline(
                dataset_id, model_class, db
            )
            
            # 3. Create Model Version Entry
            model_version = ModelVersion(
                id=str(uuid.uuid4()),
                model_id=model_id,
                version_number=version,
                artifact_path=artifact_path,
                metrics=metrics
            )
            db.add(model_version)
            
            await db.commit()
            await db.refresh(model)

            return {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "framework": framework,
                "model_class": model_class,
                "status": "active",
                "version": version,
                "note": "Baseline trained automatically"
            }

        except Exception as e:
            # Fallback if training fails (e.g. empty dataset)
            print(f"Auto-train failed: {e}")
            model.current_version = None
            await db.commit()
            return {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "framework": framework,
                "model_class": model_class,
                "status": "inactive",
                "error": str(e)
            }

    async def _train_baseline(self, dataset_id: str, model_class: str, db: AsyncSession):
        """
        Fetches dataset, trains a simple Scalar model, and uploads artifact.
        """
        # Fetch dataset path
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalars().first()
        if not dataset or not dataset.minio_path:
            raise ValueError("Dataset not found or empty")

        # Download CSV
        response = self.storage.client.get_object(self.storage.bucket, dataset.minio_path)
        df = pd.read_csv(io.BytesIO(response.read()))
        response.close()
        response.release_conn()

        if df.empty:
            raise ValueError("Dataset is empty")

        # Prep Data (Assume last column is target)
        X = df.iloc[:, :-1].select_dtypes(include="number") # Only numeric features
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train
        if "Regressor" in model_class:
            clf = RandomForestRegressor(n_estimators=10, max_depth=5)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            metrics = {
                "mse": round(float(mean_squared_error(y_test, y_pred)), 4),
                "r2": round(float(r2_score(y_test, y_pred)), 4),
            }
        else:
            clf = RandomForestClassifier(n_estimators=10, max_depth=5)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            metrics = {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "f1_score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            }

        # Save Artifact
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            joblib.dump(clf, tmp.name)
            tmp_path = tmp.name

        # Upload to MinIO
        artifact_path = f"models/{uuid.uuid4()}/model.joblib"
        self.storage.client.fput_object(self.storage.bucket, artifact_path, tmp_path)
        os.remove(tmp_path)

        return artifact_path, metrics

    async def get_active_model(
        self,
        dataset_id: str,
        db: AsyncSession,
    ) -> ModelRegistry | None:
        """Return the single active model for a dataset, or None."""
        result = await db.execute(
            select(ModelRegistry).where(
                ModelRegistry.dataset_id == dataset_id,
                ModelRegistry.status == "active",
            )
        )
        return result.scalars().first()

    async def get_all_models(
        self,
        db: AsyncSession,
    ) -> list[ModelRegistry]:
        """Return all models."""
        result = await db.execute(select(ModelRegistry))
        return result.scalars().all()

    async def activate_model(
        self,
        model_id: str,
        db: AsyncSession,
    ) -> dict:
        """
        Activate a model. Ensures only one active model per dataset:
        1. Look up the target model
        2. Deactivate any currently active model for the same dataset
        3. Set the target model to 'active'
        """
        # Fetch the target model
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalars().first()

        if model is None:
            raise ValueError(f"Model {model_id} not found")

        # Deactivate all active models for this dataset
        await db.execute(
            update(ModelRegistry)
            .where(
                ModelRegistry.dataset_id == model.dataset_id,
                ModelRegistry.status == "active",
            )
            .values(status="inactive")
        )

        # Activate the target model
        model.status = "active"
        db.add(model)
        await db.commit()
        await db.refresh(model)

        return {
            "model_id": str(model.id),
            "dataset_id": str(model.dataset_id),
            "framework": model.framework,
            "model_class": model.model_class,
            "status": model.status,
        }

    async def rollback_to_previous_version(
        self,
        model_id: str,
        db: AsyncSession,
    ) -> dict | None:
        """
        Roll back to the previous model version.
        1. Find all versions ordered by created_at DESC
        2. Locate current version, pick the one after it (= previous)
        3. Update model_registry.current_version
        Returns rollback details dict, or None if rollback impossible.
        """
        # Get model
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalars().first()
        if not model or not model.current_version:
            return None

        # Get all versions ordered newest-first
        result = await db.execute(
            select(ModelVersion)
            .where(ModelVersion.model_id == model_id)
            .order_by(ModelVersion.created_at.desc())
        )
        versions = result.scalars().all()

        # Find current version index
        current_idx = next(
            (i for i, v in enumerate(versions) if v.version_number == model.current_version),
            -1,
        )

        if current_idx == -1 or current_idx + 1 >= len(versions):
            return None  # No previous version available

        previous_version = versions[current_idx + 1]

        # Switch current_version
        old_version = model.current_version
        model.current_version = previous_version.version_number
        db.add(model)
        await db.commit()
        await db.refresh(model)

        return {
            "model_id": model_id,
            "old_version": old_version,
            "new_version": previous_version.version_number,
            "status": "rolled_back",
        }

    # ── Phase 2: Lifecycle methods ───────────────────────────────

    async def register_model_v2(
        self,
        dataset_id: str,
        framework: str,
        model_class: str,
        db: AsyncSession,
    ) -> dict:
        """Register a new model entry (without auto-training)."""

        model_id = str(uuid.uuid4())
        model = ModelRegistry(
            id=model_id,
            dataset_id=dataset_id,
            framework=framework,
            model_class=model_class,
            current_version=None,
            status="staging",
        )
        db.add(model)

        await self._log_lifecycle_event(
            db,
            model_id=model_id,
            action=LifecycleAction.MODEL_REGISTERED,
            reason="New model registered via lifecycle API",
            metadata={"dataset_id": dataset_id, "framework": framework, "model_class": model_class},
        )

        await db.commit()

        return {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "framework": framework,
            "model_class": model_class,
            "status": "staging",
            "message": "Model registered successfully",
        }

    async def add_version(
        self,
        model_id: str,
        version_number: str,
        artifact_path: str,
        db: AsyncSession,
        metrics: dict | None = None,
        training_job_id: str | None = None,
        parent_version: str | None = None,
    ) -> dict:
        """Add a new version to an existing model."""

        # Verify model exists
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalars().first()
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        # Check for duplicate version
        result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.model_id == model_id,
                ModelVersion.version_number == version_number,
            )
        )
        if result.scalars().first() is not None:
            raise ValueError(
                f"Version {version_number} already exists for model {model_id}"
            )

        version_id = str(uuid.uuid4())
        version = ModelVersion(
            id=version_id,
            model_id=model_id,
            version_number=version_number,
            artifact_path=artifact_path,
            metrics=metrics,
            training_job_id=training_job_id,
            parent_version=parent_version,
        )
        db.add(version)

        # Update current_version on the registry
        model.current_version = version_number
        db.add(model)

        await self._log_lifecycle_event(
            db,
            model_id=model_id,
            action=LifecycleAction.VERSION_ADDED,
            reason=f"Version {version_number} added",
            metadata={
                "version_number": version_number,
                "artifact_path": artifact_path,
                "metrics": metrics,
                "parent_version": parent_version,
            },
        )

        await db.commit()

        return {
            "version_id": version_id,
            "model_id": model_id,
            "version_number": version_number,
            "artifact_path": artifact_path,
            "metrics": metrics,
            "training_job_id": training_job_id,
            "parent_version": parent_version,
        }

    async def promote_version(
        self,
        model_id: str,
        version: str,
        db: AsyncSession,
    ) -> dict:
        """Promote a specific version to production."""

        # Verify model
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalars().first()
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        # Verify version exists
        result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.model_id == model_id,
                ModelVersion.version_number == version,
            )
        )
        if result.scalars().first() is None:
            raise ValueError(
                f"Version {version} not found for model {model_id}"
            )

        previous_version = model.current_version
        model.current_version = version
        model.status = "production"
        db.add(model)

        await self._log_lifecycle_event(
            db,
            model_id=model_id,
            action=LifecycleAction.PROMOTED,
            reason=f"Version {version} promoted to production",
            metadata={
                "promoted_version": version,
                "previous_version": previous_version,
            },
        )

        await db.commit()
        await db.refresh(model)

        return {
            "model_id": model_id,
            "promoted_version": version,
            "previous_version": previous_version,
            "status": "production",
            "message": f"Version {version} is now in production",
        }

    async def rollback_to_version(
        self,
        model_id: str,
        target_version: str,
        db: AsyncSession,
    ) -> dict:
        """Rollback to a specific named version."""

        # Verify model
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalars().first()
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        # Verify target version exists
        result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.model_id == model_id,
                ModelVersion.version_number == target_version,
            )
        )
        if result.scalars().first() is None:
            raise ValueError(
                f"Version {target_version} not found for model {model_id}"
            )

        previous_version = model.current_version
        if previous_version == target_version:
            raise ValueError(f"Model is already on version {target_version}")

        model.current_version = target_version
        model.status = "production"
        db.add(model)

        await self._log_lifecycle_event(
            db,
            model_id=model_id,
            action=LifecycleAction.ROLLBACK,
            reason=f"Rolled back from {previous_version} to {target_version}",
            metadata={
                "rolled_back_to": target_version,
                "previous_version": previous_version,
            },
        )

        await db.commit()
        await db.refresh(model)

        return {
            "model_id": model_id,
            "rolled_back_to": target_version,
            "previous_version": previous_version,
            "status": "production",
            "message": f"Rolled back to version {target_version}",
        }

    async def get_model_detail(
        self,
        model_id: str,
        db: AsyncSession,
    ) -> dict:
        """Return full model info including all versions."""

        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalars().first()
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        result = await db.execute(
            select(ModelVersion)
            .where(ModelVersion.model_id == model_id)
            .order_by(ModelVersion.created_at.desc())
        )
        versions = result.scalars().all()

        return {
            "model_id": str(model.id),
            "dataset_id": str(model.dataset_id),
            "framework": model.framework,
            "model_class": model.model_class,
            "current_version": model.current_version,
            "status": model.status,
            "created_at": str(model.created_at) if model.created_at else None,
            "versions": [
                {
                    "version_id": str(v.id),
                    "model_id": str(v.model_id),
                    "version_number": v.version_number,
                    "artifact_path": v.artifact_path,
                    "metrics": v.metrics,
                    "hyperparameters": getattr(v, "hyperparameters", None),
                    "training_job_id": str(v.training_job_id) if v.training_job_id else None,
                    "parent_version": v.parent_version,
                    "created_at": str(v.created_at) if v.created_at else None,
                }
                for v in versions
            ],
        }

    async def get_model_versions(
        self,
        model_id: str,
        db: AsyncSession,
    ) -> list[dict]:
        """Return all versions for a model."""

        # Verify model exists
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        if result.scalars().first() is None:
            raise ValueError(f"Model {model_id} not found")

        result = await db.execute(
            select(ModelVersion)
            .where(ModelVersion.model_id == model_id)
            .order_by(ModelVersion.created_at.desc())
        )
        versions = result.scalars().all()

        return [
            {
                "version_id": str(v.id),
                "model_id": str(v.model_id),
                "version_number": v.version_number,
                "artifact_path": v.artifact_path,
                "metrics": v.metrics,
                "hyperparameters": getattr(v, "hyperparameters", None),
                "training_job_id": str(v.training_job_id) if v.training_job_id else None,
                "parent_version": v.parent_version,
                "created_at": str(v.created_at) if v.created_at else None,
            }
            for v in versions
        ]

    async def archive_model(
        self,
        model_id: str,
        db: AsyncSession,
    ) -> dict:
        """Set model status to archived."""

        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalars().first()
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        model.status = "archived"
        db.add(model)

        await self._log_lifecycle_event(
            db,
            model_id=model_id,
            action=LifecycleAction.ARCHIVED,
            reason="Model archived",
            metadata={"previous_status": model.status},
        )

        await db.commit()

        return {
            "model_id": model_id,
            "status": "archived",
            "message": "Model archived successfully",
        }

    async def delete_model(
        self,
        model_id: str,
        db: AsyncSession,
    ) -> dict:
        """
        Delete a model and all its associated records and artifacts.
        """
        from sqlalchemy import delete
        from app.models.training_job import TrainingJob
        from app.models.monitoring_metric import MonitoringMetric
        from app.models.model_version import ModelVersion
        from app.models.inference_log import InferenceLog
        from app.models.automation_state import AutomationState
        from app.models.automation_log import AutomationLog

        # 1. Verify model exists
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalars().first()
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        # 2. Get all model versions to delete artifacts from MinIO
        result = await db.execute(
            select(ModelVersion).where(ModelVersion.model_id == model_id)
        )
        versions = result.scalars().all()
        for version in versions:
            if version.artifact_path:
                try:
                    self.storage.client.remove_object(self.storage.bucket, version.artifact_path)
                except Exception as e:
                    logger.warning(f"Failed to delete artifact {version.artifact_path}: {e}")

        # 3. Cascade deletes in the correct order (child tables first)
        
        # automation logs
        await db.execute(delete(AutomationLog).where(AutomationLog.model_id == model_id))
        
        # automation state
        await db.execute(delete(AutomationState).where(AutomationState.model_id == model_id))
        
        # inference logs
        await db.execute(delete(InferenceLog).where(InferenceLog.model_id == model_id))
        
        # monitoring metrics
        await db.execute(delete(MonitoringMetric).where(MonitoringMetric.model_id == model_id))
        
        # model versions
        await db.execute(delete(ModelVersion).where(ModelVersion.model_id == model_id))
        
        # training jobs
        await db.execute(delete(TrainingJob).where(TrainingJob.model_id == model_id))

        # 4. Finally delete the model registry entry
        await db.execute(delete(ModelRegistry).where(ModelRegistry.id == model_id))

        await db.commit()

        return {
            "model_id": model_id,
            "status": "deleted",
            "message": "Model and associated data deleted successfully",
        }

    # ── Internal helper ──────────────────────────────────────────

    async def _log_lifecycle_event(
        self,
        db: AsyncSession,
        model_id: str,
        action: LifecycleAction,
        reason: str,
        metadata: dict | None = None,
    ) -> None:
        """Write a lifecycle event to the automation_logs table."""

        log = AutomationLog(
            id=uuid.uuid4(),
            model_id=model_id,
            action=action.value,
            reason=reason,
            log_metadata=metadata,
            status="completed",
        )
        db.add(log)
        logger.info("Lifecycle event: %s for model %s — %s", action.value, model_id, reason)
