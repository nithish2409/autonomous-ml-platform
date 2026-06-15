import json
import uuid
import os
import logging

import docker

from app.core.retry import sync_retry
from app.core.metrics import training_jobs_total
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func as sa_func

from app.models.model_registry import ModelRegistry
from app.models.model_version import ModelVersion
from app.models.training_job import TrainingJob
from app.models.dataset import Dataset
from app.models.automation_log import AutomationLog

logger = logging.getLogger("training_orchestrator")

DOCKER_NETWORK = "autonomous-ml-platform_default"
TRAINING_IMAGE_TAG = "ml-training:latest"


class TrainingOrchestrator:
    """Orchestrates ML training in isolated Docker containers."""

    def __init__(self):
        self.docker_client = docker.from_env()

    # ── Docker helpers ───────────────────────────────────────────

    @sync_retry(max_retries=2, base_delay=2.0, exceptions=(docker.errors.BuildError, docker.errors.APIError))
    def build_training_image(self, framework: str) -> str:
        """Build the training Docker image. Returns the image tag."""
        build_path = "/app/training"

        image, _ = self.docker_client.images.build(
            path=build_path,
            tag=TRAINING_IMAGE_TAG,
            rm=True,
        )
        return TRAINING_IMAGE_TAG

    @sync_retry(max_retries=2, base_delay=3.0, exceptions=(docker.errors.APIError,))
    def run_training_container(
        self,
        model_id: str,
        dataset_minio_path: str,
        framework: str,
        model_class: str,
        version: str,
        target_column: str | None = None,
        hyperparameters: dict | None = None,
        split_ratio: float = 0.2,
        random_seed: int = 42,
    ) -> tuple[dict, str]:
        """
        Run the training container with env vars. Waits for completion,
        parses the JSON output from stdout.
        Returns (output_dict, full_logs_str).
        """
        env_vars = {
            "MINIO_ENDPOINT": os.getenv("MINIO_ENDPOINT"),
            "MINIO_ACCESS_KEY": os.getenv("MINIO_ACCESS_KEY"),
            "MINIO_SECRET_KEY": os.getenv("MINIO_SECRET_KEY"),
            "MINIO_BUCKET": os.getenv("MINIO_BUCKET"),
            "DATASET_PATH": dataset_minio_path,
            "MODEL_ID": model_id,
            "VERSION": version,
            "FRAMEWORK": framework,
            "MODEL_CLASS": model_class,
            "SPLIT_RATIO": str(split_ratio),
            "RANDOM_SEED": str(random_seed),
        }

        if target_column:
            env_vars["TARGET_COLUMN"] = target_column
        if hyperparameters:
            env_vars["HYPERPARAMETERS"] = json.dumps(hyperparameters)

        container = self.docker_client.containers.run(
            image=TRAINING_IMAGE_TAG,
            environment=env_vars,
            network=DOCKER_NETWORK,
            detach=True,
            remove=False,
            mem_limit="2g",
            cpu_period=100000,
            cpu_quota=200000,  # 2 CPUs max
        )

        # Wait for container to finish
        result = container.wait()
        logs = container.logs().decode("utf-8").strip()
        exit_code = result.get("StatusCode", -1)

        # Clean up container
        container.remove()

        if exit_code != 0:
            raise RuntimeError(f"Training container failed (exit {exit_code}): {logs}")

        # Parse the last line as JSON (training script prints result as last line)
        last_line = logs.strip().split("\n")[-1]
        try:
            output = json.loads(last_line)
        except json.JSONDecodeError:
            raise RuntimeError(f"Could not parse training output: {logs}")

        return output, logs

    # ── Original train method (backward compatible) ──────────────

    async def train(self, model_id: str, db: AsyncSession) -> dict:
        """
        Full training orchestration (legacy endpoint):
        1. Look up model and dataset
        2. Compute next version
        3. Create TrainingJob (pending)
        4. Build image + run container
        5. Record ModelVersion + update TrainingJob + update ModelRegistry
        """
        # Look up model
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalars().first()
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        # Look up dataset to get minio_path
        result = await db.execute(
            select(Dataset).where(Dataset.id == model.dataset_id)
        )
        dataset = result.scalars().first()
        if dataset is None:
            raise ValueError(f"Dataset {model.dataset_id} not found")

        if not dataset.minio_path:
            raise ValueError(f"Dataset {model.dataset_id} has no minio_path")

        # Compute next version number
        result = await db.execute(
            select(sa_func.count(ModelVersion.id)).where(
                ModelVersion.model_id == model_id
            )
        )
        version_count = result.scalar() or 0
        version = f"v{version_count + 1}"

        # Create training job record
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            model_id=model_id,
            config={
                "framework": model.framework,
                "model_class": model.model_class,
                "dataset_path": dataset.minio_path,
                "version": version,
            },
            status="running",
        )
        db.add(job)
        await db.commit()

        # Build image and run training container
        try:
            self.build_training_image(model.framework)
            output, _logs = self.run_training_container(
                model_id=str(model.id),
                dataset_minio_path=dataset.minio_path,
                framework=model.framework,
                model_class=model.model_class,
                version=version,
            )
        except Exception as e:
            # Mark job as failed
            job.status = "failed"
            job.result_metrics = {"error": str(e)}
            training_jobs_total.labels(status="failed").inc()
            await db.commit()
            raise

        # Record model version
        model_version = ModelVersion(
            id=str(uuid.uuid4()),
            model_id=model_id,
            version_number=version,
            artifact_path=output.get("artifact_path", ""),
            metrics=output.get("metrics"),
            training_job_id=job_id,
            parent_version=model.current_version,
        )
        db.add(model_version)

        # Update training job
        job.status = "completed"
        job.result_metrics = output.get("metrics")

        # Update model registry current_version
        model.current_version = version

        training_jobs_total.labels(status="completed").inc()
        await db.commit()

        return {
            "job_id": job_id,
            "model_id": str(model.id),
            "version": version,
            "status": "completed",
            "artifact_path": output.get("artifact_path"),
            "metrics": output.get("metrics"),
        }

    # ── Phase 3: Enhanced training ───────────────────────────────

    async def start_training(
        self,
        model_id: str,
        db: AsyncSession,
        target_column: str | None = None,
        hyperparameters: dict | None = None,
        split_ratio: float = 0.2,
        random_seed: int = 42,
    ) -> dict:
        """
        Launch a training job with full configuration:
        1. Look up model + dataset
        2. Compute next version
        3. Create TrainingJob (queued → running → completed/failed)
        4. Build image + run container with config
        5. Capture logs → automation_logs
        6. Register model version with lineage
        """
        # Look up model
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalars().first()
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        # Look up dataset
        result = await db.execute(
            select(Dataset).where(Dataset.id == model.dataset_id)
        )
        dataset = result.scalars().first()
        if dataset is None:
            raise ValueError(f"Dataset {model.dataset_id} not found")
        if not dataset.minio_path:
            raise ValueError(f"Dataset {model.dataset_id} has no minio_path")

        # Compute next version
        result = await db.execute(
            select(sa_func.count(ModelVersion.id)).where(
                ModelVersion.model_id == model_id
            )
        )
        version_count = result.scalar() or 0
        version = f"v{version_count + 1}"
        previous_version = model.current_version

        # Create training job (queued)
        job_id = str(uuid.uuid4())
        config = {
            "framework": model.framework,
            "model_class": model.model_class,
            "dataset_path": dataset.minio_path,
            "version": version,
            "target_column": target_column,
            "hyperparameters": hyperparameters,
            "split_ratio": split_ratio,
            "random_seed": random_seed,
        }
        job = TrainingJob(
            id=job_id,
            model_id=model_id,
            config=config,
            status="queued",
        )
        db.add(job)
        await db.commit()
        logger.info("Training job %s queued for model %s", job_id, model_id)

        # Transition to running
        job.status = "running"
        await db.commit()

        # Build image and run container
        container_logs = ""
        try:
            self.build_training_image(model.framework)
            output, container_logs = self.run_training_container(
                model_id=str(model.id),
                dataset_minio_path=dataset.minio_path,
                framework=model.framework,
                model_class=model.model_class,
                version=version,
                target_column=target_column,
                hyperparameters=hyperparameters,
                split_ratio=split_ratio,
                random_seed=random_seed,
            )
        except Exception as e:
            job.status = "failed"
            job.result_metrics = {"error": str(e)}
            training_jobs_total.labels(status="failed").inc()

            # Log failure
            await self._log_training_event(
                db, model_id=model_id, job_id=job_id,
                action="TRAINING_FAILED",
                reason=str(e),
                metadata={"config": config, "logs": container_logs[-2000:] if container_logs else None},
            )
            await db.commit()
            raise

        # Success: record model version with lineage
        model_version = ModelVersion(
            id=str(uuid.uuid4()),
            model_id=model_id,
            version_number=version,
            artifact_path=output.get("artifact_path", ""),
            metrics=output.get("metrics"),
            training_job_id=job_id,
            parent_version=previous_version,
        )
        db.add(model_version)

        # Update training job
        job.status = "completed"
        metrics_out = output.get("metrics", {})
        job.result_metrics = metrics_out

        # Merge auto-detected values back into config
        updated_config = dict(job.config)
        updated_config["target_column"] = metrics_out.get("target_column", updated_config.get("target_column"))
        updated_config["model_class"] = metrics_out.get("model_class_used", updated_config.get("model_class"))
        updated_config["hyperparameters"] = metrics_out.get("hyperparameters", updated_config.get("hyperparameters") or {})
        job.config = updated_config

        # Update model registry
        model.current_version = version

        training_jobs_total.labels(status="completed").inc()

        # Log success
        await self._log_training_event(
            db, model_id=model_id, job_id=job_id,
            action="TRAINING_COMPLETED",
            reason=f"Version {version} trained successfully",
            metadata={
                "version": version,
                "artifact_path": output.get("artifact_path"),
                "metrics": output.get("metrics"),
                "logs": container_logs[-2000:] if container_logs else None,
            },
        )

        await db.commit()
        logger.info("Training job %s completed for model %s (version %s)", job_id, model_id, version)

        return {
            "job_id": job_id,
            "model_id": str(model.id),
            "version": version,
            "status": "completed",
            "framework": model.framework,
            "model_class": model.model_class,
            "artifact_path": output.get("artifact_path"),
            "metrics": output.get("metrics"),
            "message": f"Training completed. Version {version} registered.",
        }

    async def get_job_status(self, job_id: str, db: AsyncSession) -> dict:
        """Get current status and details of a training job."""

        result = await db.execute(
            select(TrainingJob).where(TrainingJob.id == job_id)
        )
        job = result.scalars().first()
        if job is None:
            raise ValueError(f"Training job {job_id} not found")

        return {
            "job_id": str(job.id),
            "model_id": str(job.model_id),
            "status": job.status,
            "config": job.config,
            "metrics": job.result_metrics,
            "created_at": str(job.created_at) if job.created_at else None,
        }

    async def get_job_logs(self, job_id: str, db: AsyncSession) -> dict:
        """Get training logs for a job from automation_logs."""

        # Verify job exists
        result = await db.execute(
            select(TrainingJob).where(TrainingJob.id == job_id)
        )
        if result.scalars().first() is None:
            raise ValueError(f"Training job {job_id} not found")

        # Fetch logs from automation_logs
        result = await db.execute(
            select(AutomationLog).where(
                AutomationLog.log_metadata.contains({"job_id": job_id})
            ).order_by(AutomationLog.created_at.desc())
        )
        log_entries = result.scalars().all()

        logs = []
        for entry in log_entries:
            meta = entry.log_metadata or {}
            raw_logs = meta.get("logs", "")
            if raw_logs:
                logs.extend(raw_logs.strip().split("\n"))
            else:
                logs.append(f"[{entry.action}] {entry.reason}")

        return {
            "job_id": job_id,
            "logs": logs,
        }

    async def delete_job(self, job_id: str, db: AsyncSession) -> None:
        """Delete a training job and its associated logs."""
        # 1. Ensure job exists
        result = await db.execute(
            select(TrainingJob).where(TrainingJob.id == job_id)
        )
        job = result.scalars().first()
        if job is None:
            raise ValueError(f"Training job {job_id} not found")
            
        # 2. Cleanup logs linked to this job
        log_result = await db.execute(
            select(AutomationLog).where(
                AutomationLog.log_metadata.contains({"job_id": job_id})
            )
        )
        logs_to_delete = log_result.scalars().all()
        for log_entry in logs_to_delete:
            await db.delete(log_entry)
            
        # 3. Delete job
        await db.delete(job)
        await db.commit()
        logger.info("Deleted training job %s", job_id)

    # ── Internal helper ──────────────────────────────────────────

    async def _log_training_event(
        self,
        db: AsyncSession,
        model_id: str,
        job_id: str,
        action: str,
        reason: str,
        metadata: dict | None = None,
    ) -> None:
        """Write a training event to the automation_logs table."""

        meta = metadata or {}
        meta["job_id"] = job_id

        log = AutomationLog(
            id=uuid.uuid4(),
            model_id=model_id,
            action=action,
            reason=reason,
            log_metadata=meta,
            status="completed",
        )
        db.add(log)
        logger.info("Training event: %s for model %s job %s", action, model_id, job_id)
