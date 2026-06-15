import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from app.core.database import engine, Base
from app.core.storage import MinioClient
from app.core.logging_config import configure_logging
from prometheus_client import make_asgi_app as make_prometheus_app
from app.api import (
    datasets,
    models,
    training,
    inference,
    system,
    monitoring,
    automation,
    serving,
    model_lifecycle,
    policies,
)
from app.scheduler.monitoring_scheduler import start_scheduler, stop_scheduler
from app.models import (
    dataset,
    model_registry,
    model_version,
    monitoring_metric,
    training_job,
    automation_log,
    automation_state,
    inference_log,
    policy,
)

from fastapi.middleware.cors import CORSMiddleware

configure_logging()

app = FastAPI(title="Autonomous ML Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Phase 2 migration: add lineage columns to model_versions if missing
        await conn.execute(
            text(
                "ALTER TABLE model_versions "
                "ADD COLUMN IF NOT EXISTS training_job_id UUID REFERENCES training_jobs(id)"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE model_versions "
                "ADD COLUMN IF NOT EXISTS parent_version VARCHAR"
            )
        )
        # Phase 5 migration: add system metrics columns to monitoring_metrics
        await conn.execute(
            text(
                "ALTER TABLE monitoring_metrics "
                "ADD COLUMN IF NOT EXISTS request_count INTEGER"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE monitoring_metrics "
                "ADD COLUMN IF NOT EXISTS latency_avg FLOAT"
            )
        )
    minio_client = MinioClient()
    minio_client.ensure_bucket()
    start_scheduler()


@app.on_event("shutdown")
async def shutdown():
    stop_scheduler()


# API routers (registered BEFORE static mount so they take priority)
app.include_router(datasets.router)
app.include_router(models.router)
app.include_router(training.router)
app.include_router(inference.router)
app.include_router(system.router)
app.include_router(monitoring.router)
app.include_router(automation.router)
app.include_router(serving.router)
app.include_router(model_lifecycle.router)
app.include_router(policies.router)

# ── Prometheus /metrics scrape endpoint ──────────────────────────
# Mounted BEFORE the static catch-all so it is never shadowed.
prometheus_app = make_prometheus_app()
app.mount("/metrics", prometheus_app)

# Static frontend (catch-all, must be LAST)
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

