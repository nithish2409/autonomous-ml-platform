"""
System endpoints — health check, system status, and Prometheus metrics.
"""

import logging
import os

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.storage import MinioClient
from app.scheduler.monitoring_scheduler import scheduler

logger = logging.getLogger("system")

router = APIRouter()


# ── Health ───────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok"}


# ── System status ────────────────────────────────────────────────

@router.get("/system/status")
async def system_status(db: AsyncSession = Depends(get_db)):
    """Check connectivity to DB, MinIO, Ollama, scheduler state, and active models."""

    checks: dict = {}

    # DB
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "connected"
    except Exception as exc:
        checks["database"] = f"error: {exc}"

    # MinIO
    try:
        minio = MinioClient()
        bucket = os.getenv("MINIO_BUCKET", "ml-artifacts")
        exists = minio.client.bucket_exists(bucket)
        checks["minio"] = "connected" if exists else "bucket_missing"
    except Exception as exc:
        checks["minio"] = f"error: {exc}"

    # Ollama
    try:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
        checks["ollama"] = "connected" if resp.status_code == 200 else f"status_{resp.status_code}"
    except Exception as exc:
        checks["ollama"] = f"error: {exc}"

    # Scheduler
    checks["scheduler_running"] = scheduler.running

    # Active models
    try:
        from app.models.model_registry import ModelRegistry
        from sqlalchemy.future import select
        from sqlalchemy import func

        result = await db.execute(
            select(func.count(ModelRegistry.id)).where(ModelRegistry.status == "active")
        )
        checks["active_models"] = result.scalar() or 0
    except Exception as exc:
        checks["active_models"] = f"error: {exc}"

    return checks


# ── Manual Triggers ──────────────────────────────────────────────

from fastapi import BackgroundTasks

@router.post("/system/trigger-monitoring")
async def trigger_monitoring_cycle(background_tasks: BackgroundTasks):
    """Manually trigger the monitoring and automation cycle (useful for demos)."""
    from app.scheduler.monitoring_scheduler import trigger_monitoring_loop
    background_tasks.add_task(trigger_monitoring_loop)
    return {"status": "Monitoring cycle triggered in background"}


# ── Prometheus metrics ───────────────────────────────────────────

@router.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus text exposition endpoint."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
