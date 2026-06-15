"""Policy data model — singleton policy record with JSONB config."""

import uuid
from sqlalchemy import Column, DateTime, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from app.core.database import Base


# Default policy configuration
DEFAULT_POLICY = {
    "auto_approval": {
        "min_confidence": 85,
        "max_cost": 500.0,
        "allowed_severity": ["low", "medium"],
        "block_production": True,
    },
    "guardrails": {
        "max_gpu_per_job": 4,
        "max_daily_cost": 2000.0,
        "max_retrains_24h": 10,
        "freeze_window": False,
    },
    "escalation": {
        "notify_on_critical": True,
        "webhook_url": None,
        "email_alerts": [],
    },
}


class Policy(Base):
    __tablename__ = "policies"

    id = Column(Integer, primary_key=True, default=1)
    config = Column(JSONB, nullable=False, default=DEFAULT_POLICY)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
