import uuid
from sqlalchemy import Column, DateTime, Float, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from app.core.database import Base


class MonitoringMetric(Base):
    __tablename__ = "monitoring_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("model_registry.id"))
    drift_score = Column(Float)
    performance_delta = Column(Float)
    request_count = Column(Integer)
    latency_avg = Column(Float)
    details = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())