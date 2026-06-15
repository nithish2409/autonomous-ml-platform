import uuid
from sqlalchemy import Column, String, DateTime, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from app.core.database import Base


class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("model_registry.id"), nullable=False)
    input_summary = Column(JSONB)
    prediction_summary = Column(JSONB)
    latency_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
