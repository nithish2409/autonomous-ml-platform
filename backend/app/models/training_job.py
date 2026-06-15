import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from app.core.database import Base


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("model_registry.id"))
    config = Column(JSONB)
    status = Column(String, default="pending")
    result_metrics = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())