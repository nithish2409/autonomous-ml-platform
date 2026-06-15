import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from app.core.database import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("model_registry.id"), nullable=False)
    version_number = Column(String, nullable=False)
    artifact_path = Column(String, nullable=False)
    metrics = Column(JSONB)
    hyperparameters = Column(JSONB, nullable=True)
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id"), nullable=True)
    parent_version = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
