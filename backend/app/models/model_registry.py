import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from app.core.database import Base


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    framework = Column(String, nullable=False)
    model_class = Column(String, nullable=False)
    current_version = Column(String, nullable=True)
    status = Column(String, default="inactive")
    model_metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
