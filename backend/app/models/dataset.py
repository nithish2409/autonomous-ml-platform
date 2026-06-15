import uuid
from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from app.core.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    schema = Column(JSONB)
    baseline_stats = Column(JSONB)
    minio_path = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
