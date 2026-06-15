import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from app.core.database import Base


class AutomationLog(Base):
    __tablename__ = "automation_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("model_registry.id"))
    action = Column(String)
    reason = Column(String)
    log_metadata = Column(JSONB)
    execution_result = Column(JSONB)
    status = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())