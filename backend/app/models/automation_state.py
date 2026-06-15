from sqlalchemy import Column, DateTime, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.core.database import Base


class AutomationState(Base):
    __tablename__ = "automation_state"

    model_id = Column(UUID(as_uuid=True), ForeignKey("model_registry.id"), primary_key=True)
    last_action = Column(String)
    cooldown_until = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())