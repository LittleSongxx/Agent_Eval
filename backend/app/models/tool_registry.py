from sqlalchemy import Boolean, Column, DateTime, Integer, JSON, String, Text
from sqlalchemy.sql import func

from app.core.database import Base


class ToolDefinition(Base):
    """Lightweight catalog of tools available to Agent traces."""

    __tablename__ = "tool_definitions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(150), unique=True, nullable=False)
    description = Column(Text, nullable=False, default="")
    parameters_schema = Column(JSON, nullable=True)
    risk_level = Column(String(20), nullable=False, default="low")
    has_side_effect = Column(Boolean, nullable=False, default=False)
    idempotency_required = Column(Boolean, nullable=False, default=False)
    timeout_ms = Column(Integer, nullable=True)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
