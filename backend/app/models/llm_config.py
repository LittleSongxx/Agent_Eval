from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.sql import func

from app.core.database import Base


class LLMConfig(Base):
    __tablename__ = "llm_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    provider = Column(String(50), default="openai")
    api_base_url = Column(String(500), nullable=False)
    api_key = Column(String(500), nullable=False)
    model_name = Column(String(100), nullable=False)
    temperature = Column(Float, default=0.01)
    max_tokens = Column(Integer, default=1024)
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
