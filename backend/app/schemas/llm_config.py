from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class LLMConfigCreate(BaseModel):
    name: str
    provider: str = "openai"
    api_base_url: str
    api_key: str
    model_name: str
    temperature: float = 0.01
    max_tokens: int = 1024
    is_default: bool = False


class LLMConfigUpdate(BaseModel):
    name: Optional[str] = None
    provider: Optional[str] = None
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    is_default: Optional[bool] = None


class LLMConfigResponse(BaseModel):
    id: int
    name: str
    provider: str
    api_base_url: str
    api_key_masked: str = ""
    model_name: str
    temperature: float
    max_tokens: int
    is_default: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_orm_with_mask(cls, obj):
        key = obj.api_key or ""
        masked = key[:3] + "****" + key[-4:] if len(key) > 8 else "****"
        data = {c.name: getattr(obj, c.name) for c in obj.__table__.columns}
        data["api_key_masked"] = masked
        return cls(**data)


class LLMTestResult(BaseModel):
    success: bool
    message: str
    latency_ms: Optional[float] = None
