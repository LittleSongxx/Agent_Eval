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
    api_key: str
    model_name: str
    temperature: float
    max_tokens: int
    is_default: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class LLMTestResult(BaseModel):
    success: bool
    message: str
    latency_ms: Optional[float] = None
