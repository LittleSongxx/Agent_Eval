from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ToolDefinitionBase(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    description: str = ""
    parameters_schema: Optional[Dict[str, Any]] = None
    risk_level: str = Field(default="low", pattern="^(low|medium|high|critical)$")
    has_side_effect: bool = False
    idempotency_required: bool = False
    timeout_ms: Optional[int] = Field(default=None, ge=1, le=600000)
    enabled: bool = True


class ToolDefinitionCreate(ToolDefinitionBase):
    pass


class ToolDefinitionUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=150)
    description: Optional[str] = None
    parameters_schema: Optional[Dict[str, Any]] = None
    risk_level: Optional[str] = Field(default=None, pattern="^(low|medium|high|critical)$")
    has_side_effect: Optional[bool] = None
    idempotency_required: Optional[bool] = None
    timeout_ms: Optional[int] = Field(default=None, ge=1, le=600000)
    enabled: Optional[bool] = None


class ToolDefinitionResponse(ToolDefinitionBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class TraceLintRequest(BaseModel):
    trajectory: list[Dict[str, Any]]
    available_tools: Optional[Dict[str, Any]] = None
    tool_registry: Optional[list[Dict[str, Any]]] = None
    max_steps: int = Field(default=20, ge=1, le=200)


class TraceLintResponse(BaseModel):
    passed: bool
    score: float
    violations: list[Dict[str, Any]]
    checked_steps: int
