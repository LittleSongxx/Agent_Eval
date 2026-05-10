from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from app.schemas.metric_definition import MetricResponse


class ScenarioMetricConfig(BaseModel):
    metric_definition_id: int
    weight: float = 1.0
    pass_threshold: Optional[float] = None
    prompt_override: Optional[str] = None


class ScenarioCreate(BaseModel):
    name: str
    description: str = ""
    scene_type: str
    sample_type: str
    metrics: List[ScenarioMetricConfig] = []


class ScenarioUpdate(ScenarioCreate):
    pass


class ScenarioMetricResponse(BaseModel):
    id: int
    metric_definition_id: int
    weight: float
    pass_threshold: Optional[float] = None
    prompt_override: Optional[str] = None
    metric_definition: Optional[MetricResponse] = None

    model_config = ConfigDict(from_attributes=True)


class ScenarioResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    scene_type: str
    sample_type: str
    is_preset: bool
    metrics: List[ScenarioMetricResponse] = []
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
