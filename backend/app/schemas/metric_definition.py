from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict


class MetricCreate(BaseModel):
    name: str
    display_name: str
    metric_type: str
    config: Dict = {}
    category: str = "custom"


class MetricResponse(BaseModel):
    id: int
    name: str
    display_name: str
    metric_type: str
    config: Optional[Dict] = None
    category: Optional[str] = None
    is_builtin: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
