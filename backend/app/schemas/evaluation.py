from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from app.schemas.dataset import DatasetRowResponse


class EvalTaskCreate(BaseModel):
    name: str
    dataset_id: int
    scenario_id: int
    llm_config_id: int


class EvalTaskBrief(BaseModel):
    id: int
    name: str
    dataset_id: int
    scenario_id: int
    llm_config_id: int
    status: str
    progress: float
    total_rows: Optional[int] = None
    completed_rows: int
    summary_scores: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime
    scenario_snapshot: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class DatasetBrief(BaseModel):
    id: int
    name: str
    sample_type: str
    row_count: int
    model_config = ConfigDict(from_attributes=True)


class ScenarioBrief(BaseModel):
    id: int
    name: str
    scene_type: str
    model_config = ConfigDict(from_attributes=True)


class LLMConfigBrief(BaseModel):
    id: int
    name: str
    model_name: str
    model_config = ConfigDict(from_attributes=True)


class EvalTaskResponse(EvalTaskBrief):
    dataset: Optional[DatasetBrief] = None
    scenario: Optional[ScenarioBrief] = None
    llm_config: Optional[LLMConfigBrief] = None


class EvalRowResultResponse(BaseModel):
    id: int
    eval_task_id: int
    row_index: int
    metric_scores: Optional[Dict[str, Any]] = None
    is_pass: Optional[bool] = None
    execution_time_ms: Optional[int] = None
    error: Optional[str] = None
    dataset_row: Optional[DatasetRowResponse] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ReportSummary(BaseModel):
    eval_task: EvalTaskResponse
    total_count: int
    pass_count: int
    fail_count: int
    error_count: int
    pass_rate: float
    metric_summary: Optional[Dict[str, Any]] = None


class ReportRowsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[EvalRowResultResponse]
