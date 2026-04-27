from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

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
    manual_status: Optional[str] = None
    manual_score: Optional[float] = None
    manual_tags: Optional[List[str]] = None
    manual_note: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    dataset_row: Optional[DatasetRowResponse] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class EvalRowReviewUpdate(BaseModel):
    manual_status: Optional[Literal["pass", "fail", "needs_fix", "needs_review"]] = None
    manual_score: Optional[float] = None
    manual_tags: Optional[List[str]] = None
    manual_note: Optional[str] = None


class ReportSummary(BaseModel):
    eval_task: EvalTaskResponse
    total_count: int
    pass_count: int
    fail_count: int
    error_count: int
    pass_rate: float
    metric_summary: Optional[Dict[str, Any]] = None
    manual_review_summary: Optional[Dict[str, int]] = None


class ReportRowsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[EvalRowResultResponse]


class BlindTestTargetPayload(BaseModel):
    target_type: Literal["llm_config", "endpoint"]
    name: str
    llm_config_id: Optional[int] = None
    endpoint_url: Optional[str] = None
    transport_mode: Literal["json", "sse"] = "json"
    authorization: Optional[str] = None
    extra_headers: Optional[str] = None
    request_body_template: Optional[str] = None
    response_mode: Literal["answer_only", "answer_with_contexts"] = "answer_only"


class BlindTestTargetTestRequest(BaseModel):
    target: BlindTestTargetPayload
    test_question: str


class BlindTestTargetTestResponse(BaseModel):
    success: bool
    message: str
    answer_preview: Optional[str] = None


class BlindTestTaskCreate(BaseModel):
    name: str
    dataset_id: int
    sample_limit: Optional[int] = None
    target_a: BlindTestTargetPayload
    target_b: BlindTestTargetPayload


class BlindTestTaskBrief(BaseModel):
    id: int
    name: str
    dataset_id: int
    status: str
    progress: float
    total_rows: Optional[int] = None
    completed_rows: int
    voted_rows: int
    sample_limit: Optional[int] = None
    error_message: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    target_a: Dict[str, Any]
    target_b: Dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class BlindTestTaskResponse(BlindTestTaskBrief):
    dataset: Optional[DatasetBrief] = None


class BlindTestVoteUpdate(BaseModel):
    vote: Literal["left", "right", "tie", "both_bad", "skip"]
    vote_note: Optional[str] = None


class BlindTestRowResultResponse(BaseModel):
    id: int
    blind_test_task_id: int
    dataset_row_id: int
    row_index: int
    answer_a: Optional[str] = None
    answer_b: Optional[str] = None
    answer_a_error: Optional[str] = None
    answer_b_error: Optional[str] = None
    display_order: List[str]
    vote: Optional[str] = None
    vote_note: Optional[str] = None
    voted_at: Optional[datetime] = None
    created_at: datetime
    dataset_row: Optional[DatasetRowResponse] = None

    model_config = ConfigDict(from_attributes=True)


class BlindTestRowsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[BlindTestRowResultResponse]


class BlindTestSummaryResponse(BaseModel):
    blind_test_task: BlindTestTaskResponse
    total_count: int
    completed_count: int
    voted_count: int
    pending_vote_count: int
    model_a_wins: int
    model_b_wins: int
    ties: int
    both_bad: int
    no_answer_rows: int
