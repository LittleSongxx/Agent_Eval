from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from app.schemas.dataset import DatasetRowResponse
from app.schemas.sensitive import redact_sensitive_mapping


class MetricOverridePayload(BaseModel):
    metric_definition_id: int
    prompt_override: Optional[str] = None
    pass_threshold: Optional[float] = None
    weight: Optional[float] = None


class EvalTaskCreate(BaseModel):
    name: str
    dataset_id: int
    scenario_id: int
    llm_config_id: int
    evaluation_mode: Literal["offline", "endpoint"] = "offline"
    endpoint_target_id: Optional[int] = None
    target_config: Optional[Dict[str, Any]] = None
    response_mapping: Optional[Dict[str, Any]] = None
    result_save_mode: Literal["task_only", "write_back"] = "task_only"
    metric_overrides: Optional[List[MetricOverridePayload]] = None


class EvalDebugRequest(EvalTaskCreate):
    row_id: Optional[int] = None


class EvalDebugResponse(BaseModel):
    success: bool
    message: str
    dataset_row: Optional[DatasetRowResponse] = None
    row_data: Optional[Dict[str, Any]] = None
    endpoint_trace: Optional[Dict[str, Any]] = None
    metric_scores: Dict[str, Any] = {}
    judge_traces: List[Dict[str, Any]] = []
    is_pass: Optional[bool] = None
    execution_time_ms: Optional[int] = None
    warnings: List[str] = []
    errors: List[str] = []


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
    evaluation_mode: Optional[str] = "offline"
    endpoint_target_id: Optional[int] = None
    target_config: Optional[Dict[str, Any]] = None
    response_mapping: Optional[Dict[str, Any]] = None
    result_save_mode: Optional[str] = "task_only"

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="after")
    def hide_target_config_secrets(self):
        self.target_config = redact_sensitive_mapping(self.target_config)
        return self


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
    endpoint_trace: Optional[Dict[str, Any]] = None
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
    # 平台级聚合结果（从 metric_summary 中剥离的保留键）
    weighted_total_score: Optional[Dict[str, Any]] = None
    cost: Optional[Dict[str, Any]] = None
    judge_reliability: Optional[Dict[str, Any]] = None
    # 人工复核与自动评分的一致性（自动+人工双通道交叉验证）
    manual_auto_agreement_rate: Optional[float] = None
    manual_auto_disagreement_count: int = 0


class ReportRowsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[EvalRowResultResponse]


class ReportListItem(BaseModel):
    eval_id: int
    task_name: str
    dataset_id: int
    dataset_name: Optional[str] = None
    scenario_id: int
    scenario_name: Optional[str] = None
    evaluation_mode: Optional[str] = "offline"
    endpoint_target_id: Optional[int] = None
    endpoint_name: Optional[str] = None
    status: str
    total_count: int
    pass_count: int
    fail_count: int
    error_count: int
    pass_rate: float
    progress: float
    completed_rows: int
    total_rows: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class ReportListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[ReportListItem]


class ReportCompareMetricDelta(BaseModel):
    metric: str
    current_mean: Optional[float] = None
    baseline_mean: Optional[float] = None
    mean_delta: Optional[float] = None
    current_pass_rate: Optional[float] = None
    baseline_pass_rate: Optional[float] = None
    pass_rate_delta: Optional[float] = None
    current_error_count: int = 0
    baseline_error_count: int = 0


class ReportCompareRowItem(BaseModel):
    dataset_row_id: int
    row_index: int
    current_result_id: Optional[int] = None
    baseline_result_id: Optional[int] = None
    current_status: Optional[str] = None
    baseline_status: Optional[str] = None
    metric_deltas: Dict[str, Any] = {}
    dataset_row: Optional[DatasetRowResponse] = None


class ReportCompareSummaryDelta(BaseModel):
    current_pass_rate: float
    baseline_pass_rate: float
    pass_rate_delta: float
    current_fail_count: int
    baseline_fail_count: int
    fail_count_delta: int
    current_error_count: int
    baseline_error_count: int
    error_count_delta: int


class ReportCompareResponse(BaseModel):
    current_eval: EvalTaskResponse
    baseline_eval: EvalTaskResponse
    summary_delta: ReportCompareSummaryDelta
    metric_deltas: List[ReportCompareMetricDelta]
    row_changes: Dict[str, List[ReportCompareRowItem]]


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


class EndpointEvalTargetTestRequest(BaseModel):
    endpoint_target_id: Optional[int] = None
    target_config: Dict[str, Any]
    response_mapping: Optional[Dict[str, Any]] = None
    row_data: Dict[str, Any]


class EndpointEvalTargetTestResponse(BaseModel):
    success: bool
    message: str
    request_body: Optional[Dict[str, Any]] = None
    status_code: Optional[int] = None
    latency_ms: Optional[int] = None
    raw_response: Optional[str] = None
    extracted_fields: Optional[Dict[str, Any]] = None
    mapping_errors: Optional[Dict[str, str]] = None


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

    @model_validator(mode="after")
    def hide_target_secrets(self):
        self.target_a = redact_sensitive_mapping(self.target_a)
        self.target_b = redact_sensitive_mapping(self.target_b)
        return self


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
