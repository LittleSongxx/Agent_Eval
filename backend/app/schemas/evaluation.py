from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    # 多裁判面板：附加裁判的 LLM 配置 ID 列表（LLM 指标取均值/多数票聚合）
    judge_llm_config_ids: Optional[List[int]] = None


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
    judge_panel: Optional[List[int]] = None
    judge_snapshot: Optional[Dict[str, Any]] = None
    dataset_version: Optional[int] = None
    tool_registry_snapshot: Optional[List[Dict[str, Any]]] = None
    dataset_snapshot_digest: Optional[str] = None
    eval_fingerprint: Optional[str] = None
    judge_samples: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="after")
    def hide_target_config_secrets(self):
        self.target_config = redact_sensitive_mapping(self.target_config)
        self.judge_snapshot = redact_sensitive_mapping(self.judge_snapshot)
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


class RowAnnotationResponse(BaseModel):
    """单条标注记录。前端要展示「谁标的」就必须读这个列表。

    只读 manual_* 投影只能看到「当前生效的那一条」，看不出它是一个人说的、
    两个人一致同意的，还是仲裁的结果——而这三者证据强度差一个量级。
    """

    id: int
    annotator: str
    status: Optional[str] = None
    score: Optional[float] = None
    tags: Optional[List[str]] = None
    note: Optional[str] = None
    is_adjudication: bool = False
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class EvalRowResultResponse(BaseModel):
    id: int
    eval_task_id: int
    row_index: int
    metric_scores: Optional[Dict[str, Any]] = None
    endpoint_trace: Optional[Dict[str, Any]] = None
    is_pass: Optional[bool] = None
    execution_time_ms: Optional[int] = None
    error: Optional[str] = None
    badcase_category: Optional[str] = None
    badcase_confidence: Optional[float] = None
    badcase_source: Optional[str] = None
    manual_status: Optional[str] = None
    manual_score: Optional[float] = None
    manual_tags: Optional[List[str]] = None
    manual_note: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    # 全部标注记录（真值）。上面 manual_* 五列只是它的投影。
    annotations: List[RowAnnotationResponse] = []
    dataset_row: Optional[DatasetRowResponse] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class EvalRowReviewUpdate(BaseModel):
    manual_status: Optional[Literal["pass", "fail", "needs_fix", "needs_review"]] = None
    manual_score: Optional[float] = None
    manual_tags: Optional[List[str]] = None
    manual_note: Optional[str] = None
    # 标注者身份。不传 → 落到 DEFAULT_ANNOTATOR，旧前端行为与重构前逐字段一致。
    #
    # 这里必须写清一件事：平台还没有登录态（API 鉴权是 P2 项），标注者身份由调用方
    # 自己声明，服务端无法验证。也就是说「两位标注者独立标注」这个前提当前靠**流程**
    # 保证，不靠系统保证——同一个人可以传两个名字，伪造出一个虚高的人-人 kappa。
    # 有了鉴权之后这里应改为从会话推断，而不是继续读请求体。
    annotator: Optional[str] = None
    # True = 看过分歧后的仲裁记录。仲裁不参与人-人一致性统计（仲裁者已知双方答案，
    # 与其算一致性是循环论证），只用于决定生效标签。
    is_adjudication: bool = False


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
    # 人工 vs 自动二分类的 Cohen's kappa（修正偶然一致后的真实一致程度）
    manual_auto_kappa: Optional[float] = None
    # kappa 的 Bootstrap 95% 置信区间。点估计单独看不出抽样不确定性：
    # n=25 时 kappa=0.82 完全可能对应下界 0.55，只报点估计等于把区间伪装成定论。
    manual_auto_kappa_ci: Optional[Dict[str, Any]] = None
    # 校准提示：按 CI 下界而非点估计出（下界没过线 → 是样本量不足，不是口径有问题）
    calibration_suggestion: Optional[str] = None
    # 人-人一致性（两两 kappa + CI）。这是上面 manual_auto_kappa 的**上界参照**：
    # 判断本身主观的任务上人类可能只有 0.6，此时 judge 的 0.82 不是"更准"而是
    # "拟合了某个标注者"。只有 1 位标注者时带 insufficient_annotators=True 返回，
    # 因为"测不出来"和"一致性很好"是两回事，而缺省值最容易被读成后者。
    annotator_agreement: Optional[Dict[str, Any]] = None
    # judge kappa 是否显著超过人-人 kappa（配对检验，非两个独立区间比大小）
    judge_ceiling_check: Optional[Dict[str, Any]] = None
    # 标注者之间结论相反的行，供人工仲裁
    annotation_disagreements: Optional[List[Dict[str, Any]]] = None
    # 生效标签的来源分布：单人标注 25 条 vs 两人一致 25 条是不同量级的证据
    label_basis_summary: Optional[Dict[str, int]] = None


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
    current_cost_cny: Optional[float] = None
    baseline_cost_cny: Optional[float] = None
    cost_delta_cny: Optional[float] = None
    cost_delta_ratio: Optional[float] = None
    current_latency_p95_ms: Optional[float] = None
    baseline_latency_p95_ms: Optional[float] = None
    latency_p95_delta_ms: Optional[float] = None
    latency_p95_delta_ratio: Optional[float] = None
    error_count_delta: int


class ReportCompareComparability(BaseModel):
    """口径可比性：两个任务是否在同一把尺子下测出来的。

    对比本身不该被拦死（用更严的尺子重测一遍恰恰是常见需求），但必须显式
    告诉读数的人：差值里有多少来自被测系统，有多少来自尺子变了。
    """

    # identical / changed / unknown 三态，不能压成布尔：
    # unknown（指纹上线前的历史任务）既不是"一致"也不是"变了"，
    # 把它当成任何一端都会让读数的人做出错误归因。
    status: Literal["identical", "changed", "unknown"]
    attribution_safe: bool
    current_fingerprint: Optional[str] = None
    baseline_fingerprint: Optional[str] = None
    changed_dimensions: List[str] = []
    warning: Optional[str] = None


class ReportCompareResponse(BaseModel):
    current_eval: EvalTaskResponse
    baseline_eval: EvalTaskResponse
    comparability: ReportCompareComparability
    summary_delta: ReportCompareSummaryDelta
    metric_deltas: List[ReportCompareMetricDelta]
    row_changes: Dict[str, List[ReportCompareRowItem]]


class QualityGateMetricRule(BaseModel):
    metric: str
    minimum_mean_delta: Optional[float] = None
    minimum_pass_rate_delta: Optional[float] = None
    minimum_mean_score: Optional[float] = None


class QualityGateRequest(BaseModel):
    baseline_eval_id: int
    require_identical_fingerprint: bool = True
    minimum_pass_rate_delta: float = 0.0
    maximum_new_failures: int = 0
    maximum_new_errors: int = 0
    maximum_cost_increase_cny: Optional[float] = Field(default=None, ge=0)
    maximum_cost_increase_ratio: Optional[float] = Field(default=None, ge=0)
    maximum_latency_p95_increase_ms: Optional[float] = Field(default=None, ge=0)
    maximum_latency_p95_increase_ratio: Optional[float] = Field(default=None, ge=0)
    metric_rules: List[QualityGateMetricRule] = Field(default_factory=list)


class QualityGateViolation(BaseModel):
    rule: str
    expected: Any = None
    actual: Any = None
    message: str


class QualityGateResponse(BaseModel):
    passed: bool
    status: Literal["passed", "blocked"]
    baseline_eval_id: int
    current_eval_id: int
    violations: List[QualityGateViolation]
    comparison: ReportCompareResponse


class BadCaseUpdate(BaseModel):
    category: str = Field(min_length=1, max_length=50)


class RegressionDatasetRequest(BaseModel):
    name: Optional[str] = None
    row_ids: Optional[List[int]] = None


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
