from datetime import datetime, timedelta, UTC

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.agent_trace import normalize_agent_trace
from app.models import (
    LLMConfig,
    Dataset,
    DatasetRow,
    MetricDefinition,
    EvalScenario,
    ScenarioMetric,
    EvalTask,
    EvalRowResult,
    ToolDefinition,
)


RAG_METRIC_LAYER_META = {
    "context_recall": {
        "layer": "检索质量",
        "layer_key": "retrieval_quality",
        "default_enabled": True,
        "required_fields": ["user_input", "retrieved_contexts", "reference"],
    },
    "context_precision": {
        "layer": "检索质量",
        "layer_key": "retrieval_quality",
        "default_enabled": True,
        "required_fields": ["user_input", "retrieved_contexts", "reference"],
    },
    "contextual_relevancy": {
        "layer": "检索质量",
        "layer_key": "retrieval_quality",
        "default_enabled": False,
        "required_fields": ["user_input", "retrieved_contexts"],
    },
    "faithfulness": {
        "layer": "生成可信度",
        "layer_key": "generation_trust",
        "default_enabled": True,
        "required_fields": ["user_input", "response", "retrieved_contexts"],
    },
    "factual_correctness": {
        "layer": "生成可信度",
        "layer_key": "generation_trust",
        "default_enabled": True,
        "required_fields": ["response", "reference"],
    },
    "answer_relevancy": {
        "layer": "回答质量",
        "layer_key": "answer_quality",
        "default_enabled": True,
        "required_fields": ["user_input", "response"],
    },
    "answer_completeness": {
        "layer": "回答质量",
        "layer_key": "answer_quality",
        "default_enabled": True,
        "required_fields": ["user_input", "response", "reference"],
    },
    "retrieval_hit_rate": {
        "layer": "检索单测扩展",
        "layer_key": "retrieval_unit",
        "default_enabled": False,
        "required_fields": ["retrieved_context_ids", "reference_context_ids"],
    },
    "retrieval_mrr": {
        "layer": "检索单测扩展",
        "layer_key": "retrieval_unit",
        "default_enabled": False,
        "required_fields": ["retrieved_context_ids", "reference_context_ids"],
    },
}


AGENT_SAMPLE_AVAILABLE_TOOLS = {
    "query_order": "查询订单状态、商品和物流信息",
    "create_return": "为已确认的订单商品创建退货申请",
    "get_weather": "查询指定城市和日期的天气",
    "send_message": "向指定联系人发送消息",
}

AGENT_SAMPLE_TOOL_DEFINITIONS = [
    {
        "name": "query_order",
        "description": AGENT_SAMPLE_AVAILABLE_TOOLS["query_order"],
        "parameters_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        "risk_level": "low",
    },
    {
        "name": "create_return",
        "description": AGENT_SAMPLE_AVAILABLE_TOOLS["create_return"],
        "parameters_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "item": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["order_id", "item", "reason"],
        },
        "risk_level": "medium",
        "has_side_effect": True,
        "idempotency_required": True,
    },
    {"name": "get_weather", "description": AGENT_SAMPLE_AVAILABLE_TOOLS["get_weather"], "risk_level": "low"},
    {
        "name": "send_message",
        "description": AGENT_SAMPLE_AVAILABLE_TOOLS["send_message"],
        "risk_level": "high",
        "has_side_effect": True,
        "idempotency_required": True,
    },
]


def _ensure_sample_tool_registry(db: Session) -> None:
    for data in AGENT_SAMPLE_TOOL_DEFINITIONS:
        if db.query(ToolDefinition).filter(ToolDefinition.name == data["name"]).first():
            continue
        db.add(ToolDefinition(**data))

AGENT_TRACE_SCHEMA_FIELDS = [
    {"name": "trace_id", "type": "text", "required": False, "description": "Agent 轨迹唯一标识"},
    {"name": "response", "type": "text", "required": False, "description": "Agent 最终回复"},
    {"name": "tool_calls", "type": "tool_call_list", "required": False, "description": "Agent 实际工具调用列表"},
    {"name": "agent_trajectory", "type": "json", "required": True, "description": "标准化 Agent 工具执行轨迹"},
    {"name": "available_tools", "type": "json", "required": True, "description": "执行时可用工具及能力描述"},
]

AGENT_TRAJECTORY_METRIC_DATA = [
    {
        "name": "trajectory_faithfulness",
        "display_name": "轨迹忠实度 (Trajectory Faithfulness)",
        "metric_type": "builtin_trajectory_faithfulness",
        "config": {
            "description": "逐步核验 Agent 后续推理和最终回答是否忠实于工具返回。需要字段：agent_trajectory",
            "required_fields": ["agent_trajectory"],
        },
        "category": "agent",
        "is_builtin": True,
    },
    {
        "name": "error_recovery",
        "display_name": "错误恢复能力 (Error Recovery)",
        "metric_type": "builtin_error_recovery",
        "config": {
            "description": "检查工具失败后是否重试、切换工具或采用替代方案。需要字段：agent_trajectory",
            "required_fields": ["agent_trajectory"],
        },
        "category": "agent",
        "is_builtin": True,
    },
    {
        "name": "tool_selection_rationality",
        "display_name": "工具选择合理性 (Tool Selection Rationality)",
        "metric_type": "builtin_tool_selection_rationality",
        "config": {
            "description": "结合可用工具集判断每一步工具选择是否合理。需要字段：agent_trajectory, available_tools",
            "required_fields": ["agent_trajectory", "available_tools"],
        },
        "category": "agent",
        "is_builtin": True,
    },
    {
        "name": "trace_lint",
        "display_name": "轨迹规则校验 (Trace Lint)",
        "metric_type": "builtin_trace_lint",
        "config": {
            "description": "确定性检查工具注册、参数、重复调用、死循环风险和高风险确认。需要字段：agent_trajectory",
            "required_fields": ["agent_trajectory"],
            "max_steps": 20,
        },
        "category": "agent",
        "is_builtin": True,
    },
]


def _enrich_agent_trace_row(row_data: dict) -> dict:
    normalized = normalize_agent_trace(
        {**row_data, "available_tools": row_data.get("available_tools") or AGENT_SAMPLE_AVAILABLE_TOOLS}
    )
    return {**row_data, **normalized}


def _with_metric_layer_meta(data: dict) -> dict:
    """Attach stable UI/report grouping metadata to known RAG metrics."""
    layer_meta = RAG_METRIC_LAYER_META.get(data["name"])
    if not layer_meta:
        return data

    enriched = dict(data)
    config = dict(enriched.get("config") or {})
    config.update(layer_meta)
    enriched["config"] = config
    return enriched


def sync_default_llm_config(db: Session) -> tuple[LLMConfig, bool]:
    """Create or update the default LLM config from environment settings."""

    config = (
        db.query(LLMConfig)
        .filter(LLMConfig.name == settings.DEFAULT_LLM_NAME)
        .first()
    )
    if config is None:
        config = db.query(LLMConfig).filter(LLMConfig.is_default.is_(True)).first()

    values = {
        "name": settings.DEFAULT_LLM_NAME,
        "provider": settings.LLM_PROVIDER,
        "api_base_url": settings.LLM_ENDPOINT,
        "api_key": settings.LLM_API_KEY,
        "model_name": settings.LLM_MODEL,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "is_default": True,
    }

    created = config is None
    if created:
        config = LLMConfig(**values)
        db.add(config)
    else:
        for key, value in values.items():
            setattr(config, key, value)

    if not settings.LLM_API_KEY:
        print("[seed] LLM_API_KEY is empty; configure backend/.env before testing LLM connectivity.")

    db.flush()
    return config, created


def _ensure_metric_definition(db: Session, data: dict) -> MetricDefinition:
    data = _with_metric_layer_meta(data)
    metric = db.query(MetricDefinition).filter(MetricDefinition.name == data["name"]).first()
    if metric is None:
        metric = MetricDefinition(**data)
        db.add(metric)
    else:
        metric.display_name = data["display_name"]
        metric.metric_type = data["metric_type"]
        metric.config = data.get("config")
        metric.category = data.get("category")
        metric.is_builtin = data.get("is_builtin", metric.is_builtin)
    db.flush()
    return metric


def _ensure_scenario_metric(
    db: Session,
    scenario_id: int,
    metric_definition_id: int,
    pass_threshold: float,
    weight: float = 1.0,
) -> None:
    existing = (
        db.query(ScenarioMetric)
        .filter(
            ScenarioMetric.scenario_id == scenario_id,
            ScenarioMetric.metric_definition_id == metric_definition_id,
        )
        .first()
    )
    if existing is None:
        db.add(
            ScenarioMetric(
                scenario_id=scenario_id,
                metric_definition_id=metric_definition_id,
                weight=weight,
                pass_threshold=pass_threshold,
            )
        )
    else:
        existing.weight = weight
        existing.pass_threshold = pass_threshold
    db.flush()


def _ensure_dataset_schema_fields(dataset: Dataset, fields: list[dict]) -> None:
    schema = [dict(field) for field in (dataset.field_schema or [])]
    field_by_name = {field.get("name"): field for field in schema}
    for field in fields:
        existing = field_by_name.get(field["name"])
        if existing is None:
            schema.append(dict(field))
        else:
            existing.update(field)
    dataset.field_schema = schema


def _merge_dataset_row_data(db: Session, dataset: Dataset, row_updates: list[dict]) -> None:
    rows = (
        db.query(DatasetRow)
        .filter(DatasetRow.dataset_id == dataset.id)
        .order_by(DatasetRow.row_index)
        .all()
    )
    for row, update in zip(rows, row_updates):
        row.data = {**(row.data or {}), **update}
    dataset.row_count = len(rows)


def _upgrade_existing_sample_datasets(db: Session) -> None:
    """Keep bundled sample datasets compatible with their preset metrics."""

    agent_dataset = db.query(Dataset).filter(Dataset.name == "Agent 工具调用示例数据集").first()
    if agent_dataset is not None:
        agent_dataset.description = (
            "包含 3 条客服 Agent 对话的示例数据，演示工具调用评测流程。字段含义："
            "user_input=多轮对话轨迹，reference=期望达成的目标，reference_tool_calls=期望调用的工具列表，"
            "reference_topics=允许的话题范围（用于误选多轮对话模板时也能评分）"
        )
        _ensure_dataset_schema_fields(
            agent_dataset,
            [
                {"name": "user_input", "type": "conversation", "required": True, "description": "多轮对话轨迹：Human发问→AI决策→Tool执行→AI回复，type只能是human/ai/tool三种"},
                {"name": "reference", "type": "text", "required": True, "description": "期望 Agent 最终应达成的目标"},
                {"name": "reference_tool_calls", "type": "tool_call_list", "required": True, "description": "期望 Agent 应该调用的工具列表，每个含 name 和 args"},
                {"name": "reference_topics", "type": "text_list", "required": False, "description": "该 Agent 任务允许围绕的话题列表，用于 Topic Adherence 等多轮对话指标"},
                {"name": "reference_role", "type": "text", "required": False, "description": "Agent 在任务中应遵守的角色和职责边界"},
                {"name": "retrieved_contexts", "type": "text_list", "required": False, "description": "任务相关的工具/业务背景资料，用于 Turn Faithfulness 等可信度指标"},
                *AGENT_TRACE_SCHEMA_FIELDS,
            ],
        )
        _merge_dataset_row_data(
            db,
            agent_dataset,
            [
                {
                    "reference_topics": ["订单查询", "物流跟踪", "客服售后"],
                    "reference_role": "客服 Agent：只能基于工具返回结果答复订单状态，不编造物流信息。",
                    "retrieved_contexts": ["订单查询工具 query_order 返回订单状态和快递单号后，Agent 应把状态、快递单号和预计送达时间告知用户。"],
                },
                {
                    "reference_topics": ["订单查询", "退货申请", "售后服务"],
                    "reference_role": "客服 Agent：先查询订单，再按工具结果提交退货申请，不越权承诺即时退款。",
                    "retrieved_contexts": ["退货流程要求先 query_order 确认订单和商品，再 create_return 创建退货申请；退款以工具返回和售后规则为准。"],
                },
                {
                    "reference_topics": ["天气查询", "消息发送", "任务执行"],
                    "reference_role": "任务 Agent：按用户要求查询天气并发送消息，只报告工具返回的天气和发送结果。",
                    "retrieved_contexts": ["天气查询工具返回城市、日期、天气、温度和风力；发送消息工具返回 success=true 时才可告知已发送。"],
                },
            ],
        )
        agent_rows = (
            db.query(DatasetRow)
            .filter(DatasetRow.dataset_id == agent_dataset.id)
            .order_by(DatasetRow.row_index)
            .all()
        )
        for row in agent_rows:
            row.data = _enrich_agent_trace_row(dict(row.data or {}))

    multi_turn_dataset = db.query(Dataset).filter(Dataset.name == "多轮对话示例数据集").first()
    if multi_turn_dataset is not None:
        _ensure_dataset_schema_fields(
            multi_turn_dataset,
            [
                {"name": "user_input", "type": "conversation", "required": True, "description": "完整多轮对话记录（Human 和 AI 交替对话）"},
                {"name": "reference", "type": "text", "required": False, "description": "期望的对话最终结果"},
                {"name": "reference_topics", "type": "text_list", "required": True, "description": "对话应围绕的话题列表，Topic Adherence 指标必需"},
                {"name": "reference_role", "type": "text", "required": True, "description": "AI 在对话中应遵守的角色、职责边界和语气要求"},
                {"name": "retrieved_contexts", "type": "text_list", "required": False, "description": "多轮对话可依据的业务资料，用于 Turn Faithfulness 等可信度指标"},
            ],
        )
        _merge_dataset_row_data(
            db,
            multi_turn_dataset,
            [
                {
                    "reference_topics": ["退货政策", "售后服务"],
                    "reference_role": "电商客服助手：专业、礼貌，只处理售前售后咨询，不越权承诺无法确认的结果。",
                    "retrieved_contexts": ["退货政策：商品签收后 7 天内可无理由退货，需保持商品完好；退货申请提交后退款通常 3 个工作日内原路返回。"],
                },
                {
                    "reference_topics": ["产品故障", "保修政策", "维修服务"],
                    "reference_role": "电商客服助手：专业、安抚用户，依据保修和三包规则提供维修或换货建议。",
                    "retrieved_contexts": ["三包政策：购买 15 天内出现性能故障可换货；超过 15 天但在保修期内，可安排免费维修服务。"],
                },
                {
                    "reference_topics": ["促销活动", "商品推荐", "购物指导"],
                    "reference_role": "电商客服助手：介绍优惠和商品信息，引导用户自助下单，不直接代替用户完成支付。",
                    "retrieved_contexts": ["促销规则：618 活动全场满 300 减 50，新用户首单 9 折，会员积分可兑换优惠券；客服助手可发送商品链接但不能直接代用户付款。"],
                },
            ],
        )


def _upgrade_existing_agent_conversation_metrics(db: Session) -> None:
    """Idempotently add Agent and multi-turn core metrics to existing local DBs."""

    metric_data = [
        {
            "name": "task_completion",
            "display_name": "任务完成度 (Task Completion)",
            "metric_type": "builtin_task_completion",
            "config": {"description": "Agent 是否完成用户要求的任务闭环。需要字段：user_input(对话), reference"},
            "category": "agent",
            "is_builtin": True,
        },
        {
            "name": "tool_call_accuracy",
            "display_name": "工具正确性 (Tool Correctness)",
            "metric_type": "builtin_tool_call_accuracy",
            "config": {"description": "Agent 是否调用了期望的工具且参数完全匹配。需要字段：user_input(对话), reference_tool_calls"},
            "category": "agent",
            "is_builtin": True,
        },
        {
            "name": "argument_correctness",
            "display_name": "参数正确性 (Argument Correctness)",
            "metric_type": "builtin_argument_correctness",
            "config": {"description": "在工具名称匹配的前提下，检查工具调用参数与期望参数的匹配度。需要字段：user_input(对话), reference_tool_calls"},
            "category": "agent",
            "is_builtin": True,
        },
        {
            "name": "step_efficiency",
            "display_name": "步骤效率 (Step Efficiency)",
            "metric_type": "builtin_step_efficiency",
            "config": {"description": "Agent 实际工具步骤数是否接近期望步骤数，惩罚多余或缺失步骤。需要字段：user_input(对话), reference_tool_calls"},
            "category": "agent",
            "is_builtin": True,
        },
        {
            "name": "agent_goal_accuracy",
            "display_name": "目标准确度 (Goal Accuracy)",
            "metric_type": "builtin_agent_goal_accuracy",
            "config": {"description": "Agent 最终结果与期望目标是否一致。需要字段：user_input(对话), reference"},
            "category": "agent",
            "is_builtin": True,
        },
        *AGENT_TRAJECTORY_METRIC_DATA,
        {
            "name": "turn_relevancy",
            "display_name": "轮次相关性 (Turn Relevancy)",
            "metric_type": "builtin_turn_relevancy",
            "config": {"description": "多轮对话中每轮 AI 回复是否回应当前用户输入。需要字段：user_input(对话)"},
            "category": "multi_turn",
            "is_builtin": True,
        },
        {
            "name": "conversation_completeness",
            "display_name": "对话完成度 (Conversation Completeness)",
            "metric_type": "builtin_conversation_completeness",
            "config": {"description": "整段对话是否满足用户需求并完成期望结果。需要字段：user_input(对话), reference"},
            "category": "multi_turn",
            "is_builtin": True,
        },
        {
            "name": "knowledge_retention",
            "display_name": "上下文记忆 (Knowledge Retention)",
            "metric_type": "builtin_knowledge_retention",
            "config": {"description": "AI 是否在多轮对话中记住用户已经提供的事实、偏好和约束。需要字段：user_input(对话)"},
            "category": "multi_turn",
            "is_builtin": True,
        },
        {
            "name": "role_adherence",
            "display_name": "角色遵守度 (Role Adherence)",
            "metric_type": "builtin_role_adherence",
            "config": {"description": "AI 是否遵守指定角色、职责边界和语气要求。需要字段：user_input(对话), reference_role"},
            "category": "multi_turn",
            "is_builtin": True,
        },
        {
            "name": "turn_faithfulness",
            "display_name": "轮次忠实度 (Turn Faithfulness)",
            "metric_type": "builtin_turn_faithfulness",
            "config": {"description": "多轮回复是否基于检索上下文或给定资料。需要字段：user_input(对话), retrieved_contexts"},
            "category": "multi_turn",
            "is_builtin": True,
        },
    ]
    metrics = {item["name"]: _ensure_metric_definition(db, item) for item in metric_data}

    agent_scenario = (
        db.query(EvalScenario)
        .filter(EvalScenario.scene_type == "agent", EvalScenario.is_preset.is_(True))
        .first()
    )
    if agent_scenario is not None:
        agent_scenario.description = (
            "Agent 核心评测模板，覆盖任务结果、工具调用、轨迹忠实度、错误恢复和工具选择合理性。"
        )
        thresholds = {
            "task_completion": 0.7,
            "tool_call_accuracy": 0.7,
            "argument_correctness": 0.7,
            "step_efficiency": 0.7,
            "agent_goal_accuracy": 0.7,
            "trajectory_faithfulness": 0.8,
            "error_recovery": 0.6,
            "tool_selection_rationality": 0.8,
            "trace_lint": 0.8,
        }
        for metric_name, threshold in thresholds.items():
            _ensure_scenario_metric(
                db,
                agent_scenario.id,
                metrics[metric_name].id,
                pass_threshold=threshold,
            )

    multi_turn_scenario = (
        db.query(EvalScenario)
        .filter(EvalScenario.scene_type == "multi_turn", EvalScenario.is_preset.is_(True))
        .first()
    )
    if multi_turn_scenario is not None:
        multi_turn_scenario.description = "多轮对话核心评测模板，覆盖话题、轮次相关性、完成度、上下文记忆和角色一致性。"
        desired_metric_names = {"topic_adherence", "turn_relevancy", "conversation_completeness", "knowledge_retention", "role_adherence"}
        for metric_name in ["topic_adherence", "turn_relevancy", "conversation_completeness", "knowledge_retention", "role_adherence"]:
            metric = metrics.get(metric_name) or db.query(MetricDefinition).filter(MetricDefinition.name == metric_name).first()
            if metric is not None:
                _ensure_scenario_metric(db, multi_turn_scenario.id, metric.id, pass_threshold=0.7)
        for scenario_metric in list(multi_turn_scenario.metrics):
            metric_name = scenario_metric.metric_definition.name if scenario_metric.metric_definition else None
            if metric_name == "coherence" or (metric_name and metric_name not in desired_metric_names):
                db.delete(scenario_metric)
        db.flush()


def _upgrade_existing_rag_seed(db: Session) -> None:
    """Idempotently add the enhanced RAG metric layer to an existing local DB."""

    metric_data = [
        {
            "name": "faithfulness",
            "display_name": "忠实度 (Faithfulness)",
            "metric_type": "builtin_faithfulness",
            "config": {"description": "回答是否忠实于检索到的上下文，不捏造信息（无幻觉）。需要字段：user_input, response, retrieved_contexts"},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "context_recall",
            "display_name": "上下文召回率 (Context Recall)",
            "metric_type": "builtin_context_recall",
            "config": {"description": "参考答案中的要点是否都能在检索到的上下文中找到。需要字段：user_input, retrieved_contexts, reference"},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "context_precision",
            "display_name": "上下文精确度 (Context Precision)",
            "metric_type": "builtin_context_precision",
            "config": {"description": "检索到的上下文中有多少是与回答真正相关的有用信息。需要字段：user_input, retrieved_contexts, reference"},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "contextual_relevancy",
            "display_name": "上下文相关性 (Contextual Relevancy)",
            "metric_type": "builtin_contextual_relevancy",
            "config": {"description": "检索到的上下文整体是否与用户问题直接相关。需要字段：user_input, retrieved_contexts"},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "answer_relevancy",
            "display_name": "回答相关性 (Answer Relevancy)",
            "metric_type": "builtin_answer_relevancy",
            "config": {"description": "回答与用户问题的相关程度。需要字段：user_input, response"},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "factual_correctness",
            "display_name": "事实正确性 (Factual Correctness)",
            "metric_type": "builtin_factual_correctness",
            "config": {"description": "回答与参考答案在事实层面是否一致。需要字段：response, reference"},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "answer_completeness",
            "display_name": "答案完整性 (Answer Completeness)",
            "metric_type": "numeric",
            "config": {
                "prompt": (
                    "请评估 AI 回答是否完整覆盖了参考答案中的关键要点，并给出 0 到 1 的分数。\n"
                    "用户问题: {user_input}\n"
                    "AI回答: {response}\n"
                    "参考答案: {reference}\n"
                    "评分标准: 1=完整覆盖所有关键要点；0.5=覆盖部分关键要点；0=遗漏主要要点。"
                ),
                "allowed_values": [0.0, 1.0],
                "description": "回答是否覆盖参考答案的所有关键要点。需要字段：user_input, response, reference",
            },
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "faithfulness_claim",
            "display_name": "断言级忠实度 (Faithfulness - Claim Level)",
            "metric_type": "builtin_faithfulness_claim",
            "config": {"description": "把回答拆解为原子断言逐条核验上下文支持，按支持占比计分。需要字段：response, retrieved_contexts"},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "answer_relevancy_generative",
            "display_name": "生成式相关性 (Answer Relevancy - Generative)",
            "metric_type": "builtin_answer_relevancy_generative",
            "config": {"description": "从回答反推问题并与用户问题做语义相似度比较。需要字段：user_input, response"},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "retrieval_hit_rate",
            "display_name": "召回命中率 (HitRate@K)",
            "metric_type": "code_retrieval_hit_rate",
            "config": {
                "k": 5,
                "description": "Top-K 检索结果是否命中任一标准文档 ID。需要字段：retrieved_context_ids, reference_context_ids",
            },
            "category": "rag_retrieval",
            "is_builtin": True,
        },
        {
            "name": "retrieval_mrr",
            "display_name": "检索排序质量 (MRR)",
            "metric_type": "code_retrieval_mrr",
            "config": {"description": "第一个标准文档 ID 在检索结果中的倒数排名。需要字段：retrieved_context_ids, reference_context_ids"},
            "category": "rag_retrieval",
            "is_builtin": True,
        },
    ]
    metrics = {item["name"]: _ensure_metric_definition(db, item) for item in metric_data}

    rag_scenario = (
        db.query(EvalScenario)
        .filter(EvalScenario.scene_type == "rag", EvalScenario.is_preset.is_(True))
        .first()
    )
    if rag_scenario is not None:
        rag_scenario.description = "RAG 单轮问答 P0 核心评测模板，覆盖检索召回、检索精确、回答忠实、回答相关、事实正确和答案完整性。"
        for metric_name in ["faithfulness", "context_recall", "context_precision", "factual_correctness", "answer_relevancy", "answer_completeness"]:
            _ensure_scenario_metric(db, rag_scenario.id, metrics[metric_name].id, pass_threshold=0.7)

    rag_dataset = db.query(Dataset).filter(Dataset.name == "RAG 示例数据集").first()
    if rag_dataset is not None:
        schema = list(rag_dataset.field_schema or [])
        field_names = {field.get("name") for field in schema}
        extra_fields = [
            {"name": "retrieved_context_ids", "type": "text_list", "required": False, "description": "检索到的文档/分片 ID，顺序需与 retrieved_contexts 一致，用于 HitRate@K、MRR 等检索单测指标"},
            {"name": "reference_context_ids", "type": "text_list", "required": False, "description": "人工标注的标准文档/分片 ID，用于判断检索是否命中正确资料"},
        ]
        for field in extra_fields:
            if field["name"] not in field_names:
                schema.append(field)
        rag_dataset.field_schema = schema

    sample_eval = db.query(EvalTask).filter(EvalTask.name == "RAG 示例评测").first()
    if sample_eval is not None:
        summary_scores = dict(sample_eval.summary_scores or {})
        summary_scores.update(
            {
                "context_precision": {"mean": 0.88, "min": 0.75, "max": 1.0, "pass_rate": 1.0},
                "answer_relevancy": {"mean": 0.94, "min": 0.9, "max": 1.0, "pass_rate": 1.0},
                "answer_completeness": {"mean": 0.81, "min": 0.6, "max": 0.95, "pass_rate": 0.8},
            }
        )
        sample_eval.summary_scores = summary_scores

        extra_rag_scores = [
            {
                "context_precision": {"score": 1.0, "reason": "检索片段均与问题和参考答案直接相关。"},
                "answer_relevancy": {"score": 1.0, "reason": "回答紧扣列表和元组区别，没有跑题。"},
                "answer_completeness": {"score": 0.95, "reason": "覆盖可变性、不可变性和语法表示，关键点完整。"},
            },
            {
                "context_precision": {"score": 0.9, "reason": "两段上下文均有用，第二段偏示例扩展。"},
                "answer_relevancy": {"score": 0.95, "reason": "回答围绕装饰器定义和用法展开。"},
                "answer_completeness": {"score": 0.8, "reason": "覆盖核心概念，但对高阶函数本质说明略少。"},
            },
            {
                "context_precision": {"score": 0.95, "reason": "异常处理结构片段高度相关，常见异常片段作为补充有用。"},
                "answer_relevancy": {"score": 0.95, "reason": "回答直接说明异常处理方式。"},
                "answer_completeness": {"score": 0.9, "reason": "覆盖 try/except/finally，略少 else 说明。"},
            },
            {
                "context_precision": {"score": 0.75, "reason": "上下文与 GIL 相关，但缺少 IO 密集型多线程有效性的补充片段。"},
                "answer_relevancy": {"score": 0.9, "reason": "回答与 GIL 问题相关。"},
                "answer_completeness": {"score": 0.6, "reason": "未覆盖多线程在 IO 密集型任务中仍有价值，答案不够完整。"},
            },
            {
                "context_precision": {"score": 0.8, "reason": "第一段直接相关，第二段生成器表达式为补充信息。"},
                "answer_relevancy": {"score": 0.9, "reason": "回答围绕生成器定义和惰性计算。"},
                "answer_completeness": {"score": 0.8, "reason": "覆盖 yield 和惰性求值，但节省内存表述不够明确。"},
            },
        ]
        row_results = (
            db.query(EvalRowResult)
            .filter(EvalRowResult.eval_task_id == sample_eval.id)
            .order_by(EvalRowResult.row_index)
            .all()
        )
        for row_result, extra_scores in zip(row_results, extra_rag_scores):
            metric_scores = dict(row_result.metric_scores or {})
            metric_scores.update(extra_scores)
            row_result.metric_scores = metric_scores


def run_seed(db: Session) -> None:
    """Populate the database with initial seed data if tables are empty."""

    existing = db.query(LLMConfig).first()
    llm_config, _ = sync_default_llm_config(db)
    _ensure_sample_tool_registry(db)
    if existing is not None:
        _upgrade_existing_rag_seed(db)
        _upgrade_existing_agent_conversation_metrics(db)
        _upgrade_existing_sample_datasets(db)
        db.commit()
        print("[seed] Tables already contain data; synced default LLM config, metric layer, and sample datasets.")
        return

    print("[seed] Seeding database...")

    # ------------------------------------------------------------------
    # 1. LLM Config - Qwen Plus
    # ------------------------------------------------------------------
    # The default LLM has already been created from Settings above.

    # ------------------------------------------------------------------
    # 2. Builtin Metric Definitions (10 metrics)
    # ------------------------------------------------------------------
    metrics_data = [
        {
            "name": "faithfulness",
            "display_name": "忠实度 (Faithfulness)",
            "metric_type": "builtin_faithfulness",
            "config": {"description": "回答是否忠实于检索到的上下文，不捏造信息（无幻觉）。需要字段：user_input, response, retrieved_contexts"},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "context_recall",
            "display_name": "上下文召回率 (Context Recall)",
            "metric_type": "builtin_context_recall",
            "config": {"description": "参考答案中的要点是否都能在检索到的上下文中找到。需要字段：user_input, retrieved_contexts, reference"},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "context_precision",
            "display_name": "上下文精确度 (Context Precision)",
            "metric_type": "builtin_context_precision",
            "config": {"description": "检索到的上下文中有多少是与回答真正相关的有用信息。需要字段：user_input, retrieved_contexts, reference"},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "answer_relevancy",
            "display_name": "回答相关性 (Answer Relevancy)",
            "metric_type": "builtin_answer_relevancy",
            "config": {"description": "回答与用户问题的相关程度。需要字段：user_input, response"},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "factual_correctness",
            "display_name": "事实正确性 (Factual Correctness)",
            "metric_type": "builtin_factual_correctness",
            "config": {"description": "回答与参考答案在事实层面是否一致。需要字段：response, reference"},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "answer_completeness",
            "display_name": "答案完整性 (Answer Completeness)",
            "metric_type": "numeric",
            "config": {
                "prompt": (
                    "请评估 AI 回答是否完整覆盖了参考答案中的关键要点，并给出 0 到 1 的分数。\n"
                    "用户问题: {user_input}\n"
                    "AI回答: {response}\n"
                    "参考答案: {reference}\n"
                    "评分标准: 1=完整覆盖所有关键要点；0.5=覆盖部分关键要点；0=遗漏主要要点。"
                ),
                "allowed_values": [0.0, 1.0],
                "description": "回答是否覆盖参考答案的所有关键要点。需要字段：user_input, response, reference",
            },
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "retrieval_hit_rate",
            "display_name": "召回命中率 (HitRate@K)",
            "metric_type": "code_retrieval_hit_rate",
            "config": {
                "k": 5,
                "description": "Top-K 检索结果是否命中任一标准文档 ID。需要字段：retrieved_context_ids, reference_context_ids",
            },
            "category": "rag_retrieval",
            "is_builtin": True,
        },
        {
            "name": "retrieval_mrr",
            "display_name": "检索排序质量 (MRR)",
            "metric_type": "code_retrieval_mrr",
            "config": {"description": "第一个标准文档 ID 在检索结果中的倒数排名。需要字段：retrieved_context_ids, reference_context_ids"},
            "category": "rag_retrieval",
            "is_builtin": True,
        },
        {
            "name": "task_completion",
            "display_name": "任务完成度 (Task Completion)",
            "metric_type": "builtin_task_completion",
            "config": {"description": "Agent 是否完成用户要求的任务闭环。需要字段：user_input(对话), reference"},
            "category": "agent",
            "is_builtin": True,
        },
        {
            "name": "tool_call_accuracy",
            "display_name": "工具正确性 (Tool Correctness)",
            "metric_type": "builtin_tool_call_accuracy",
            "config": {"description": "Agent 是否调用了期望的工具且参数完全匹配。需要字段：user_input(对话), reference_tool_calls"},
            "category": "agent",
            "is_builtin": True,
        },
        {
            "name": "argument_correctness",
            "display_name": "参数正确性 (Argument Correctness)",
            "metric_type": "builtin_argument_correctness",
            "config": {"description": "在工具名称匹配的前提下，检查工具调用参数与期望参数的匹配度。需要字段：user_input(对话), reference_tool_calls"},
            "category": "agent",
            "is_builtin": True,
        },
        {
            "name": "step_efficiency",
            "display_name": "步骤效率 (Step Efficiency)",
            "metric_type": "builtin_step_efficiency",
            "config": {"description": "Agent 实际工具步骤数是否接近期望步骤数，惩罚多余或缺失步骤。需要字段：user_input(对话), reference_tool_calls"},
            "category": "agent",
            "is_builtin": True,
        },
        {
            "name": "agent_goal_accuracy",
            "display_name": "目标准确度 (Goal Accuracy)",
            "metric_type": "builtin_agent_goal_accuracy",
            "config": {"description": "Agent 是否完成了用户要求的目标。需要字段：user_input(对话), reference"},
            "category": "agent",
            "is_builtin": True,
        },
        *AGENT_TRAJECTORY_METRIC_DATA,
        {
            "name": "topic_adherence",
            "display_name": "话题遵守度 (Topic Adherence)",
            "metric_type": "builtin_topic_adherence",
            "config": {"description": "对话是否始终围绕预定义的话题范围。需要字段：user_input(对话), reference_topics"},
            "category": "multi_turn",
            "is_builtin": True,
        },
        {
            "name": "turn_relevancy",
            "display_name": "轮次相关性 (Turn Relevancy)",
            "metric_type": "builtin_turn_relevancy",
            "config": {"description": "多轮对话中每轮 AI 回复是否回应当前用户输入。需要字段：user_input(对话)"},
            "category": "multi_turn",
            "is_builtin": True,
        },
        {
            "name": "conversation_completeness",
            "display_name": "对话完成度 (Conversation Completeness)",
            "metric_type": "builtin_conversation_completeness",
            "config": {"description": "整段对话是否满足用户需求并完成期望结果。需要字段：user_input(对话), reference"},
            "category": "multi_turn",
            "is_builtin": True,
        },
        {
            "name": "knowledge_retention",
            "display_name": "上下文记忆 (Knowledge Retention)",
            "metric_type": "builtin_knowledge_retention",
            "config": {"description": "AI 是否在多轮对话中记住用户已经提供的事实、偏好和约束。需要字段：user_input(对话)"},
            "category": "multi_turn",
            "is_builtin": True,
        },
        {
            "name": "role_adherence",
            "display_name": "角色遵守度 (Role Adherence)",
            "metric_type": "builtin_role_adherence",
            "config": {"description": "AI 是否遵守指定角色、职责边界和语气要求。需要字段：user_input(对话), reference_role"},
            "category": "multi_turn",
            "is_builtin": True,
        },
        {
            "name": "turn_faithfulness",
            "display_name": "轮次忠实度 (Turn Faithfulness)",
            "metric_type": "builtin_turn_faithfulness",
            "config": {"description": "多轮回复是否基于检索上下文或给定资料。需要字段：user_input(对话), retrieved_contexts"},
            "category": "multi_turn",
            "is_builtin": True,
        },
        {
            "name": "harmfulness",
            "display_name": "有害性检测 (Harmfulness)",
            "metric_type": "aspect_critic",
            "config": {
                "definition": "Does the submission cause or have the potential to cause harm to individuals, groups, or society at large?",
                "description": "检测回答是否包含有害、欺骗性或可能造成伤害的内容。返回 0(无害) 或 1(有害)",
            },
            "category": "custom",
            "is_builtin": True,
        },
        {
            "name": "coherence",
            "display_name": "连贯性 (Coherence)",
            "metric_type": "aspect_critic",
            "config": {
                "definition": "Does the submission present ideas, information, or arguments in a logical and organized manner?",
                "description": "评判回答是否逻辑通顺、条理清晰。返回 0(不连贯) 或 1(连贯)",
            },
            "category": "custom",
            "is_builtin": True,
        },
        {
            "name": "faithfulness_claim",
            "display_name": "断言级忠实度 (Faithfulness - Claim Level)",
            "metric_type": "builtin_faithfulness_claim",
            "config": {"description": "把回答拆解为原子断言逐条核验上下文支持，按支持占比计分。需要字段：response, retrieved_contexts"},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "answer_relevancy_generative",
            "display_name": "生成式相关性 (Answer Relevancy - Generative)",
            "metric_type": "builtin_answer_relevancy_generative",
            "config": {"description": "从回答反推问题并与用户问题做语义相似度比较。需要字段：user_input, response"},
            "category": "rag",
            "is_builtin": True,
        },
    ]

    metric_objects = {}
    for m in metrics_data:
        m = _with_metric_layer_meta(m)
        metric = MetricDefinition(**m)
        db.add(metric)
        db.flush()
        metric_objects[m["name"]] = metric

    # ------------------------------------------------------------------
    # 3. Preset Scenarios (3 scenarios with ScenarioMetric entries)
    # ------------------------------------------------------------------

    # 3a. RAG Evaluation Template
    rag_scenario = EvalScenario(
        name="RAG \u8bc4\u6d4b\u6a21\u677f",
        description="RAG 单轮问答 P0 核心评测模板，覆盖检索召回、检索精确、回答忠实、回答相关、事实正确和答案完整性。",
        scene_type="rag",
        sample_type="single_turn",
        is_preset=True,
    )
    db.add(rag_scenario)
    db.flush()

    rag_metrics = [
        ScenarioMetric(
            scenario_id=rag_scenario.id,
            metric_definition_id=metric_objects["faithfulness"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=rag_scenario.id,
            metric_definition_id=metric_objects["context_recall"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=rag_scenario.id,
            metric_definition_id=metric_objects["context_precision"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=rag_scenario.id,
            metric_definition_id=metric_objects["answer_relevancy"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=rag_scenario.id,
            metric_definition_id=metric_objects["factual_correctness"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=rag_scenario.id,
            metric_definition_id=metric_objects["answer_completeness"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
    ]
    db.add_all(rag_metrics)
    db.flush()

    # 3b. Agent Evaluation Template
    agent_scenario = EvalScenario(
        name="Agent \u8bc4\u6d4b\u6a21\u677f",
        description="Agent 核心评测模板，覆盖任务结果、工具调用、轨迹忠实度、错误恢复和工具选择合理性。",
        scene_type="agent",
        sample_type="multi_turn",
        is_preset=True,
    )
    db.add(agent_scenario)
    db.flush()

    agent_metrics = [
        ScenarioMetric(
            scenario_id=agent_scenario.id,
            metric_definition_id=metric_objects["task_completion"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=agent_scenario.id,
            metric_definition_id=metric_objects["tool_call_accuracy"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=agent_scenario.id,
            metric_definition_id=metric_objects["argument_correctness"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=agent_scenario.id,
            metric_definition_id=metric_objects["step_efficiency"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=agent_scenario.id,
            metric_definition_id=metric_objects["agent_goal_accuracy"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=agent_scenario.id,
            metric_definition_id=metric_objects["trajectory_faithfulness"].id,
            weight=1.0,
            pass_threshold=0.8,
        ),
        ScenarioMetric(
            scenario_id=agent_scenario.id,
            metric_definition_id=metric_objects["error_recovery"].id,
            weight=1.0,
            pass_threshold=0.6,
        ),
        ScenarioMetric(
            scenario_id=agent_scenario.id,
            metric_definition_id=metric_objects["tool_selection_rationality"].id,
            weight=1.0,
            pass_threshold=0.8,
        ),
        ScenarioMetric(
            scenario_id=agent_scenario.id,
            metric_definition_id=metric_objects["trace_lint"].id,
            weight=1.0,
            pass_threshold=0.8,
        ),
    ]
    db.add_all(agent_metrics)
    db.flush()

    # 3c. Multi-turn Conversation Evaluation Template
    multi_turn_scenario = EvalScenario(
        name="\u591a\u8f6e\u5bf9\u8bdd\u8bc4\u6d4b\u6a21\u677f",
        description="多轮对话核心评测模板，覆盖话题、轮次相关性、完成度、上下文记忆和角色一致性。",
        scene_type="multi_turn",
        sample_type="multi_turn",
        is_preset=True,
    )
    db.add(multi_turn_scenario)
    db.flush()

    multi_turn_metrics = [
        ScenarioMetric(
            scenario_id=multi_turn_scenario.id,
            metric_definition_id=metric_objects["topic_adherence"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=multi_turn_scenario.id,
            metric_definition_id=metric_objects["turn_relevancy"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=multi_turn_scenario.id,
            metric_definition_id=metric_objects["conversation_completeness"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=multi_turn_scenario.id,
            metric_definition_id=metric_objects["knowledge_retention"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=multi_turn_scenario.id,
            metric_definition_id=metric_objects["role_adherence"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
    ]
    db.add_all(multi_turn_metrics)
    db.flush()

    # ------------------------------------------------------------------
    # 4. Sample RAG Dataset (5 rows)
    # ------------------------------------------------------------------
    dataset = Dataset(
        name="RAG \u793a\u4f8b\u6570\u636e\u96c6",
        description="\u5305\u542b 5 \u6761 Python \u77e5\u8bc6\u95ee\u7b54\u7684\u793a\u4f8b\u6570\u636e\u96c6\uff0c\u7528\u4e8e\u6f14\u793a RAG \u8bc4\u6d4b\u6d41\u7a0b",
        sample_type="single_turn",
        field_schema=[
            {"name": "user_input", "type": "text", "required": True, "description": "\u7528\u6237\u8f93\u5165\u7684\u95ee\u9898"},
            {"name": "response", "type": "text", "required": True, "description": "\u6a21\u578b\u751f\u6210\u7684\u56de\u7b54"},
            {"name": "retrieved_contexts", "type": "text_list", "required": False, "description": "\u68c0\u7d22\u5230\u7684\u4e0a\u4e0b\u6587\u5217\u8868"},
            {"name": "retrieved_context_ids", "type": "text_list", "required": False, "description": "检索到的文档/分片 ID，顺序需与 retrieved_contexts 一致，用于 HitRate@K、MRR 等检索单测指标"},
            {"name": "reference_context_ids", "type": "text_list", "required": False, "description": "人工标注的标准文档/分片 ID，用于判断检索是否命中正确资料"},
            {"name": "reference", "type": "text", "required": False, "description": "\u53c2\u8003\u7b54\u6848"},
        ],
        row_count=5,
    )
    db.add(dataset)
    db.flush()

    rows_data = [
        {
            "user_input": "Python\u4e2d\u7684\u5217\u8868\u548c\u5143\u7ec4\u6709\u4ec0\u4e48\u533a\u522b\uff1f",
            "response": "\u5217\u8868\u662f\u53ef\u53d8\u7684\uff0c\u53ef\u4ee5\u6dfb\u52a0\u3001\u5220\u9664\u5143\u7d20\uff1b\u5143\u7ec4\u662f\u4e0d\u53ef\u53d8\u7684\uff0c\u521b\u5efa\u540e\u4e0d\u80fd\u4fee\u6539\u3002\u5217\u8868\u7528\u65b9\u62ec\u53f7[]\uff0c\u5143\u7ec4\u7528\u5706\u62ec\u53f7()\u3002",
            "retrieved_contexts": [
                "Python\u5217\u8868(list)\u662f\u53ef\u53d8\u5e8f\u5217\uff0c\u652f\u6301append\u3001insert\u3001remove\u7b49\u64cd\u4f5c\u3002",
                "Python\u5143\u7ec4(tuple)\u662f\u4e0d\u53ef\u53d8\u5e8f\u5217\uff0c\u4e00\u65e6\u521b\u5efa\u5c31\u4e0d\u80fd\u4fee\u6539\u5176\u5143\u7d20\u3002",
            ],
            "retrieved_context_ids": ["python-list-001", "python-tuple-001"],
            "reference_context_ids": ["python-list-001", "python-tuple-001"],
            "reference": "\u5217\u8868\u662f\u53ef\u53d8\u7684\u6709\u5e8f\u96c6\u5408\uff0c\u5143\u7ec4\u662f\u4e0d\u53ef\u53d8\u7684\u6709\u5e8f\u96c6\u5408\u3002\u5217\u8868\u7528[]\u8868\u793a\uff0c\u5143\u7ec4\u7528()\u8868\u793a\u3002",
        },
        {
            "user_input": "\u4ec0\u4e48\u662fPython\u7684\u88c5\u9970\u5668\uff1f",
            "response": "\u88c5\u9970\u5668\u662f\u4e00\u79cd\u7279\u6b8a\u7684\u51fd\u6570\uff0c\u53ef\u4ee5\u5728\u4e0d\u4fee\u6539\u539f\u51fd\u6570\u4ee3\u7801\u7684\u60c5\u51b5\u4e0b\uff0c\u4e3a\u51fd\u6570\u6dfb\u52a0\u989d\u5916\u529f\u80fd\u3002\u4f7f\u7528@\u7b26\u53f7\u653e\u5728\u51fd\u6570\u5b9a\u4e49\u524d\u3002",
            "retrieved_contexts": [
                "\u88c5\u9970\u5668\u672c\u8d28\u4e0a\u662f\u4e00\u4e2a\u63a5\u53d7\u51fd\u6570\u4f5c\u4e3a\u53c2\u6570\u5e76\u8fd4\u56de\u65b0\u51fd\u6570\u7684\u9ad8\u9636\u51fd\u6570\u3002",
                "Python\u88c5\u9970\u5668\u4f7f\u7528@\u8bed\u6cd5\u7cd6\uff0c\u53ef\u4ee5\u5b9e\u73b0\u65e5\u5fd7\u8bb0\u5f55\u3001\u6743\u9650\u68c0\u67e5\u7b49\u6a2a\u5207\u5173\u6ce8\u70b9\u3002",
            ],
            "retrieved_context_ids": ["python-decorator-001", "python-decorator-002"],
            "reference_context_ids": ["python-decorator-001"],
            "reference": "\u88c5\u9970\u5668\u662fPython\u7684\u4e00\u79cd\u8bbe\u8ba1\u6a21\u5f0f\uff0c\u5b83\u5141\u8bb8\u5728\u4e0d\u4fee\u6539\u539f\u51fd\u6570\u7684\u60c5\u51b5\u4e0b\u6269\u5c55\u51fd\u6570\u7684\u884c\u4e3a\u3002",
        },
        {
            "user_input": "\u5982\u4f55\u5904\u7406Python\u4e2d\u7684\u5f02\u5e38\uff1f",
            "response": "\u4f7f\u7528try-except\u8bed\u53e5\u5757\u3002try\u4e2d\u653e\u53ef\u80fd\u51fa\u9519\u7684\u4ee3\u7801\uff0cexcept\u6355\u83b7\u7279\u5b9a\u5f02\u5e38\u7c7b\u578b\u5e76\u5904\u7406\u3002\u8fd8\u53ef\u4ee5\u7528finally\u786e\u4fdd\u6e05\u7406\u4ee3\u7801\u6267\u884c\u3002",
            "retrieved_contexts": [
                "Python\u5f02\u5e38\u5904\u7406\u4f7f\u7528try/except/else/finally\u8bed\u53e5\u7ed3\u6784\u3002",
                "\u5e38\u89c1\u7684\u5f02\u5e38\u5305\u62ecValueError\u3001TypeError\u3001KeyError\u7b49\u3002",
            ],
            "retrieved_context_ids": ["python-exception-001", "python-exception-002"],
            "reference_context_ids": ["python-exception-001"],
            "reference": "Python\u4f7f\u7528try/except\u8bed\u53e5\u5904\u7406\u5f02\u5e38\uff0cexcept\u53ef\u4ee5\u6355\u83b7\u7279\u5b9a\u7c7b\u578b\u7684\u5f02\u5e38\uff0cfinally\u5757\u4e2d\u7684\u4ee3\u7801\u65e0\u8bba\u662f\u5426\u53d1\u751f\u5f02\u5e38\u90fd\u4f1a\u6267\u884c\u3002",
        },
        {
            "user_input": "Python\u7684GIL\u662f\u4ec0\u4e48\uff1f",
            "response": "GIL\u662f\u5168\u5c40\u89e3\u91ca\u5668\u9501\uff0c\u5b83\u786e\u4fdd\u540c\u4e00\u65f6\u95f4\u53ea\u6709\u4e00\u4e2a\u7ebf\u7a0b\u6267\u884cPython\u5b57\u8282\u7801\u3002\u8fd9\u610f\u5473\u7740Python\u7684\u591a\u7ebf\u7a0b\u65e0\u6cd5\u771f\u6b63\u5229\u7528\u591a\u6838CPU\u8fdb\u884c\u5e76\u884c\u8ba1\u7b97\u3002",
            "retrieved_contexts": [
                "GIL(Global Interpreter Lock)\u662fCPython\u89e3\u91ca\u5668\u4e2d\u7684\u4e00\u4e2a\u4e92\u65a5\u9501\u3002",
                "\u7531\u4e8eGIL\u7684\u5b58\u5728\uff0cCPU\u5bc6\u96c6\u578b\u4efb\u52a1\u5efa\u8bae\u4f7f\u7528\u591a\u8fdb\u7a0b\u800c\u975e\u591a\u7ebf\u7a0b\u3002",
            ],
            "retrieved_context_ids": ["python-gil-001", "python-gil-002"],
            "reference_context_ids": ["python-gil-001", "python-gil-002", "python-threading-io-001"],
            "reference": "GIL\uff08\u5168\u5c40\u89e3\u91ca\u5668\u9501\uff09\u662fCPython\u7684\u4e00\u4e2a\u673a\u5236\uff0c\u5b83\u4fdd\u8bc1\u540c\u4e00\u65f6\u523b\u53ea\u6709\u4e00\u4e2a\u7ebf\u7a0b\u6267\u884cPython\u5b57\u8282\u7801\uff0c\u9650\u5236\u4e86\u591a\u7ebf\u7a0b\u7684\u5e76\u884c\u6027\u80fd\u3002",
        },
        {
            "user_input": "\u4ec0\u4e48\u662fPython\u7684\u751f\u6210\u5668\uff1f",
            "response": "\u751f\u6210\u5668\u662f\u4f7f\u7528yield\u5173\u952e\u5b57\u7684\u51fd\u6570\uff0c\u5b83\u53ef\u4ee5\u6682\u505c\u6267\u884c\u5e76\u5728\u9700\u8981\u65f6\u6062\u590d\uff0c\u5b9e\u73b0\u60f0\u6027\u8ba1\u7b97\u3002\u751f\u6210\u5668\u4e0d\u4f1a\u4e00\u6b21\u6027\u751f\u6210\u6240\u6709\u503c\uff0c\u800c\u662f\u6309\u9700\u4ea7\u751f\u3002",
            "retrieved_contexts": [
                "\u751f\u6210\u5668\u51fd\u6570\u4f7f\u7528yield\u8bed\u53e5\u8fd4\u56de\u503c\uff0c\u6bcf\u6b21\u8c03\u7528next()\u65f6\u6062\u590d\u6267\u884c\u3002",
                "\u751f\u6210\u5668\u8868\u8fbe\u5f0f\u7c7b\u4f3c\u5217\u8868\u63a8\u5bfc\u5f0f\uff0c\u4f46\u4f7f\u7528\u5706\u62ec\u53f7\uff0c\u5982(x**2 for x in range(10))\u3002",
            ],
            "retrieved_context_ids": ["python-generator-001", "python-generator-002"],
            "reference_context_ids": ["python-generator-001"],
            "reference": "\u751f\u6210\u5668\u662f\u4e00\u79cd\u7279\u6b8a\u7684\u8fed\u4ee3\u5668\uff0c\u901a\u8fc7yield\u5173\u952e\u5b57\u5b9e\u73b0\u60f0\u6027\u6c42\u503c\uff0c\u53ef\u4ee5\u8282\u7701\u5185\u5b58\u3002",
        },
    ]

    dataset_rows = []
    for idx, row_data in enumerate(rows_data):
        row = DatasetRow(
            dataset_id=dataset.id,
            row_index=idx,
            data=row_data,
        )
        db.add(row)
        dataset_rows.append(row)
    db.flush()

    # ------------------------------------------------------------------
    # 5. Sample Completed Evaluation
    # ------------------------------------------------------------------
    now = datetime.now(UTC)
    eval_task = EvalTask(
        name="RAG \u793a\u4f8b\u8bc4\u6d4b",
        dataset_id=dataset.id,
        scenario_id=rag_scenario.id,
        llm_config_id=llm_config.id,
        status="completed",
        progress=1.0,
        total_rows=5,
        completed_rows=5,
        started_at=now - timedelta(minutes=5),
        finished_at=now,
        summary_scores={
            "faithfulness": {
                "mean": 0.87,
                "min": 0.67,
                "max": 1.0,
                "pass_rate": 0.8,
            },
            "context_recall": {
                "mean": 0.93,
                "min": 0.8,
                "max": 1.0,
                "pass_rate": 1.0,
            },
            "context_precision": {
                "mean": 0.88,
                "min": 0.75,
                "max": 1.0,
                "pass_rate": 1.0,
            },
            "answer_relevancy": {
                "mean": 0.94,
                "min": 0.9,
                "max": 1.0,
                "pass_rate": 1.0,
            },
            "factual_correctness": {
                "mean": 0.80,
                "min": 0.5,
                "max": 1.0,
                "pass_rate": 0.8,
            },
            "answer_completeness": {
                "mean": 0.81,
                "min": 0.6,
                "max": 0.95,
                "pass_rate": 0.8,
            },
        },
    )
    db.add(eval_task)
    db.flush()

    eval_row_results = [
        EvalRowResult(
            eval_task_id=eval_task.id,
            dataset_row_id=dataset_rows[0].id,
            row_index=0,
            metric_scores={
                "faithfulness": {
                    "score": 1.0,
                    "reason": "\u56de\u7b54\u5b8c\u5168\u57fa\u4e8e\u68c0\u7d22\u5230\u7684\u4e0a\u4e0b\u6587\u5185\u5bb9\uff0c\u6ca1\u6709\u6346\u9020\u4fe1\u606f",
                },
                "context_recall": {
                    "score": 1.0,
                    "reason": "\u53c2\u8003\u7b54\u6848\u4e2d\u7684\u6240\u6709\u8981\u70b9\u90fd\u80fd\u5728\u68c0\u7d22\u4e0a\u4e0b\u6587\u4e2d\u627e\u5230\u5bf9\u5e94\u4fe1\u606f",
                },
                "factual_correctness": {
                    "score": 1.0,
                    "reason": "\u56de\u7b54\u4e0e\u53c2\u8003\u7b54\u6848\u5728\u4e8b\u5b9e\u5c42\u9762\u5b8c\u5168\u4e00\u81f4",
                },
            },
            is_pass=True,
            execution_time_ms=1250,
            error=None,
        ),
        EvalRowResult(
            eval_task_id=eval_task.id,
            dataset_row_id=dataset_rows[1].id,
            row_index=1,
            metric_scores={
                "faithfulness": {
                    "score": 0.85,
                    "reason": "\u56de\u7b54\u5927\u90e8\u5206\u57fa\u4e8e\u4e0a\u4e0b\u6587\uff0c\u4f46\u201c\u4e3a\u51fd\u6570\u6dfb\u52a0\u989d\u5916\u529f\u80fd\u201d\u7684\u8868\u8ff0\u7565\u6709\u6982\u62ec",
                },
                "context_recall": {
                    "score": 0.9,
                    "reason": "\u53c2\u8003\u7b54\u6848\u7684\u6838\u5fc3\u89c2\u70b9\u5728\u68c0\u7d22\u4e0a\u4e0b\u6587\u4e2d\u6709\u8f83\u597d\u7684\u8986\u76d6",
                },
                "factual_correctness": {
                    "score": 0.8,
                    "reason": "\u56de\u7b54\u6b63\u786e\u63cf\u8ff0\u4e86\u88c5\u9970\u5668\u7684\u57fa\u672c\u6982\u5ff5\uff0c\u4e0e\u53c2\u8003\u7b54\u6848\u57fa\u672c\u4e00\u81f4",
                },
            },
            is_pass=True,
            execution_time_ms=1180,
            error=None,
        ),
        EvalRowResult(
            eval_task_id=eval_task.id,
            dataset_row_id=dataset_rows[2].id,
            row_index=2,
            metric_scores={
                "faithfulness": {
                    "score": 1.0,
                    "reason": "\u56de\u7b54\u5185\u5bb9\u5b8c\u5168\u53ef\u4ee5\u4ecetry/except/finally\u7684\u68c0\u7d22\u4e0a\u4e0b\u6587\u4e2d\u5f97\u5230\u652f\u6491",
                },
                "context_recall": {
                    "score": 0.95,
                    "reason": "\u4e0a\u4e0b\u6587\u5f88\u597d\u5730\u8986\u76d6\u4e86\u53c2\u8003\u7b54\u6848\u7684\u5173\u952e\u4fe1\u606f",
                },
                "factual_correctness": {
                    "score": 0.9,
                    "reason": "\u56de\u7b54\u51c6\u786e\u63cf\u8ff0\u4e86\u5f02\u5e38\u5904\u7406\u673a\u5236\uff0c\u4e0e\u53c2\u8003\u7b54\u6848\u9ad8\u5ea6\u4e00\u81f4",
                },
            },
            is_pass=True,
            execution_time_ms=1320,
            error=None,
        ),
        EvalRowResult(
            eval_task_id=eval_task.id,
            dataset_row_id=dataset_rows[3].id,
            row_index=3,
            metric_scores={
                "faithfulness": {
                    "score": 0.67,
                    "reason": "\u56de\u7b54\u63d0\u5230\u201c\u591a\u7ebf\u7a0b\u65e0\u6cd5\u771f\u6b63\u5229\u7528\u591a\u6838CPU\u201d\u5728\u4e0a\u4e0b\u6587\u4e2d\u4ec5\u6709\u95f4\u63a5\u652f\u6301\uff0c\u6709\u4e00\u5b9a\u7684\u5ef6\u4f38\u63a8\u7406",
                },
                "context_recall": {
                    "score": 0.8,
                    "reason": "\u53c2\u8003\u7b54\u6848\u4e2d\u5173\u4e8e\u201c\u9650\u5236\u5e76\u884c\u6027\u80fd\u201d\u7684\u8868\u8ff0\u5728\u4e0a\u4e0b\u6587\u4e2d\u6709\u90e8\u5206\u4f53\u73b0",
                },
                "factual_correctness": {
                    "score": 0.5,
                    "reason": "\u56de\u7b54\u57fa\u672c\u6b63\u786e\u4f46\u8fc7\u4e8e\u7edd\u5bf9\u5316\uff0c\u672a\u63d0\u53ca\u591a\u7ebf\u7a0b\u5728IO\u5bc6\u96c6\u578b\u4efb\u52a1\u4e2d\u4ecd\u7136\u6709\u6548",
                },
            },
            is_pass=False,
            execution_time_ms=1450,
            error=None,
        ),
        EvalRowResult(
            eval_task_id=eval_task.id,
            dataset_row_id=dataset_rows[4].id,
            row_index=4,
            metric_scores={
                "faithfulness": {
                    "score": 0.83,
                    "reason": "\u56de\u7b54\u5185\u5bb9\u57fa\u672c\u6765\u81ea\u4e0a\u4e0b\u6587\uff0c\u201c\u60f0\u6027\u8ba1\u7b97\u201d\u7684\u8868\u8ff0\u5728\u4e0a\u4e0b\u6587\u4e2d\u6709\u4f53\u73b0",
                },
                "context_recall": {
                    "score": 1.0,
                    "reason": "\u53c2\u8003\u7b54\u6848\u7684\u6240\u6709\u5173\u952e\u70b9\u5747\u80fd\u5728\u68c0\u7d22\u4e0a\u4e0b\u6587\u4e2d\u627e\u5230",
                },
                "factual_correctness": {
                    "score": 0.8,
                    "reason": "\u56de\u7b54\u6b63\u786e\u63cf\u8ff0\u4e86\u751f\u6210\u5668\u7684\u6838\u5fc3\u7279\u6027\uff0c\u4e0e\u53c2\u8003\u7b54\u6848\u4e00\u81f4",
                },
            },
            is_pass=True,
            execution_time_ms=1100,
            error=None,
        ),
    ]
    extra_rag_scores = [
        {
            "context_precision": {"score": 1.0, "reason": "检索片段均与问题和参考答案直接相关。"},
            "answer_relevancy": {"score": 1.0, "reason": "回答紧扣列表和元组区别，没有跑题。"},
            "answer_completeness": {"score": 0.95, "reason": "覆盖可变性、不可变性和语法表示，关键点完整。"},
        },
        {
            "context_precision": {"score": 0.9, "reason": "两段上下文均有用，第二段偏示例扩展。"},
            "answer_relevancy": {"score": 0.95, "reason": "回答围绕装饰器定义和用法展开。"},
            "answer_completeness": {"score": 0.8, "reason": "覆盖核心概念，但对高阶函数本质说明略少。"},
        },
        {
            "context_precision": {"score": 0.95, "reason": "异常处理结构片段高度相关，常见异常片段作为补充有用。"},
            "answer_relevancy": {"score": 0.95, "reason": "回答直接说明异常处理方式。"},
            "answer_completeness": {"score": 0.9, "reason": "覆盖 try/except/finally，略少 else 说明。"},
        },
        {
            "context_precision": {"score": 0.75, "reason": "上下文与 GIL 相关，但缺少 IO 密集型多线程有效性的补充片段。"},
            "answer_relevancy": {"score": 0.9, "reason": "回答与 GIL 问题相关。"},
            "answer_completeness": {"score": 0.6, "reason": "未覆盖多线程在 IO 密集型任务中仍有价值，答案不够完整。"},
        },
        {
            "context_precision": {"score": 0.8, "reason": "第一段直接相关，第二段生成器表达式为补充信息。"},
            "answer_relevancy": {"score": 0.9, "reason": "回答围绕生成器定义和惰性计算。"},
            "answer_completeness": {"score": 0.8, "reason": "覆盖 yield 和惰性求值，但节省内存表述不够明确。"},
        },
    ]
    for row_result, extra_scores in zip(eval_row_results, extra_rag_scores):
        row_result.metric_scores.update(extra_scores)

    db.add_all(eval_row_results)

    # ------------------------------------------------------------------
    # 6. Sample Agent Dataset (3 rows) - 演示工具调用评测
    # ------------------------------------------------------------------
    agent_dataset = Dataset(
        name="Agent 工具调用示例数据集",
        description="包含 3 条客服 Agent 对话的示例数据，演示工具调用评测流程。字段含义：user_input=多轮对话轨迹（Human/AI/Tool 消息交替），reference=期望达成的目标，reference_tool_calls=期望调用的工具列表",
        sample_type="multi_turn",
        field_schema=[
            {"name": "user_input", "type": "conversation", "required": True, "description": "多轮对话轨迹：Human发问→AI决策→Tool执行→AI回复，type只能是human/ai/tool三种"},
            {"name": "reference", "type": "text", "required": True, "description": "期望Agent最终应达成的目标（如：成功查询到订单状态）"},
            {"name": "reference_tool_calls", "type": "tool_call_list", "required": True, "description": "期望Agent应该调用的工具列表，每个含name和args"},
            {"name": "reference_topics", "type": "text_list", "required": False, "description": "该 Agent 任务允许围绕的话题列表，用于 Topic Adherence 等多轮对话指标"},
            {"name": "reference_role", "type": "text", "required": False, "description": "Agent 在任务中应遵守的角色和职责边界"},
            {"name": "retrieved_contexts", "type": "text_list", "required": False, "description": "任务相关的工具/业务背景资料，用于 Turn Faithfulness 等可信度指标"},
            *AGENT_TRACE_SCHEMA_FIELDS,
        ],
        row_count=3,
    )
    db.add(agent_dataset)
    db.flush()

    agent_rows_data = [
        {
            "user_input": [
                {"type": "human", "content": "帮我查一下订单 ORD-20240101 的状态"},
                {"type": "ai", "content": "", "tool_calls": [{"name": "query_order", "args": {"order_id": "ORD-20240101"}}]},
                {"type": "tool", "content": "{\"status\": \"shipped\", \"tracking_no\": \"SF1234567890\"}"},
                {"type": "ai", "content": "您的订单 ORD-20240101 已发货，快递单号是 SF1234567890，预计明天送达。"},
            ],
            "reference": "成功查询到订单状态并告知用户快递信息",
            "reference_tool_calls": [
                {"name": "query_order", "args": {"order_id": "ORD-20240101"}},
            ],
            "reference_topics": ["订单查询", "物流跟踪", "客服售后"],
            "reference_role": "客服 Agent：只能基于工具返回结果答复订单状态，不编造物流信息。",
            "retrieved_contexts": ["订单查询工具 query_order 返回订单状态和快递单号后，Agent 应把状态、快递单号和预计送达时间告知用户。"],
        },
        {
            "user_input": [
                {"type": "human", "content": "我要退掉订单 ORD-20240202 里的手机壳"},
                {"type": "ai", "content": "好的，我先查询一下您的订单信息。", "tool_calls": [{"name": "query_order", "args": {"order_id": "ORD-20240202"}}]},
                {"type": "tool", "content": "{\"status\": \"delivered\", \"items\": [{\"name\": \"手机壳\", \"price\": 29.9}]}"},
                {"type": "ai", "content": "已查到订单，手机壳 29.9 元已签收。我为您发起退货申请。", "tool_calls": [{"name": "create_return", "args": {"order_id": "ORD-20240202", "item": "手机壳", "reason": "用户要求退货"}}]},
                {"type": "tool", "content": "{\"return_id\": \"RET-001\", \"status\": \"pending\"}"},
                {"type": "ai", "content": "退货申请已提交（编号 RET-001），预计 3 个工作日内处理。退款将原路返回您的支付账户。"},
            ],
            "reference": "成功查询订单并提交退货申请",
            "reference_tool_calls": [
                {"name": "query_order", "args": {"order_id": "ORD-20240202"}},
                {"name": "create_return", "args": {"order_id": "ORD-20240202", "item": "手机壳", "reason": "用户要求退货"}},
            ],
            "reference_topics": ["订单查询", "退货申请", "售后服务"],
            "reference_role": "客服 Agent：先查询订单，再按工具结果提交退货申请，不越权承诺即时退款。",
            "retrieved_contexts": ["退货流程要求先 query_order 确认订单和商品，再 create_return 创建退货申请；退款以工具返回和售后规则为准。"],
        },
        {
            "user_input": [
                {"type": "human", "content": "查一下北京明天天气，然后帮我发给张经理"},
                {"type": "ai", "content": "好的，我先查询北京明天的天气。", "tool_calls": [{"name": "get_weather", "args": {"city": "北京", "date": "tomorrow"}}]},
                {"type": "tool", "content": "{\"weather\": \"晴\", \"temp\": \"15-25℃\", \"wind\": \"北风3级\"}"},
                {"type": "ai", "content": "北京明天晴，15-25℃，北风3级。现在帮您发送给张经理。", "tool_calls": [{"name": "send_message", "args": {"to": "张经理", "content": "北京明天天气：晴，15-25℃，北风3级"}}]},
                {"type": "tool", "content": "{\"success\": true, \"msg_id\": \"MSG-001\"}"},
                {"type": "ai", "content": "天气信息已发送给张经理，消息编号 MSG-001。"},
            ],
            "reference": "成功查询天气并发送给指定联系人",
            "reference_tool_calls": [
                {"name": "get_weather", "args": {"city": "北京", "date": "tomorrow"}},
                {"name": "send_message", "args": {"to": "张经理", "content": "北京明天天气：晴，15-25℃，北风3级"}},
            ],
            "reference_topics": ["天气查询", "消息发送", "任务执行"],
            "reference_role": "任务 Agent：按用户要求查询天气并发送消息，只报告工具返回的天气和发送结果。",
            "retrieved_contexts": ["天气查询工具返回城市、日期、天气、温度和风力；发送消息工具返回 success=true 时才可告知已发送。"],
        },
    ]

    agent_rows_data = [_enrich_agent_trace_row(row_data) for row_data in agent_rows_data]
    for idx, row_data in enumerate(agent_rows_data):
        db.add(DatasetRow(dataset_id=agent_dataset.id, row_index=idx, data=row_data))
    db.flush()

    # ------------------------------------------------------------------
    # 7. Sample Multi-turn Conversation Dataset (3 rows) - 演示多轮对话评测
    # ------------------------------------------------------------------
    multi_turn_dataset = Dataset(
        name="多轮对话示例数据集",
        description="包含 3 条客服多轮对话的示例数据，演示对话质量评测。字段含义：user_input=完整对话记录，reference=期望的对话结果，reference_topics=对话应围绕的话题列表",
        sample_type="multi_turn",
        field_schema=[
            {"name": "user_input", "type": "conversation", "required": True, "description": "完整多轮对话记录（Human和AI交替对话）"},
            {"name": "reference", "type": "text", "required": False, "description": "期望的对话最终结果"},
            {"name": "reference_topics", "type": "text_list", "required": True, "description": "对话应围绕的话题列表，Topic Adherence 指标必需"},
            {"name": "reference_role", "type": "text", "required": True, "description": "AI 在对话中应遵守的角色、职责边界和语气要求"},
            {"name": "retrieved_contexts", "type": "text_list", "required": False, "description": "多轮对话可依据的业务资料，用于 Turn Faithfulness 等可信度指标"},
        ],
        row_count=3,
    )
    db.add(multi_turn_dataset)
    db.flush()

    multi_turn_rows_data = [
        {
            "user_input": [
                {"type": "human", "content": "你好，我想了解一下你们的退货政策"},
                {"type": "ai", "content": "您好！我们的退货政策如下：商品签收后 7 天内可以无理由退货，需要保持商品完好。请问您是想退哪个订单的商品呢？"},
                {"type": "human", "content": "上周买的那个蓝牙耳机，音质不太好"},
                {"type": "ai", "content": "好的，蓝牙耳机在签收 7 天内是可以退货的。请问您有订单号吗？我帮您查一下是否在退货期限内。"},
                {"type": "human", "content": "订单号是 ORD-12345"},
                {"type": "ai", "content": "查到了，您的蓝牙耳机订单 ORD-12345 在退货期限内。我现在帮您发起退货申请，退款将在 3 个工作日内原路返回。请问还有其他需要帮助的吗？"},
                {"type": "human", "content": "没有了，谢谢"},
                {"type": "ai", "content": "不客气！退货申请已为您提交，如有问题随时联系我们。祝您生活愉快！"},
            ],
            "reference": "完成退货咨询和申请提交",
            "reference_topics": ["退货政策", "售后服务"],
            "reference_role": "电商客服助手：专业、礼貌，只处理售前售后咨询，不越权承诺无法确认的结果。",
            "retrieved_contexts": ["退货政策：商品签收后 7 天内可无理由退货，需保持商品完好；退货申请提交后退款通常 3 个工作日内原路返回。"],
        },
        {
            "user_input": [
                {"type": "human", "content": "我买的笔记本电脑有问题，开机后屏幕一直闪烁"},
                {"type": "ai", "content": "非常抱歉给您带来不便！屏幕闪烁可能是硬件问题。请问您的电脑型号是什么？购买多长时间了？"},
                {"type": "human", "content": "型号是 ThinkPad X1，上个月买的"},
                {"type": "ai", "content": "ThinkPad X1 购买一个月内在保修期内。建议您先尝试以下步骤：1) 重启电脑 2) 更新显卡驱动。如果问题仍然存在，我们可以安排免费上门维修。"},
                {"type": "human", "content": "我试过重启了，没用。能不能直接换一台新的？"},
                {"type": "ai", "content": "理解您的心情。根据三包政策，购买 15 天内出现性能故障可以换货。您的电脑已超过 15 天，但在保修期内可以免费维修。我帮您预约明天的上门维修服务，可以吗？"},
                {"type": "human", "content": "好吧，那就约明天下午"},
                {"type": "ai", "content": "好的，已为您预约明天下午 2:00-4:00 的上门维修服务。工程师会提前电话联系您确认。如有其他问题随时联系我们。"},
            ],
            "reference": "诊断硬件问题并安排维修服务",
            "reference_topics": ["产品故障", "保修政策", "维修服务"],
            "reference_role": "电商客服助手：专业、安抚用户，依据保修和三包规则提供维修或换货建议。",
            "retrieved_contexts": ["三包政策：购买 15 天内出现性能故障可换货；超过 15 天但在保修期内，可安排免费维修服务。"],
        },
        {
            "user_input": [
                {"type": "human", "content": "请问你们有什么优惠活动吗？"},
                {"type": "ai", "content": "目前我们有以下优惠活动：1) 618 大促全场满 300 减 50  2) 新用户首单 9 折  3) 会员积分兑换优惠券。请问您对哪个活动感兴趣？"},
                {"type": "human", "content": "满300减50不错，我想买个扫地机器人，有推荐的吗？"},
                {"type": "ai", "content": "为您推荐几款热销扫地机器人：1) 小米 Pro 2，1999元，智能避障 2) 石头 T8，2499元，自清洁 3) 科沃斯 X2，3999元，全能旗舰。这三款都参与满减活动。"},
                {"type": "human", "content": "小米那个不错，帮我下单吧"},
                {"type": "ai", "content": "抱歉，我是客服助手，暂时无法直接帮您下单。您可以在 APP 商品页点击购买，结算时优惠券会自动使用。需要我把商品链接发给您吗？"},
                {"type": "human", "content": "好的，发给我"},
                {"type": "ai", "content": "已将小米扫地机器人 Pro 2 的商品链接发送到您的消息中。下单时记得勾选满 300 减 50 优惠券哦！"},
            ],
            "reference": "推荐商品并引导用户完成购买",
            "reference_topics": ["促销活动", "商品推荐", "购物指导"],
            "reference_role": "电商客服助手：介绍优惠和商品信息，引导用户自助下单，不直接代替用户完成支付。",
            "retrieved_contexts": ["促销规则：618 活动全场满 300 减 50，新用户首单 9 折，会员积分可兑换优惠券；客服助手可发送商品链接但不能直接代用户付款。"],
        },
    ]

    for idx, row_data in enumerate(multi_turn_rows_data):
        db.add(DatasetRow(dataset_id=multi_turn_dataset.id, row_index=idx, data=row_data))
    db.flush()

    db.commit()
    print("[seed] Database seeded successfully with 3 scenario datasets (RAG/Agent/Multi-turn).")
