"""
Evaluation engine for the AI Evaluation Platform.

The engine owns metric execution instead of depending on a specific third-party
metric framework. LLM-based metrics use the configured OpenAI-compatible model
directly and persist both the score and the judge's concrete reason.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import typing as t
from datetime import datetime, timezone

from app.core.scenario_snapshot import snapshot_to_scenario_metrics

logger = logging.getLogger(__name__)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
DEFAULT_SUMMARY_PASS_THRESHOLD = 0.7


_BUILTIN_LLM_METRIC_SPECS: dict[str, dict[str, t.Any]] = {
    "builtin_faithfulness": {
        "required_fields": ["response", "retrieved_contexts"],
        "criteria": (
            "判断 AI 回答是否忠实于检索上下文。回答中的关键事实必须能从 retrieved_contexts "
            "得到直接或合理支持；捏造、不被上下文支持或与上下文冲突的内容需要扣分。"
        ),
    },
    "builtin_context_recall": {
        "required_fields": ["retrieved_contexts", "reference"],
        "criteria": (
            "判断检索上下文是否覆盖参考答案中的关键事实和必要依据。reference 中的重要要点如果在 "
            "retrieved_contexts 中找不到对应支持，需要扣分。"
        ),
    },
    "builtin_context_precision": {
        "required_fields": ["user_input", "retrieved_contexts"],
        "criteria": (
            "判断检索上下文与用户问题及参考答案是否相关、有用。无关、重复、噪声或无法支撑回答的片段越多，"
            "分数越低。"
        ),
    },
    "builtin_contextual_relevancy": {
        "required_fields": ["user_input", "retrieved_contexts"],
        "criteria": (
            "判断检索上下文整体是否与用户问题直接相关。上下文应围绕用户问题提供可用信息；无关、泛化、"
            "只沾边或无法帮助回答的问题片段需要扣分。"
        ),
    },
    "builtin_factual_correctness": {
        "required_fields": ["response", "reference"],
        "criteria": (
            "判断 AI 回答与参考答案在事实层面是否一致。事实错误、遗漏关键限定条件、与 reference 冲突的内容"
            "需要扣分。"
        ),
    },
    "builtin_answer_relevancy": {
        "required_fields": ["user_input", "response"],
        "criteria": (
            "判断 AI 回答是否直接回应用户问题。跑题、泛泛而谈、答非所问或只回答了问题的一小部分需要扣分。"
        ),
    },
    "builtin_agent_goal_accuracy": {
        "required_fields": ["user_input", "reference"],
        "criteria": (
            "判断 Agent 在整段对话中是否完成了用户目标，并与 reference 中的期望结果一致。工具调用、最终回复"
            "和中间步骤都可以作为证据。"
        ),
    },
    "builtin_task_completion": {
        "required_fields": ["user_input", "reference"],
        "criteria": (
            "判断 Agent 是否完成了用户要达成的任务闭环。重点看最终状态是否已经满足 reference 描述的任务目标，"
            "而不只看是否给出看似合理的回复。"
        ),
    },
    "builtin_topic_adherence": {
        "required_fields": ["user_input", "reference_topics"],
        "criteria": (
            "判断多轮对话是否始终围绕 reference_topics 指定的话题范围展开。明显偏离主题、引入无关内容或没有"
            "回应当前轮次主题需要扣分。"
        ),
    },
    "builtin_turn_relevancy": {
        "required_fields": ["user_input"],
        "criteria": (
            "逐轮判断 AI 回复是否回应了当前轮用户输入，并且没有忽略用户追问、答非所问或把上一轮上下文错误带入。"
        ),
    },
    "builtin_conversation_completeness": {
        "required_fields": ["user_input", "reference"],
        "criteria": (
            "判断整段多轮对话是否满足用户需求并完成 reference 描述的对话目标。遗漏关键步骤、没有收束问题或"
            "未给出可执行结论需要扣分。"
        ),
    },
    "builtin_knowledge_retention": {
        "required_fields": ["user_input"],
        "criteria": (
            "判断 AI 是否在多轮对话中持续记住用户已经提供的事实、约束、偏好、订单号、时间等信息。忘记、混淆"
            "或自相矛盾需要扣分。"
        ),
    },
    "builtin_role_adherence": {
        "required_fields": ["user_input", "reference_role"],
        "criteria": (
            "判断 AI 是否始终遵守 reference_role 描述的角色、职责边界和语气要求。越权承诺、角色漂移、"
            "使用不合适语气或执行角色不允许的动作需要扣分。"
        ),
    },
    "builtin_turn_faithfulness": {
        "required_fields": ["user_input", "retrieved_contexts"],
        "criteria": (
            "判断多轮对话中每轮 AI 回复是否基于对应的检索上下文或已给定资料。跨轮捏造事实、与上下文冲突或"
            "无法从资料支持的回答需要扣分。"
        ),
    },
}


_NAMED_LLM_METRIC_SPECS: dict[str, dict[str, t.Any]] = {
    "answer_completeness": {
        "required_fields": ["response", "reference"],
        "criteria": (
            "判断 AI 回答是否完整覆盖参考答案中的关键要点。遗漏主要结论、条件、步骤或重要限定需要扣分；"
            "表达顺序不同但语义完整可以给高分。"
        ),
    },
}


class _MetricResult:
    """Small result object shared by all metric implementations."""

    def __init__(self, value: t.Any, reason: str):
        self.value = value
        self.reason = reason


class OpenAIJudgeClient:
    """OpenAI-compatible JSON judge client used by native LLM metrics."""

    def __init__(self, llm_config):
        from openai import AsyncOpenAI

        self.model = llm_config.model_name
        self.temperature = llm_config.temperature if llm_config.temperature is not None else 0.01
        self.max_tokens = min(int(llm_config.max_tokens or 1024), 1200)
        self.client = AsyncOpenAI(
            base_url=llm_config.api_base_url,
            api_key=llm_config.api_key,
            timeout=180,
            max_retries=2,
        )

    async def judge_json(self, payload: dict[str, t.Any]) -> dict[str, t.Any]:
        system_prompt = (
            "你是一个严谨的 AI 评测裁判。只能依据用户提供的样本字段评分，不要引入外部知识或想象证据。"
            "必须返回严格 JSON，格式为 {\"score\": 数值或标签, \"reason\": \"1到2句中文理由\"}。"
            "reason 必须指出具体支撑点或扣分点，不要写空泛结论。"
        )
        user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.warning("Judge JSON mode failed, retrying without response_format: %s", exc)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        content = response.choices[0].message.content or ""
        return _parse_json_object(content)


class NativeBuiltinLLMMetric:
    """Built-in metric implemented by our own judge prompt."""

    def __init__(self, metric_def, spec: dict[str, t.Any]):
        self.name = metric_def.name
        self.display_name = metric_def.display_name
        self.metric_type = metric_def.metric_type
        self.config = metric_def.config or {}
        self.required_fields = list(
            self.config.get("required_fields") or spec.get("required_fields") or []
        )
        self.criteria = spec["criteria"]
        self.description = self.config.get("description")

    async def ascore(self, row_data: dict[str, t.Any], judge: OpenAIJudgeClient) -> _MetricResult:
        missing = _missing_required_fields(row_data, self.required_fields)
        if missing:
            return _MetricResult(None, f"缺少必需字段: {', '.join(missing)}。")

        payload = {
            "metric": {
                "name": self.name,
                "display_name": self.display_name,
                "score_range": "0 到 1，1 表示完全满足指标，0 表示完全不满足",
                "criteria": self.criteria,
                "description": self.description,
                "required_fields": self.required_fields,
            },
            "sample": _sample_payload(row_data),
            "instruction": (
                "请给出 0 到 1 的浮点分数，并用中文说明理由。理由必须基于 sample 中的具体字段，"
                "指出哪里支持高分或哪里导致扣分。"
            ),
        }
        result = await judge.judge_json(payload)
        score = _coerce_float(result.get("score"))
        reason = _clean_reason(result.get("reason"))
        if score is None:
            raise ValueError(f"Judge did not return a numeric score for {self.name}")
        if not reason:
            raise ValueError(f"Judge did not return a reason for {self.name}")
        return _MetricResult(round(_clamp(score, 0.0, 1.0), 4), reason)


class NativePromptMetric:
    """User-configured prompt metric implemented by the platform judge."""

    def __init__(self, metric_def, mode: str):
        self.name = metric_def.name
        self.display_name = metric_def.display_name
        self.mode = mode
        self.config = metric_def.config or {}

    async def ascore(self, row_data: dict[str, t.Any], judge: OpenAIJudgeClient) -> _MetricResult:
        if self.mode == "numeric":
            return await self._score_numeric(row_data, judge)
        if self.mode == "discrete":
            return await self._score_discrete(row_data, judge)
        if self.mode == "aspect_critic":
            return await self._score_aspect(row_data, judge)
        raise ValueError(f"Unsupported prompt metric mode: {self.mode}")

    async def _score_numeric(
        self, row_data: dict[str, t.Any], judge: OpenAIJudgeClient
    ) -> _MetricResult:
        raw_range = self.config.get("allowed_values", [0.0, 1.0])
        lower, upper = float(raw_range[0]), float(raw_range[1])
        prompt = _render_prompt(self.config.get("prompt", ""), row_data)
        payload = {
            "metric": {
                "name": self.name,
                "display_name": self.display_name,
                "score_range": f"{lower} 到 {upper}",
                "criteria": self.config.get("description") or prompt,
            },
            "prompt": prompt,
            "sample": _sample_payload(row_data),
            "instruction": "请按 prompt 的标准评分，返回 JSON: {\"score\": 数值, \"reason\": \"中文理由\"}。",
        }
        result = await judge.judge_json(payload)
        score = _coerce_float(result.get("score"))
        reason = _clean_reason(result.get("reason"))
        if score is None:
            raise ValueError(f"Judge did not return a numeric score for {self.name}")
        if not reason:
            raise ValueError(f"Judge did not return a reason for {self.name}")
        return _MetricResult(round(_clamp(score, lower, upper), 4), reason)

    async def _score_discrete(
        self, row_data: dict[str, t.Any], judge: OpenAIJudgeClient
    ) -> _MetricResult:
        allowed_values = [str(v) for v in self.config.get("allowed_values", ["pass", "fail"])]
        prompt = _render_prompt(self.config.get("prompt", ""), row_data)
        payload = {
            "metric": {
                "name": self.name,
                "display_name": self.display_name,
                "allowed_values": allowed_values,
                "criteria": self.config.get("description") or prompt,
            },
            "prompt": prompt,
            "sample": _sample_payload(row_data),
            "instruction": "score 必须严格使用 allowed_values 中的一个值，并返回中文 reason。",
        }
        result = await judge.judge_json(payload)
        score = str(result.get("score", "")).strip()
        reason = _clean_reason(result.get("reason"))
        if score not in allowed_values:
            score = _match_allowed_value(score, allowed_values)
        if not score:
            raise ValueError(f"Judge did not return an allowed value for {self.name}")
        if not reason:
            raise ValueError(f"Judge did not return a reason for {self.name}")
        return _MetricResult(score, reason)

    async def _score_aspect(
        self, row_data: dict[str, t.Any], judge: OpenAIJudgeClient
    ) -> _MetricResult:
        definition = self.config.get("definition") or self.config.get("description") or self.display_name
        payload = {
            "metric": {
                "name": self.name,
                "display_name": self.display_name,
                "score_range": "0 或 1，1 表示样本满足 definition，0 表示不满足",
                "definition": definition,
                "description": self.config.get("description"),
            },
            "sample": _sample_payload(row_data),
            "instruction": "判断样本是否满足 definition，返回 0 或 1，并说明具体理由。",
        }
        result = await judge.judge_json(payload)
        score = _coerce_float(result.get("score"))
        reason = _clean_reason(result.get("reason"))
        if score is None:
            raise ValueError(f"Judge did not return a numeric score for {self.name}")
        if not reason:
            raise ValueError(f"Judge did not return a reason for {self.name}")
        return _MetricResult(1.0 if score >= 0.5 else 0.0, reason)


class RetrievalHitRateAtK:
    """Deterministic retrieval metric: whether any expected document id is in top-k."""

    def __init__(self, k: int = 5):
        self.k = k

    def score(
        self,
        retrieved_context_ids: list[str] | None = None,
        reference_context_ids: list[str] | None = None,
        **_: t.Any,
    ) -> _MetricResult:
        row_data = {
            "retrieved_context_ids": retrieved_context_ids,
            "reference_context_ids": reference_context_ids,
        }
        retrieved = list(row_data.get("retrieved_context_ids") or [])[: self.k]
        expected = set(row_data.get("reference_context_ids") or [])
        if not expected:
            return _MetricResult(None, "缺少 reference_context_ids，无法计算 HitRate@K。")
        hit = bool(set(retrieved) & expected)
        return _MetricResult(
            1.0 if hit else 0.0,
            f"HitRate@{self.k}: top-{self.k} 检索 ID 为 {retrieved}，标准 ID 为 {sorted(expected)}。",
        )

    async def ascore(self, row_data: dict[str, t.Any], _judge: OpenAIJudgeClient) -> _MetricResult:
        return self.score(
            retrieved_context_ids=row_data.get("retrieved_context_ids"),
            reference_context_ids=row_data.get("reference_context_ids"),
        )


class RetrievalMRR:
    """Deterministic retrieval metric: reciprocal rank of first expected document id."""

    def score(
        self,
        retrieved_context_ids: list[str] | None = None,
        reference_context_ids: list[str] | None = None,
        **_: t.Any,
    ) -> _MetricResult:
        retrieved = list(retrieved_context_ids or [])
        expected = set(reference_context_ids or [])
        if not expected:
            return _MetricResult(None, "缺少 reference_context_ids，无法计算 MRR。")
        for idx, doc_id in enumerate(retrieved, start=1):
            if doc_id in expected:
                return _MetricResult(
                    round(1.0 / idx, 4),
                    f"第一个命中的标准文档 ID 为 {doc_id!r}，位于第 {idx} 位。",
                )
        return _MetricResult(0.0, "retrieved_context_ids 中没有命中任何标准文档 ID。")

    async def ascore(self, row_data: dict[str, t.Any], _judge: OpenAIJudgeClient) -> _MetricResult:
        return self.score(
            retrieved_context_ids=row_data.get("retrieved_context_ids"),
            reference_context_ids=row_data.get("reference_context_ids"),
        )


class ToolCallAccuracyMetric:
    """Deterministic Agent metric comparing actual and expected tool calls."""

    async def ascore(self, row_data: dict[str, t.Any], _judge: OpenAIJudgeClient) -> _MetricResult:
        expected = list(row_data.get("reference_tool_calls") or [])
        actual = _extract_actual_tool_calls(row_data.get("user_input"))
        if not expected:
            return _MetricResult(None, "缺少 reference_tool_calls，无法计算工具调用准确度。")

        unmatched = list(actual)
        matched = 0
        for expected_call in expected:
            match_idx = _find_tool_call_match(expected_call, unmatched)
            if match_idx is not None:
                matched += 1
                unmatched.pop(match_idx)

        score = round(matched / len(expected), 4)
        return _MetricResult(
            score,
            f"期望工具调用 {len(expected)} 个，实际调用 {len(actual)} 个，完全匹配 {matched} 个。",
        )


class ArgumentCorrectnessMetric:
    """Deterministic Agent metric focused on arguments after the tool name matches."""

    async def ascore(self, row_data: dict[str, t.Any], _judge: OpenAIJudgeClient) -> _MetricResult:
        expected = list(row_data.get("reference_tool_calls") or [])
        actual = _extract_actual_tool_calls(row_data.get("user_input"))
        if not expected:
            return _MetricResult(None, "缺少 reference_tool_calls，无法计算参数正确性。")

        unmatched = list(actual)
        per_call_scores: list[float] = []
        matched_names = 0
        for expected_call in expected:
            match_idx, arg_score = _find_best_tool_name_match(expected_call, unmatched)
            if match_idx is not None:
                matched_names += 1
                per_call_scores.append(arg_score)
                unmatched.pop(match_idx)
            else:
                per_call_scores.append(0.0)

        score = round(sum(per_call_scores) / len(expected), 4)
        return _MetricResult(
            score,
            f"期望工具调用 {len(expected)} 个，找到同名实际调用 {matched_names} 个，参数平均匹配度 {score}。",
        )


class StepEfficiencyMetric:
    """Deterministic Agent metric penalizing missing or unnecessary tool steps."""

    async def ascore(self, row_data: dict[str, t.Any], _judge: OpenAIJudgeClient) -> _MetricResult:
        expected = list(row_data.get("reference_tool_calls") or [])
        actual = _extract_actual_tool_calls(row_data.get("user_input"))
        if not expected:
            return _MetricResult(None, "缺少 reference_tool_calls，无法计算步骤效率。")

        expected_count = len(expected)
        actual_count = len(actual)
        if actual_count == 0:
            return _MetricResult(0.0, f"期望 {expected_count} 个工具步骤，但实际没有调用工具。")
        if actual_count == expected_count:
            score = 1.0
        elif actual_count > expected_count:
            score = expected_count / actual_count
        else:
            score = actual_count / expected_count
        score = round(score, 4)
        return _MetricResult(
            score,
            f"期望工具步骤 {expected_count} 个，实际工具步骤 {actual_count} 个；步骤数量越接近期望越高分。",
        )


def build_metric(metric_def, llm=None) -> tuple[str, t.Any]:
    """
    Instantiate a metric executor from a MetricDefinition database row.

    The default path is native and OpenAI-compatible. This keeps the platform
    independent from third-party metric internals while still supporting configurable judge
    prompts and deterministic retrieval metrics.
    """
    metric_type: str = metric_def.metric_type
    config: dict = metric_def.config or {}

    spec = _NAMED_LLM_METRIC_SPECS.get(metric_def.name) or _BUILTIN_LLM_METRIC_SPECS.get(metric_type)
    if spec:
        return ("llm", NativeBuiltinLLMMetric(metric_def, spec))

    if metric_type == "code_retrieval_hit_rate":
        return ("simple", RetrievalHitRateAtK(k=int(config.get("k", 5))))

    if metric_type == "code_retrieval_mrr":
        return ("simple", RetrievalMRR())

    if metric_type == "builtin_tool_call_accuracy":
        return ("simple", ToolCallAccuracyMetric())

    if metric_type == "builtin_argument_correctness":
        return ("simple", ArgumentCorrectnessMetric())

    if metric_type == "builtin_step_efficiency":
        return ("simple", StepEfficiencyMetric())

    if metric_type in {"numeric", "discrete", "aspect_critic"}:
        return ("llm", NativePromptMetric(metric_def, mode=metric_type))

    raise ValueError(f"Unknown metric_type: {metric_type!r}")


async def run_evaluation(task_id: int, session_factory) -> None:
    """
    Execute a full evaluation run for the given EvalTask.

    Designed to be launched with ``asyncio.create_task(...)`` from the API
    endpoint. ``session_factory`` is the sync ``SessionLocal`` class, so the
    background runner opens its own session.
    """
    from app.models.dataset import DatasetRow
    from app.models.evaluation import EvalTask
    from app.models.metric_definition import MetricDefinition
    from app.models.scenario import ScenarioMetric

    db = session_factory()

    def _log(task_obj, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        task_obj.logs = (task_obj.logs or "") + line
        logger.info(msg)

    try:
        task = db.query(EvalTask).get(task_id)
        if task is None:
            logger.error("EvalTask %d not found -- aborting", task_id)
            return

        dataset = task.dataset
        scenario = task.scenario
        scenario_snapshot = task.scenario_snapshot or {}
        llm_config = task.llm_config

        scenario_metrics = snapshot_to_scenario_metrics(scenario_snapshot)
        if not scenario_metrics:
            scenario_metrics = (
                db.query(ScenarioMetric)
                .filter(ScenarioMetric.scenario_id == scenario.id)
                .all()
            )
            for sm in scenario_metrics:
                _ = sm.metric_definition

        dataset_rows: list[DatasetRow] = (
            db.query(DatasetRow)
            .filter(DatasetRow.dataset_id == dataset.id)
            .order_by(DatasetRow.row_index)
            .all()
        )
        total_rows = len(dataset_rows)
        if total_rows == 0:
            task.status = "failed"
            task.total_rows = 0
            task.completed_rows = 0
            task.progress = 0.0
            task.started_at = datetime.now(timezone.utc)
            _log(task, "✗ 数据集没有数据行")
            db.commit()
            raise RuntimeError(f"Dataset {dataset.id} has no rows")

        task.status = "running"
        task.total_rows = total_rows
        task.completed_rows = 0
        task.progress = 0.0
        task.started_at = datetime.now(timezone.utc)
        _log(task, "========== 评测任务启动 ==========")
        _log(task, f"任务: {task.name} (ID={task_id})")
        _log(task, f"数据集: {dataset.name} ({total_rows} 条)")
        _log(task, f"场景: {scenario_snapshot.get('name') or scenario.name}")
        _log(task, f"评判 LLM: {llm_config.model_name} @ {llm_config.api_base_url}")
        _log(task, f"指标数: {len(scenario_metrics)} 个")
        _log(task, f"进度: 0/{total_rows} (0%)")
        db.commit()

        _log(task, "正在构建原生评判 LLM 客户端...")
        db.commit()
        judge_client = OpenAIJudgeClient(llm_config)
        _log(task, "✓ 评判 LLM 客户端构建成功")
        db.commit()

        metrics: list[tuple[str, t.Any, ScenarioMetric]] = []
        for sm in scenario_metrics:
            metric_def: MetricDefinition = sm.metric_definition
            try:
                kind, metric_instance = build_metric(metric_def)
                metrics.append((metric_def.name, metric_instance, sm))
                _log(task, f"✓ 指标 [{metric_def.display_name}] 构建成功 (类型: {kind})")
            except Exception as exc:
                _log(task, f"✗ 指标 [{metric_def.display_name}] 构建失败: {exc}")

        if not metrics:
            _log(task, "✗ 错误: 没有任何指标构建成功，无法执行评测")
            db.commit()
            raise RuntimeError("No metrics could be built for this scenario")

        _log(task, f"========== 开始逐行评测 ({total_rows} 条) ==========")
        db.commit()

        all_row_scores: list[dict[str, t.Any]] = []

        for idx, dataset_row in enumerate(dataset_rows):
            db.refresh(task)
            if task.status == "cancelled":
                _log(task, f"⚠ 用户取消评测，已完成 {idx}/{total_rows} 行")
                db.commit()
                break

            row_start = time.time()
            row_data: dict = dataset_row.data or {}
            metric_scores: dict[str, t.Any] = {}
            row_error: str | None = None

            user_input_preview = str(row_data.get("user_input", ""))[:60]
            _log(task, f"── 行 #{idx + 1}/{total_rows}: {user_input_preview}...")
            db.commit()

            for metric_name, metric_instance, _sm in metrics:
                db.refresh(task)
                if task.status == "cancelled":
                    _log(task, f"⚠ 用户取消评测，当前行停止在指标 [{metric_name}]")
                    db.commit()
                    break

                _log(task, f"  ▸ 评测指标 [{metric_name}]...")
                db.commit()
                try:
                    result = await metric_instance.ascore(row_data, judge_client)
                    val = result.value
                    if isinstance(val, (int, float)):
                        val = round(float(val), 4)
                    reason = str(result.reason or "")[:800]
                    metric_scores[metric_name] = {"score": val, "reason": reason}
                    if val is None:
                        _log(task, f"  ✗ [{metric_name}] 无法评分: {reason[:200]}")
                    else:
                        _log(task, f"  ✓ [{metric_name}] = {val}")
                except Exception as exc:
                    metric_scores[metric_name] = {"score": None, "reason": str(exc)[:800]}
                    _log(task, f"  ✗ [{metric_name}] 失败: {str(exc)[:200]}")
                db.commit()
                await asyncio.sleep(0)

            db.refresh(task)
            if task.status == "cancelled":
                _log(task, f"⚠ 用户取消评测，已完成 {idx}/{total_rows} 行")
                db.commit()
                break

            is_pass = _determine_pass(metric_scores, metrics)
            execution_time_ms = int((time.time() - row_start) * 1000)

            _persist_row_result(db, task, dataset_row, metric_scores, row_error, is_pass, execution_time_ms)

            task.completed_rows = idx + 1
            task.progress = round((idx + 1) / total_rows, 4)

            status_icon = "✓ 通过" if is_pass else "✗ 不通过"
            _log(task, f"  → 结果: {status_icon} ({execution_time_ms}ms)")
            _log(
                task,
                f"  → 进度: {task.completed_rows}/{total_rows} ({round(task.progress * 100, 1)}%)",
            )
            db.commit()

            all_row_scores.append(metric_scores)
            await asyncio.sleep(0)

        db.refresh(task)
        if task.status == "cancelled":
            _log(task, "========== 评测已取消 ==========")
            task.finished_at = datetime.now(timezone.utc)
            db.commit()
        else:
            _log(task, "========== 计算汇总统计 ==========")
            db.commit()
            summary_scores = _compute_summary_scores(all_row_scores, metrics)
            task.status = "completed"
            task.finished_at = datetime.now(timezone.utc)
            task.summary_scores = summary_scores
            task.progress = 1.0

            pass_count = sum(1 for s in all_row_scores if _determine_pass(s, metrics))
            _log(task, f"通过: {pass_count}/{total_rows} ({round(pass_count / total_rows * 100, 1)}%)")
            for m_name, info in summary_scores.items():
                mean = info.get("mean")
                pr = info.get("pass_rate")
                _log(task, f"  {m_name}: 均值={mean}, 通过率={pr}")
            _log(task, "========== 评测完成 ==========")
            db.commit()

    except Exception as exc:
        logger.exception("Evaluation failed for task %d", task_id)
        try:
            db.rollback()
            task = db.query(EvalTask).get(task_id)
            if task is not None:
                task.status = "failed"
                task.error_message = str(exc)[:2000]
                task.finished_at = datetime.now(timezone.utc)
                _log(task, f"✗✗✗ 评测失败: {str(exc)[:500]}")
                _log(task, "========== 评测异常终止 ==========")
                db.commit()
        except Exception:
            logger.exception("Failed to persist error status for task %d", task_id)
    finally:
        db.close()


def _persist_row_result(
    db,
    task,
    dataset_row,
    metric_scores: dict,
    error: str | None,
    is_pass: bool,
    execution_time_ms: int,
) -> None:
    """Create an EvalRowResult and add it to the session (caller commits)."""
    from app.models.evaluation import EvalRowResult

    row_result = EvalRowResult(
        eval_task_id=task.id,
        dataset_row_id=dataset_row.id,
        row_index=dataset_row.row_index,
        metric_scores=metric_scores,
        is_pass=is_pass,
        execution_time_ms=execution_time_ms,
        error=error,
    )
    db.add(row_result)


def _determine_pass(metric_scores: dict, metrics: list) -> bool:
    """
    Determine whether a row passes based on per-metric pass_threshold.

    A row passes when all metrics that have a defined threshold meet or exceed
    that threshold. Metrics with errors (score=None) cause failure.
    """
    for metric_name, _metric_instance, scenario_metric in metrics:
        threshold = scenario_metric.pass_threshold
        if threshold is None:
            continue
        raw = metric_scores.get(metric_name)
        if raw is None:
            return False
        score = raw.get("score") if isinstance(raw, dict) else raw
        if score is None:
            return False
        if isinstance(score, str):
            numeric_score = 1.0 if score.lower() in ("pass", "yes", "true", "1") else 0.0
        else:
            numeric_score = float(score)
        if numeric_score < threshold:
            return False
    return True


def _compute_summary_scores(all_row_scores: list[dict[str, t.Any]], metrics: list) -> dict:
    """
    Compute per-metric aggregate statistics across all evaluated rows.

    Returns a dict like:
    {
        "faithfulness": {
            "mean": 0.85, "min": 0.6, "max": 1.0,
            "pass_rate": 0.9, "count": 10, "error_count": 0
        }
    }
    """
    thresholds = {name: sm.pass_threshold for name, _, sm in metrics}
    summary: dict = {}

    for metric_name, _, _ in metrics:
        numeric_values: list[float] = []
        error_count = 0

        for row_scores in all_row_scores:
            raw = row_scores.get(metric_name)
            if raw is None:
                error_count += 1
                continue
            score = raw.get("score") if isinstance(raw, dict) else raw
            if score is None:
                error_count += 1
                continue
            if isinstance(score, str):
                numeric_values.append(
                    1.0 if score.lower() in ("pass", "yes", "true", "1") else 0.0
                )
            else:
                numeric_values.append(float(score))

        if numeric_values:
            mean_val = sum(numeric_values) / len(numeric_values)
            min_val = min(numeric_values)
            max_val = max(numeric_values)

            threshold = thresholds.get(metric_name)
            effective_threshold = threshold if threshold is not None else DEFAULT_SUMMARY_PASS_THRESHOLD
            pass_count = sum(1 for v in numeric_values if v >= effective_threshold)
            pass_rate = pass_count / len(numeric_values)
        else:
            mean_val = None
            min_val = None
            max_val = None
            pass_rate = None

        summary[metric_name] = {
            "mean": round(mean_val, 4) if mean_val is not None else None,
            "min": round(min_val, 4) if min_val is not None else None,
            "max": round(max_val, 4) if max_val is not None else None,
            "pass_rate": round(pass_rate, 4) if pass_rate is not None else None,
            "pass_threshold": thresholds.get(metric_name),
            "effective_pass_threshold": (
                thresholds.get(metric_name)
                if thresholds.get(metric_name) is not None
                else DEFAULT_SUMMARY_PASS_THRESHOLD
            ),
            "count": len(numeric_values),
            "error_count": error_count,
        }

    return summary


def _parse_json_object(content: str) -> dict[str, t.Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(content)
        if not match:
            raise ValueError(f"Judge response is not JSON: {content[:200]}")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Judge response JSON must be an object")
    return parsed


def _coerce_float(value: t.Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _clean_reason(value: t.Any) -> str:
    if value is None:
        return ""
    reason = str(value).strip()
    return reason[:800]


def _missing_required_fields(row_data: dict[str, t.Any], required_fields: list[str]) -> list[str]:
    missing: list[str] = []
    for field in required_fields:
        value = row_data.get(field)
        if value is None or value == "" or value == [] or value == {}:
            missing.append(field)
    return missing


def _sample_payload(row_data: dict[str, t.Any]) -> dict[str, t.Any]:
    important_fields = [
        "user_input",
        "response",
        "reference",
        "retrieved_contexts",
        "reference_contexts",
        "retrieved_context_ids",
        "reference_context_ids",
        "reference_tool_calls",
        "reference_topics",
        "reference_role",
        "rubrics",
    ]
    payload = {
        key: _json_safe(row_data[key])
        for key in important_fields
        if key in row_data and row_data[key] is not None
    }
    for key, value in row_data.items():
        if key not in payload and len(payload) < 16:
            payload[key] = _json_safe(value)
    return payload


def _json_safe(value: t.Any, limit: int = 2400) -> t.Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _limit_text(value, limit)
    if isinstance(value, list):
        per_item_limit = max(300, limit // max(len(value), 1))
        return [_json_safe(item, per_item_limit) for item in value[:12]]
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, 700)
            for key, item in list(value.items())[:20]
        }
    return _limit_text(str(value), limit)


def _limit_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...[truncated]"


def _render_prompt(prompt: str, row_data: dict[str, t.Any]) -> str:
    if not prompt:
        return ""
    safe_vars = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        for key, value in row_data.items()
    }
    try:
        return prompt.format_map(_SafeFormatDict(safe_vars))
    except Exception:
        return prompt


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _match_allowed_value(value: str, allowed_values: list[str]) -> str:
    normalized = value.strip().lower()
    for allowed in allowed_values:
        if normalized == allowed.lower():
            return allowed
    for allowed in allowed_values:
        if allowed.lower() in normalized:
            return allowed
    return ""


def _extract_actual_tool_calls(user_input: t.Any) -> list[dict[str, t.Any]]:
    if not isinstance(user_input, list):
        return []
    calls: list[dict[str, t.Any]] = []
    for message in user_input:
        if not isinstance(message, dict):
            continue
        for tool_call in message.get("tool_calls") or []:
            if isinstance(tool_call, dict):
                calls.append(
                    {
                        "name": tool_call.get("name"),
                        "args": tool_call.get("args") or {},
                    }
                )
    return calls


def _find_tool_call_match(
    expected_call: dict[str, t.Any],
    actual_calls: list[dict[str, t.Any]],
) -> int | None:
    expected_name = expected_call.get("name")
    expected_args = expected_call.get("args") or {}
    for idx, actual_call in enumerate(actual_calls):
        if actual_call.get("name") != expected_name:
            continue
        if (actual_call.get("args") or {}) == expected_args:
            return idx
    return None


def _find_best_tool_name_match(
    expected_call: dict[str, t.Any],
    actual_calls: list[dict[str, t.Any]],
) -> tuple[int | None, float]:
    expected_name = expected_call.get("name")
    best_idx: int | None = None
    best_score = 0.0
    for idx, actual_call in enumerate(actual_calls):
        if actual_call.get("name") != expected_name:
            continue
        score = _argument_match_score(expected_call.get("args") or {}, actual_call.get("args") or {})
        if best_idx is None or score > best_score:
            best_idx = idx
            best_score = score
    return best_idx, best_score


def _argument_match_score(expected_args: dict[str, t.Any], actual_args: dict[str, t.Any]) -> float:
    if not expected_args:
        return 1.0 if not actual_args else 0.8
    matched = 0
    for key, expected_value in expected_args.items():
        if key in actual_args and actual_args[key] == expected_value:
            matched += 1
    return matched / len(expected_args)
