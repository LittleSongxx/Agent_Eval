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
import statistics
import time
import typing as t
from datetime import datetime, timezone

from app.core.config import settings
from app.core.embedding_client import OpenAICompatibleEmbeddingClient
from app.core.endpoint_eval import extract_eval_fields, invoke_endpoint
from app.core.prompt_manager import (
    ASPECT_CRITIC_INSTRUCTION,
    BUILTIN_LLM_METRIC_SPECS,
    BUILTIN_LLM_SCORE_INSTRUCTION,
    CLAIM_DECOMPOSITION_PROMPT,
    CLAIM_VERIFICATION_PROMPT,
    DISCRETE_SCORE_INSTRUCTION,
    GENERATIVE_QUESTION_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT_COT,
    NAMED_LLM_METRIC_SPECS,
    NONCOMMITTAL_QUESTION_MARKERS,
    NUMERIC_SCORE_INSTRUCTION,
    STRUCTURED_JSON_SYSTEM_PROMPT,
    extract_prompt_variables,
    render_prompt,
    resolve_prompt_variable,
)
from app.core.scenario_snapshot import snapshot_to_scenario_metrics

logger = logging.getLogger(__name__)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
DEFAULT_SUMMARY_PASS_THRESHOLD = 0.7

# 汇总统计里保留给平台级聚合结果的键，不会出现在指标维度统计中
RESERVED_SUMMARY_KEYS = ("weighted_total_score", "cost", "judge_reliability", "latency")

# 进程内任务心跳：daemon 工作线程在处理每一行/每个指标时刷新时间戳，
# API 层据此判断任务是否仍存活，避免把长任务误判为陈旧任务
task_heartbeats: dict[int, float] = {}

# 取消状态轮询间隔（秒）。抽成模块常量而不是写死在 wait_for 里，是为了让
# 取消测试能把它调小到毫秒级——否则每个取消用例都得真等 1 秒。
CANCEL_POLL_INTERVAL_SECONDS = 1.0
_DECIMAL_SCORE_RANGE_RE = re.compile(
    r"(?:0\.\d+\s*(?:到|至|~|-)\s*(?:0\.\d+|1(?:\.0)?))|(?:0\s*(?:到|至|~|-)\s*1(?:\.0)?)"
)
_BINARY_SCORE_RE = re.compile(r"(?:返回|给|打|只能|必须)?\s*0\s*(?:或|/|和|、)\s*1")


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
        self.api_base_url = llm_config.api_base_url
        self.api_key = llm_config.api_key
        self.temperature = llm_config.temperature if llm_config.temperature is not None else 0.01
        self.max_tokens = min(int(llm_config.max_tokens or 1024), 1200)
        self.last_messages: list[dict[str, str]] | None = None
        self.last_raw_response: str | None = None
        self.row_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.client = AsyncOpenAI(
            base_url=llm_config.api_base_url,
            api_key=llm_config.api_key,
            timeout=180,
            max_retries=2,
        )

    def reset_row_usage(self) -> None:
        """Clear per-row token accounting (called before each metric evaluation)."""
        self.row_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def take_row_usage(self) -> dict[str, int]:
        """Return the tokens consumed for the current row/metric and reset."""
        usage = dict(self.row_usage)
        self.reset_row_usage()
        return usage

    def _accumulate_usage(self, usage) -> None:
        if usage is None:
            return
        self.row_usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.row_usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        self.row_usage["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)

    async def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, t.Any]:
        """与 judge 判定解耦的结构化调用（断言拆解/核验、问题反推等中间步骤）。

        复用同一 OpenAI 兼容客户端与 token 核算，但系统提示词由调用方指定，
        不受 JUDGE_SYSTEM_PROMPT 的评分格式约束。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        self.last_messages = messages
        self.last_raw_response = None
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.warning("chat_json JSON mode failed, retrying without response_format: %s", exc)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        self._accumulate_usage(getattr(response, "usage", None))
        content = response.choices[0].message.content or ""
        self.last_raw_response = content
        return _parse_json_object(content)

    async def judge_json(self, payload: dict[str, t.Any]) -> dict[str, t.Any]:
        user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        system_prompt = JUDGE_SYSTEM_PROMPT_COT if settings.JUDGE_COT_MODE else JUDGE_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        self.last_messages = messages
        self.last_raw_response = None
        logger.info(
            "========== JUDGE_PROMPT_BEGIN ==========\n%s\n========== JUDGE_PROMPT_END ==========",
            json.dumps(messages, ensure_ascii=False, indent=2),
        )

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

        self._accumulate_usage(getattr(response, "usage", None))
        content = response.choices[0].message.content or ""
        self.last_raw_response = content
        return _parse_json_object(content)


class NativeBuiltinLLMMetric:
    """Built-in metric implemented by our own judge prompt."""

    def __init__(
        self,
        metric_def,
        spec: dict[str, t.Any],
        prompt_override: str | None = None,
        pass_threshold: float | None = None,
        weight: float | None = None,
        effective_criteria: str | None = None,
    ):
        self.id = getattr(metric_def, "id", None)
        self.name = metric_def.name
        self.display_name = metric_def.display_name
        self.metric_type = metric_def.metric_type
        self.config = metric_def.config or {}
        self.pass_threshold = pass_threshold
        self.weight = weight if weight is not None else 1.0
        self.required_fields = list(
            self.config.get("required_fields") or spec.get("required_fields") or []
        )
        # 下发给裁判的可见范围。与 required_fields 是两件事：后者是"必须存在
        # 才打分"的校验门槛，前者是"裁判能看到什么"。两个集合本就不相等——
        # 检索侧指标不能看 response，Agent/多轮指标要看 response 但不能把它
        # 列为必需（回复由接口在运行时注入）。详见 prompt_manager 的说明。
        # 没有声明时退回 required_fields，保证新增指标默认是收紧的而非放开的。
        self.judge_fields = list(
            self.config.get("judge_fields")
            or spec.get("judge_fields")
            or self.required_fields
        )
        # 优先使用任务快照里冻结的判定标准。缺省回落到"override 或当前 spec 常量"
        # 只为兼容快照生成前创建的历史任务——新任务一律带 effective_criteria，
        # 否则改一次 prompt_manager 的常量就会让历史任务的口径静默漂移。
        self.criteria = (
            (effective_criteria or "").strip()
            or (prompt_override or "").strip()
            or spec["criteria"]
        )
        self.description = self.config.get("description")

    async def ascore(self, row_data: dict[str, t.Any], judge: OpenAIJudgeClient) -> _MetricResult:
        missing = _missing_required_fields(row_data, self.required_fields)
        if missing:
            return _MetricResult(None, f"缺少必需字段: {', '.join(missing)}。")

        context = build_template_context(row_data, row_data.get("_endpoint_trace"), self, self)
        criteria = render_prompt(self.criteria, context)
        payload = {
            "metric": {
                "name": self.name,
                "display_name": self.display_name,
                "score_range": "0 到 1，1 表示完全满足指标，0 表示完全不满足",
                "criteria": criteria,
                "description": self.description,
                "required_fields": self.required_fields,
            },
            "sample": _sample_payload(row_data, allow_fields=self.judge_fields),
            "instruction": BUILTIN_LLM_SCORE_INSTRUCTION,
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

    def __init__(
        self,
        metric_def,
        mode: str,
        prompt_override: str | None = None,
        pass_threshold: float | None = None,
        weight: float | None = None,
    ):
        self.id = getattr(metric_def, "id", None)
        self.name = metric_def.name
        self.display_name = metric_def.display_name
        self.metric_type = metric_def.metric_type
        self.mode = mode
        self.config = metric_def.config or {}
        self.prompt_override = (prompt_override or "").strip() or None
        self.pass_threshold = pass_threshold
        self.weight = weight if weight is not None else 1.0

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
        prompt_template = self.prompt_override or self.config.get("prompt", "")
        prompt, sample = _render_custom_prompt_and_sample(prompt_template, row_data, self, self)
        payload = {
            "metric": {
                "name": self.name,
                "display_name": self.display_name,
                "score_range": f"{lower} 到 {upper}",
                "criteria": self.config.get("description") or prompt,
            },
            "prompt": prompt,
            "sample": sample,
            "instruction": NUMERIC_SCORE_INSTRUCTION,
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
        prompt_template = self.prompt_override or self.config.get("prompt", "")
        prompt, sample = _render_custom_prompt_and_sample(prompt_template, row_data, self, self)
        payload = {
            "metric": {
                "name": self.name,
                "display_name": self.display_name,
                "allowed_values": allowed_values,
                "criteria": self.config.get("description") or prompt,
            },
            "prompt": prompt,
            "sample": sample,
            "instruction": DISCRETE_SCORE_INSTRUCTION,
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
        definition_template = (
            self.prompt_override
            or self.config.get("definition")
            or self.config.get("description")
            or self.display_name
        )
        definition, sample = _render_custom_prompt_and_sample(
            str(definition_template), row_data, self, self
        )
        payload = {
            "metric": {
                "name": self.name,
                "display_name": self.display_name,
                "score_range": "0 或 1，1 表示样本满足 definition，0 表示不满足",
                "definition": definition,
                "description": self.config.get("description"),
            },
            "sample": sample,
            "instruction": ASPECT_CRITIC_INSTRUCTION,
        }
        result = await judge.judge_json(payload)
        score = _coerce_float(result.get("score"))
        reason = _clean_reason(result.get("reason"))
        if score is None:
            raise ValueError(f"Judge did not return a numeric score for {self.name}")
        if not reason:
            raise ValueError(f"Judge did not return a reason for {self.name}")
        return _MetricResult(1.0 if score >= 0.5 else 0.0, reason)


class ClaimFaithfulnessMetric:
    """断言级忠实度：把回答拆成原子断言，逐条核验上下文支持，按支持占比计分。

    与整体判定的 faithfulness 互为补充：整体判定看"是否大体忠实"，
    断言级判定暴露"哪一句是编的"，是平台原生实现（不依赖第三方评测框架）。
    """

    def __init__(
        self,
        metric_def,
        prompt_override: str | None = None,
        pass_threshold: float | None = None,
        weight: float | None = None,
    ):
        self.id = getattr(metric_def, "id", None)
        self.name = metric_def.name
        self.display_name = metric_def.display_name
        self.metric_type = metric_def.metric_type
        self.config = metric_def.config or {}
        self.pass_threshold = pass_threshold
        self.weight = weight if weight is not None else 1.0
        self.required_fields = ["response", "retrieved_contexts"]

    async def ascore(self, row_data: dict[str, t.Any], judge: OpenAIJudgeClient) -> _MetricResult:
        missing = _missing_required_fields(row_data, self.required_fields)
        if missing:
            return _MetricResult(None, f"缺少必需字段: {', '.join(missing)}。")
        response = str(row_data.get("response") or "")
        contexts = list(row_data.get("retrieved_contexts") or [])
        if not contexts:
            return _MetricResult(None, "缺少 retrieved_contexts，无法核验断言。")

        try:
            decomp = await judge.chat_json(
                STRUCTURED_JSON_SYSTEM_PROMPT,
                CLAIM_DECOMPOSITION_PROMPT.format(response=response),
            )
        except Exception as exc:
            return _MetricResult(None, f"断言拆解失败: {str(exc)[:200]}")

        claims = [str(item).strip() for item in (decomp.get("claims") or []) if str(item).strip()]
        if not claims:
            # 无断言 = 回答未基于检索上下文给出任何可核验内容（如答非所问），
            # 忠实性按 0 处理：回答没有扎根于上下文，与 RAGAS 行为一致
            return _MetricResult(0.0, "回答未包含可核验断言（未基于检索上下文作答），忠实性按 0 处理。")

        contexts_text = "\n".join(contexts)
        supported = 0
        details: list[dict[str, t.Any]] = []
        for claim in claims:
            try:
                verdict = await judge.chat_json(
                    STRUCTURED_JSON_SYSTEM_PROMPT,
                    CLAIM_VERIFICATION_PROMPT.format(contexts=contexts_text, claim=claim),
                )
                is_supported = str(verdict.get("verdict") or "").strip().lower() == "supported"
                if is_supported:
                    supported += 1
                details.append(
                    {
                        "claim": claim,
                        "supported": is_supported,
                        "reason": str(verdict.get("reason") or "")[:200],
                    }
                )
            except Exception as exc:
                details.append({"claim": claim, "supported": False, "reason": f"核验失败: {str(exc)[:100]}"})

        score = round(supported / len(claims), 4)
        preview = "；".join(
            f"{'✓' if item['supported'] else '✗'} {item['claim'][:24]}" for item in details[:6]
        )
        reason = (
            f"共拆解 {len(claims)} 条断言，被检索上下文支持 {supported} 条"
            f"（{round(score * 100)}%）。{preview}"
        )
        return _MetricResult(score, reason)


class CitationAccuracyMetric:
    """引用准确性：回答中的引用标记是否指向真正支持该内容的文档。

    评测目标：检测"引用文档 A 但内容实际来自文档 B"的引用错位问题。
    这在金融、法律、医疗等需要可溯源的场景下是严重问题。

    实现方式：
    1. 提取回答中的引用标记（如 [1], [Doc A], 根据文档X）
    2. 提取被引用的内容片段（引用标记前后的句子）
    3. 用 LLM Judge 判断：被引用的文档是否真正支持该内容
    4. 计算引用准确率 = 正确引用数 / 总引用数

    与 RAGAS v0.4 CitationRecall/CitationPrecision 对标。
    """

    def __init__(
        self,
        metric_def,
        prompt_override: str | None = None,
        pass_threshold: float | None = None,
        weight: float | None = None,
    ):
        self.id = getattr(metric_def, "id", None)
        self.name = metric_def.name
        self.display_name = metric_def.display_name
        self.metric_type = metric_def.metric_type
        self.config = metric_def.config or {}
        self.pass_threshold = pass_threshold
        self.weight = weight if weight is not None else 1.0
        # 需要：response（含引用标记）、retrieved_contexts（文档列表）
        self.required_fields = ["response", "retrieved_contexts"]

    async def ascore(self, row_data: dict[str, t.Any], judge: OpenAIJudgeClient) -> _MetricResult:
        missing = _missing_required_fields(row_data, self.required_fields)
        if missing:
            return _MetricResult(None, f"缺少必需字段: {', '.join(missing)}。")

        response = str(row_data.get("response") or "")
        contexts = list(row_data.get("retrieved_contexts") or [])

        if not contexts:
            return _MetricResult(None, "缺少 retrieved_contexts，无法验证引用。")

        # 提取引用标记和对应的内容片段
        citations = self._extract_citations(response)

        if not citations:
            # 无引用标记 = 回答未标注引用来源，引用准确性按 N/A 处理
            return _MetricResult(None, "回答未包含引用标记（如 [1], [Doc A]），无法评测引用准确性。")

        # 构建文档索引映射（支持 [1], [Doc A] 等格式）
        doc_index = self._build_doc_index(contexts)

        correct = 0
        details: list[dict[str, t.Any]] = []

        for citation in citations:
            cite_mark = citation["mark"]  # 如 "[1]"
            content = citation["content"]  # 引用标记附近的内容

            # 找到被引用的文档
            cited_doc = self._resolve_citation(cite_mark, doc_index, contexts)

            if cited_doc is None:
                details.append({
                    "citation": cite_mark,
                    "content": content[:50],
                    "correct": False,
                    "reason": f"引用标记 {cite_mark} 未能匹配到任何文档"
                })
                continue

            # 用 LLM Judge 判断：cited_doc 是否真正支持 content
            try:
                verification_prompt = f"""判断以下文档是否支持所引用的内容。

被引用的内容：
{content}

被引用的文档：
{cited_doc}

如果文档明确支持该内容（包含相同或等价的事实），返回 {{"verdict": "supported", "reason": "具体理由"}}；
否则返回 {{"verdict": "not_supported", "reason": "具体理由"}}。"""

                verdict = await judge.chat_json(
                    STRUCTURED_JSON_SYSTEM_PROMPT,
                    verification_prompt,
                )

                is_correct = str(verdict.get("verdict") or "").strip().lower() == "supported"
                if is_correct:
                    correct += 1

                details.append({
                    "citation": cite_mark,
                    "content": content[:50],
                    "correct": is_correct,
                    "reason": str(verdict.get("reason") or "")[:150],
                })
            except Exception as exc:
                details.append({
                    "citation": cite_mark,
                    "content": content[:50],
                    "correct": False,
                    "reason": f"验证失败: {str(exc)[:100]}"
                })

        score = round(correct / len(citations), 4)
        preview = "；".join(
            f"{'✓' if item['correct'] else '✗'} {item['citation']}" for item in details[:6]
        )
        reason = (
            f"共提取 {len(citations)} 个引用，引用准确 {correct} 个"
            f"（{round(score * 100)}%）。{preview}"
        )
        return _MetricResult(score, reason)

    def _extract_citations(self, response: str) -> list[dict[str, str]]:
        """提取回答中的引用标记和对应内容。

        支持格式：
        - [1], [2], [3] （数字索引）
        - [Doc A], [文档 B] （文档标识）
        - 根据文档1, 根据文档A （中文格式）
        """
        citations = []

        # 正则匹配：[数字] 或 [文档名]
        pattern = r'\[([^\]]+)\]'

        # 按句子分割，找到每个引用标记所在的句子
        sentences = re.split(r'[。！？\.\!\?]', response)

        for sentence in sentences:
            matches = re.finditer(pattern, sentence)
            for match in matches:
                cite_mark = match.group(0)  # 完整标记 "[1]"
                cite_id = match.group(1)    # 标记内容 "1"

                citations.append({
                    "mark": cite_mark,
                    "id": cite_id,
                    "content": sentence.strip()  # 该引用标记所在的句子
                })

        # 去重（同一个引用标记在同一句话中可能出现多次）
        seen = set()
        unique_citations = []
        for c in citations:
            key = (c["mark"], c["content"])
            if key not in seen:
                seen.add(key)
                unique_citations.append(c)

        return unique_citations

    def _build_doc_index(self, contexts: list[str]) -> dict[str, str]:
        """构建文档索引映射。

        Returns:
            {"1": "文档1内容", "2": "文档2内容", ...}
        """
        doc_index = {}
        for idx, doc in enumerate(contexts, start=1):
            # 支持数字索引
            doc_index[str(idx)] = doc
            # 支持字母索引（A, B, C, ...）
            if idx <= 26:
                doc_index[chr(64 + idx)] = doc  # A=65
        return doc_index

    def _resolve_citation(
        self,
        cite_mark: str,
        doc_index: dict[str, str],
        contexts: list[str]
    ) -> str | None:
        """将引用标记解析到具体文档。

        Args:
            cite_mark: 如 "[1]", "[Doc A]"
            doc_index: 文档索引映射
            contexts: 原始文档列表

        Returns:
            被引用的文档内容，如果无法解析则返回 None
        """
        # 提取标记内的 ID
        inner = cite_mark.strip("[]")

        # 直接匹配
        if inner in doc_index:
            return doc_index[inner]

        # 尝试去掉前缀（如 "Doc 1" -> "1", "文档A" -> "A"）
        cleaned = re.sub(r'^(Doc|文档|document)\s*', '', inner, flags=re.IGNORECASE).strip()
        if cleaned in doc_index:
            return doc_index[cleaned]

        # 无法解析
        return None


class TrajectoryFaithfulnessMetric:
    """轨迹忠实度：Agent 推理步骤与工具返回结果是否一致。

    评测目标：检测 Agent 在推理时是否忠实地基于工具返回结果，而非凭空编造。
    这在 Agent 调试和 badcase 分析中是核心问题。

    实现方式：
    1. 解析 agent_trajectory：[{step, thought, tool, tool_input, tool_output, action}]
    2. 对每个步骤，用 LLM Judge 判断：thought 是否与 tool_output 一致
    3. 计算忠实度 = 一致步骤数 / 总步骤数

    与 TRAJECT-Bench、VAKRA 等业界前沿对标。
    """

    def __init__(
        self,
        metric_def,
        prompt_override: str | None = None,
        pass_threshold: float | None = None,
        weight: float | None = None,
    ):
        self.id = getattr(metric_def, "id", None)
        self.name = metric_def.name
        self.display_name = metric_def.display_name
        self.metric_type = metric_def.metric_type
        self.config = metric_def.config or {}
        self.pass_threshold = pass_threshold
        self.weight = weight if weight is not None else 1.0
        self.required_fields = ["agent_trajectory"]

    async def ascore(self, row_data: dict[str, t.Any], judge: OpenAIJudgeClient) -> _MetricResult:
        missing = _missing_required_fields(row_data, self.required_fields)
        if missing:
            return _MetricResult(None, f"缺少必需字段: {', '.join(missing)}。")

        trajectory = list(row_data.get("agent_trajectory") or [])

        if not trajectory:
            return _MetricResult(None, "agent_trajectory 为空，无法评测轨迹忠实度。")

        faithful_count = 0
        details: list[dict[str, t.Any]] = []

        for step_data in trajectory:
            step_num = step_data.get("step", 0)
            thought = str(step_data.get("thought") or "")
            tool_output = str(step_data.get("tool_output") or "")

            # 如果该步骤没有工具调用，跳过（纯思考步骤无需验证）
            if not step_data.get("tool") or not tool_output:
                continue

            # 用 LLM Judge 判断：thought 是否忠实反映 tool_output
            try:
                verification_prompt = f"""判断 Agent 的推理是否忠实于工具返回结果。

工具返回结果：
{tool_output}

Agent 推理内容：
{thought}

如果 Agent 推理内容与工具返回结果一致（没有编造信息、没有曲解结果），返回 {{"verdict": "faithful", "reason": "具体理由"}}；
否则返回 {{"verdict": "not_faithful", "reason": "具体理由"}}。"""

                verdict = await judge.chat_json(
                    STRUCTURED_JSON_SYSTEM_PROMPT,
                    verification_prompt,
                )

                is_faithful = str(verdict.get("verdict") or "").strip().lower() == "faithful"
                if is_faithful:
                    faithful_count += 1

                details.append({
                    "step": step_num,
                    "tool": step_data.get("tool"),
                    "faithful": is_faithful,
                    "reason": str(verdict.get("reason") or "")[:150],
                })
            except Exception as exc:
                details.append({
                    "step": step_num,
                    "tool": step_data.get("tool"),
                    "faithful": False,
                    "reason": f"验证失败: {str(exc)[:100]}"
                })

        # 计算忠实度
        evaluated_steps = len([d for d in details if d])
        if evaluated_steps == 0:
            return _MetricResult(None, "没有需要验证的工具调用步骤。")

        score = round(faithful_count / evaluated_steps, 4)
        preview = "；".join(
            f"Step {item['step']} {'✓' if item['faithful'] else '✗'}" for item in details[:5]
        )
        reason = (
            f"共评测 {evaluated_steps} 个工具调用步骤，忠实度 {faithful_count} 个"
            f"（{round(score * 100)}%）。{preview}"
        )
        return _MetricResult(score, reason)


class ErrorRecoveryMetric:
    """错误恢复能力：工具调用失败后是否有合理的降级或重试。

    评测目标：检测 Agent 遇到错误时的恢复能力。
    优秀的 Agent 应该能够：
    1. 识别错误（工具返回错误信息）
    2. 采取恢复措施（重试、降级、换工具）
    3. 最终完成任务或给出合理解释

    实现方式：
    1. 识别轨迹中的错误步骤（tool_output 包含 "error", "failed" 等关键词）
    2. 检查后续步骤是否有恢复动作（重试、换工具、解释原因）
    3. 计算恢复率 = 成功恢复的错误数 / 总错误数
    """

    def __init__(
        self,
        metric_def,
        prompt_override: str | None = None,
        pass_threshold: float | None = None,
        weight: float | None = None,
    ):
        self.id = getattr(metric_def, "id", None)
        self.name = metric_def.name
        self.display_name = metric_def.display_name
        self.metric_type = metric_def.metric_type
        self.config = metric_def.config or {}
        self.pass_threshold = pass_threshold
        self.weight = weight if weight is not None else 1.0
        self.required_fields = ["agent_trajectory"]

    async def ascore(self, row_data: dict[str, t.Any], judge: OpenAIJudgeClient) -> _MetricResult:
        missing = _missing_required_fields(row_data, self.required_fields)
        if missing:
            return _MetricResult(None, f"缺少必需字段: {', '.join(missing)}。")

        trajectory = list(row_data.get("agent_trajectory") or [])

        if not trajectory:
            return _MetricResult(None, "agent_trajectory 为空，无法评测错误恢复能力。")

        # 识别错误步骤
        error_steps = []
        for i, step_data in enumerate(trajectory):
            tool_output = str(step_data.get("tool_output") or "").lower()
            # 检测错误关键词
            if any(keyword in tool_output for keyword in ["error", "failed", "exception", "错误", "失败"]):
                error_steps.append((i, step_data))

        if not error_steps:
            return _MetricResult(None, "轨迹中未检测到错误步骤，无需评测错误恢复能力。")

        recovered_count = 0
        details: list[dict[str, t.Any]] = []

        for error_idx, error_step in error_steps:
            step_num = error_step.get("step", error_idx + 1)
            tool_name = error_step.get("tool", "")

            # 检查后续步骤是否有恢复动作
            has_recovery = False
            recovery_action = ""

            # 查看后续最多 3 个步骤
            for next_step_data in trajectory[error_idx + 1:error_idx + 4]:
                next_thought = str(next_step_data.get("thought") or "").lower()
                next_tool = next_step_data.get("tool", "")

                # 检测恢复行为
                if "重试" in next_thought or "retry" in next_thought:
                    has_recovery = True
                    recovery_action = "重试相同操作"
                    break
                elif next_tool and next_tool != tool_name:
                    has_recovery = True
                    recovery_action = f"切换到工具 {next_tool}"
                    break
                elif any(keyword in next_thought for keyword in ["换", "改用", "尝试", "alternative"]):
                    has_recovery = True
                    recovery_action = "寻找替代方案"
                    break

            if has_recovery:
                recovered_count += 1

            details.append({
                "step": step_num,
                "tool": tool_name,
                "recovered": has_recovery,
                "recovery_action": recovery_action or "无恢复动作",
            })

        score = round(recovered_count / len(error_steps), 4)
        preview = "；".join(
            f"Step {item['step']} {'✓ ' + item['recovery_action'] if item['recovered'] else '✗ 直接终止'}"
            for item in details[:3]
        )
        reason = (
            f"检测到 {len(error_steps)} 个错误步骤，成功恢复 {recovered_count} 个"
            f"（{round(score * 100)}%）。{preview}"
        )
        return _MetricResult(score, reason)


class ToolSelectionRationalityMetric:
    """工具选择合理性：当前步骤是否选择了最优工具。

    评测目标：检测 Agent 是否选择了完成当前任务的最优工具。
    次优工具选择会导致：
    1. 效率低下（能一步完成的任务用了多步）
    2. 结果不准确（工具能力不足）
    3. 成本浪费（调用了不必要的工具）

    实现方式：
    1. 提取每个步骤的：任务目标、可用工具列表、实际选择的工具
    2. 用 LLM Judge 判断：是否存在更优的工具选择
    3. 计算合理性 = 选择最优工具的步骤数 / 总步骤数
    """

    def __init__(
        self,
        metric_def,
        prompt_override: str | None = None,
        pass_threshold: float | None = None,
        weight: float | None = None,
    ):
        self.id = getattr(metric_def, "id", None)
        self.name = metric_def.name
        self.display_name = metric_def.display_name
        self.metric_type = metric_def.metric_type
        self.config = metric_def.config or {}
        self.pass_threshold = pass_threshold
        self.weight = weight if weight is not None else 1.0
        self.required_fields = ["agent_trajectory", "available_tools"]

    async def ascore(self, row_data: dict[str, t.Any], judge: OpenAIJudgeClient) -> _MetricResult:
        missing = _missing_required_fields(row_data, self.required_fields)
        if missing:
            return _MetricResult(None, f"缺少必需字段: {', '.join(missing)}。")

        trajectory = list(row_data.get("agent_trajectory") or [])
        available_tools = dict(row_data.get("available_tools") or {})

        if not trajectory:
            return _MetricResult(None, "agent_trajectory 为空，无法评测工具选择合理性。")

        if not available_tools:
            return _MetricResult(None, "available_tools 为空，无法判断工具选择是否最优。")

        rational_count = 0
        details: list[dict[str, t.Any]] = []

        for step_data in trajectory:
            step_num = step_data.get("step", 0)
            thought = str(step_data.get("thought") or "")
            selected_tool = step_data.get("tool", "")

            # 如果该步骤没有工具调用，跳过
            if not selected_tool:
                continue

            # 构建工具列表描述
            tools_desc = "\n".join(
                f"- {name}: {desc}" for name, desc in available_tools.items()
            )

            # 用 LLM Judge 判断工具选择是否最优
            try:
                verification_prompt = f"""判断 Agent 的工具选择是否最优。

任务目标：
{thought}

实际选择的工具：
{selected_tool}

可用工具列表：
{tools_desc}

如果 Agent 选择了最优工具（没有更好的替代工具能更高效/准确地完成任务），返回 {{"verdict": "optimal", "reason": "具体理由"}}；
如果存在更优的工具选择，返回 {{"verdict": "suboptimal", "better_tool": "工具名", "reason": "为什么更优"}}。"""

                verdict = await judge.chat_json(
                    STRUCTURED_JSON_SYSTEM_PROMPT,
                    verification_prompt,
                )

                is_optimal = str(verdict.get("verdict") or "").strip().lower() == "optimal"
                if is_optimal:
                    rational_count += 1

                details.append({
                    "step": step_num,
                    "selected_tool": selected_tool,
                    "optimal": is_optimal,
                    "better_tool": verdict.get("better_tool", "") if not is_optimal else "",
                    "reason": str(verdict.get("reason") or "")[:150],
                })
            except Exception as exc:
                details.append({
                    "step": step_num,
                    "selected_tool": selected_tool,
                    "optimal": True,  # 默认认为合理（验证失败不应惩罚）
                    "reason": f"验证失败: {str(exc)[:100]}"
                })
                rational_count += 1

        evaluated_steps = len([d for d in details if d])
        if evaluated_steps == 0:
            return _MetricResult(None, "没有需要评测的工具调用步骤。")

        score = round(rational_count / evaluated_steps, 4)
        preview = "；".join(
            f"Step {item['step']} {item['selected_tool']} {'✓' if item['optimal'] else '✗→' + item['better_tool']}"
            for item in details[:5]
        )
        reason = (
            f"共评测 {evaluated_steps} 个工具选择，最优选择 {rational_count} 个"
            f"（{round(score * 100)}%）。{preview}"
        )
        return _MetricResult(score, reason)


class GenerativeAnswerRelevancyMetric:
    """生成式相关性：从回答反推它可能回答的问题，与用户问题做语义相似度。

    与整体判定的 answer_relevancy 互为补充：整体判定看"是否回应了问题"，
    生成式判定看"语义贴合度"，两通道分歧大的样本即需人工复核的信号。
    平台原生实现（反推 prompt + OpenAI 兼容 embedding），不依赖第三方框架。
    """

    def __init__(
        self,
        metric_def,
        prompt_override: str | None = None,
        pass_threshold: float | None = None,
        weight: float | None = None,
    ):
        self.id = getattr(metric_def, "id", None)
        self.name = metric_def.name
        self.display_name = metric_def.display_name
        self.metric_type = metric_def.metric_type
        self.config = metric_def.config or {}
        self.pass_threshold = pass_threshold
        self.weight = weight if weight is not None else 1.0
        self.required_fields = ["user_input", "response"]

    async def ascore(self, row_data: dict[str, t.Any], judge: OpenAIJudgeClient) -> _MetricResult:
        missing = _missing_required_fields(row_data, self.required_fields)
        if missing:
            return _MetricResult(None, f"缺少必需字段: {', '.join(missing)}。")
        user_input = str(row_data.get("user_input") or "")
        response = str(row_data.get("response") or "")
        strictness = max(1, int(settings.GENERATIVE_RELEVANCY_STRICTNESS))

        try:
            gen = await judge.chat_json(
                STRUCTURED_JSON_SYSTEM_PROMPT,
                GENERATIVE_QUESTION_PROMPT.format(n=strictness, response=response),
            )
        except Exception as exc:
            return _MetricResult(None, f"问题反推失败: {str(exc)[:200]}")

        questions = [str(item).strip() for item in (gen.get("questions") or []) if str(item).strip()]
        if not questions:
            return _MetricResult(0.0, "反推问题为空，按相关性为 0 处理。")
        noncommittal = sum(
            1 for q in questions if any(marker in q for marker in NONCOMMITTAL_QUESTION_MARKERS)
        )
        if noncommittal == len(questions):
            return _MetricResult(0.0, "反推问题全部为回避式，回答未正面回应问题。")

        try:
            embedding_client = OpenAICompatibleEmbeddingClient(
                judge.api_base_url,
                judge.api_key,
                settings.LLM_EMBEDDING_MODEL,
            )
            vectors = await embedding_client.embed_texts([user_input] + questions)
        except Exception as exc:
            return _MetricResult(None, f"语义相似度计算失败: {str(exc)[:200]}")

        user_vector = vectors[0]
        similarities = [
            OpenAICompatibleEmbeddingClient.cosine_similarity(user_vector, vector)
            for vector in vectors[1:]
        ]
        score = max(similarities) if similarities else 0.0
        reason = (
            f"反推 {len(questions)} 个问题（回避式 {noncommittal} 个），"
            f"与用户问题最大语义相似度 {round(score, 4)}。"
        )
        return _MetricResult(round(score, 4), reason)


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


def build_metric(
    metric_def,
    llm=None,
    prompt_override: str | None = None,
    scenario_metric=None,
) -> tuple[str, t.Any]:
    """
    Instantiate a metric executor from a MetricDefinition database row.

    The default path is native and OpenAI-compatible. This keeps the platform
    independent from third-party metric internals while still supporting configurable judge
    prompts and deterministic retrieval metrics.
    """
    metric_type: str = metric_def.metric_type
    config: dict = metric_def.config or {}

    # 平台原生复合指标（多步结构化判定：断言拆解 / 生成式语义比较 / 引用准确性）优先于通用
    # Judge 路径——它们的 spec 条目仅用于指标元数据，实际执行走专用执行器
    if metric_type == "builtin_faithfulness_claim":
        return (
            "llm",
            ClaimFaithfulnessMetric(
                metric_def,
                prompt_override=prompt_override,
                pass_threshold=getattr(scenario_metric, "pass_threshold", None),
                weight=getattr(scenario_metric, "weight", None),
            ),
        )

    if metric_type == "builtin_answer_relevancy_generative":
        return (
            "llm",
            GenerativeAnswerRelevancyMetric(
                metric_def,
                prompt_override=prompt_override,
                pass_threshold=getattr(scenario_metric, "pass_threshold", None),
                weight=getattr(scenario_metric, "weight", None),
            ),
        )

    if metric_type == "builtin_citation_accuracy":
        return (
            "llm",
            CitationAccuracyMetric(
                metric_def,
                prompt_override=prompt_override,
                pass_threshold=getattr(scenario_metric, "pass_threshold", None),
                weight=getattr(scenario_metric, "weight", None),
            ),
        )

    if metric_type == "builtin_citation_accuracy":
        return (
            "llm",
            CitationAccuracyMetric(
                metric_def,
                prompt_override=prompt_override,
                pass_threshold=getattr(scenario_metric, "pass_threshold", None),
                weight=getattr(scenario_metric, "weight", None),
            ),
        )

    spec = NAMED_LLM_METRIC_SPECS.get(metric_def.name) or BUILTIN_LLM_METRIC_SPECS.get(metric_type)
    if spec:
        return (
            "llm",
            NativeBuiltinLLMMetric(
                metric_def,
                spec,
                prompt_override=prompt_override,
                pass_threshold=getattr(scenario_metric, "pass_threshold", None),
                weight=getattr(scenario_metric, "weight", None),
                # 任务快照冻结的判定标准。由 snapshot_to_scenario_metrics 水合而来，
                # 因此走任务路径的调用天然带上，脚本/单测直连时为 None 并回落到 spec。
                effective_criteria=getattr(scenario_metric, "effective_criteria", None),
            ),
        )

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
        return (
            "llm",
            NativePromptMetric(
                metric_def,
                mode=metric_type,
                prompt_override=prompt_override,
                pass_threshold=getattr(scenario_metric, "pass_threshold", None),
                weight=getattr(scenario_metric, "weight", None),
            ),
        )

    raise ValueError(f"Unknown metric_type: {metric_type!r}")


def get_metric_prompt_template(metric_def, prompt_override: str | None = None) -> str:
    """Return the business-owned prompt text used by a configurable metric."""
    config = metric_def.config or {}
    override = (prompt_override or "").strip()
    if override:
        return override
    if metric_def.metric_type == "aspect_critic":
        return str(config.get("definition") or config.get("description") or metric_def.display_name or "")
    return str(config.get("prompt") or config.get("definition") or config.get("description") or "")


def build_template_context(
    row_data: dict[str, t.Any],
    endpoint_trace: dict[str, t.Any] | None,
    metric_def,
    scenario_metric=None,
) -> dict[str, t.Any]:
    """Build the unified variable context available to Judge prompt templates."""

    dataset_data = row_data.get("_dataset_data")
    dataset = _public_row_data(dataset_data if isinstance(dataset_data, dict) else row_data)
    endpoint_trace = endpoint_trace or {}
    extracted_fields = endpoint_trace.get("extracted_fields") or {}
    endpoint: dict[str, t.Any] = dict(extracted_fields if isinstance(extracted_fields, dict) else {})
    for key in ("raw_response", "status_code", "latency_ms", "request_body", "mapping_errors"):
        if key in endpoint_trace:
            endpoint[key] = endpoint_trace.get(key)

    metric_config = getattr(metric_def, "config", None) or {}
    metric = {
        "id": getattr(metric_def, "id", None),
        "name": getattr(metric_def, "name", None),
        "display_name": getattr(metric_def, "display_name", None),
        "metric_type": getattr(metric_def, "metric_type", None),
        "description": metric_config.get("description"),
        "score_range": _metric_score_range(metric_def),
        "pass_threshold": getattr(scenario_metric, "pass_threshold", None),
        "weight": getattr(scenario_metric, "weight", None),
        "required_fields": getattr(metric_def, "required_fields", None) or metric_config.get("required_fields"),
    }
    flat = _public_row_data(row_data)
    flat.update(endpoint)
    return {
        **flat,
        "dataset": dataset,
        "endpoint": endpoint,
        "metric": metric,
    }


def diagnose_metric_prompt(
    metric_def,
    row_data: dict[str, t.Any],
    prompt_override: str | None = None,
    scenario_metric=None,
    evaluation_mode: str | None = None,
) -> list[str]:
    """Find common conflicts between metric type, business prompt and sample fields."""
    warnings: list[str] = []
    metric_type = metric_def.metric_type
    template = get_metric_prompt_template(metric_def, prompt_override)
    normalized = template.replace("～", "~")

    if metric_type == "aspect_critic" and _DECIMAL_SCORE_RANGE_RE.search(normalized):
        warnings.append(
            f"指标 [{metric_def.display_name}] 是 0/1 判断(aspect_critic)，但业务规则中出现连续分或分档区间。"
            "建议改为 numeric 指标，或把业务规则改成严格返回 0/1。"
        )
    if metric_type == "numeric" and _BINARY_SCORE_RE.search(normalized) and not _DECIMAL_SCORE_RANGE_RE.search(normalized):
        warnings.append(
            f"指标 [{metric_def.display_name}] 是数值评分(numeric)，但业务规则看起来要求 0/1。"
            "建议改为 aspect_critic，或明确允许 0~1 连续分。"
        )
    if metric_type == "discrete" and _DECIMAL_SCORE_RANGE_RE.search(normalized):
        warnings.append(
            f"指标 [{metric_def.display_name}] 是标签判断(discrete)，但业务规则中出现数值分档。"
            "建议改为 numeric，或把 allowed_values 的每个标签判定条件写清楚。"
        )

    context = build_template_context(
        row_data,
        row_data.get("_endpoint_trace"),
        metric_def,
        scenario_metric,
    )
    for field in sorted(extract_prompt_variables(template)):
        if field.startswith("endpoint.") and evaluation_mode != "endpoint":
            warnings.append(
                f"指标 [{metric_def.display_name}] 的业务提示词引用了 {{{field}}}，但当前是已有结果评测模式，接口变量不可用。"
            )
            continue
        found, value = resolve_prompt_variable(context, field)
        if value is None or value == "" or value == [] or value == {}:
            warnings.append(
                f"指标 [{metric_def.display_name}] 的业务提示词引用了 {{{field}}}，但当前样本没有该字段或字段为空。"
            )
    return warnings


def _is_llm_metric(metric_instance: t.Any) -> bool:
    """LLM-based metrics can be sampled/paneled/swapped; deterministic metrics have zero variance."""
    return isinstance(
        metric_instance,
        (
            NativeBuiltinLLMMetric,
            NativePromptMetric,
            ClaimFaithfulnessMetric,
            GenerativeAnswerRelevancyMetric,
        ),
    )


async def _score_metric_with_sampling(
    metric_instance: t.Any,
    row_data: dict[str, t.Any],
    judge: OpenAIJudgeClient,
) -> tuple[_MetricResult, dict[str, t.Any]]:
    """
    Score one metric, optionally sampling the judge multiple times.

    When ``EVAL_JUDGE_SAMPLES > 1``, the LLM judge is called N times for the
    same row. Numeric scores are aggregated by mean and the sample standard
    deviation is reported as a stability signal; discrete labels fall back to
    majority voting. Deterministic metrics are always evaluated once.
    """
    if not _is_llm_metric(metric_instance) or int(settings.EVAL_JUDGE_SAMPLES) <= 1:
        result = await metric_instance.ascore(row_data, judge)
        return result, {}

    sample_count = int(settings.EVAL_JUDGE_SAMPLES)
    results: list[_MetricResult] = [
        await metric_instance.ascore(row_data, judge) for _ in range(sample_count)
    ]
    stats: dict[str, t.Any] = {"sample_count": sample_count}

    numeric = [
        float(result.value) for result in results
        if isinstance(result.value, (int, float)) and result.value is not None
    ]
    if len(numeric) == len(results) and numeric:
        mean = sum(numeric) / len(numeric)
        std = statistics.pstdev(numeric) if len(numeric) > 1 else 0.0
        # 选择分数最接近均值的采样作为展示理由，保证 reason 与 score 口径一致
        best = min(results, key=lambda result: abs(float(result.value) - mean))
        stats["score_std"] = round(std, 4)
        stats["sample_scores"] = [round(value, 4) for value in numeric]
        return _MetricResult(mean, best.reason), stats

    # 非数值标签（如 discrete）：取多数票
    from collections import Counter

    counter = Counter(str(result.value) for result in results)
    majority_label, _ = counter.most_common(1)[0]
    best = next(result for result in results if str(result.value) == majority_label)
    stats["score_std"] = None
    stats["sample_scores"] = [str(result.value) for result in results]
    return _MetricResult(majority_label, best.reason), stats


def _aggregate_judge_results(
    results: list[tuple[str, _MetricResult, dict[str, t.Any]]],
) -> tuple[t.Any, str, dict[str, t.Any]]:
    """聚合多裁判评分：数值取均值（记录裁判间平均绝对偏差 MAD），离散标签取多数票。"""
    values = [result.value for _name, result, _stats in results]
    names = [name for name, _result, _stats in results]

    if all(value is None for value in values):
        return None, "所有裁判均未返回有效分数。", {
            "judge_count": len(results),
            "judge_scores": {name: None for name in names},
            "judge_mad": None,
        }

    numeric = [float(value) for value in values if isinstance(value, (int, float)) and value is not None]
    if numeric and len(numeric) == sum(1 for value in values if value is not None):
        # 数值型（允许个别裁判缺分）：对有效分值取均值，MAD 仅基于有效分值
        mean = sum(numeric) / len(numeric)
        mad = sum(abs(value - mean) for value in numeric) / len(numeric)
        numeric_results = [item for item in results if item[1].value is not None]
        best = min(numeric_results, key=lambda item: abs(float(item[1].value) - mean))
        stats = {
            "judge_count": len(results),
            "judge_scores": {name: round(float(value), 4) if value is not None else None for name, value in zip(names, values)},
            "judge_mad": round(mad, 4),
        }
        return mean, best[1].reason, stats

    from collections import Counter

    counter = Counter(str(value) for value in values)
    majority_label, _count = counter.most_common(1)[0]
    best = next(item for item in results if str(item[1].value) == majority_label)
    stats = {
        "judge_count": len(results),
        "judge_scores": {name: str(value) for name, value in zip(names, values)},
        "judge_mad": None,
    }
    return majority_label, best[1].reason, stats


def _swap_row_data(row_data: dict[str, t.Any]) -> dict[str, t.Any]:
    """换序互评输入：反转列表字段（如 retrieved_contexts）顺序。

    对顺序不敏感的指标（如忠实性），换序后分数应基本不变；分数明显波动
    即提示 Judge 存在位置偏置，样本应标记为低置信度。
    """
    swapped = dict(row_data)
    for key, value in row_data.items():
        if isinstance(value, list):
            swapped[key] = list(reversed(value))
    return swapped


def _last_log_lines(logs: str, count: int = 8) -> str:
    lines = [line for line in (logs or "").split("\n") if line.strip()]
    return "\n".join(lines[-count:])


def _broadcast_progress(task) -> None:
    """Push task progress to WebSocket subscribers without blocking the runner."""
    try:
        from app.core.ws_manager import manager

        manager.send_json(
            task.id,
            {
                "type": "task_progress",
                "task_id": task.id,
                "status": task.status,
                "progress": task.progress or 0,
                "total_rows": task.total_rows or 0,
                "completed_rows": task.completed_rows or 0,
                "log_tail": _last_log_lines(task.logs or ""),
            },
        )
    except Exception:
        logger.debug("WS progress push failed for task %s", getattr(task, "id", None), exc_info=True)


def _percentile_ms(sorted_values: list[int], q: float) -> int | None:
    """已排序耗时列表的分位数（nearest-rank）。

    刻意不引入 numpy：评测样本量在数百量级，nearest-rank 与线性插值的差异
    远小于 Judge 本身的延迟抖动，为一个分位数加一个二进制依赖不划算。
    """
    if not sorted_values:
        return None
    idx = max(0, min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def _build_latency_summary(
    row_latencies_ms: list[int],
    metric_latencies_ms: dict[str, list[int]],
    wall_clock_ms: int,
    concurrency: int,
) -> dict[str, t.Any]:
    """把逐行/逐指标耗时聚合成分位数。

    原先 `execution_time_ms` 只逐行写进 `eval_row_results`，从未聚合——
    任务级报告里没有任何延迟数字，"这套评测跑一轮多久"只能靠人去翻行记录
    自己算。而均值对长尾完全不敏感：24 行 2s + 1 行 60s 的均值只有 4.3s，
    看起来毫无问题，p95 会直接把那根长尾暴露出来。

    `speedup_estimate` 用"逐行耗时之和 / 实际墙钟"估算并发带来的加速比，
    它同时是一个自检信号：并发数 > 1 而加速比 ≈ 1，说明并发实际没生效
    （例如全部卡在同一个限流上）。
    """
    rows_sorted = sorted(row_latencies_ms)
    serial_total = sum(row_latencies_ms)
    summary: dict[str, t.Any] = {
        "row_count": len(rows_sorted),
        "row_p50_ms": _percentile_ms(rows_sorted, 0.50),
        "row_p95_ms": _percentile_ms(rows_sorted, 0.95),
        "row_max_ms": rows_sorted[-1] if rows_sorted else None,
        "row_mean_ms": int(serial_total / len(rows_sorted)) if rows_sorted else None,
        "wall_clock_ms": wall_clock_ms,
        "row_concurrency": concurrency,
        # 逐行耗时之和 ÷ 墙钟：并发生效时 > 1，未生效时 ≈ 1
        "speedup_estimate": (
            round(serial_total / wall_clock_ms, 2) if wall_clock_ms > 0 and serial_total else None
        ),
    }
    per_metric: dict[str, t.Any] = {}
    for metric_name, values in metric_latencies_ms.items():
        if not values:
            continue
        values_sorted = sorted(values)
        per_metric[metric_name] = {
            "count": len(values_sorted),
            "p50_ms": _percentile_ms(values_sorted, 0.50),
            "p95_ms": _percentile_ms(values_sorted, 0.95),
            "max_ms": values_sorted[-1],
        }
    if per_metric:
        summary["per_metric"] = per_metric
        # 哪个指标最慢是优化的第一落点：生成式指标要反推问题 + 调 embedding，
        # 通常是长尾来源，但这句话必须靠数字支撑而不是靠猜
        slowest = max(per_metric.items(), key=lambda kv: kv[1]["p95_ms"] or 0)
        summary["slowest_metric"] = {"name": slowest[0], "p95_ms": slowest[1]["p95_ms"]}
    return summary


async def _evaluate_single_row(
    *,
    task_id: int,
    row_index: int,
    row_position: int,
    total_rows: int,
    base_row_data: dict[str, t.Any],
    metrics: list[tuple[str, t.Any, t.Any]],
    judge_client: OpenAIJudgeClient,
    panel_judges: list[OpenAIJudgeClient],
    panel_judge_names: list[str],
    primary_judge_name: str,
    evaluation_mode: str,
    target_config: dict[str, t.Any],
    response_mapping: dict[str, t.Any],
    cancel_event: asyncio.Event,
) -> dict[str, t.Any]:
    """评测单行，不碰数据库——所有落库都由调度侧串行完成。

    抽成纯函数是并发化的前提。原来这段逻辑内联在 for 循环里，每写一条日志、
    每做一次取消检查都紧跟一个 `db.commit()`。多行并发跑的时候那样写有两个
    后果：一是不同行的中间状态会交叉提交到同一个 session，二是日志按时间
    穿插成"行1的指标A、行3的指标B、行1的指标C"，没法读。所以这里只返回
    数据和一整块日志，由调度侧按行原子写入。

    唯一保留的进程内副作用是心跳。`task_heartbeats` 是模块级 dict，asyncio
    单线程下并发写同一个 key 是安全的；而心跳必须在行内刷新——否则并发跑
    超过 recovery 窗口的任务会被当成陈旧任务杀掉，这个坑在串行版里被
    "每个指标都刷一次"掩盖着。
    """
    logs: list[str] = []
    started = time.time()
    metric_scores: dict[str, t.Any] = {}
    metric_latency_ms: dict[str, int] = {}
    endpoint_trace: dict[str, t.Any] | None = None
    row_error: str | None = None
    extracted_fields: dict[str, t.Any] = {}

    user_input_preview = str(base_row_data.get("user_input", ""))[:60]
    logs.append(f"── 行 #{row_position}/{total_rows}: {user_input_preview}...")

    # 已经取消时直接返回：这一行还没开始，不该产生结果记录
    if cancel_event.is_set():
        return {"row_index": row_index, "cancelled": True, "logs": logs}

    row_data: dict[str, t.Any] = {**base_row_data, "_dataset_data": base_row_data}

    if evaluation_mode == "endpoint":
        try:
            logs.append("  ▸ 调用被测业务接口...")
            response_payload = await invoke_endpoint(base_row_data, target_config or {})
            extracted_fields, mapping_errors = extract_eval_fields(
                response_payload, response_mapping or {}
            )
            endpoint_trace = {
                "status": "success",
                "status_code": response_payload.get("status_code"),
                "latency_ms": response_payload.get("latency_ms"),
                "request_body": response_payload.get("request_body"),
                "raw_response": response_payload.get("raw_response"),
                "extracted_fields": extracted_fields,
                "mapping_errors": mapping_errors,
            }
            row_data = {
                **base_row_data,
                **extracted_fields,
                "_dataset_data": base_row_data,
                "_endpoint_trace": endpoint_trace,
            }
            missing_fields = _missing_fields_for_metrics(row_data, metrics)
            if missing_fields:
                row_error = f"字段缺失导致无法完整评分: {', '.join(missing_fields)}"
            logs.append(
                f"  ✓ 接口调用成功，提取字段: {', '.join(extracted_fields.keys()) or '无'}"
            )
            if mapping_errors:
                logs.append(f"  ⚠ 字段映射提示: {mapping_errors}")
        except Exception as exc:
            row_error = f"接口调用失败: {str(exc)[:800]}"
            endpoint_trace = {"status": "error", "error": row_error}
            for metric_name, _metric_instance, _sm in metrics:
                metric_scores[metric_name] = {"score": None, "reason": row_error}
            logs.append(f"  ✗ {row_error[:200]}")

    for metric_name, metric_instance, _sm in metrics:
        if row_error and metric_scores.get(metric_name):
            continue
        if cancel_event.is_set():
            logs.append(f"⚠ 用户取消评测，当前行停止在指标 [{metric_name}]")
            break

        logs.append(f"  ▸ 评测指标 [{metric_name}]...")
        task_heartbeats[task_id] = time.time()
        metric_started = time.time()

        is_llm_metric = _is_llm_metric(metric_instance)
        judges = [judge_client] + (panel_judges if is_llm_metric else [])
        judge_names = [primary_judge_name] + (panel_judge_names if is_llm_metric else [])

        metric_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            # token 计数归零放进 try：这一圈原先在 try 之外，裁判客户端只要在这里
            # 抛异常（缺方法、连接对象已关闭等），整个任务就带着裸 traceback 变成
            # failed。而这个 try/except 存在的意义本来就是"单指标失败只坏这一格"，
            # 归零属于该指标的准备步骤，没有理由被排除在外。
            for jc in judges:
                jc.reset_row_usage()

            if len(judges) == 1:
                result, judge_stats = await _score_metric_with_sampling(
                    metric_instance, row_data, judge_client
                )
                val = result.value
                if isinstance(val, (int, float)):
                    val = round(float(val), 4)
                reason = str(result.reason or "")[:800]
                metric_scores[metric_name] = {"score": val, "reason": reason}
                if judge_stats:
                    metric_scores[metric_name].update(judge_stats)
            else:
                panel_results: list[tuple[str, _MetricResult, dict[str, t.Any]]] = []
                for jc, judge_name in zip(judges, judge_names):
                    judge_result, judge_stats = await _score_metric_with_sampling(
                        metric_instance, row_data, jc
                    )
                    panel_results.append((judge_name, judge_result, judge_stats))
                value, reason, panel_stats = _aggregate_judge_results(panel_results)
                val = round(float(value), 4) if isinstance(value, (int, float)) else value
                metric_scores[metric_name] = {"score": val, "reason": str(reason)[:800]}
                metric_scores[metric_name].update(panel_stats)

            for jc in judges:
                usage = jc.take_row_usage()
                metric_usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                metric_usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                metric_usage["total_tokens"] += int(usage.get("total_tokens") or 0)
            if metric_usage.get("total_tokens", 0) > 0:
                metric_scores[metric_name]["judge_tokens"] = metric_usage

            # 换序互评：反转列表字段后由主裁判复评，检测位置偏置
            if settings.EVAL_SWAP_CHECK and is_llm_metric:
                swap_result, _swap_stats = await _score_metric_with_sampling(
                    metric_instance, _swap_row_data(row_data), judge_client
                )
                swap_val = swap_result.value
                if isinstance(val, (int, float)) and isinstance(swap_val, (int, float)):
                    consistency = 1.0 - abs(float(val) - float(swap_val))
                    metric_scores[metric_name]["swap_score"] = round(float(swap_val), 4)
                    metric_scores[metric_name]["swap_consistency"] = round(
                        max(0.0, min(1.0, consistency)), 4
                    )
                # 换序复评的 token 也计入成本
                swap_usage = judge_client.take_row_usage()
                metric_usage["prompt_tokens"] += int(swap_usage.get("prompt_tokens") or 0)
                metric_usage["completion_tokens"] += int(swap_usage.get("completion_tokens") or 0)
                metric_usage["total_tokens"] += int(swap_usage.get("total_tokens") or 0)
                if metric_usage.get("total_tokens", 0) > 0:
                    metric_scores[metric_name]["judge_tokens"] = metric_usage

            if val is None:
                logs.append(f"  ✗ [{metric_name}] 无法评分: {str(reason)[:200]}")
            else:
                logs.append(f"  ✓ [{metric_name}] = {val}")
        except Exception as exc:
            metric_scores[metric_name] = {"score": None, "reason": str(exc)[:800]}
            logs.append(f"  ✗ [{metric_name}] 失败: {str(exc)[:200]}")

        metric_latency_ms[metric_name] = int((time.time() - metric_started) * 1000)

    if cancel_event.is_set():
        return {"row_index": row_index, "cancelled": True, "logs": logs}

    is_pass = _determine_pass(metric_scores, metrics)
    execution_time_ms = int((time.time() - started) * 1000)
    status_icon = "✓ 通过" if is_pass else "✗ 不通过"
    logs.append(f"  → 结果: {status_icon} ({execution_time_ms}ms)")

    return {
        "row_index": row_index,
        "cancelled": False,
        "metric_scores": metric_scores,
        "metric_latency_ms": metric_latency_ms,
        "endpoint_trace": endpoint_trace,
        "extracted_fields": extracted_fields,
        "row_error": row_error,
        "is_pass": is_pass,
        "execution_time_ms": execution_time_ms,
        "logs": logs,
    }


async def run_evaluation(task_id: int, session_factory) -> None:
    """
    Execute a full evaluation run for the given EvalTask.

    Designed to be launched with ``asyncio.create_task(...)`` from the API
    endpoint. ``session_factory`` is the sync ``SessionLocal`` class, so the
    background runner opens its own session.
    """
    from app.models.dataset import DatasetRow
    from app.models.evaluation import EvalTask
    from app.models.llm_config import LLMConfig
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
        evaluation_mode = task.evaluation_mode or "offline"

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
        _log(task, f"评测模式: {'接口实时评测' if evaluation_mode == 'endpoint' else '已有结果评测'}")
        _log(task, f"评判 LLM: {llm_config.model_name} @ {llm_config.api_base_url}")
        _log(task, f"指标数: {len(scenario_metrics)} 个")
        _log(task, f"进度: 0/{total_rows} (0%)")
        db.commit()
        _broadcast_progress(task)

        _log(task, "正在构建原生评判 LLM 客户端...")
        db.commit()
        judge_client = OpenAIJudgeClient(llm_config)
        _log(task, "✓ 评判 LLM 客户端构建成功")
        db.commit()

        # 多裁判面板：附加裁判与主裁判独立打分，LLM 指标取均值/多数票，
        # 并用裁判间平均绝对偏差量化"换一个模型还认不认这个分"
        panel_judges: list[OpenAIJudgeClient] = []
        panel_judge_names: list[str] = []
        # 并发跑多行时每个并发槽位需要一整套独立裁判客户端（见下方 bundle 池），
        # 所以这里把构建成功的配置留下来，而不只留客户端实例
        panel_configs: list[t.Any] = []
        for panel_config_id in list(task.judge_panel or []):
            panel_config = db.query(LLMConfig).filter(LLMConfig.id == panel_config_id).first()
            if panel_config is None:
                _log(task, f"⚠ 裁判配置 #{panel_config_id} 不存在，已跳过")
                continue
            try:
                panel_judges.append(OpenAIJudgeClient(panel_config))
                panel_judge_names.append(panel_config.name or f"judge-{panel_config_id}")
                panel_configs.append(panel_config)
                _log(task, f"✓ 附加裁判 [{panel_config.name}] ({panel_config.model_name}) 构建成功")
            except Exception as exc:
                _log(task, f"⚠ 裁判 [{panel_config.name}] 构建失败: {str(exc)[:200]}")
        if panel_judges:
            _log(task, f"多裁判面板就绪: 主裁判 {llm_config.model_name} + {len(panel_judges)} 个附加裁判")
            db.commit()

        metrics: list[tuple[str, t.Any, ScenarioMetric]] = []
        for sm in scenario_metrics:
            metric_def: MetricDefinition = sm.metric_definition
            try:
                kind, metric_instance = build_metric(
                    metric_def,
                    prompt_override=getattr(sm, "prompt_override", None),
                    scenario_metric=sm,
                )
                metrics.append((metric_def.name, metric_instance, sm))
                _log(task, f"✓ 指标 [{metric_def.display_name}] 构建成功 (类型: {kind})")
            except Exception as exc:
                _log(task, f"✗ 指标 [{metric_def.display_name}] 构建失败: {exc}")

        if not metrics:
            _log(task, "✗ 错误: 没有任何指标构建成功，无法执行评测")
            db.commit()
            raise RuntimeError("No metrics could be built for this scenario")

        row_concurrency = max(1, int(settings.EVAL_ROW_CONCURRENCY))
        # 并发上限不该超过行数：4 个槽位跑 2 行只会白建 2 套裁判客户端
        row_concurrency = min(row_concurrency, total_rows)
        if row_concurrency > 1:
            _log(
                task,
                f"========== 开始并发评测 ({total_rows} 条, 并发 {row_concurrency}) ==========",
            )
            # 并发下日志按"行完成"整块落库而非实时逐行追加，这里说清楚，
            # 否则看日志的人会以为任务卡住了
            _log(task, "提示: 并发模式下每行日志在该行评测完成时整块写入")
        else:
            _log(task, f"========== 开始逐行评测 ({total_rows} 条) ==========")
        db.commit()

        # 按 row_index 存放，最后排序后再算汇总——完成顺序不能影响汇总结果
        row_scores_by_index: dict[int, dict[str, t.Any]] = {}
        task_token_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        row_latencies_ms: list[int] = []
        metric_latencies_ms: dict[str, list[int]] = {}
        rows_by_index = {row.row_index: row for row in dataset_rows}
        wall_clock_started = time.time()

        cancel_event = asyncio.Event()
        watcher_stop = asyncio.Event()

        async def _watch_for_cancel() -> None:
            """独立 session 轮询取消状态，间隔见 CANCEL_POLL_INTERVAL_SECONDS。

            为什么要单开一个 session：调度侧的 session 正在跑写事务，用同一个
            连接反复 refresh 会把读到的状态和未提交的写混在一起。取消检查只
            需要读，单独一个连接最干净。

            为什么要轮询：串行版靠"每个指标前 db.refresh(task)"检查取消，
            并发化后那样做会变成 N×M 次刷库。轮询把它降到每秒一次，
            响应延迟从"下一个指标边界"变成"最多一个轮询间隔"，实际更快。
            """
            watch_db = session_factory()
            try:
                while not watcher_stop.is_set():
                    try:
                        await asyncio.wait_for(
                            watcher_stop.wait(),
                            timeout=CANCEL_POLL_INTERVAL_SECONDS,
                        )
                        return
                    except asyncio.TimeoutError:
                        pass
                    try:
                        # 结束上一次读事务再读，确保拿到的是最新提交。
                        # 实测（pysqlite 默认 isolation_level=''）：驱动不会为
                        # 纯 SELECT 发真正的 BEGIN，所以不 rollback 也能读到新
                        # 提交——这行是防御性的，防的是日后把连接改成真正会
                        # 开读事务的配置（显式 BEGIN / 其他驱动），那时缺了它
                        # 就会一直读到旧快照，取消永远检测不到。
                        watch_db.rollback()
                        status = (
                            watch_db.query(EvalTask.status)
                            .filter(EvalTask.id == task_id)
                            .scalar()
                        )
                        if status == "cancelled":
                            cancel_event.set()
                            return
                    except Exception:
                        logger.debug("Cancel watcher query failed", exc_info=True)
            finally:
                watch_db.close()

        # 并发路径上一次 ORM 属性访问都不能有。session 默认 expire_on_commit=True，
        # 而调度侧每完成一行就 commit 一次——commit 之后 task 的属性全部过期，
        # 协程里再读 task.target_config 就会触发一次同步查库，而这次查库发生在
        # 其他行正在飞的时候。所以这些值在并发开始前一次性取成纯 dict。
        frozen_target_config = dict(task.target_config or {})
        frozen_response_mapping = dict(task.response_mapping or {})
        frozen_primary_judge_name = llm_config.model_name
        frozen_save_mode = task.result_save_mode

        async def _run_row(row_position: int, row_index: int, base_row_data: dict) -> dict:
            """借一套裁判客户端跑一行，跑完归还。

            bundle 池同时充当并发闸门：池里只有 row_concurrency 套客户端，
            第 N+1 行会阻塞在 `pool.get()` 上。用一个 Queue 同时解决"限并发"
            和"每行独占一套客户端"，比 Semaphore + 另一个池少一层结构。

            独占是必须的：`OpenAIJudgeClient.row_usage` 是实例级可变状态，
            两行共用一个客户端时 `reset_row_usage()` / `take_row_usage()`
            会互相清掉对方的 token 计数，成本统计直接失真。
            """
            if cancel_event.is_set():
                return {"row_index": row_index, "cancelled": True, "logs": []}
            bundle = await pool.get()
            try:
                if cancel_event.is_set():
                    return {"row_index": row_index, "cancelled": True, "logs": []}
                return await _evaluate_single_row(
                    task_id=task_id,
                    row_index=row_index,
                    row_position=row_position,
                    total_rows=total_rows,
                    base_row_data=base_row_data,
                    metrics=metrics,
                    judge_client=bundle["judge"],
                    panel_judges=bundle["panel"],
                    panel_judge_names=panel_judge_names,
                    primary_judge_name=frozen_primary_judge_name,
                    evaluation_mode=evaluation_mode,
                    target_config=frozen_target_config,
                    response_mapping=frozen_response_mapping,
                    cancel_event=cancel_event,
                )
            finally:
                pool.put_nowait(bundle)

        # 每个并发槽位一套独立裁判客户端。第一套复用已构建好的实例，
        # 避免并发数为 1 时（默认串行）多建一套连接池。
        pool: asyncio.Queue = asyncio.Queue()
        pool.put_nowait({"judge": judge_client, "panel": panel_judges})
        for _ in range(row_concurrency - 1):
            pool.put_nowait(
                {
                    "judge": OpenAIJudgeClient(llm_config),
                    "panel": [OpenAIJudgeClient(cfg) for cfg in panel_configs],
                }
            )

        # ORM 属性在协程里访问可能触发懒加载，所以行数据在调度侧先取成纯 dict
        row_payloads = [
            (position, row.row_index, dict(row.data or {}))
            for position, row in enumerate(dataset_rows, start=1)
        ]

        watcher = asyncio.create_task(_watch_for_cancel())
        pending = [
            asyncio.ensure_future(_run_row(position, row_index, payload))
            for position, row_index, payload in row_payloads
        ]

        try:
            for finished in asyncio.as_completed(pending):
                result = await finished
                row_index = result["row_index"]

                # 整块写入这一行的日志：并发下逐条追加会把不同行的日志穿插
                # 成没法阅读的样子
                for line in result.get("logs") or []:
                    _log(task, line)

                if result.get("cancelled"):
                    db.commit()
                    continue

                dataset_row = rows_by_index[row_index]
                metric_scores = result["metric_scores"]
                extracted_fields = result.get("extracted_fields") or {}

                # 回写必须在调度侧做：worker 拿不到 session，只把提取到的字段带回来
                if (
                    evaluation_mode == "endpoint"
                    and frozen_save_mode == "write_back"
                    and extracted_fields
                ):
                    dataset_row.data = {**(dataset_row.data or {}), **extracted_fields}
                    _ensure_dataset_schema_fields(dataset, extracted_fields)

                _persist_row_result(
                    db,
                    task,
                    dataset_row,
                    metric_scores,
                    result.get("row_error"),
                    result["is_pass"],
                    result["execution_time_ms"],
                    endpoint_trace=result.get("endpoint_trace"),
                )

                row_scores_by_index[row_index] = metric_scores
                row_latencies_ms.append(result["execution_time_ms"])
                for metric_name, elapsed in (result.get("metric_latency_ms") or {}).items():
                    metric_latencies_ms.setdefault(metric_name, []).append(elapsed)

                for entry in metric_scores.values():
                    if not isinstance(entry, dict):
                        continue
                    row_usage = entry.get("judge_tokens")
                    if isinstance(row_usage, dict):
                        task_token_usage["prompt_tokens"] += int(row_usage.get("prompt_tokens") or 0)
                        task_token_usage["completion_tokens"] += int(row_usage.get("completion_tokens") or 0)
                        task_token_usage["total_tokens"] += int(row_usage.get("total_tokens") or 0)

                task.completed_rows = len(row_scores_by_index)
                task.progress = round(task.completed_rows / total_rows, 4)
                task_heartbeats[task_id] = time.time()
                _log(
                    task,
                    f"  → 进度: {task.completed_rows}/{total_rows} ({round(task.progress * 100, 1)}%)",
                )
                # 每行一次 commit。串行版每写一行日志就 commit 一次，
                # 25 行 × 6 指标下是几百次写事务，绝大多数只为了让日志实时可见
                db.commit()
                _broadcast_progress(task)
        finally:
            watcher_stop.set()
            watcher.cancel()
            for future in pending:
                if not future.done():
                    future.cancel()

        wall_clock_ms = int((time.time() - wall_clock_started) * 1000)
        all_row_scores: list[dict[str, t.Any]] = [
            row_scores_by_index[idx] for idx in sorted(row_scores_by_index)
        ]

        # 显式 refresh：取消可能在最后一行完成之后、写终态之前才落库。
        # 当前 session 是 expire_on_commit=True，逐行 commit 已经让 task 属性
        # 过期，读 task.status 本身就会重查——也就是说这行在当前配置下是冗余的。
        # 留着是因为它守的是"跑满也不能覆盖 cancelled"这个语义：一旦日后把
        # session 改成 expire_on_commit=False（很合理的优化，正好能省掉并发
        # 路径上那些 frozen_* hoist），没有这行就会把用户的取消写成 completed。
        # tests/test_eval_cancellation.py 里有一条专门用 expire_on_commit=False
        # 跑的用例钉住这一点。
        db.refresh(task)
        if task.status == "cancelled":
            _log(task, "========== 评测已取消 ==========")
            task.finished_at = datetime.now(timezone.utc)
            db.commit()
            _broadcast_progress(task)
        else:
            _log(task, "========== 计算汇总统计 ==========")
            db.commit()
            summary_scores = _compute_summary_scores(all_row_scores, metrics)
            _attach_task_level_aggregates(summary_scores, task_token_usage)
            # 延迟分位数：execution_time_ms 原先只逐行落库、从未聚合，
            # 任务级报告里查不到"这轮评测跑了多久、慢在哪个指标"
            summary_scores["latency"] = _build_latency_summary(
                row_latencies_ms, metric_latencies_ms, wall_clock_ms, row_concurrency
            )
            task.status = "completed"
            task.finished_at = datetime.now(timezone.utc)
            task.summary_scores = summary_scores
            task.progress = 1.0

            pass_count = sum(1 for s in all_row_scores if _determine_pass(s, metrics))
            _log(task, f"通过: {pass_count}/{total_rows} ({round(pass_count / total_rows * 100, 1)}%)")
            for m_name, info in summary_scores.items():
                if m_name in RESERVED_SUMMARY_KEYS:
                    continue
                mean = info.get("mean")
                pr = info.get("pass_rate")
                _log(task, f"  {m_name}: 均值={mean}, 通过率={pr}")
            latency = summary_scores["latency"]
            _log(
                task,
                f"耗时: 墙钟 {latency['wall_clock_ms']}ms / 行 p50 {latency['row_p50_ms']}ms"
                f" / 行 p95 {latency['row_p95_ms']}ms"
                + (
                    f" / 并发 {row_concurrency} 加速比 {latency['speedup_estimate']}x"
                    if row_concurrency > 1
                    else ""
                ),
            )
            if latency.get("slowest_metric"):
                _log(
                    task,
                    f"最慢指标: {latency['slowest_metric']['name']}"
                    f" (p95 {latency['slowest_metric']['p95_ms']}ms)",
                )
            _log(task, "========== 评测完成 ==========")
            db.commit()
            _broadcast_progress(task)

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
                _broadcast_progress(task)
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
    endpoint_trace: dict | None = None,
) -> None:
    """Create an EvalRowResult and add it to the session (caller commits)."""
    from app.models.evaluation import EvalRowResult

    row_result = EvalRowResult(
        eval_task_id=task.id,
        dataset_row_id=dataset_row.id,
        row_index=dataset_row.row_index,
        metric_scores=metric_scores,
        endpoint_trace=endpoint_trace,
        is_pass=is_pass,
        execution_time_ms=execution_time_ms,
        error=error,
    )
    db.add(row_result)


def _missing_fields_for_metrics(row_data: dict, metrics: list) -> list[str]:
    required_fields: set[str] = set()
    for _metric_name, metric_instance, _scenario_metric in metrics:
        required_fields.update(getattr(metric_instance, "required_fields", []) or [])
    return _missing_required_fields(row_data, sorted(required_fields))


def _ensure_dataset_schema_fields(dataset, extracted_fields: dict[str, t.Any]) -> None:
    existing_schema = list(dataset.field_schema or [])
    existing_names = {field.get("name") for field in existing_schema if isinstance(field, dict)}
    for field_name, value in extracted_fields.items():
        if field_name in existing_names:
            continue
        existing_schema.append(
            {
                "name": field_name,
                "type": _infer_schema_type(value),
                "required": False,
                "description": "接口实时评测回写字段",
            }
        )
    dataset.field_schema = existing_schema


def _infer_schema_type(value: t.Any) -> str:
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "json"
    if isinstance(value, (int, float)):
        return "number"
    return "text"


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
            "pass_rate": 0.9, "count": 10, "error_count": 0,
            "judge_std_mean": 0.02, "low_confidence_count": 1
        },
        "weighted_total_score": {"weighted_mean": 0.81, ...}
    }

    ``weighted_total_score`` aggregates numeric metrics using the scenario
    weights frozen in the task snapshot, so different experiments with the same
    scenario remain comparable on a single headline number.
    """
    thresholds = {name: sm.pass_threshold for name, _, sm in metrics}
    summary: dict = {}

    for metric_name, _, _ in metrics:
        numeric_values: list[float] = []
        error_count = 0
        std_values: list[float] = []
        low_confidence_count = 0

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

            # 多采样场景下记录 judge 稳定性信号
            row_std = raw.get("score_std") if isinstance(raw, dict) else None
            if isinstance(row_std, (int, float)):
                std_values.append(float(row_std))
                if float(row_std) > settings.JUDGE_STD_THRESHOLD:
                    low_confidence_count += 1

        swap_consistencies: list[float] = []
        swap_inconsistent_count = 0
        judge_mads: list[float] = []
        for row_scores in all_row_scores:
            raw = row_scores.get(metric_name)
            if isinstance(raw, dict):
                swap_consistency = raw.get("swap_consistency")
                if isinstance(swap_consistency, (int, float)):
                    swap_consistencies.append(float(swap_consistency))
                    if float(swap_consistency) < 0.9:
                        swap_inconsistent_count += 1
                judge_mad = raw.get("judge_mad")
                if isinstance(judge_mad, (int, float)):
                    judge_mads.append(float(judge_mad))

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
            "judge_std_mean": (
                round(sum(std_values) / len(std_values), 4) if std_values else None
            ),
            "low_confidence_count": low_confidence_count,
            "swap_consistency_mean": (
                round(sum(swap_consistencies) / len(swap_consistencies), 4)
                if swap_consistencies
                else None
            ),
            "swap_inconsistent_count": swap_inconsistent_count,
            "judge_mad_mean": (
                round(sum(judge_mads) / len(judge_mads), 4) if judge_mads else None
            ),
        }

    # 总体加权总分：仅统计有数值均值的指标，权重来自任务创建时冻结的场景快照
    weighted_parts: list[tuple[str, float, float]] = []
    for metric_name, _instance, scenario_metric in metrics:
        info = summary.get(metric_name) or {}
        if info.get("mean") is None:
            continue
        try:
            weight = float(getattr(scenario_metric, "weight", None) or 0.0)
        except (TypeError, ValueError):
            weight = 0.0
        if weight > 0:
            weighted_parts.append((metric_name, float(info["mean"]), weight))

    if weighted_parts:
        weight_sum = sum(weight for _name, _mean, weight in weighted_parts)
        weighted_mean = sum(mean * weight for _name, mean, weight in weighted_parts) / weight_sum
        summary["weighted_total_score"] = {
            "weighted_mean": round(weighted_mean, 4),
            "metric_count": len(weighted_parts),
            "weights": {name: weight for name, _mean, weight in weighted_parts},
        }

    return summary


def _attach_task_level_aggregates(summary_scores: dict, token_usage: dict[str, int]) -> None:
    """Attach task-level aggregates (cost, judge reliability) to the summary dict."""
    if token_usage.get("total_tokens", 0) > 0:
        prompt_tokens = int(token_usage.get("prompt_tokens") or 0)
        completion_tokens = int(token_usage.get("completion_tokens") or 0)
        estimated_cost = (
            prompt_tokens / 1000 * settings.LLM_INPUT_PRICE_PER_1K
            + completion_tokens / 1000 * settings.LLM_OUTPUT_PRICE_PER_1K
        )
        summary_scores["cost"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(token_usage.get("total_tokens") or 0),
            "estimated_cost": round(estimated_cost, 6),
            "currency": "CNY",
            "pricing_per_1k": {
                "input": settings.LLM_INPUT_PRICE_PER_1K,
                "output": settings.LLM_OUTPUT_PRICE_PER_1K,
            },
        }

    std_means = [
        info.get("judge_std_mean")
        for info in summary_scores.values()
        if isinstance(info, dict) and info.get("judge_std_mean") is not None
    ]
    swap_means = [
        info.get("swap_consistency_mean")
        for info in summary_scores.values()
        if isinstance(info, dict) and info.get("swap_consistency_mean") is not None
    ]
    judge_mad_means = [
        info.get("judge_mad_mean")
        for info in summary_scores.values()
        if isinstance(info, dict) and info.get("judge_mad_mean") is not None
    ]
    reliability: dict[str, t.Any] = {}
    if std_means:
        low_confidence = sum(
            int(info.get("low_confidence_count") or 0)
            for info in summary_scores.values()
            if isinstance(info, dict)
        )
        reliability.update(
            {
                "sampled_metric_count": len(std_means),
                "mean_std": round(sum(std_means) / len(std_means), 4),
                "low_confidence_row_count": low_confidence,
                "std_threshold": settings.JUDGE_STD_THRESHOLD,
            }
        )
    if swap_means:
        reliability.update(
            {
                "swap_metric_count": len(swap_means),
                "mean_swap_consistency": round(sum(swap_means) / len(swap_means), 4),
                "swap_inconsistent_rows": sum(
                    int(info.get("swap_inconsistent_count") or 0)
                    for info in summary_scores.values()
                    if isinstance(info, dict)
                ),
            }
        )
    if judge_mad_means:
        reliability.update(
            {
                "panel_metric_count": len(judge_mad_means),
                "mean_judge_mad": round(sum(judge_mad_means) / len(judge_mad_means), 4),
            }
        )
    if reliability:
        summary_scores["judge_reliability"] = reliability


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


def _public_row_data(row_data: dict[str, t.Any]) -> dict[str, t.Any]:
    return {
        key: value
        for key, value in (row_data or {}).items()
        if not str(key).startswith("_")
    }


def _metric_score_range(metric_def) -> str | None:
    config = getattr(metric_def, "config", None) or {}
    metric_type = getattr(metric_def, "metric_type", "")
    if metric_type == "numeric":
        allowed = config.get("allowed_values") or [0, 1]
        if isinstance(allowed, list) and len(allowed) >= 2:
            return f"{allowed[0]} 到 {allowed[1]}"
    if metric_type == "discrete":
        allowed = config.get("allowed_values") or ["pass", "fail"]
        return " / ".join(str(item) for item in allowed)
    if metric_type == "aspect_critic":
        return "0 或 1"
    return "0 到 1"


def _excluded_sample_fields(referenced_fields: set[str]) -> set[str]:
    excluded: set[str] = set()
    for field in referenced_fields:
        if "." not in field:
            excluded.add(field)
            continue
        namespace, name = field.split(".", 1)
        if namespace in {"dataset", "endpoint"} and name:
            excluded.add(name)
    return excluded


def _render_custom_prompt_and_sample(
    prompt_template: str,
    row_data: dict[str, t.Any],
    metric_def=None,
    scenario_metric=None,
) -> tuple[str, dict[str, t.Any]]:
    """Render a custom Judge prompt and omit embedded fields from sample.

    Custom metric prompts often contain variables such as ``{user_input}``,
    ``{reference}`` and ``{response}``. After rendering, those field values are
    already part of the business prompt, so sending them again in ``sample``
    makes the raw Judge Prompt noisy and can confuse the expected judging
    source of truth. Non-referenced fields are still kept in ``sample`` as
    useful structured context.
    """

    referenced_fields = extract_prompt_variables(prompt_template)
    context = build_template_context(
        row_data,
        row_data.get("_endpoint_trace"),
        metric_def,
        scenario_metric,
    )
    prompt = render_prompt(prompt_template, context)
    return prompt, _sample_payload(row_data, exclude_fields=_excluded_sample_fields(referenced_fields))


# 永不下发给 Judge 的字段：数据集生产过程留下的元信息。
# generation_meta 里可能带有样本的真值标记（如 {"kind": "hallucinated"}）或
# 内部编号，一旦进入 Judge Prompt 就等于把答案泄露给裁判，评分不再是对回答
# 质量的独立判断，指标区分度会被虚高。裁判只应看到被评测系统的输入与输出。
_JUDGE_HIDDEN_FIELDS = {"generation_meta"}


def _sample_payload(
    row_data: dict[str, t.Any],
    exclude_fields: set[str] | None = None,
    allow_fields: t.Sequence[str] | None = None,
) -> dict[str, t.Any]:
    """Build the ``sample`` block of a Judge Prompt.

    ``allow_fields`` 是白名单口径：给定时，**只有**名单内的字段会下发给裁判，
    兜底循环整段跳过。内置指标走这条路径（名单来自 spec 的 ``judge_fields``），
    以保证"指标只看它声明要看的东西"。

    不给 ``allow_fields`` 时退回黑名单口径：重要字段优先、其余字段兜底补齐。
    自定义 prompt 指标走这条路径——它的可见范围由用户自己写的模板决定，
    平台不替用户裁剪。
    """

    exclude_fields = set(exclude_fields or set()) | _JUDGE_HIDDEN_FIELDS
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

    if allow_fields is not None:
        # 白名单模式：严格按声明口径下发，顺序沿用 important_fields 以保持
        # Judge Prompt 稳定（同一指标的 prompt 不因 dict 顺序变化而抖动），
        # 名单里的非常规字段追加在后面。
        allowed = [key for key in allow_fields if key not in exclude_fields]
        ordered = [key for key in important_fields if key in allowed]
        ordered += [key for key in allowed if key not in ordered]
        return {
            key: _json_safe(row_data[key])
            for key in ordered
            if key in row_data and row_data[key] is not None
        }

    payload = {
        key: _json_safe(row_data[key])
        for key in important_fields
        if key in row_data and key not in exclude_fields and row_data[key] is not None
    }
    for key, value in row_data.items():
        if str(key).startswith("_"):
            continue
        if key not in exclude_fields and key not in payload and len(payload) < 16:
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
