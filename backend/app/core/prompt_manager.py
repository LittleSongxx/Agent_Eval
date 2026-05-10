"""Centralized system prompts and prompt/template rendering helpers.

This module intentionally stores only platform-owned prompt text and default
templates. Business-owned content must stay in the database, for example
``MetricDefinition.config.prompt``, ``ScenarioMetric.prompt_override`` and
``EndpointTarget.request_body_template``. Keeping that boundary clear prevents
business scoring standards from being mixed with system output-format rules.
"""

from __future__ import annotations

import json
import re
import typing as t


# System-only Judge instruction. Business users should not override this because
# the evaluation engine depends on the stable JSON shape to persist scores.
JUDGE_SYSTEM_PROMPT = (
    "你是一个严谨的 AI 评测裁判。只能依据用户提供的样本字段评分，不要引入外部知识或想象证据。"
    "必须返回严格 JSON，格式为 {\"score\": 数值或标签, \"reason\": \"1到2句中文理由\"}。"
    "reason 必须指出具体支撑点或扣分点，不要写空泛结论。"
)


# Metric instructions are also system-owned: business prompts decide what to
# judge, while these strings decide how the Judge must return the result.
NUMERIC_SCORE_INSTRUCTION = "请按 prompt 的标准评分，返回 JSON: {\"score\": 数值, \"reason\": \"中文理由\"}。"
DISCRETE_SCORE_INSTRUCTION = "score 必须严格使用 allowed_values 中的一个值，并返回中文 reason。"
ASPECT_CRITIC_INSTRUCTION = "判断样本是否满足 definition，返回 0 或 1，并说明具体理由。"
BUILTIN_LLM_SCORE_INSTRUCTION = (
    "请给出 0 到 1 的浮点分数，并用中文说明理由。理由必须基于 sample 中的具体字段，"
    "指出哪里支持高分或哪里导致扣分。"
)


# Built-in metric criteria are platform defaults. Scenario-level
# ``prompt_override`` can replace a criterion for a specific business scenario.
BUILTIN_LLM_METRIC_SPECS: dict[str, dict[str, t.Any]] = {
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


NAMED_LLM_METRIC_SPECS: dict[str, dict[str, t.Any]] = {
    "answer_completeness": {
        "required_fields": ["response", "reference"],
        "criteria": (
            "判断 AI 回答是否完整覆盖参考答案中的关键要点。遗漏主要结论、条件、步骤或重要限定需要扣分；"
            "表达顺序不同但语义完整可以给高分。"
        ),
    },
}


# Generic endpoint default. Specific vendors such as DeepSeek are examples in
# the UI/database, not hard requirements for the backend engine.
DEFAULT_ENDPOINT_BODY_TEMPLATE = '{"question":"{{user_input}}"}'
DEFAULT_BLIND_TEST_ENDPOINT_BODY_TEMPLATE = '{"question":"{{question}}"}'
DEFAULT_RAG_TARGET_REQUEST_BODY_TEMPLATE = '{"question":"{{question}}","kb_codes":[],"payload":{"files":[]}}'
DEFAULT_RESPONSE_PATH = "data.answer"


PROMPT_VARIABLE_RE = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})")


# RAG dataset generation prompts are platform defaults used to synthesize and
# validate candidate questions from uploaded documents.
QUESTION_GENERATION_PROMPT = """你是一个严谨的 RAG 测试集构建助手。请只基于给定文档分片内容，生成 {n} 个高质量问题及标准答案。

## 文档
- 文件名: {filename}
- 分片ID: {chunk_key}

## 分片内容
{content}

## 要求
1. 问题必须能直接依据分片内容回答，不能依赖外部知识。
2. 问题要覆盖事实、规则、步骤、例外等不同角度，避免重复。
3. 标准答案必须简洁准确，严格以分片内容为依据。
4. 生成数量严格为 {n} 个。
5. 不要生成“文档里提到了什么”这类元问题。
6. 优先生成真实业务会问的问题，而不是照抄标题。
7. 每道题尽量只考查一个明确知识点，避免含糊表述。

## 输出格式
只输出 JSON：
{{
  "items": [
    {{
      "question": "问题文本",
      "reference": "标准答案"
    }}
  ]
}}
"""


QUESTION_VALIDATION_PROMPT = """你是一个严格的数据集质检助手。请检查下面每个候选问题是否真的适合作为 RAG 评测样本。

## 分片内容
{content}

## 候选问题
{items_json}

## 评估标准
1. 问题必须能仅凭分片内容回答。
2. 标准答案必须被分片直接支持，不能有外推。
3. 问题不能是“文档提到了什么”“这一节讲了什么”之类元问题。
4. 问题应具体、清晰、像真实用户会问的问题。
5. 重复或高度近似的问题只保留质量更高的那个。

## 输出格式
只输出 JSON：
{{
  "items": [
    {{
      "question": "原问题",
      "keep": true,
      "score": 0.0,
      "reason": "一句中文理由"
    }}
  ]
}}
"""


DEFAULT_TARGET_CONTEXT_PROMPT = """请使用你自己的正常 RAG / 知识库检索流程回答用户问题，并严格输出 JSON。

输出字段要求：
- answer: 最终回答
- retrieved_contexts: 你本次回答实际使用或返回给模型的检索片段数组
- retrieved_context_ids: 与 retrieved_contexts 对应的文档/分片 ID 数组

不要输出额外解释。"""


DEFAULT_TARGET_ANSWER_ONLY_PROMPT = """请回答用户问题，并严格输出 JSON：
{
  "answer": "你的回答"
}
不要输出额外解释。"""


def render_prompt(prompt: str, row_data: dict[str, t.Any]) -> str:
    """Render business scoring prompts using ``{field}`` variables.

    This is used for Judge prompts stored in metric/scenario configuration.
    Missing variables are rendered as empty strings so one partially populated
    row cannot break a full batch evaluation. Only simple ``{field}`` tokens
    are replaced; unrelated JSON/example braces in the prompt are left intact.
    """

    if not prompt:
        return ""
    safe_vars = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        for key, value in row_data.items()
    }

    def replace_var(match: re.Match[str]) -> str:
        return safe_vars.get(match.group(1), "")

    return PROMPT_VARIABLE_RE.sub(replace_var, prompt)


def extract_prompt_variables(prompt: str) -> set[str]:
    """Return row field names referenced by a business Judge prompt.

    The evaluator uses this to avoid sending the same field twice: if a custom
    prompt already embeds ``{response}``, the structured ``sample.response`` can
    be omitted from that Judge request while other non-referenced fields remain
    available.
    """

    if not prompt:
        return set()
    return set(PROMPT_VARIABLE_RE.findall(prompt))


def render_template_value(value: t.Any, context: dict[str, t.Any]) -> t.Any:
    """Render endpoint request templates using ``{{field}}`` variables.

    Endpoint templates need a double-brace syntax because request bodies are
    usually JSON examples. If the whole string is exactly one placeholder, the
    original value type is preserved; otherwise values are string-substituted.
    """

    if isinstance(value, str):
        stripped = value.strip()
        flat_context = _flatten_context(context)
        for key, raw_value in flat_context.items():
            if stripped == f"{{{{{key}}}}}":
                return raw_value
        rendered = value
        for key, raw_value in flat_context.items():
            replacement = raw_value if isinstance(raw_value, str) else json.dumps(raw_value, ensure_ascii=False)
            rendered = rendered.replace(f"{{{{{key}}}}}", replacement)
        return rendered
    if isinstance(value, list):
        return [render_template_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: render_template_value(item, context) for key, item in value.items()}
    return value


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _flatten_context(context: dict[str, t.Any]) -> dict[str, t.Any]:
    flattened = dict(context)
    row_data = context.get("row_data")
    if isinstance(row_data, dict):
        for key, value in row_data.items():
            flattened[f"row_data.{key}"] = value
    return flattened
