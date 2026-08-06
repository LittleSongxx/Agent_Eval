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

# 强制推理模式（JUDGE_COT_MODE=true 时启用）：先推理后打分，提升判定一致性
JUDGE_SYSTEM_PROMPT_COT = (
    JUDGE_SYSTEM_PROMPT
    + "在给出分数前，先在 reason 中写出逐步推理过程：1) 依据哪些样本字段；"
    "2) 逐条对照判定标准；3) 得出分数结论。reason 中推理在前、结论在后。"
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
#
# 两个字段各管一件事，不要混用：
#
# ``required_fields``  —— 评分前的**校验门槛**：数据集里必须存在这些字段，否则该行
#     直接判为"字段缺失"不打分。它只能包含静态数据集里就该有的字段。Agent /
#     多轮指标不把 ``response`` 列进来，是因为那类评测跑的是真实接口，回复在运行
#     时才由 ``extract_eval_fields`` 注入，提前要求会让 dataset 模式全部失败。
#
# ``judge_fields``     —— 下发给裁判的**可见范围**：只有这些字段会进 Judge Prompt
#     的 sample。这是"指标只看它该看的东西"的唯一执行点。
#
# 两者必须分开，因为它们的集合本来就不相等：
#   - context_recall 的可见范围**不能**含 ``response``——它衡量的是"检索到的上下文
#     够不够覆盖参考答案"，与回答写得好不好无关。曾经因为兜底逻辑把整行都下发，
#     裁判看到错误回答后把生成质量算进了检索分（见 docs/meta-evaluation.md §5.3
#     缺陷 B，A/B 实测：去掉 response 后 0.7875 → 1.0000，与 RAGAS 完全一致）。
#   - turn_relevancy 的可见范围**必须**含 ``response``——它判的就是"AI 回复有没有
#     回应当轮问题"，但 ``response`` 不能进 required_fields（理由见上）。
#
# 未声明 ``judge_fields`` 时回退到 ``required_fields``，保证新指标默认是收紧的而
# 不是放开的：漏配只会让裁判少看到字段（分数可疑、易发现），不会重新泄露。
BUILTIN_LLM_METRIC_SPECS: dict[str, dict[str, t.Any]] = {
    "builtin_faithfulness": {
        "required_fields": ["response", "retrieved_contexts"],
        # 生成侧：判断回答是否被上下文支持。不下发 reference——否则裁判会拿
        # 参考答案比对事实对错，那是 factual_correctness 的职责，不是忠实度。
        "judge_fields": ["response", "retrieved_contexts"],
        "criteria": (
            "判断 AI 回答是否忠实于检索上下文。回答中的关键事实必须能从 retrieved_contexts "
            "得到直接或合理支持；捏造、不被上下文支持或与上下文冲突的内容需要扣分。"
        ),
    },
    "builtin_context_recall": {
        "required_fields": ["retrieved_contexts", "reference"],
        # 检索侧指标：绝不能看见 response，否则生成质量会串进检索分。
        "judge_fields": ["retrieved_contexts", "reference"],
        "criteria": (
            "判断检索上下文是否覆盖参考答案中的关键事实和必要依据。reference 中的重要要点如果在 "
            "retrieved_contexts 中找不到对应支持，需要扣分。"
        ),
    },
    "builtin_context_precision": {
        "required_fields": ["user_input", "retrieved_contexts"],
        # criteria 明确要"与用户问题及参考答案是否相关"，故 reference 属于可见范围；
        # 但它不进 required_fields——没有 reference 时该指标仍可只按问题相关性打分。
        # 同样不含 response：判的是上下文有没有用，不是回答写得好不好。
        "judge_fields": ["user_input", "retrieved_contexts", "reference"],
        "criteria": (
            "判断检索上下文与用户问题及参考答案是否相关、有用。无关、重复、噪声或无法支撑回答的片段越多，"
            "分数越低。"
        ),
    },
    "builtin_contextual_relevancy": {
        "required_fields": ["user_input", "retrieved_contexts"],
        # 检索侧：只看上下文与问题是否相关，不含 response。
        "judge_fields": ["user_input", "retrieved_contexts"],
        "criteria": (
            "判断检索上下文整体是否与用户问题直接相关。上下文应围绕用户问题提供可用信息；无关、泛化、"
            "只沾边或无法帮助回答的问题片段需要扣分。"
        ),
    },
    "builtin_factual_correctness": {
        "required_fields": ["response", "reference"],
        # 生成侧：拿参考答案比对事实。不下发 retrieved_contexts——事实对错以
        # reference 为准，检索到了什么不影响判断。
        "judge_fields": ["response", "reference"],
        "criteria": (
            "判断 AI 回答与参考答案在事实层面是否一致。事实错误、遗漏关键限定条件、与 reference 冲突的内容"
            "需要扣分。"
        ),
    },
    "builtin_answer_relevancy": {
        "required_fields": ["user_input", "response"],
        # 只判"是否回应了问题"，不判对错。因此 reference 必须挡住：裁判一旦看到
        # 参考答案，就会把事实错误也算进扣分，指标口径悄悄从"相关性"漂移成
        # "正确性"。这是缺陷 B 在生成侧的那一半——曾导致本指标与 RAGAS 的
        # answer_relevancy 只有 0.44 相关（RAGAS 侧不惩罚事实错误）。
        "judge_fields": ["user_input", "response"],
        "criteria": (
            "判断 AI 回答是否直接且完整地回应用户问题。跑题、泛泛而谈、答非所问需要扣分。"
            "完整性的判定标准：回答在句子中间被截断（如以省略号、未完成句结尾，或只给出问题"
            "部分要点的结论），视为只回答了问题的一小部分，须显著扣分（此类回答不得高于 0.3）；"
            "回答已覆盖问题全部要点时不得因格式（如列表）或文字简洁扣分。"
            "只依据回答文本自身判断完整性，不得用裁判自身常识推断'完整答案还应包含什么'来扣分。"
        ),
    },
    "builtin_faithfulness_claim": {
        "required_fields": ["response", "retrieved_contexts"],
        # 该指标不走 sample 下发（自己直接读字段做断言拆解），judge_fields 在此
        # 只作口径声明，保证每个内置指标的可见范围都是显式写明、可被测试枚举的。
        "judge_fields": ["response", "retrieved_contexts"],
        "criteria": (
            "将 AI 回答拆解为若干独立断言（claim），逐条判断每个断言是否被 retrieved_contexts 支持，"
            "以被支持断言占比作为分数。未覆盖、与上下文冲突或无法验证的断言一律视为不支持。"
        ),
    },
    "builtin_answer_relevancy_generative": {
        "required_fields": ["user_input", "response"],
        # 同上：自己直接读字段做反推问题 + 向量相似度，不走 sample 下发。
        "judge_fields": ["user_input", "response"],
        "criteria": (
            "生成式相关性判定：从 AI 回答反推它可能回答的问题，再与用户问题做语义相似度比较，"
            "相似度越高说明回答越贴合问题。回答空泛、答非所问时反推问题与用户问题相似度低。"
        ),
    },
    "builtin_agent_goal_accuracy": {
        "required_fields": ["user_input", "reference"],
        # 可见范围比校验门槛宽：criteria 明确说"工具调用、最终回复和中间步骤都可以
        # 作为证据"，所以 response / tool_calls 必须下发。它们不进 required_fields，
        # 是因为 Agent 评测跑真实接口，这两个字段运行时才注入（见文件顶部说明）。
        "judge_fields": [
            "user_input",
            "reference",
            "response",
            "tool_calls",
            "reference_tool_calls",
        ],
        "criteria": (
            "判断 Agent 在整段对话中是否完成了用户目标，并与 reference 中的期望结果一致。工具调用、最终回复"
            "和中间步骤都可以作为证据。"
        ),
    },
    "builtin_task_completion": {
        "required_fields": ["user_input", "reference"],
        # 同上：判断任务闭环要看最终回复和实际执行的工具调用，两者都不在
        # required_fields 里（运行时注入），但必须对裁判可见。
        "judge_fields": [
            "user_input",
            "reference",
            "response",
            "tool_calls",
            "reference_tool_calls",
        ],
        "criteria": (
            "判断 Agent 是否完成了用户要达成的任务闭环。重点看最终状态是否已经满足 reference 描述的任务目标，"
            "而不只看是否给出看似合理的回复。"
        ),
    },
    "builtin_topic_adherence": {
        "required_fields": ["user_input", "reference_topics"],
        # 多轮：判"有没有跑题"必须看得见 AI 的回复。response 不在 required_fields
        # 里是因为多轮评测走真实接口、回复运行时才注入（见文件顶部说明）。
        "judge_fields": ["user_input", "reference_topics", "response"],
        "criteria": (
            "判断多轮对话是否始终围绕 reference_topics 指定的话题范围展开。明显偏离主题、引入无关内容或没有"
            "回应当前轮次主题需要扣分。"
        ),
    },
    "builtin_turn_relevancy": {
        "required_fields": ["user_input"],
        # 多轮：逐轮判回复是否回应了当前轮输入，response 是被判对象本身。
        "judge_fields": ["user_input", "response"],
        "criteria": (
            "逐轮判断 AI 回复是否回应了当前轮用户输入，并且没有忽略用户追问、答非所问或把上一轮上下文错误带入。"
        ),
    },
    "builtin_conversation_completeness": {
        "required_fields": ["user_input", "reference"],
        # 多轮：整段对话是否完成 reference 描述的目标，需要看到实际回复。
        "judge_fields": ["user_input", "reference", "response"],
        "criteria": (
            "判断整段多轮对话是否满足用户需求并完成 reference 描述的对话目标。遗漏关键步骤、没有收束问题或"
            "未给出可执行结论需要扣分。"
        ),
    },
    "builtin_knowledge_retention": {
        "required_fields": ["user_input"],
        # 多轮：判断 AI 有没有记住用户先前给过的信息，必须看到 AI 的回复本身。
        "judge_fields": ["user_input", "response"],
        "criteria": (
            "判断 AI 是否在多轮对话中持续记住用户已经提供的事实、约束、偏好、订单号、时间等信息。忘记、混淆"
            "或自相矛盾需要扣分。"
        ),
    },
    "builtin_role_adherence": {
        "required_fields": ["user_input", "reference_role"],
        # 多轮：判断 AI 回复是否守住角色边界，必须看到 AI 的回复本身。
        "judge_fields": ["user_input", "reference_role", "response"],
        "criteria": (
            "判断 AI 是否始终遵守 reference_role 描述的角色、职责边界和语气要求。越权承诺、角色漂移、"
            "使用不合适语气或执行角色不允许的动作需要扣分。"
        ),
    },
    "builtin_turn_faithfulness": {
        "required_fields": ["user_input", "retrieved_contexts"],
        # 多轮忠实度：判断每轮回复是否有资料支持，必须同时看到回复和上下文。
        # 与检索侧的 context_* 不同——这个指标本来就是生成侧的，看 response 是本职。
        "judge_fields": ["user_input", "retrieved_contexts", "response"],
        "criteria": (
            "判断多轮对话中每轮 AI 回复是否基于对应的检索上下文或已给定资料。跨轮捏造事实、与上下文冲突或"
            "无法从资料支持的回答需要扣分。"
        ),
    },
}


NAMED_LLM_METRIC_SPECS: dict[str, dict[str, t.Any]] = {
    "answer_completeness": {
        "required_fields": ["response", "reference"],
        # 生成侧：以参考答案为基准看要点覆盖度，不需要检索上下文。
        "judge_fields": ["response", "reference"],
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


PROMPT_VARIABLE_RE = re.compile(
    r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\}(?!\})"
)


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


# 结构化中间步骤（断言拆解/核验、问题反推）共用的中性系统提示词
STRUCTURED_JSON_SYSTEM_PROMPT = "你是一个严谨的文本分析助手。必须只输出严格 JSON，不要输出任何其他内容。"


# ---- 断言级忠实度（claim-level faithfulness）----
CLAIM_DECOMPOSITION_PROMPT = """请把下面的 AI 回答拆解为若干条"原子断言"，每条断言必须是能独立验证真伪的一句话陈述。

要求：
1. 断言之间不能互相包含或重复；
2. 只去掉纯过渡语和语气词；评价性、意见性陈述也要保留——它们同样可以被上下文支持或反驳（上下文未提及即视为不支持）；
3. 如果回答确实不包含任何陈述（例如只有"你好""好的"），返回空列表。

## 回答
{response}

## 输出格式
只输出 JSON：
{{
  "claims": ["断言1", "断言2", ...]
}}"""

CLAIM_VERIFICATION_PROMPT = """请判断下面的"断言"是否被给定的"检索上下文"直接或合理地支持。

判定标准：
- 支持（supported）：上下文中有明确信息支撑该断言；
- 不支持（unsupported）：断言与上下文冲突、上下文没有提及、或需要外部知识才能成立。

## 检索上下文
{contexts}

## 断言
{claim}

## 输出格式
只输出 JSON：
{{
  "verdict": "supported" 或 "unsupported",
  "reason": "1 句中文理由"
}}"""

# ---- 生成式相关性（generative answer relevancy）----
GENERATIVE_QUESTION_PROMPT = """请从下面的 AI 回答出发，反推出 {n} 个"这个回答最可能在回答的问题"。

要求：
1. 问题要具体，贴近真实用户会问的措辞；
2. 如果回答内容空泛、敷衍、或明显答非所问，请全部输出"回避式问题"（如"这个问题我不清楚"）。
3. 输出数量严格等于 {n}。

## 回答
{response}

## 输出格式
只输出 JSON：
{{
  "questions": ["问题1", "问题2", ...]
}}"""

NONCOMMITTAL_QUESTION_MARKERS = ("不清楚", "不知道", "无法回答", "不太明白", "请提供更多信息", "我不确定")


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


def render_prompt(prompt: str, context: dict[str, t.Any]) -> str:
    """Render business scoring prompts using ``{field}`` or ``{group.field}`` variables.

    This is used for Judge prompts stored in metric/scenario configuration.
    Missing variables are rendered as empty strings so one partially populated
    row cannot break a full batch evaluation. Unrelated JSON/example braces in
    the prompt are left intact.
    """

    if not prompt:
        return ""

    def replace_var(match: re.Match[str]) -> str:
        found, value = resolve_prompt_variable(context, match.group(1))
        if not found or value is None:
            return ""
        return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)

    return PROMPT_VARIABLE_RE.sub(replace_var, prompt)


def extract_prompt_variables(prompt: str) -> set[str]:
    """Return variable names referenced by a business Judge prompt.

    The evaluator uses this to avoid sending the same field twice: if a custom
    prompt already embeds ``{response}``, the structured ``sample.response`` can
    be omitted from that Judge request while other non-referenced fields remain
    available.
    """

    if not prompt:
        return set()
    return set(PROMPT_VARIABLE_RE.findall(prompt))


def resolve_prompt_variable(context: dict[str, t.Any], variable: str) -> tuple[bool, t.Any]:
    """Resolve a prompt variable from a nested context."""

    if variable in context:
        return True, context.get(variable)
    current: t.Any = context
    for part in variable.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False, None
    return True, current


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
