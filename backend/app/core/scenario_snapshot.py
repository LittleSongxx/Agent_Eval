from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

from app.core.config import settings
from app.core.prompt_manager import (
    BUILTIN_LLM_METRIC_SPECS,
    BUILTIN_LLM_SCORE_INSTRUCTION,
    CLAIM_DECOMPOSITION_PROMPT,
    CLAIM_VERIFICATION_PROMPT,
    GENERATIVE_QUESTION_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT_COT,
    NAMED_LLM_METRIC_SPECS,
    STRUCTURED_JSON_SYSTEM_PROMPT,
    TOOL_SELECTION_RATIONALITY_PROMPT,
    TRAJECTORY_FAITHFULNESS_PROMPT,
)

# 复合指标不读 criteria：它们的判定逻辑写在这些模块级 prompt 常量里，
# 所以 criteria 落库还不足以锁住口径——这些常量同样得进指纹。
_COMPOSITE_METRIC_PROMPTS: dict[str, tuple[str, ...]] = {
    "builtin_faithfulness_claim": (
        CLAIM_DECOMPOSITION_PROMPT,
        CLAIM_VERIFICATION_PROMPT,
        STRUCTURED_JSON_SYSTEM_PROMPT,
    ),
    "builtin_answer_relevancy_generative": (
        GENERATIVE_QUESTION_PROMPT,
        STRUCTURED_JSON_SYSTEM_PROMPT,
    ),
    "builtin_trajectory_faithfulness": (
        TRAJECTORY_FAITHFULNESS_PROMPT,
        STRUCTURED_JSON_SYSTEM_PROMPT,
    ),
    "builtin_tool_selection_rationality": (
        TOOL_SELECTION_RATIONALITY_PROMPT,
        STRUCTURED_JSON_SYSTEM_PROMPT,
    ),
}


def _digest(*parts: str | None) -> str:
    joined = "␞".join(part or "" for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def engine_prompt_digest(metric_type: str | None) -> str:
    """Digest the engine-owned prompts a metric type actually executes with.

    ``effective_criteria`` covers the generic judge path, but two things live
    outside it: the composite metrics (claim decomposition/verification, reverse
    question generation) ignore ``criteria`` entirely, and every LLM metric is
    wrapped by the judge system prompt and scoring instruction. Editing any of
    those changes results with no visible config change, which is the same class
    of silent drift as the criteria constants.
    """
    judge_system = JUDGE_SYSTEM_PROMPT_COT if settings.JUDGE_COT_MODE else JUDGE_SYSTEM_PROMPT
    return _digest(
        judge_system,
        BUILTIN_LLM_SCORE_INSTRUCTION,
        *_COMPOSITE_METRIC_PROMPTS.get(metric_type or "", ()),
    )


def resolve_effective_criteria(
    metric_name: str | None,
    metric_type: str | None,
    prompt_override: str | None,
) -> str | None:
    """Resolve the judge criteria a metric will actually score with.

    The precedence mirrors ``NativeBuiltinLLMMetric.__init__``: an explicit
    ``prompt_override`` wins, otherwise the built-in spec constant applies.
    Materializing the result at task-creation time is what makes an old task
    reproducible — the spec constants are module-level Python and change with
    the code, so a task that only stored ``prompt_override=None`` would silently
    be re-read against whatever the current revision happens to say.

    Returns ``None`` for metrics with no criteria concept (deterministic
    retrieval/agent metrics, and user prompt metrics that carry their prompt in
    ``config`` instead).
    """
    override = (prompt_override or "").strip()
    if override:
        return override
    spec = NAMED_LLM_METRIC_SPECS.get(metric_name or "") or BUILTIN_LLM_METRIC_SPECS.get(
        metric_type or ""
    )
    if not spec:
        return None
    criteria = spec.get("criteria")
    return criteria if isinstance(criteria, str) else None


def _metric_override_map(metric_overrides: list[Any] | None) -> dict[int, dict[str, Any]]:
    overrides: dict[int, dict[str, Any]] = {}
    for item in metric_overrides or []:
        if hasattr(item, "model_dump"):
            data = item.model_dump(exclude_unset=True)
        elif isinstance(item, dict):
            data = dict(item)
        else:
            data = {
                key: getattr(item, key)
                for key in ("metric_definition_id", "weight", "pass_threshold", "prompt_override")
                if hasattr(item, key)
            }
        metric_definition_id = data.get("metric_definition_id")
        if metric_definition_id is None:
            continue
        overrides[int(metric_definition_id)] = data
    return overrides


def build_scenario_snapshot(scenario, metric_overrides: list[Any] | None = None) -> dict[str, Any]:
    """Freeze the scenario config that a task should evaluate with."""

    overrides = _metric_override_map(metric_overrides)
    metrics = []
    for scenario_metric in sorted(scenario.metrics or [], key=lambda item: item.id or 0):
        metric_definition = scenario_metric.metric_definition
        if metric_definition is None:
            continue
        override = overrides.get(int(scenario_metric.metric_definition_id), {})
        prompt_override = (
            override["prompt_override"]
            if "prompt_override" in override
            else getattr(scenario_metric, "prompt_override", None)
        )
        metrics.append(
            {
                "id": scenario_metric.id,
                "metric_definition_id": scenario_metric.metric_definition_id,
                "weight": override.get("weight", scenario_metric.weight),
                "pass_threshold": (
                    override["pass_threshold"]
                    if "pass_threshold" in override
                    else scenario_metric.pass_threshold
                ),
                "prompt_override": prompt_override,
                # 冻结这次评测真正使用的判定标准。prompt_override 为空时它等于
                # 当前代码里的 spec 常量——不落库的话，改一次常量就会让历史任务
                # 的口径静默漂移，而任务本身完全看不出变化。
                "effective_criteria": resolve_effective_criteria(
                    metric_definition.name,
                    metric_definition.metric_type,
                    prompt_override,
                ),
                # criteria 之外还有引擎自带的 prompt（复合指标的拆解/核验、
                # 裁判系统提示词）。它们改了同样会动分数，但配置上看不出来，
                # 所以一起冻结成摘要用于漂移检测。
                "engine_prompt_digest": engine_prompt_digest(metric_definition.metric_type),
                "metric_definition": {
                    "id": metric_definition.id,
                    "name": metric_definition.name,
                    "display_name": metric_definition.display_name,
                    "metric_type": metric_definition.metric_type,
                    "config": metric_definition.config or {},
                    "category": metric_definition.category,
                    "is_builtin": metric_definition.is_builtin,
                },
            }
        )

    return {
        "id": scenario.id,
        "name": scenario.name,
        "description": scenario.description,
        "scene_type": scenario.scene_type,
        "sample_type": scenario.sample_type,
        "is_preset": scenario.is_preset,
        "metrics": metrics,
    }


def build_judge_snapshot(llm_config, judge_panel: list[int] | None = None) -> dict[str, Any]:
    """Freeze the judge identity a task scores with — never the API key.

    ``LLMConfig`` rows are edited in place with no versioning, so a task that
    only kept ``llm_config_id`` cannot tell you afterwards which model actually
    graded it. Only non-secret fields are captured; ``api_key`` is deliberately
    excluded so snapshots stay safe to return over the API and to commit in
    report fixtures.
    """
    if llm_config is None:
        return {}
    return {
        "llm_config_id": getattr(llm_config, "id", None),
        "name": getattr(llm_config, "name", None),
        "provider": getattr(llm_config, "provider", None),
        "api_base_url": getattr(llm_config, "api_base_url", None),
        "model_name": getattr(llm_config, "model_name", None),
        "temperature": getattr(llm_config, "temperature", None),
        "max_tokens": getattr(llm_config, "max_tokens", None),
        "judge_panel": list(judge_panel or []),
    }


def compute_eval_fingerprint(
    scenario_snapshot: dict[str, Any] | None,
    judge_snapshot: dict[str, Any] | None,
    dataset_id: int | None,
    dataset_version: int | None,
    judge_samples: int | None = None,
    tool_registry_snapshot: list[dict[str, Any]] | None = None,
) -> str:
    """Hash every dimension that can move a score, so two tasks are comparable iff equal.

    A rubric is *data + ruler + judge*. Freezing the dataset version alone was
    only the first of the three: the ruler (criteria, thresholds, weights) and
    the judge (model, temperature) move results just as much. The digest lets
    the compare endpoint state which dimensions changed instead of implying the
    whole delta came from the system under test.
    """
    metrics = []
    for item in (scenario_snapshot or {}).get("metrics") or []:
        metric_definition = item.get("metric_definition") or {}
        metrics.append(
            {
                "name": metric_definition.get("name"),
                "metric_type": metric_definition.get("metric_type"),
                "weight": item.get("weight"),
                "pass_threshold": item.get("pass_threshold"),
                "effective_criteria": item.get("effective_criteria"),
                "engine_prompt_digest": item.get("engine_prompt_digest"),
                "config": metric_definition.get("config") or {},
            }
        )
    metrics.sort(key=lambda entry: (entry["name"] or "", entry["metric_type"] or ""))

    judge = dict(judge_snapshot or {})
    judge.pop("api_key", None)
    judge.pop("name", None)  # display-only; renaming a config must not change the digest

    material = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "metrics": metrics,
        "judge": {
            "model_name": judge.get("model_name"),
            "api_base_url": judge.get("api_base_url"),
            "temperature": judge.get("temperature"),
            "max_tokens": judge.get("max_tokens"),
            "judge_panel": sorted(judge.get("judge_panel") or []),
        },
        "judge_samples": judge_samples,
        "tool_registry": tool_registry_snapshot or [],
    }
    payload = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def describe_fingerprint_diff(
    current: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
) -> list[str]:
    """List which comparability dimensions differ between two tasks.

    Cross-rubric comparison is legitimate (you often *want* to know what a
    stricter ruler does), so this reports rather than blocks.
    """
    current = current or {}
    baseline = baseline or {}
    changed: list[str] = []

    if current.get("dataset_version") != baseline.get("dataset_version"):
        changed.append("dataset_version")

    cur_judge = (current.get("judge_snapshot") or {})
    base_judge = (baseline.get("judge_snapshot") or {})
    for field in ("model_name", "api_base_url", "temperature", "max_tokens"):
        if cur_judge.get(field) != base_judge.get(field):
            changed.append(f"judge.{field}")
    if sorted(cur_judge.get("judge_panel") or []) != sorted(base_judge.get("judge_panel") or []):
        changed.append("judge.judge_panel")

    if current.get("tool_registry_snapshot") != baseline.get("tool_registry_snapshot"):
        changed.append("tool_registry")

    def _criteria_map(snapshot: dict[str, Any] | None) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in (snapshot or {}).get("metrics") or []:
            name = (item.get("metric_definition") or {}).get("name")
            if name:
                result[name] = item
        return result

    cur_metrics = _criteria_map(current.get("scenario_snapshot"))
    base_metrics = _criteria_map(baseline.get("scenario_snapshot"))
    for name in sorted(set(cur_metrics) & set(base_metrics)):
        cur_item, base_item = cur_metrics[name], base_metrics[name]
        if cur_item.get("effective_criteria") != base_item.get("effective_criteria"):
            changed.append(f"criteria.{name}")
        if cur_item.get("engine_prompt_digest") != base_item.get("engine_prompt_digest"):
            changed.append(f"engine_prompt.{name}")
        if cur_item.get("pass_threshold") != base_item.get("pass_threshold"):
            changed.append(f"threshold.{name}")
        if cur_item.get("weight") != base_item.get("weight"):
            changed.append(f"weight.{name}")

    return changed


def snapshot_to_scenario_metrics(snapshot: dict[str, Any] | None) -> list[Any]:
    """Hydrate frozen metric config into lightweight objects used by the engine."""

    if not snapshot:
        return []

    hydrated = []
    for item in snapshot.get("metrics") or []:
        metric_definition = item.get("metric_definition") or {}
        hydrated.append(
            SimpleNamespace(
                id=item.get("id"),
                scenario_id=snapshot.get("id"),
                metric_definition_id=item.get("metric_definition_id"),
                weight=item.get("weight", 1.0),
                pass_threshold=item.get("pass_threshold"),
                prompt_override=item.get("prompt_override"),
                effective_criteria=item.get("effective_criteria"),
                metric_definition=SimpleNamespace(
                    id=metric_definition.get("id"),
                    name=metric_definition.get("name"),
                    display_name=metric_definition.get("display_name")
                    or metric_definition.get("name")
                    or "Unknown Metric",
                    metric_type=metric_definition.get("metric_type"),
                    config=metric_definition.get("config") or {},
                    category=metric_definition.get("category"),
                    is_builtin=metric_definition.get("is_builtin", False),
                ),
            )
        )
    return hydrated
