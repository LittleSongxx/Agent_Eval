"""Small, deterministic Bad Case taxonomy shared by reports and regression sets."""

from __future__ import annotations

from typing import Any

BADCASE_CATEGORIES = (
    "retrieval", "generation", "tool_selection", "tool_arguments",
    "trace_faithfulness", "error_recovery", "trace_lint", "endpoint_error", "unknown",
)

CATEGORY_LABELS = {
    "retrieval": "检索问题", "generation": "生成/事实问题",
    "tool_selection": "工具选择问题", "tool_arguments": "工具参数问题",
    "trace_faithfulness": "轨迹忠实度问题", "error_recovery": "错误恢复问题",
    "trace_lint": "轨迹规则问题", "endpoint_error": "接口异常", "unknown": "其他",
}


def classify_badcase(metric_scores: dict[str, Any] | None, error: str | None = None, endpoint_trace: dict[str, Any] | None = None) -> dict[str, Any]:
    """Infer one primary category from the most actionable failed signal."""
    if error or (endpoint_trace or {}).get("status") == "error":
        return {"category": "endpoint_error", "confidence": 1.0, "source": "runtime"}
    scores = metric_scores or {}
    priority = (
        ("trace_lint", "trace_lint"), ("error_recovery", "error_recovery"),
        ("trajectory_faithfulness", "trace_faithfulness"),
        ("tool_selection_rationality", "tool_selection"),
        ("argument_correctness", "tool_arguments"), ("tool_call_accuracy", "tool_selection"),
        ("context_recall", "retrieval"), ("context_precision", "retrieval"),
        ("contextual_relevancy", "retrieval"), ("faithfulness", "generation"),
        ("factual_correctness", "generation"), ("answer_relevancy", "generation"),
        ("answer_completeness", "generation"),
    )
    for metric, category in priority:
        value = scores.get(metric)
        score = value.get("score") if isinstance(value, dict) else value
        if score is None:
            continue
        try:
            if float(score) < 0.7:
                return {"category": category, "confidence": 0.85, "source": metric}
        except (TypeError, ValueError):
            if str(score).lower() in {"fail", "false", "no", "0"}:
                return {"category": category, "confidence": 0.75, "source": metric}
    return {"category": "unknown", "confidence": 0.0, "source": "none"}
