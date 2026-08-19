"""Deterministic Agent trace checks used before/alongside LLM judging."""

from __future__ import annotations

import json
from typing import Any


def lint_agent_trace(
    trajectory: list[dict[str, Any]] | None,
    available_tools: dict[str, Any] | None = None,
    tool_registry: list[dict[str, Any]] | None = None,
    max_steps: int = 20,
) -> dict[str, Any]:
    """Return explainable violations without making an LLM call."""

    trace = list(trajectory or [])
    registry = {str(item.get("name")): item for item in (tool_registry or []) if item.get("name")}
    declared_available = set((available_tools or {}).keys())
    registry_available = {name for name, item in registry.items() if item.get("enabled", True)}
    # 有运行时 available_tools 时，以它为准；只有没有运行时声明时才回退到目录。
    # 否则“已登记但本次未暴露给 Agent”的工具会被错误放行。
    available = declared_available or registry_available
    violations: list[dict[str, Any]] = []
    seen_calls: set[str] = set()
    failed_signatures: set[str] = set()
    finished = False

    def add(rule: str, step: int | None, message: str, severity: str = "error") -> None:
        violations.append({"rule": rule, "step": step, "message": message, "severity": severity})

    if not trace:
        add("empty_trace", None, "agent_trajectory 为空")
    if len(trace) > max_steps:
        add("max_steps_exceeded", max_steps, f"轨迹包含 {len(trace)} 步，超过上限 {max_steps}")

    for index, step in enumerate(trace):
        step_number = int(step.get("step", index + 1) or index + 1)
        tool = str(step.get("tool") or "").strip()
        action = str(step.get("action") or "").lower()
        if finished and tool:
            add("call_after_finish", step_number, "finish 后仍继续调用工具")
        if not tool:
            if action in {"finish", "final", "done"}:
                finished = True
            continue

        args = step.get("tool_input")
        if not isinstance(args, dict):
            add("invalid_arguments", step_number, "工具参数必须是 JSON 对象")
            args = {}
        if available and tool not in available:
            add("unregistered_tool", step_number, f"工具 {tool} 不在 available_tools / registry 中")
        definition = registry.get(tool)
        if definition and definition.get("parameters_schema"):
            schema = definition.get("parameters_schema") or {}
            required = schema.get("required") or []
            missing = [name for name in required if name not in args]
            if missing:
                add("missing_required_arguments", step_number, f"工具 {tool} 缺少必填参数: {', '.join(missing)}")
        signature = json.dumps({"tool": tool, "args": args}, ensure_ascii=False, sort_keys=True)
        if signature in seen_calls:
            add("duplicate_call", step_number, f"重复调用工具 {tool}", "warning")
        seen_calls.add(signature)
        output = str(step.get("tool_output") or "").lower()
        if step.get("is_error") or any(word in output for word in ("error", "failed", "exception", "失败", "错误")):
            if signature in failed_signatures:
                add("repeated_failed_call", step_number, f"相同参数重复失败: {tool}")
            failed_signatures.add(signature)
        if definition and definition.get("has_side_effect") and definition.get("risk_level") in {"high", "critical"}:
            if not step.get("confirmed") and not step.get("user_confirmed"):
                add("high_risk_without_confirmation", step_number, f"高风险副作用工具 {tool} 缺少确认记录")

    penalty = sum(1.0 if item["severity"] == "error" else 0.5 for item in violations)
    score = max(0.0, round(1.0 - penalty / max(len(trace), 1), 4))
    return {
        "passed": not violations,
        "score": score,
        "violations": violations,
        "checked_steps": len(trace),
    }
