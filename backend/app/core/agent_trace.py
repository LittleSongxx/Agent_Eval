"""Normalize Agent execution traces into the platform's dataset contract.

The platform evaluates rows rather than framework-specific callback objects.  This
module is deliberately dependency-free so trace conversion can be unit tested
without a database or an LLM provider.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


class AgentTraceValidationError(ValueError):
    """Raised when an imported trace cannot be made unambiguous."""


_EVENT_TYPE_ALIASES = {
    "human": "human",
    "user": "human",
    "user_message": "human",
    "assistant": "assistant",
    "ai": "assistant",
    "assistant_message": "assistant",
    "tool": "tool",
    "tool_result": "tool",
}


def normalize_agent_trace(trace: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Convert one documented or legacy trace into a normalized dataset row.

    Supported input forms include the canonical ``events`` contract and the
    repository's existing ``agent_trajectory_samples.json`` fixture.  A stable
    trace id is derived when one is not supplied, which makes repeated imports
    idempotent.
    """

    if not isinstance(trace, dict):
        raise AgentTraceValidationError(f"第 {index + 1} 条轨迹必须是 JSON 对象")

    raw_events = trace.get("events")
    if raw_events is None and isinstance(trace.get("user_input"), list):
        raw_events = trace.get("user_input")
    if raw_events is not None and not isinstance(raw_events, list):
        raise AgentTraceValidationError(f"第 {index + 1} 条轨迹的 events 必须是数组")

    raw_trajectory = trace.get("agent_trajectory")
    if raw_trajectory is not None and not isinstance(raw_trajectory, list):
        raise AgentTraceValidationError(f"第 {index + 1} 条轨迹的 agent_trajectory 必须是数组")

    events = _normalize_events(raw_events, index) if raw_events is not None else []
    supplied_trajectory = _normalize_trajectory(raw_trajectory, index) if raw_trajectory is not None else []
    trajectory = supplied_trajectory

    if events:
        derived_trajectory = _trajectory_from_events(events)
        if derived_trajectory:
            if supplied_trajectory and _trajectory_signature(supplied_trajectory) != _trajectory_signature(derived_trajectory):
                raise AgentTraceValidationError(
                    f"第 {index + 1} 条轨迹同时提供的 events 与 agent_trajectory 内容冲突"
                )
            trajectory = derived_trajectory
    if not trajectory:
        raise AgentTraceValidationError(
            f"第 {index + 1} 条轨迹必须提供非空 events 或 agent_trajectory"
        )

    available_tools = _normalize_available_tools(trace.get("available_tools"), index)
    reference_tool_calls = _normalize_tool_calls(trace.get("reference_tool_calls"), index, "reference_tool_calls")

    task = _first_non_empty(trace, "input", "task", "user_input", "question")
    if isinstance(task, list):
        # A caller may already provide a conversation as user_input.  Keep it as
        # the source conversation only when no event contract is present.
        conversation = _normalize_conversation(task, index)
        task_text = _first_human_content(conversation) or json.dumps(task, ensure_ascii=False)
    else:
        task_text = str(task or "").strip()
        conversation = _conversation_from_events(events) if events else []
        if not conversation and task_text:
            conversation.append({"type": "human", "content": task_text})

    if not task_text and conversation:
        task_text = _first_human_content(conversation) or ""
    if not task_text:
        raise AgentTraceValidationError(f"第 {index + 1} 条轨迹缺少 input/task/user_input")

    response = _first_non_empty(trace, "output", "response", "final_answer")
    if isinstance(response, (dict, list)):
        response = json.dumps(response, ensure_ascii=False)
    response = str(response or "").strip()
    if not response:
        response = _final_assistant_content(events, trajectory)

    if not conversation:
        conversation = _conversation_from_trajectory(task_text, trajectory, response)
    elif not any(item.get("type") == "human" for item in conversation):
        conversation.insert(0, {"type": "human", "content": task_text})
    if response and not any(
        item.get("type") == "ai" and item.get("content") == response for item in conversation
    ):
        conversation.append({"type": "ai", "content": response})

    actual_tool_calls = _actual_tool_calls_from_trajectory(trajectory)
    trace_id = str(trace.get("trace_id") or "").strip()
    normalized: dict[str, Any] = {
        "trace_id": trace_id,
        "user_input": conversation,
        "response": response,
        "tool_calls": actual_tool_calls,
        "agent_trajectory": trajectory,
        "available_tools": available_tools,
    }
    if trace.get("reference") is not None:
        normalized["reference"] = trace.get("reference")
    if reference_tool_calls:
        normalized["reference_tool_calls"] = reference_tool_calls
    for key in ("reference_topics", "reference_role", "retrieved_contexts", "metadata"):
        if trace.get(key) is not None:
            normalized[key] = trace.get(key)

    if not normalized["trace_id"]:
        material = dict(normalized)
        material.pop("trace_id", None)
        digest = hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()
        normalized["trace_id"] = f"trace-{digest[:16]}"
    return normalized


def canonical_agent_trace(row: dict[str, Any]) -> str:
    """Return a deduplication key independent of a caller-provided trace id."""

    material = dict(row or {})
    material.pop("trace_id", None)
    return _canonical_json(material)


def normalize_available_tools(value: Any) -> dict[str, str]:
    """Normalize dict, simple-list, and OpenAI function-tool schemas."""

    return _normalize_available_tools(value, 0)


def normalize_agent_trajectory(value: Any) -> list[dict[str, Any]]:
    """Normalize a response-mapped trajectory or event list in endpoint mode."""

    if not isinstance(value, list) or not value:
        raise AgentTraceValidationError("agent_trajectory 必须是非空数组")
    if all(isinstance(item, dict) and ("step" in item or "tool_input" in item) for item in value):
        return _normalize_trajectory(value, 0)
    events = _normalize_events(value, 0)
    trajectory = _trajectory_from_events(events)
    if not trajectory:
        raise AgentTraceValidationError("agent_trajectory 中没有可评测的事件")
    return trajectory


def _normalize_events(events: list[Any], index: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            raise AgentTraceValidationError(
                f"第 {index + 1} 条轨迹的第 {event_index + 1} 个 event 必须是对象"
            )
        raw_type = str(event.get("type") or event.get("role") or "").strip().lower()
        event_type = _EVENT_TYPE_ALIASES.get(raw_type)
        if event_type is None:
            raise AgentTraceValidationError(
                f"第 {index + 1} 条轨迹的 event 类型 {raw_type or '<空>'} 不支持"
            )
        content = event.get("content", event.get("text", ""))
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        item: dict[str, Any] = {"type": event_type, "content": content}
        if event_type == "assistant":
            calls = event.get("tool_calls")
            if calls is not None:
                item["tool_calls"] = _normalize_tool_calls(calls, index, "events.tool_calls")
        elif event_type == "tool":
            item["tool_call_id"] = event.get("tool_call_id") or event.get("call_id")
            item["name"] = event.get("name") or event.get("tool")
            item["is_error"] = bool(event.get("is_error", False))
        normalized.append(item)
    return normalized


def _normalize_trajectory(trajectory: list[Any], index: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for step_index, step in enumerate(trajectory):
        if not isinstance(step, dict):
            raise AgentTraceValidationError(
                f"第 {index + 1} 条轨迹的第 {step_index + 1} 个轨迹步骤必须是对象"
            )
        tool = step.get("tool") or step.get("name")
        if tool is not None:
            tool = str(tool).strip() or None
        tool_input = step.get("tool_input", step.get("arguments", step.get("args")))
        if tool_input is not None and not isinstance(tool_input, dict):
            raise AgentTraceValidationError(
                f"第 {index + 1} 条轨迹的第 {step_index + 1} 个步骤 tool_input 必须是对象"
            )
        tool_output = step.get("tool_output", step.get("output"))
        if tool_output is not None and not isinstance(tool_output, str):
            tool_output = json.dumps(tool_output, ensure_ascii=False)
        normalized.append(
            {
                "step": int(step.get("step", step_index + 1)),
                "thought": str(step.get("thought", step.get("content", "")) or ""),
                "tool": tool,
                "tool_input": tool_input,
                "tool_output": tool_output,
                "is_error": bool(step.get("is_error", False)),
                "action": str(step.get("action") or ("continue" if tool else "finish")),
            }
        )
    return normalized


def _trajectory_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trajectory: list[dict[str, Any]] = []
    used_tool_events: set[int] = set()
    step = 1
    for event_index, event in enumerate(events):
        if event["type"] != "assistant":
            continue
        calls = event.get("tool_calls") or []
        if not calls:
            if event.get("content"):
                trajectory.append(
                    {
                        "step": step,
                        "thought": event["content"],
                        "tool": None,
                        "tool_input": None,
                        "tool_output": None,
                        "action": "finish",
                    }
                )
                step += 1
            continue
        if len(calls) > 1 and any(not call.get("id") for call in calls):
            raise AgentTraceValidationError(
                "同一 assistant event 包含并行工具调用时，每个 tool_call 必须带唯一 id"
            )
        for call in calls:
            output = None
            is_error = False
            candidates: list[int] = []
            available_tool_indices: list[int] = []
            for later_index, later in enumerate(events[event_index + 1 :], start=event_index + 1):
                if later["type"] == "assistant":
                    # A tool result after the next assistant turn belongs to a
                    # later call; considering it here creates false matches.
                    break
                if later["type"] != "tool":
                    continue
                if later_index in used_tool_events:
                    continue
                available_tool_indices.append(later_index)
                same_id = call.get("id") and later.get("tool_call_id") == call.get("id")
                same_name = call.get("name") and later.get("name") == call.get("name")
                if same_id:
                    candidates = [later_index]
                    break
                if not call.get("id") and same_name:
                    candidates.append(later_index)
            if not candidates and not call.get("id"):
                remaining = available_tool_indices
                if len(remaining) == 1:
                    candidates = remaining
                elif len(remaining) > 1:
                    raise AgentTraceValidationError(
                        f"工具 {call.get('name') or '<unknown>'} 缺少 call id，存在多个可能的返回结果"
                    )
            if call.get("id") and not candidates:
                raise AgentTraceValidationError(
                    f"工具 {call.get('name') or '<unknown>'} 的 call id {call.get('id')} 找不到对应返回结果"
                )
            if len(candidates) > 1:
                raise AgentTraceValidationError(
                    f"工具 {call.get('name') or '<unknown>'} 缺少 call id，无法唯一匹配返回结果"
                )
            if candidates:
                matched_index = candidates[0]
                output = events[matched_index].get("content")
                is_error = bool(events[matched_index].get("is_error", False))
                used_tool_events.add(matched_index)
            trajectory.append(
                {
                    "step": step,
                    "thought": event.get("content", ""),
                    "tool": call.get("name"),
                    "tool_input": call.get("args", {}),
                    "tool_output": output,
                    "is_error": is_error,
                    "action": "continue",
                }
            )
            step += 1
    return trajectory


def _conversation_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conversation: list[dict[str, Any]] = []
    for event in events:
        item_type = {"human": "human", "assistant": "ai", "tool": "tool"}[event["type"]]
        item = {"type": item_type, "content": event.get("content", "")}
        if event.get("tool_calls"):
            item["tool_calls"] = [
                {"name": call.get("name"), "args": call.get("args", {})}
                for call in event["tool_calls"]
            ]
        if event.get("name"):
            item["name"] = event["name"]
        conversation.append(item)
    return conversation


def _conversation_from_trajectory(task: str, trajectory: list[dict[str, Any]], response: str) -> list[dict[str, Any]]:
    conversation: list[dict[str, Any]] = [{"type": "human", "content": task}]
    for step in trajectory:
        item: dict[str, Any] = {"type": "ai", "content": step.get("thought", "")}
        if step.get("tool"):
            item["tool_calls"] = [{"name": step["tool"], "args": step.get("tool_input") or {}}]
            conversation.append(item)
            conversation.append(
                {
                    "type": "tool",
                    "name": step["tool"],
                    "content": step.get("tool_output") or "",
                }
            )
        elif step.get("thought"):
            conversation.append(item)
    if response and not any(item.get("type") == "ai" and item.get("content") == response for item in conversation):
        conversation.append({"type": "ai", "content": response})
    return conversation


def _normalize_tool_calls(value: Any, index: int, field: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise AgentTraceValidationError(f"第 {index + 1} 条轨迹的 {field} 必须是数组")
    calls: list[dict[str, Any]] = []
    for call_index, call in enumerate(value):
        if not isinstance(call, dict):
            raise AgentTraceValidationError(f"第 {index + 1} 条轨迹的 {field}[{call_index}] 必须是对象")
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(call.get("name") or call.get("tool") or function.get("name") or "").strip()
        if not name:
            raise AgentTraceValidationError(f"第 {index + 1} 条轨迹的 {field}[{call_index}] 缺少 name")
        args = call.get(
            "args",
            call.get("arguments", call.get("tool_input", function.get("arguments", {}))),
        )
        if args is None:
            args = {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as exc:
                raise AgentTraceValidationError(
                    f"第 {index + 1} 条轨迹的 {field}[{call_index}] arguments 不是合法 JSON"
                ) from exc
        if not isinstance(args, dict):
            raise AgentTraceValidationError(f"第 {index + 1} 条轨迹的 {field}[{call_index}] 参数必须是对象")
        item = {"name": name, "args": args}
        call_id = call.get("id") or call.get("tool_call_id")
        if call_id:
            item["id"] = call_id
        calls.append(item)
    return calls


def _normalize_available_tools(value: Any, index: int) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        result: dict[str, str] = {}
        for name, description in value.items():
            clean_name = str(name).strip()
            if not clean_name:
                raise AgentTraceValidationError(f"第 {index + 1} 条轨迹的 available_tools 存在空工具名")
            if isinstance(description, dict):
                description = description.get("description", "")
            result[clean_name] = str(description or "")
        return result
    if isinstance(value, list):
        result = {}
        for tool_index, tool in enumerate(value):
            if not isinstance(tool, dict):
                raise AgentTraceValidationError(f"第 {index + 1} 条轨迹的 available_tools[{tool_index}] 必须是对象")
            function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
            name = str(tool.get("name") or tool.get("tool") or function.get("name") or "").strip()
            if not name:
                raise AgentTraceValidationError(f"第 {index + 1} 条轨迹的 available_tools[{tool_index}] 缺少 name")
            result[name] = str(tool.get("description") or function.get("description") or "")
        return result
    raise AgentTraceValidationError(f"第 {index + 1} 条轨迹的 available_tools 必须是对象或数组")


def _actual_tool_calls_from_trajectory(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"name": step["tool"], "args": step.get("tool_input") or {}}
        for step in trajectory
        if step.get("tool")
    ]


def _normalize_conversation(value: list[Any], index: int) -> list[dict[str, Any]]:
    events = _normalize_events(value, index)
    return _conversation_from_events(events)


def _final_assistant_content(events: list[dict[str, Any]], trajectory: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "assistant" and event.get("content"):
            return str(event["content"])
    for step in reversed(trajectory):
        if not step.get("tool") and step.get("thought"):
            return str(step["thought"])
    return ""


def _first_human_content(conversation: list[dict[str, Any]]) -> str:
    for item in conversation:
        if item.get("type") == "human" and item.get("content"):
            return str(item["content"])
    return ""


def _first_non_empty(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, "", []):
            return value
    return None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _trajectory_signature(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare execution facts while ignoring formatting/default fields."""

    return [
        {
            "tool": item.get("tool"),
            "tool_input": item.get("tool_input"),
            "tool_output": item.get("tool_output"),
            "is_error": bool(item.get("is_error", False)),
        }
        for item in trajectory
    ]
