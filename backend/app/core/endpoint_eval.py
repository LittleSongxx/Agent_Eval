from __future__ import annotations

import json
import time
import typing as t

import httpx

from app.core.prompt_manager import (
    DEFAULT_ENDPOINT_BODY_TEMPLATE,
    DEFAULT_RESPONSE_PATH,
    render_template_value,
)


def parse_json_object(value: str | None, default: dict[str, t.Any] | None = None) -> dict[str, t.Any]:
    raw = (value or "").strip()
    if not raw:
        return dict(default or {})
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("JSON 配置必须是对象")
    return loaded


def get_path(payload: t.Any, path: str | None) -> t.Any:
    raw_path = (path or "").strip()
    if not raw_path:
        return None
    current = payload
    for part in raw_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current


def set_path(payload: dict[str, t.Any], path: str, value: t.Any) -> None:
    parts = [part for part in (path or "").split(".") if part]
    if not parts:
        return
    current: dict[str, t.Any] = payload
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def build_request_body(row_data: dict[str, t.Any], body_template: str | None) -> dict[str, t.Any]:
    template = parse_json_object(body_template, default=json.loads(DEFAULT_ENDPOINT_BODY_TEMPLATE))
    rendered = render_template_value(
        template,
        {
            "user_input": row_data.get("user_input"),
            "question": row_data.get("user_input"),
            "row_data": row_data,
        },
    )
    if not isinstance(rendered, dict):
        raise ValueError("请求体模板渲染后必须是 JSON 对象")
    return rendered


def build_headers(target_config: dict[str, t.Any]) -> dict[str, str]:
    transport_mode = str(target_config.get("transport_mode") or "json")
    headers = {
        "content-type": "application/json",
        "accept": "text/event-stream" if transport_mode == "sse" else "application/json",
        "connection": "close",
    }
    authorization = str(target_config.get("authorization") or "").strip()
    if authorization:
        headers["authorization"] = authorization
    for key, value in parse_json_object(str(target_config.get("extra_headers") or ""), default={}).items():
        if value is not None:
            headers[str(key)] = str(value)
    return headers


async def invoke_endpoint(row_data: dict[str, t.Any], target_config: dict[str, t.Any]) -> dict[str, t.Any]:
    url = str(target_config.get("endpoint_url") or "").strip()
    if not url:
        raise ValueError("被测接口地址为空")

    transport_mode = str(target_config.get("transport_mode") or "json").strip() or "json"
    timeout = float(target_config.get("timeout_seconds") or 180)
    request_body = build_request_body(row_data, str(target_config.get("request_body_template") or ""))
    headers = build_headers(target_config)
    started_at = time.time()
    limits = httpx.Limits(max_keepalive_connections=0, max_connections=10)

    async with httpx.AsyncClient(timeout=timeout, limits=limits, http2=False) as client:
        if transport_mode == "sse":
            response_payload = await _invoke_sse(client, url, headers, request_body)
        else:
            response_payload = await _invoke_json(client, url, headers, request_body)

    response_payload["latency_ms"] = int((time.time() - started_at) * 1000)
    response_payload["request_body"] = request_body
    return response_payload


def extract_eval_fields(
    response_payload: dict[str, t.Any],
    response_mapping: dict[str, t.Any] | None,
) -> tuple[dict[str, t.Any], dict[str, str]]:
    mapping = response_mapping or {}
    parsed = response_payload.get("parsed_response")
    raw_text = response_payload.get("raw_response")
    extracted: dict[str, t.Any] = {}
    errors: dict[str, str] = {}

    path_by_field = {
        "response": mapping.get("response_path") or DEFAULT_RESPONSE_PATH,
        "retrieved_contexts": mapping.get("retrieved_contexts_path"),
        "tool_calls": mapping.get("tool_calls_path"),
        "retrieved_context_ids": mapping.get("retrieved_context_ids_path"),
    }

    for field, path in path_by_field.items():
        if not path:
            continue
        value = get_path(parsed, str(path)) if parsed is not None else None
        if value is None and field == "response":
            value = response_payload.get("answer")
        if value is None:
            errors[field] = f"响应字段路径不存在: {path}"
            continue
        extracted[field] = value

    if "response" not in extracted and isinstance(raw_text, str) and raw_text.strip():
        extracted["response"] = raw_text.strip()
    return extracted, errors


async def _invoke_json(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    request_body: dict[str, t.Any],
) -> dict[str, t.Any]:
    response = await client.post(url, headers=headers, json=request_body)
    raw_text = response.text
    response.raise_for_status()
    parsed_response: t.Any = None
    content_type = (response.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        parsed_response = response.json()
    else:
        try:
            parsed_response = response.json()
        except Exception:
            parsed_response = None
    answer = _first_path(parsed_response, ["answer", "data.answer", "choices.0.message.content"])
    return {
        "status_code": response.status_code,
        "content_type": content_type,
        "raw_response": raw_text[:20000],
        "parsed_response": parsed_response if isinstance(parsed_response, (dict, list)) else None,
        "answer": answer,
    }


async def _invoke_sse(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    request_body: dict[str, t.Any],
) -> dict[str, t.Any]:
    answer_parts: list[str] = []
    final_answer: str | None = None
    raw_events: list[str] = []
    status_code = 0
    content_type = ""
    async with client.stream("POST", url, headers=headers, json=request_body) as response:
        status_code = response.status_code
        content_type = response.headers.get("content-type") or ""
        response.raise_for_status()
        async for raw_line in response.aiter_lines():
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            payload_text = line[5:].strip()
            if not payload_text or payload_text == "[DONE]":
                continue
            raw_events.append(payload_text)
            payload = _safe_json(payload_text)
            text = _first_path(payload, ["answer", "data.answer", "content", "text", "delta", "choices.0.delta.content"])
            if isinstance(text, str) and text:
                answer_parts.append(text)
            maybe_final = _first_path(payload, ["final_answer", "data.final_answer"])
            if isinstance(maybe_final, str) and maybe_final:
                final_answer = maybe_final
    answer = final_answer or "".join(answer_parts).strip()
    return {
        "status_code": status_code,
        "content_type": content_type,
        "raw_response": "\n".join(raw_events)[:20000],
        "parsed_response": {"answer": answer, "events": raw_events[:50]},
        "answer": answer,
    }


def _first_path(payload: t.Any, paths: list[str]) -> t.Any:
    for path in paths:
        value = get_path(payload, path)
        if value not in (None, ""):
            return value
    return None


def _safe_json(value: str) -> t.Any:
    try:
        return json.loads(value)
    except Exception:
        return {"text": value}

