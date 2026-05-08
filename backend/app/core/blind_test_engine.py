from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy.orm import Session, joinedload

from app.core.database import SessionLocal
from app.models.dataset import DatasetRow
from app.models.evaluation import BlindTestRowResult, BlindTestTask
from app.models.llm_config import LLMConfig

BLIND_TEST_CONCURRENCY = 3
DEFAULT_ENDPOINT_BODY_TEMPLATE = '{"question":"{{question}}"}'


def _append_log(task: BlindTestTask, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    current = task.logs or ""
    task.logs = f"{current}[{timestamp}] {message}\n"


def _normalize_messages(user_input: Any, row_data: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = str(row_data.get("system_prompt") or "").strip()
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if isinstance(user_input, list):
        for item in user_input:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or item.get("type") or "user").lower()
            if role == "human":
                role = "user"
            elif role == "bot":
                role = "assistant"
            if role not in {"system", "user", "assistant"}:
                role = "user"
            content = str(item.get("content") or item.get("text") or "").strip()
            if content:
                messages.append({"role": role, "content": content})
        if messages:
            return messages

    content = user_input if isinstance(user_input, str) else json.dumps(user_input, ensure_ascii=False)
    messages.append({"role": "user", "content": content})
    return messages


def _parse_json_text(value: str | None, default: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = (value or "").strip()
    if not raw:
        return dict(default or {})
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("JSON 配置必须是对象")
    return loaded


def _render_template_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        for key, raw_value in context.items():
            placeholder = f"{{{{{key}}}}}"
            if stripped == placeholder:
                return raw_value
        rendered = value
        for key, raw_value in context.items():
            placeholder = f"{{{{{key}}}}}"
            replacement = raw_value if isinstance(raw_value, str) else json.dumps(raw_value, ensure_ascii=False)
            rendered = rendered.replace(placeholder, replacement)
        return rendered
    if isinstance(value, list):
        return [_render_template_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _render_template_value(item, context) for key, item in value.items()}
    return value


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                loaded = json.loads(text[start : end + 1])
                return loaded if isinstance(loaded, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _extract_from_dict(payload: dict[str, Any], candidates: list[str]) -> Any:
    for path in candidates:
        current: Any = payload
        ok = True
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
                current = current[int(part)]
            else:
                ok = False
                break
        if ok and current not in (None, ""):
            return current
    return None


def _normalize_event_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        maybe = _extract_json_object(payload)
        if maybe:
            return maybe
        return {"text": payload}
    return {}


def _collect_stream_message(state: dict[str, Any], payload: dict[str, Any]) -> None:
    final_answer = _extract_from_dict(payload, ["answer", "data.answer", "choices.0.message.content"])
    if isinstance(final_answer, str) and final_answer.strip():
        state["final_answer"] = final_answer.strip()

    delta_answer = _extract_from_dict(payload, ["content", "text", "delta", "data.content", "data.text", "data.delta", "choices.0.delta.content"])
    if isinstance(delta_answer, str) and delta_answer.strip():
        state["answer_parts"].append(delta_answer)


class OpenAIResponseClient:
    def __init__(self, llm_config: LLMConfig):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            base_url=llm_config.api_base_url,
            api_key=llm_config.api_key,
            timeout=180,
            max_retries=2,
        )
        self.model = llm_config.model_name
        self.temperature = llm_config.temperature if llm_config.temperature is not None else 0.2
        self.max_tokens = min(int(llm_config.max_tokens or 2048), 2000)

    async def answer(self, row_data: dict[str, Any]) -> str:
        user_input = row_data.get("user_input")
        messages = _normalize_messages(user_input, row_data)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class EndpointResponseClient:
    def __init__(self, target: dict[str, Any]):
        self.url = str(target.get("endpoint_url") or "").strip()
        self.transport_mode = str(target.get("transport_mode") or "json").strip() or "json"
        self.authorization = str(target.get("authorization") or "").strip()
        self.extra_headers = str(target.get("extra_headers") or "")
        self.body_template = str(target.get("request_body_template") or DEFAULT_ENDPOINT_BODY_TEMPLATE)
        self.timeout = 180
        self.max_retries = 3

    def build_headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream" if self.transport_mode == "sse" else "application/json",
            "connection": "close",
        }
        if self.authorization:
            headers["authorization"] = self.authorization
        for key, value in _parse_json_text(self.extra_headers, default={}).items():
            if value is not None:
                headers[str(key)] = str(value)
        return headers

    def build_body(self, row_data: dict[str, Any]) -> dict[str, Any]:
        default = json.loads(DEFAULT_ENDPOINT_BODY_TEMPLATE)
        template = _parse_json_text(self.body_template, default=default)
        user_input = row_data.get("user_input")
        question = user_input if isinstance(user_input, str) else json.dumps(user_input, ensure_ascii=False)
        context = {
            "question": question,
            "user_input": user_input,
            "row_data": row_data,
        }
        rendered = _render_template_value(template, context)
        if not isinstance(rendered, dict):
            raise ValueError("目标接口请求体模板渲染后必须是 JSON 对象")
        return rendered

    async def answer(self, row_data: dict[str, Any]) -> str:
        if not self.url:
            raise ValueError("目标接口 URL 为空")
        headers = self.build_headers()
        body = self.build_body(row_data)
        limits = httpx.Limits(max_keepalive_connections=0, max_connections=10)
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout, limits=limits, http2=False) as client:
                    if self.transport_mode == "sse":
                        return await self._answer_sse(client, headers, body)
                    return await self._answer_json(client, headers, body)
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(min(0.8 * attempt, 2.5))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("接口调用失败")

    async def _answer_json(self, client: httpx.AsyncClient, headers: dict[str, str], body: dict[str, Any]) -> str:
        response = await client.post(self.url, headers=headers, json=body)
        response.raise_for_status()
        content_type = (response.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            payload = response.json()
            if isinstance(payload, dict):
                return str(_extract_from_dict(payload, ["answer", "data.answer", "choices.0.message.content"]) or "").strip()
        return response.text.strip()

    async def _answer_sse(self, client: httpx.AsyncClient, headers: dict[str, str], body: dict[str, Any]) -> str:
        state: dict[str, Any] = {"answer_parts": [], "final_answer": None}
        async with client.stream("POST", self.url, headers=headers, json=body) as response:
            response.raise_for_status()
            data_lines: list[str] = []
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line:
                    if data_lines:
                        payload_text = "\n".join(data_lines).strip()
                        data_lines = []
                        if payload_text != "[DONE]":
                            _collect_stream_message(state, _normalize_event_payload(payload_text))
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if data_lines:
                payload_text = "\n".join(data_lines).strip()
                if payload_text and payload_text != "[DONE]":
                    _collect_stream_message(state, _normalize_event_payload(payload_text))
        return str(state.get("final_answer") or "").strip() or "".join(state["answer_parts"]).strip()


def _build_target_client(session: Session, target: dict[str, Any]) -> OpenAIResponseClient | EndpointResponseClient:
    target_type = target.get("target_type")
    if target_type == "llm_config":
        llm_config_id = target.get("llm_config_id")
        if not llm_config_id:
            raise ValueError("模型对比对象缺少 llm_config_id")
        llm_config = session.query(LLMConfig).filter(LLMConfig.id == int(llm_config_id)).first()
        if llm_config is None:
            raise ValueError("盲测引用的 LLM 配置不存在")
        return OpenAIResponseClient(llm_config)
    if target_type == "endpoint":
        return EndpointResponseClient(target)
    raise ValueError(f"不支持的盲测目标类型: {target_type}")


def _make_display_order(task_id: int, row_id: int) -> list[str]:
    digest = hashlib.md5(f"{task_id}:{row_id}".encode("utf-8")).hexdigest()
    return ["a", "b"] if int(digest[:2], 16) % 2 == 0 else ["b", "a"]


def _summarize_votes(task: BlindTestTask) -> dict[str, Any]:
    total = len(task.row_results)
    completed = sum(1 for row in task.row_results if row.answer_a or row.answer_b or row.answer_a_error or row.answer_b_error)
    voted = sum(1 for row in task.row_results if row.vote and row.vote != "skip")
    both_bad = sum(1 for row in task.row_results if row.vote == "both_bad")
    ties = sum(1 for row in task.row_results if row.vote == "tie")
    model_a_wins = 0
    model_b_wins = 0
    no_answer_rows = 0
    for row in task.row_results:
        if not (row.answer_a or row.answer_b):
            no_answer_rows += 1
            continue
        if row.vote not in {"left", "right"}:
            continue
        left_side = row.display_order[0] if row.display_order else "a"
        winner = left_side if row.vote == "left" else ("b" if left_side == "a" else "a")
        if winner == "a":
            model_a_wins += 1
        else:
            model_b_wins += 1
    return {
        "total_count": total,
        "completed_count": completed,
        "voted_count": voted,
        "pending_vote_count": max(total - voted, 0),
        "model_a_wins": model_a_wins,
        "model_b_wins": model_b_wins,
        "ties": ties,
        "both_bad": both_bad,
        "no_answer_rows": no_answer_rows,
    }


async def run_blind_test(task_id: int, session_factory=SessionLocal) -> None:
    session: Session = session_factory()
    try:
        task = (
            session.query(BlindTestTask)
            .options(joinedload(BlindTestTask.row_results).joinedload(BlindTestRowResult.dataset_row))
            .filter(BlindTestTask.id == task_id)
            .first()
        )
        if task is None:
            return
        if task.status == "cancelled":
            return

        task.status = "running"
        task.started_at = datetime.now(timezone.utc)
        task.progress = 0.0
        task.completed_rows = 0
        task.logs = ""
        _append_log(task, f"开始生成盲测答案，共 {task.total_rows or 0} 条")
        session.commit()

        client_a = _build_target_client(session, task.target_a or {})
        client_b = _build_target_client(session, task.target_b or {})

        row_results = (
            session.query(BlindTestRowResult)
            .options(joinedload(BlindTestRowResult.dataset_row))
            .filter(BlindTestRowResult.blind_test_task_id == task.id)
            .order_by(BlindTestRowResult.row_index.asc())
            .all()
        )

        semaphore = asyncio.Semaphore(BLIND_TEST_CONCURRENCY)

        async def _generate(row_result_id: int) -> tuple[int, dict[str, Any]]:
            local_session: Session = session_factory()
            try:
                row_result = (
                    local_session.query(BlindTestRowResult)
                    .options(joinedload(BlindTestRowResult.dataset_row))
                    .filter(BlindTestRowResult.id == row_result_id)
                    .first()
                )
                if row_result is None:
                    return row_result_id, {}
                row_data = dict(row_result.dataset_row.data or {})
                async with semaphore:
                    answer_a, answer_b = await asyncio.gather(
                        _safe_answer(client_a, row_data),
                        _safe_answer(client_b, row_data),
                    )
                return row_result_id, {
                    "answer_a": answer_a.get("answer"),
                    "answer_b": answer_b.get("answer"),
                    "answer_a_error": answer_a.get("error"),
                    "answer_b_error": answer_b.get("error"),
                }
            finally:
                local_session.close()

        results = await asyncio.gather(*[_generate(row.id) for row in row_results])

        for row_id, payload in results:
            row_result = session.query(BlindTestRowResult).filter(BlindTestRowResult.id == row_id).first()
            if row_result is None:
                continue
            row_result.answer_a = payload.get("answer_a")
            row_result.answer_b = payload.get("answer_b")
            row_result.answer_a_error = payload.get("answer_a_error")
            row_result.answer_b_error = payload.get("answer_b_error")
            task.completed_rows += 1
            if task.total_rows:
                task.progress = round(task.completed_rows / task.total_rows, 4)
            _append_log(
                task,
                f"样本 #{row_result.row_index + 1}: "
                f"A={'成功' if row_result.answer_a else '失败'} "
                f"B={'成功' if row_result.answer_b else '失败'}",
            )
            session.commit()
            session.refresh(task)
            if task.status == "cancelled":
                _append_log(task, "任务已取消")
                session.commit()
                return

        task.progress = 1.0
        task.status = "completed"
        task.finished_at = datetime.now(timezone.utc)
        task.summary = _summarize_votes(task)
        _append_log(task, "盲测答案生成完成，等待人工投票")
        session.commit()
    except Exception as exc:
        session.rollback()
        task = session.query(BlindTestTask).filter(BlindTestTask.id == task_id).first()
        if task is not None:
            task.status = "failed"
            task.error_message = str(exc)
            task.finished_at = datetime.now(timezone.utc)
            _append_log(task, f"任务失败: {exc}")
            session.commit()
    finally:
        session.close()


async def _safe_answer(client: OpenAIResponseClient | EndpointResponseClient, row_data: dict[str, Any]) -> dict[str, Any]:
    try:
        answer = await client.answer(row_data)
        return {"answer": answer}
    except Exception as exc:
        return {"answer": None, "error": str(exc)}
