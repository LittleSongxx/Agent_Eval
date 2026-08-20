import asyncio
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.core.database import get_db, SessionLocal
from app.core.scenario_snapshot import (
    build_dataset_snapshot,
    build_judge_snapshot,
    build_judge_runtime_snapshot,
    build_scenario_snapshot,
    compute_eval_fingerprint,
    dataset_snapshot_digest,
    snapshot_to_scenario_metrics,
)
from app.models.evaluation import EvalTask
from app.models.dataset import Dataset, DatasetRow
from app.models.endpoint_target import EndpointTarget
from app.models.scenario import EvalScenario, ScenarioMetric
from app.models.llm_config import LLMConfig
from app.models.tool_registry import ToolDefinition
from app.schemas.evaluation import EvalDebugRequest, EvalDebugResponse, EvalTaskCreate, EvalTaskResponse
from app.schemas.evaluation import EndpointEvalTargetTestRequest, EndpointEvalTargetTestResponse

router = APIRouter(prefix="/evaluations", tags=["Evaluations"])

EVAL_TASK_STALE_MINUTES = 30


def recover_stale_eval_tasks(db: Session, older_than_minutes: int = EVAL_TASK_STALE_MINUTES) -> int:
    """Mark tasks stuck in running/pending beyond a heartbeat window as failed.

    Judge 调用超时或进程异常都会让任务停留在 running；与 RAG 生成任务的
    陈旧回收机制对齐，避免任务永久悬挂。进程内存活的任务会通过心跳表
    刷新时间戳，不会被误杀。
    """
    from app.core.evaluation_engine import task_heartbeats

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=older_than_minutes)
    stale = (
        db.query(EvalTask)
        .filter(
            EvalTask.status.in_(["pending", "running"]),
            EvalTask.started_at.isnot(None),
            EvalTask.started_at < cutoff,
        )
        .all()
    )
    recovered = 0
    now = time.time()
    for task in stale:
        last_beat = task_heartbeats.get(task.id)
        if last_beat is not None and (now - last_beat) < older_than_minutes * 60:
            continue
        task.status = "failed"
        task.error_message = (
            f"评测任务超过 {older_than_minutes} 分钟未完成，已自动标记为失败（可能为进程中断）"
        )
        task.finished_at = datetime.now(timezone.utc)
        recovered += 1
    db.commit()
    return recovered


def _launch_evaluation(task_id: int) -> None:
    """Run evaluation outside the FastAPI event loop so progress APIs stay responsive."""
    from app.core.evaluation_engine import run_evaluation

    # 记录评测所在的后端进程 PID：启动 recovery 时（lifespan startup）只回收
    # "worker 进程已死"的任务，TestClient 等误启动的 lifespan 不会误杀运行中的任务
    db = SessionLocal()
    try:
        task = db.query(EvalTask).filter(EvalTask.id == task_id).first()
        if task is not None:
            task.worker_pid = os.getpid()
            db.commit()
    finally:
        db.close()

    def runner() -> None:
        asyncio.run(run_evaluation(task_id, SessionLocal))

    thread = threading.Thread(
        target=runner,
        name=f"eval-task-{task_id}",
        daemon=True,
    )
    thread.start()


def _validate_endpoint_payload(payload: EvalTaskCreate) -> None:
    if payload.evaluation_mode != "endpoint":
        return
    target_config = payload.target_config or {}
    if not str(target_config.get("endpoint_url") or "").strip():
        raise HTTPException(status_code=400, detail="接口实时评测必须配置 endpoint_url")
    if str(target_config.get("transport_mode") or "json") not in {"json", "sse"}:
        raise HTTPException(status_code=400, detail="transport_mode 仅支持 json 或 sse")
    if payload.result_save_mode not in {"task_only", "write_back"}:
        raise HTTPException(status_code=400, detail="result_save_mode 不合法")


def _snapshot_endpoint_config(payload: EvalTaskCreate, db: Session) -> tuple[dict | None, dict | None]:
    if payload.evaluation_mode != "endpoint":
        return payload.target_config, payload.response_mapping
    if not payload.endpoint_target_id:
        return payload.target_config, payload.response_mapping

    target = db.query(EndpointTarget).filter(EndpointTarget.id == payload.endpoint_target_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Endpoint target not found")
    target_config = {
        "endpoint_url": target.endpoint_url,
        "transport_mode": target.transport_mode or "json",
        "authorization": target.authorization or "",
        "extra_headers": target.extra_headers or "{}",
        "request_body_template": target.request_body_template or "",
    }
    return target_config, target.response_mapping or {}


def _resolve_endpoint_test_config(
    payload: EndpointEvalTargetTestRequest,
    db: Session,
) -> tuple[dict, dict]:
    target_config = dict(payload.target_config or {})
    response_mapping = dict(payload.response_mapping or {})
    if not payload.endpoint_target_id:
        return target_config, response_mapping

    target = db.query(EndpointTarget).filter(EndpointTarget.id == payload.endpoint_target_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Endpoint target not found")

    stored_config = {
        "endpoint_url": target.endpoint_url,
        "transport_mode": target.transport_mode or "json",
        "authorization": target.authorization or "",
        "extra_headers": target.extra_headers or "{}",
        "request_body_template": target.request_body_template or "",
    }
    stored_config.update({key: value for key, value in target_config.items() if value is not None})
    if not str(target_config.get("authorization") or "").strip():
        stored_config["authorization"] = target.authorization or ""

    return stored_config, response_mapping or target.response_mapping or {}


@router.get("", response_model=List[EvalTaskResponse])
def list_evaluations(db: Session = Depends(get_db)):
    recover_stale_eval_tasks(db)
    return db.query(EvalTask).order_by(EvalTask.created_at.desc()).all()


@router.post("", response_model=EvalTaskResponse, status_code=201)
async def create_evaluation(payload: EvalTaskCreate, db: Session = Depends(get_db)):
    target_config, response_mapping = _snapshot_endpoint_config(payload, db)
    if payload.evaluation_mode == "endpoint":
        payload.target_config = target_config
        payload.response_mapping = response_mapping
    _validate_endpoint_payload(payload)
    # Validate foreign keys
    if not db.query(Dataset).filter(Dataset.id == payload.dataset_id).first():
        raise HTTPException(status_code=404, detail="Dataset not found")
    scenario = (
        db.query(EvalScenario)
        .options(
            joinedload(EvalScenario.metrics).joinedload(ScenarioMetric.metric_definition)
        )
        .filter(EvalScenario.id == payload.scenario_id)
        .first()
    )
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    llm_config = db.query(LLMConfig).filter(LLMConfig.id == payload.llm_config_id).first()
    if not llm_config:
        raise HTTPException(status_code=404, detail="LLM config not found")

    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    dataset_rows = (
        db.query(DatasetRow)
        .filter(DatasetRow.dataset_id == payload.dataset_id)
        .order_by(DatasetRow.row_index)
        .all()
    )
    actual_row_count = len(dataset_rows)
    dataset_snapshot = build_dataset_snapshot(dataset_rows)
    dataset_digest = dataset_snapshot_digest(dataset_snapshot)
    # Keep the denormalized count aligned with the actual rows before taking
    # the immutable task snapshot.
    dataset.row_count = actual_row_count
    tool_registry_snapshot = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters_schema": tool.parameters_schema,
            "risk_level": tool.risk_level,
            "has_side_effect": tool.has_side_effect,
            "idempotency_required": tool.idempotency_required,
            "timeout_ms": tool.timeout_ms,
            "enabled": tool.enabled,
        }
        for tool in db.query(ToolDefinition).filter(ToolDefinition.enabled.is_(True)).order_by(ToolDefinition.name).all()
    ]

    # 多裁判面板：校验附加裁判配置存在，任务创建时冻结面板
    judge_panel: list[int] = []
    panel_configs: list[LLMConfig] = []
    for panel_config_id in list(payload.judge_llm_config_ids or []):
        if panel_config_id == payload.llm_config_id or panel_config_id in judge_panel:
            # The primary Judge is already included; duplicate panel IDs would
            # otherwise silently give one config extra voting weight.
            continue
        panel_config = db.query(LLMConfig).filter(LLMConfig.id == panel_config_id).first()
        if not panel_config:
            raise HTTPException(status_code=404, detail=f"裁判 LLM 配置 {panel_config_id} 不存在")
        judge_panel.append(panel_config_id)
        panel_configs.append(panel_config)

    # 评测口径 = 数据 + 尺子 + 裁判。三者都得在建任务时冻结，否则事后无法
    # 判断两个任务的分差来自被测系统还是来自口径本身。
    scenario_snapshot = build_scenario_snapshot(scenario, payload.metric_overrides)
    judge_snapshot = build_judge_snapshot(llm_config, judge_panel, panel_configs)
    judge_runtime_snapshot = build_judge_runtime_snapshot(llm_config, panel_configs)

    task = EvalTask(
        name=payload.name,
        dataset_id=payload.dataset_id,
        scenario_id=payload.scenario_id,
        llm_config_id=payload.llm_config_id,
        status="pending",
        total_rows=actual_row_count,
        scenario_snapshot=scenario_snapshot,
        evaluation_mode=payload.evaluation_mode,
        endpoint_target_id=payload.endpoint_target_id,
        target_config=target_config,
        response_mapping=response_mapping,
        result_save_mode=payload.result_save_mode,
        judge_panel=judge_panel or None,
        dataset_version=dataset.version,
        judge_snapshot=judge_snapshot,
        judge_runtime_snapshot=judge_runtime_snapshot,
        tool_registry_snapshot=tool_registry_snapshot,
        dataset_snapshot=dataset_snapshot,
        dataset_snapshot_digest=dataset_digest,
        judge_samples=int(settings.EVAL_JUDGE_SAMPLES),
        eval_fingerprint=compute_eval_fingerprint(
            scenario_snapshot,
            judge_snapshot,
            payload.dataset_id,
            dataset.version,
            judge_samples=int(settings.EVAL_JUDGE_SAMPLES),
            tool_registry_snapshot=tool_registry_snapshot,
            dataset_snapshot=dataset_snapshot,
        ),
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    if settings.RUN_EVAL_ON_CREATE:
        _launch_evaluation(task.id)

    return task


@router.post("/test-endpoint", response_model=EndpointEvalTargetTestResponse)
async def test_endpoint_eval_target(
    payload: EndpointEvalTargetTestRequest,
    db: Session = Depends(get_db),
):
    from app.core.endpoint_eval import extract_eval_fields, invoke_endpoint

    try:
        target_config, response_mapping = _resolve_endpoint_test_config(payload, db)
        response_payload = await invoke_endpoint(payload.row_data or {}, target_config)
        extracted_fields, mapping_errors = extract_eval_fields(
            response_payload,
            response_mapping,
        )
        return EndpointEvalTargetTestResponse(
            success=True,
            message="接口调用成功",
            request_body=response_payload.get("request_body"),
            status_code=response_payload.get("status_code"),
            latency_ms=response_payload.get("latency_ms"),
            raw_response=response_payload.get("raw_response"),
            extracted_fields=extracted_fields,
            mapping_errors=mapping_errors,
        )
    except Exception as exc:
        return EndpointEvalTargetTestResponse(
            success=False,
            message=str(exc),
        )


@router.post("/debug", response_model=EvalDebugResponse)
async def debug_evaluation_flow(payload: EvalDebugRequest, db: Session = Depends(get_db)):
    """Run one dataset row through the same endpoint + Judge path without persisting a task."""
    from app.core.endpoint_eval import extract_eval_fields, invoke_endpoint
    from app.core.evaluation_engine import (
        OpenAIJudgeClient,
        _determine_pass,
        _missing_fields_for_metrics,
        build_metric,
        diagnose_metric_prompt,
    )

    started = time.time()
    errors: list[str] = []
    warnings: list[str] = []
    metric_scores: dict = {}
    judge_traces: list[dict] = []
    endpoint_trace: dict | None = None

    target_config, response_mapping = _snapshot_endpoint_config(payload, db)
    if payload.evaluation_mode == "endpoint":
        payload.target_config = target_config
        payload.response_mapping = response_mapping
    _validate_endpoint_payload(payload)

    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    row_query = db.query(DatasetRow).filter(DatasetRow.dataset_id == payload.dataset_id)
    if payload.row_id:
        row_query = row_query.filter(DatasetRow.id == payload.row_id)
    dataset_row = row_query.order_by(DatasetRow.row_index.asc()).first()
    if not dataset_row:
        raise HTTPException(status_code=404, detail="Dataset row not found")

    scenario = (
        db.query(EvalScenario)
        .options(
            joinedload(EvalScenario.metrics).joinedload(ScenarioMetric.metric_definition)
        )
        .filter(EvalScenario.id == payload.scenario_id)
        .first()
    )
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    if not scenario.metrics:
        raise HTTPException(status_code=400, detail="Scenario has no metrics")

    llm_config = db.query(LLMConfig).filter(LLMConfig.id == payload.llm_config_id).first()
    if not llm_config:
        raise HTTPException(status_code=404, detail="LLM config not found")

    base_row_data = dict(dataset_row.data or {})
    row_data = {**base_row_data, "_dataset_data": base_row_data}
    if payload.evaluation_mode == "endpoint":
        try:
            response_payload = await invoke_endpoint(base_row_data, target_config or {})
            extracted_fields, mapping_errors = extract_eval_fields(
                response_payload,
                response_mapping or {},
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
        except Exception as exc:
            error = f"接口调用失败: {str(exc)[:800]}"
            errors.append(error)
            endpoint_trace = {"status": "error", "error": error}

    scenario_snapshot = build_scenario_snapshot(scenario, payload.metric_overrides)
    scenario_metrics = snapshot_to_scenario_metrics(scenario_snapshot)
    metrics: list[tuple[str, object, ScenarioMetric]] = []
    for scenario_metric in scenario_metrics:
        metric_def = scenario_metric.metric_definition
        try:
            _kind, metric_instance = build_metric(
                metric_def,
                prompt_override=getattr(scenario_metric, "prompt_override", None),
                scenario_metric=scenario_metric,
            )
            metrics.append((metric_def.name, metric_instance, scenario_metric))
        except Exception as exc:
            errors.append(f"指标 [{metric_def.display_name}] 构建失败: {str(exc)[:300]}")

    if not metrics:
        raise HTTPException(status_code=400, detail="No metrics could be built")

    missing_fields = _missing_fields_for_metrics(row_data, metrics)
    if missing_fields:
        warnings.append(f"字段缺失可能导致无法完整评分: {', '.join(missing_fields)}")

    judge_client = OpenAIJudgeClient(llm_config)
    for metric_name, metric_instance, scenario_metric in metrics:
        metric_def = scenario_metric.metric_definition
        metric_warnings = diagnose_metric_prompt(
            metric_def,
            row_data,
            prompt_override=getattr(scenario_metric, "prompt_override", None),
            scenario_metric=scenario_metric,
            evaluation_mode=payload.evaluation_mode,
        )
        warnings.extend(metric_warnings)
        trace = {
            "metric_name": metric_name,
            "metric_display_name": metric_def.display_name,
            "metric_type": metric_def.metric_type,
            "prompt_messages": None,
            "judge_raw_response": None,
            "warnings": metric_warnings,
            "error": None,
        }
        try:
            judge_client.last_messages = None
            judge_client.last_raw_response = None
            result = await metric_instance.ascore(row_data, judge_client)
            value = result.value
            if isinstance(value, (int, float)):
                value = round(float(value), 4)
            metric_scores[metric_name] = {
                "score": value,
                "reason": str(result.reason or "")[:800],
            }
        except Exception as exc:
            error = str(exc)[:800]
            metric_scores[metric_name] = {"score": None, "reason": error}
            trace["error"] = error
            errors.append(f"指标 [{metric_name}] 评分失败: {error}")
        finally:
            trace["prompt_messages"] = judge_client.last_messages
            trace["judge_raw_response"] = judge_client.last_raw_response
            trace["parsed_result"] = metric_scores.get(metric_name)
            judge_traces.append(trace)
        await asyncio.sleep(0)

    is_pass = _determine_pass(metric_scores, metrics)
    execution_time_ms = int((time.time() - started) * 1000)
    return EvalDebugResponse(
        success=(
            len([item for item in metric_scores.values() if item.get("score") is None]) == 0
            and not errors
            and not warnings
        ),
        message="调试完成" if not errors and not warnings else "调试完成，但存在错误或风险",
        dataset_row=dataset_row,
        row_data={key: value for key, value in row_data.items() if not str(key).startswith("_")},
        endpoint_trace=endpoint_trace,
        metric_scores=metric_scores,
        judge_traces=judge_traces,
        is_pass=is_pass,
        execution_time_ms=execution_time_ms,
        warnings=warnings,
        errors=errors,
    )


@router.get("/{task_id}", response_model=EvalTaskResponse)
def get_evaluation(task_id: int, db: Session = Depends(get_db)):
    recover_stale_eval_tasks(db)
    task = db.query(EvalTask).filter(EvalTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Evaluation task not found")
    return task


@router.get("/{task_id}/logs")
def get_evaluation_logs(task_id: int, db: Session = Depends(get_db)):
    recover_stale_eval_tasks(db)
    task = db.query(EvalTask).filter(EvalTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Evaluation task not found")
    return {
        "task_id": task_id,
        "status": task.status,
        "progress": task.progress or 0,
        "total_rows": task.total_rows or 0,
        "completed_rows": task.completed_rows or 0,
        "logs": task.logs or "",
    }


@router.post("/{task_id}/cancel", response_model=EvalTaskResponse)
def cancel_evaluation(task_id: int, db: Session = Depends(get_db)):
    task = db.query(EvalTask).filter(EvalTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Evaluation task not found")
    task.status = "cancelled"
    db.commit()
    db.refresh(task)
    return task
