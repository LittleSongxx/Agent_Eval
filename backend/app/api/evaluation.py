import asyncio
import threading
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.core.database import get_db, SessionLocal
from app.core.scenario_snapshot import build_scenario_snapshot
from app.models.evaluation import EvalTask
from app.models.dataset import Dataset, DatasetRow
from app.models.endpoint_target import EndpointTarget
from app.models.scenario import EvalScenario, ScenarioMetric
from app.models.llm_config import LLMConfig
from app.schemas.evaluation import EvalTaskCreate, EvalTaskResponse
from app.schemas.evaluation import EndpointEvalTargetTestRequest, EndpointEvalTargetTestResponse

router = APIRouter(prefix="/evaluations", tags=["Evaluations"])


def _launch_evaluation(task_id: int) -> None:
    """Run evaluation outside the FastAPI event loop so progress APIs stay responsive."""
    from app.core.evaluation_engine import run_evaluation

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


@router.get("", response_model=List[EvalTaskResponse])
def list_evaluations(db: Session = Depends(get_db)):
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
    if not db.query(LLMConfig).filter(LLMConfig.id == payload.llm_config_id).first():
        raise HTTPException(status_code=404, detail="LLM config not found")

    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    actual_row_count = (
        db.query(DatasetRow)
        .filter(DatasetRow.dataset_id == payload.dataset_id)
        .count()
    )
    dataset.row_count = actual_row_count

    task = EvalTask(
        name=payload.name,
        dataset_id=payload.dataset_id,
        scenario_id=payload.scenario_id,
        llm_config_id=payload.llm_config_id,
        status="pending",
        total_rows=actual_row_count,
        scenario_snapshot=build_scenario_snapshot(scenario),
        evaluation_mode=payload.evaluation_mode,
        endpoint_target_id=payload.endpoint_target_id,
        target_config=target_config,
        response_mapping=response_mapping,
        result_save_mode=payload.result_save_mode,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    if settings.RUN_EVAL_ON_CREATE:
        _launch_evaluation(task.id)

    return task


@router.post("/test-endpoint", response_model=EndpointEvalTargetTestResponse)
async def test_endpoint_eval_target(payload: EndpointEvalTargetTestRequest):
    from app.core.endpoint_eval import extract_eval_fields, invoke_endpoint

    try:
        response_payload = await invoke_endpoint(payload.row_data or {}, payload.target_config or {})
        extracted_fields, mapping_errors = extract_eval_fields(
            response_payload,
            payload.response_mapping or {},
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


@router.get("/{task_id}", response_model=EvalTaskResponse)
def get_evaluation(task_id: int, db: Session = Depends(get_db)):
    task = db.query(EvalTask).filter(EvalTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Evaluation task not found")
    return task


@router.get("/{task_id}/logs")
def get_evaluation_logs(task_id: int, db: Session = Depends(get_db)):
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
