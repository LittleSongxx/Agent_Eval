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
from app.models.scenario import EvalScenario, ScenarioMetric
from app.models.llm_config import LLMConfig
from app.schemas.evaluation import EvalTaskCreate, EvalTaskResponse

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


@router.get("", response_model=List[EvalTaskResponse])
def list_evaluations(db: Session = Depends(get_db)):
    return db.query(EvalTask).order_by(EvalTask.created_at.desc()).all()


@router.post("", response_model=EvalTaskResponse, status_code=201)
async def create_evaluation(payload: EvalTaskCreate, db: Session = Depends(get_db)):
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
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    if settings.RUN_EVAL_ON_CREATE:
        _launch_evaluation(task.id)

    return task


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
