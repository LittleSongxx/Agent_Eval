import asyncio
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db, SessionLocal
from app.models.evaluation import EvalTask
from app.models.dataset import Dataset
from app.models.scenario import EvalScenario
from app.models.llm_config import LLMConfig
from app.schemas.evaluation import EvalTaskCreate, EvalTaskResponse

router = APIRouter(prefix="/evaluations", tags=["Evaluations"])


@router.get("", response_model=List[EvalTaskResponse])
def list_evaluations(db: Session = Depends(get_db)):
    return db.query(EvalTask).order_by(EvalTask.created_at.desc()).all()


@router.post("", response_model=EvalTaskResponse, status_code=201)
async def create_evaluation(payload: EvalTaskCreate, db: Session = Depends(get_db)):
    # Validate foreign keys
    if not db.query(Dataset).filter(Dataset.id == payload.dataset_id).first():
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not db.query(EvalScenario).filter(EvalScenario.id == payload.scenario_id).first():
        raise HTTPException(status_code=404, detail="Scenario not found")
    if not db.query(LLMConfig).filter(LLMConfig.id == payload.llm_config_id).first():
        raise HTTPException(status_code=404, detail="LLM config not found")

    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()

    task = EvalTask(
        name=payload.name,
        dataset_id=payload.dataset_id,
        scenario_id=payload.scenario_id,
        llm_config_id=payload.llm_config_id,
        status="pending",
        total_rows=dataset.row_count,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    # Launch background evaluation
    from app.core.evaluation_engine import run_evaluation

    asyncio.create_task(run_evaluation(task.id, SessionLocal))

    return task


@router.get("/{task_id}", response_model=EvalTaskResponse)
def get_evaluation(task_id: int, db: Session = Depends(get_db)):
    task = db.query(EvalTask).filter(EvalTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Evaluation task not found")
    return task


@router.post("/{task_id}/cancel", response_model=EvalTaskResponse)
def cancel_evaluation(task_id: int, db: Session = Depends(get_db)):
    task = db.query(EvalTask).filter(EvalTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Evaluation task not found")
    task.status = "cancelled"
    db.commit()
    db.refresh(task)
    return task
