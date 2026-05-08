import asyncio
import threading
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload

from app.core.blind_test_engine import run_blind_test, _summarize_votes
from app.core.database import SessionLocal, get_db
from app.models.dataset import Dataset, DatasetRow
from app.models.evaluation import BlindTestRowResult, BlindTestTask
from app.models.llm_config import LLMConfig
from app.schemas.evaluation import (
    BlindTestRowsResponse,
    BlindTestSummaryResponse,
    BlindTestTargetTestRequest,
    BlindTestTargetTestResponse,
    BlindTestTargetPayload,
    BlindTestTaskCreate,
    BlindTestTaskResponse,
    BlindTestVoteUpdate,
    BlindTestRowResultResponse,
)

router = APIRouter(prefix="/blind-tests", tags=["Blind Tests"])


def _launch_blind_test(task_id: int) -> None:
    def runner() -> None:
        asyncio.run(run_blind_test(task_id, SessionLocal))

    thread = threading.Thread(target=runner, name=f"blind-test-{task_id}", daemon=True)
    thread.start()


def _validate_target(payload: BlindTestTargetPayload, db: Session) -> dict:
    target = payload.model_dump()
    if payload.target_type == "llm_config":
        if not payload.llm_config_id:
            raise HTTPException(status_code=422, detail="模型类型的盲测目标必须选择 LLM 配置")
        llm_config = db.query(LLMConfig).filter(LLMConfig.id == payload.llm_config_id).first()
        if llm_config is None:
            raise HTTPException(status_code=404, detail="盲测引用的 LLM 配置不存在")
        target["name"] = payload.name or llm_config.name
    else:
        if not payload.endpoint_url:
            raise HTTPException(status_code=422, detail="接口类型的盲测目标必须填写 URL")
        target["name"] = payload.name or "接口对比对象"
    return target


def _compute_summary(task: BlindTestTask) -> BlindTestSummaryResponse:
    summary = _summarize_votes(task)
    task.summary = summary
    return BlindTestSummaryResponse(
        blind_test_task=BlindTestTaskResponse.model_validate(task),
        **summary,
    )


@router.get("", response_model=List[BlindTestTaskResponse])
def list_blind_tests(db: Session = Depends(get_db)):
    tasks = (
        db.query(BlindTestTask)
        .options(joinedload(BlindTestTask.dataset))
        .order_by(BlindTestTask.created_at.desc())
        .all()
    )
    return tasks


@router.post("/test-target", response_model=BlindTestTargetTestResponse)
async def test_blind_test_target(payload: BlindTestTargetTestRequest, db: Session = Depends(get_db)):
    try:
        target = _validate_target(payload.target, db)
        row_data = {"user_input": payload.test_question}
        from app.core.blind_test_engine import _build_target_client

        client = _build_target_client(db, target)
        answer = await client.answer(row_data)
        return BlindTestTargetTestResponse(
            success=True,
            message="Target reachable",
            answer_preview=(answer or "")[:800],
        )
    except Exception as exc:
        return BlindTestTargetTestResponse(
            success=False,
            message=str(exc),
            answer_preview=None,
        )


@router.post("", response_model=BlindTestTaskResponse, status_code=201)
def create_blind_test(payload: BlindTestTaskCreate, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    field_names = {field.get("name") for field in (dataset.field_schema or []) if isinstance(field, dict)}
    if "user_input" not in field_names:
        raise HTTPException(status_code=422, detail="盲测数据集必须包含 user_input 字段")

    rows = (
        db.query(DatasetRow)
        .filter(DatasetRow.dataset_id == payload.dataset_id)
        .order_by(DatasetRow.row_index.asc())
        .all()
    )
    if not rows:
        raise HTTPException(status_code=422, detail="数据集暂无样本，无法创建盲测任务")

    selected_rows = rows[: payload.sample_limit] if payload.sample_limit and payload.sample_limit > 0 else rows
    target_a = _validate_target(payload.target_a, db)
    target_b = _validate_target(payload.target_b, db)

    task = BlindTestTask(
        name=payload.name,
        dataset_id=payload.dataset_id,
        status="pending",
        progress=0.0,
        total_rows=len(selected_rows),
        completed_rows=0,
        voted_rows=0,
        sample_limit=payload.sample_limit,
        target_a=target_a,
        target_b=target_b,
    )
    db.add(task)
    db.flush()

    for row in selected_rows:
        blind_row = BlindTestRowResult(
            blind_test_task_id=task.id,
            dataset_row_id=row.id,
            row_index=row.row_index,
            display_order=["a", "b"] if (task.id + row.id) % 2 == 0 else ["b", "a"],
        )
        db.add(blind_row)

    db.commit()
    db.refresh(task)
    _launch_blind_test(task.id)
    return (
        db.query(BlindTestTask)
        .options(joinedload(BlindTestTask.dataset))
        .filter(BlindTestTask.id == task.id)
        .first()
    )


@router.get("/{task_id}", response_model=BlindTestTaskResponse)
def get_blind_test(task_id: int, db: Session = Depends(get_db)):
    task = (
        db.query(BlindTestTask)
        .options(joinedload(BlindTestTask.dataset))
        .filter(BlindTestTask.id == task_id)
        .first()
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Blind test task not found")
    return task


@router.get("/{task_id}/summary", response_model=BlindTestSummaryResponse)
def get_blind_test_summary(task_id: int, db: Session = Depends(get_db)):
    task = (
        db.query(BlindTestTask)
        .options(joinedload(BlindTestTask.dataset), joinedload(BlindTestTask.row_results))
        .filter(BlindTestTask.id == task_id)
        .first()
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Blind test task not found")
    summary = _compute_summary(task)
    db.commit()
    return summary


@router.get("/{task_id}/rows", response_model=BlindTestRowsResponse)
def list_blind_test_rows(
    task_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = Query(None),
    db: Session = Depends(get_db),
):
    task = db.query(BlindTestTask).filter(BlindTestTask.id == task_id).first()
    if task is None:
        raise HTTPException(status_code=404, detail="Blind test task not found")

    query = (
        db.query(BlindTestRowResult)
        .options(joinedload(BlindTestRowResult.dataset_row))
        .filter(BlindTestRowResult.blind_test_task_id == task_id)
    )
    if status == "pending_vote":
        query = query.filter(BlindTestRowResult.vote.is_(None))
    elif status == "voted":
        query = query.filter(BlindTestRowResult.vote.isnot(None))
    elif status == "ready":
        query = query.filter(
            (BlindTestRowResult.answer_a.isnot(None)) | (BlindTestRowResult.answer_b.isnot(None))
        )
    total = query.count()
    items = (
        query.order_by(BlindTestRowResult.row_index.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return BlindTestRowsResponse(total=total, page=page, page_size=page_size, items=items)


@router.post("/{task_id}/rows/{row_id}/vote", response_model=BlindTestRowResultResponse)
def vote_blind_test_row(task_id: int, row_id: int, payload: BlindTestVoteUpdate, db: Session = Depends(get_db)):
    row = (
        db.query(BlindTestRowResult)
        .options(joinedload(BlindTestRowResult.blind_test_task))
        .filter(
            BlindTestRowResult.blind_test_task_id == task_id,
            BlindTestRowResult.id == row_id,
        )
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Blind test row not found")
    row.vote = payload.vote
    row.vote_note = payload.vote_note
    row.voted_at = datetime.now(timezone.utc)
    task = row.blind_test_task
    if task is not None:
        task.voted_rows = (
            db.query(BlindTestRowResult)
            .filter(
                BlindTestRowResult.blind_test_task_id == task_id,
                BlindTestRowResult.vote.isnot(None),
                BlindTestRowResult.vote != "skip",
            )
            .count()
        )
        task.summary = _summarize_votes(task)
    db.commit()
    db.refresh(row)
    return row


@router.post("/{task_id}/cancel", response_model=BlindTestTaskResponse)
def cancel_blind_test(task_id: int, db: Session = Depends(get_db)):
    task = db.query(BlindTestTask).filter(BlindTestTask.id == task_id).first()
    if task is None:
        raise HTTPException(status_code=404, detail="Blind test task not found")
    task.status = "cancelled"
    task.finished_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(task)
    return task
