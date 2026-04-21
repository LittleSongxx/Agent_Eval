from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload

from app.core.database import get_db
from app.models.evaluation import EvalTask, EvalRowResult
from app.schemas.evaluation import (
    EvalTaskResponse,
    EvalRowResultResponse,
    ReportSummary,
    ReportRowsResponse,
)

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/{eval_id}/summary", response_model=ReportSummary)
def get_report_summary(eval_id: int, db: Session = Depends(get_db)):
    task = (
        db.query(EvalTask)
        .options(
            joinedload(EvalTask.dataset),
            joinedload(EvalTask.scenario),
            joinedload(EvalTask.llm_config),
        )
        .filter(EvalTask.id == eval_id)
        .first()
    )
    if not task:
        raise HTTPException(status_code=404, detail="Evaluation task not found")

    row_results = (
        db.query(EvalRowResult)
        .filter(EvalRowResult.eval_task_id == eval_id)
        .all()
    )

    total_count = len(row_results)
    pass_count = sum(1 for r in row_results if r.is_pass is True)
    fail_count = sum(1 for r in row_results if r.is_pass is False)
    error_count = sum(1 for r in row_results if r.error is not None)
    pass_rate = (pass_count / total_count) if total_count > 0 else 0.0

    metric_summary = task.summary_scores if task.summary_scores else {}

    return ReportSummary(
        eval_task=EvalTaskResponse.model_validate(task),
        total_count=total_count,
        pass_count=pass_count,
        fail_count=fail_count,
        error_count=error_count,
        pass_rate=round(pass_rate, 4),
        metric_summary=metric_summary,
    )


@router.get("/{eval_id}/rows", response_model=ReportRowsResponse)
def get_report_rows(
    eval_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = Query(None),
    db: Session = Depends(get_db),
):
    task = db.query(EvalTask).filter(EvalTask.id == eval_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Evaluation task not found")

    query = (
        db.query(EvalRowResult)
        .options(joinedload(EvalRowResult.dataset_row))
        .filter(EvalRowResult.eval_task_id == eval_id)
    )

    # Optional filter by pass/fail status
    if status == "pass":
        query = query.filter(EvalRowResult.is_pass == True)  # noqa: E712
    elif status == "fail":
        query = query.filter(EvalRowResult.is_pass == False)  # noqa: E712
    elif status == "error":
        query = query.filter(EvalRowResult.error.isnot(None))

    total = query.count()
    offset = (page - 1) * page_size
    items = (
        query
        .order_by(EvalRowResult.row_index)
        .offset(offset)
        .limit(page_size)
        .all()
    )

    return ReportRowsResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=items,
    )


@router.get("/{eval_id}/rows/{row_id}", response_model=EvalRowResultResponse)
def get_report_row_detail(
    eval_id: int, row_id: int, db: Session = Depends(get_db)
):
    row_result = (
        db.query(EvalRowResult)
        .options(joinedload(EvalRowResult.dataset_row))
        .filter(
            EvalRowResult.eval_task_id == eval_id,
            EvalRowResult.id == row_id,
        )
        .first()
    )
    if not row_result:
        raise HTTPException(status_code=404, detail="Row result not found")
    return row_result
