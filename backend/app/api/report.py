from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload

from app.core.database import get_db
from app.models.evaluation import EvalTask, EvalRowResult
from app.core.evaluation_engine import DEFAULT_SUMMARY_PASS_THRESHOLD
from app.schemas.evaluation import (
    EvalTaskResponse,
    EvalRowResultResponse,
    EvalRowReviewUpdate,
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

    metric_summary = _normalize_metric_summary(task.summary_scores or {}, row_results)
    manual_review_summary = {
        "reviewed_count": sum(1 for r in row_results if r.manual_status is not None),
        "manual_pass_count": sum(1 for r in row_results if r.manual_status == "pass"),
        "manual_fail_count": sum(1 for r in row_results if r.manual_status == "fail"),
        "manual_needs_fix_count": sum(1 for r in row_results if r.manual_status == "needs_fix"),
        "manual_needs_review_count": sum(1 for r in row_results if r.manual_status == "needs_review"),
    }

    return ReportSummary(
        eval_task=EvalTaskResponse.model_validate(task),
        total_count=total_count,
        pass_count=pass_count,
        fail_count=fail_count,
        error_count=error_count,
        pass_rate=round(pass_rate, 4),
        metric_summary=metric_summary,
        manual_review_summary=manual_review_summary,
    )


def _normalize_metric_summary(metric_summary: dict, row_results: list[EvalRowResult]) -> dict:
    """Fill display pass rates for old reports whose metrics had no threshold."""

    normalized = {
        name: dict(info or {})
        for name, info in (metric_summary or {}).items()
    }
    if not normalized:
        return normalized

    for metric_name, info in normalized.items():
        if info.get("pass_rate") is not None:
            continue
        threshold = info.get("effective_pass_threshold")
        if threshold is None:
            threshold = info.get("pass_threshold")
        if threshold is None:
            threshold = DEFAULT_SUMMARY_PASS_THRESHOLD

        values = []
        for row in row_results:
            raw = (row.metric_scores or {}).get(metric_name)
            score = raw.get("score") if isinstance(raw, dict) else raw
            if score is None:
                continue
            if isinstance(score, str):
                values.append(1.0 if score.lower() in ("pass", "yes", "true", "1") else 0.0)
            else:
                values.append(float(score))

        if values:
            pass_count = sum(1 for value in values if value >= threshold)
            info["pass_rate"] = round(pass_count / len(values), 4)
            info["effective_pass_threshold"] = threshold
        else:
            info["pass_rate"] = None

    return normalized


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


@router.patch("/{eval_id}/rows/{row_id}/review", response_model=EvalRowResultResponse)
def update_report_row_review(
    eval_id: int,
    row_id: int,
    payload: EvalRowReviewUpdate,
    db: Session = Depends(get_db),
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

    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(row_result, key, value)
    row_result.reviewed_at = None if not data else datetime.now(timezone.utc)
    db.commit()
    db.refresh(row_result)
    return row_result
