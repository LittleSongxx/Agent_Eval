from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import Integer, func
from sqlalchemy.orm import Session, joinedload, selectinload

from app.core.database import get_db
from app.models.evaluation import EvalTask, EvalRowResult
from app.core.agreement import (
    bootstrap_kappa_ci,
    build_calibration_suggestion,
    cohens_kappa,
    judge_ceiling_comparison,
    pairwise_annotator_kappa,
)
from app.core.annotation import (
    ANNOTATION_PAYLOAD_FIELDS,
    BASIS_NONE,
    DEFAULT_ANNOTATOR,
    delete_annotation,
    disagreement_rows,
    effective_annotation,
    labels_by_annotator,
    upsert_annotation,
)
from app.core.evaluation_engine import DEFAULT_SUMMARY_PASS_THRESHOLD
from app.core.scenario_snapshot import describe_fingerprint_diff
from app.schemas.evaluation import (
    EvalTaskResponse,
    EvalRowResultResponse,
    EvalRowReviewUpdate,
    ReportCompareResponse,
    ReportListResponse,
    ReportSummary,
    ReportRowsResponse,
)

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("", response_model=ReportListResponse)
def list_reports(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = Query(None),
    evaluation_mode: str | None = Query(None),
    keyword: str | None = Query(None),
    db: Session = Depends(get_db),
):
    query = (
        db.query(EvalTask)
        .options(
            joinedload(EvalTask.dataset),
            joinedload(EvalTask.scenario),
            joinedload(EvalTask.endpoint_target),
        )
    )
    if status:
        query = query.filter(EvalTask.status == status)
    if evaluation_mode:
        query = query.filter(EvalTask.evaluation_mode == evaluation_mode)
    if keyword:
        query = query.filter(EvalTask.name.ilike(f"%{keyword.strip()}%"))

    total = query.count()
    tasks = (
        query
        .order_by(EvalTask.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    task_ids = [task.id for task in tasks]
    row_stats = _load_report_row_stats(db, task_ids)

    items = []
    for task in tasks:
        stats = row_stats.get(task.id, {})
        total_count = int(stats.get("total_count") or 0)
        pass_count = int(stats.get("pass_count") or 0)
        fail_count = int(stats.get("fail_count") or 0)
        error_count = int(stats.get("error_count") or 0)
        pass_rate = pass_count / total_count if total_count else 0.0
        endpoint_name = task.endpoint_target.name if task.endpoint_target else None
        items.append(
            {
                "eval_id": task.id,
                "task_name": task.name,
                "dataset_id": task.dataset_id,
                "dataset_name": task.dataset.name if task.dataset else None,
                "scenario_id": task.scenario_id,
                "scenario_name": (task.scenario_snapshot or {}).get("name")
                or (task.scenario.name if task.scenario else None),
                "evaluation_mode": task.evaluation_mode or "offline",
                "endpoint_target_id": task.endpoint_target_id,
                "endpoint_name": endpoint_name,
                "status": task.status,
                "total_count": total_count,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "error_count": error_count,
                "pass_rate": round(pass_rate, 4),
                "progress": task.progress or 0,
                "completed_rows": task.completed_rows or 0,
                "total_rows": task.total_rows,
                "error_message": task.error_message,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "finished_at": task.finished_at,
            }
        )

    return ReportListResponse(total=total, page=page, page_size=page_size, items=items)


def _load_report_row_stats(db: Session, task_ids: list[int]) -> dict[int, dict]:
    if not task_ids:
        return {}
    rows = (
        db.query(
            EvalRowResult.eval_task_id,
            func.count(EvalRowResult.id).label("total_count"),
            func.sum(EvalRowResult.is_pass.is_(True).cast(Integer)).label("pass_count"),
            func.sum(EvalRowResult.is_pass.is_(False).cast(Integer)).label("fail_count"),
            func.sum(EvalRowResult.error.isnot(None).cast(Integer)).label("error_count"),
        )
        .filter(EvalRowResult.eval_task_id.in_(task_ids))
        .group_by(EvalRowResult.eval_task_id)
        .all()
    )
    return {
        row.eval_task_id: {
            "total_count": row.total_count,
            "pass_count": row.pass_count,
            "fail_count": row.fail_count,
            "error_count": row.error_count,
        }
        for row in rows
    }


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
        # 一致性统计要遍历每行的全部标注：不预加载会退化成 1+N 次查询
        # （25 行标注时 25 次额外 SELECT，报告页每次刷新都付这个代价）
        .options(selectinload(EvalRowResult.annotations))
        .filter(EvalRowResult.eval_task_id == eval_id)
        .all()
    )

    total_count = len(row_results)
    pass_count = sum(1 for r in row_results if r.is_pass is True)
    fail_count = sum(1 for r in row_results if r.is_pass is False)
    error_count = sum(1 for r in row_results if r.error is not None)
    pass_rate = (pass_count / total_count) if total_count > 0 else 0.0

    raw_summary = task.summary_scores or {}
    # 平台级聚合结果（加权总分/成本/Judge 稳定性）先取出，再归一化指标维度统计
    weighted_total_score = raw_summary.get("weighted_total_score")
    cost = raw_summary.get("cost")
    judge_reliability = raw_summary.get("judge_reliability")
    metric_summary = _normalize_metric_summary(raw_summary, row_results)

    manual_review_summary = {
        "reviewed_count": sum(1 for r in row_results if r.manual_status is not None),
        "manual_pass_count": sum(1 for r in row_results if r.manual_status == "pass"),
        "manual_fail_count": sum(1 for r in row_results if r.manual_status == "fail"),
        "manual_needs_fix_count": sum(1 for r in row_results if r.manual_status == "needs_fix"),
        "manual_needs_review_count": sum(1 for r in row_results if r.manual_status == "needs_review"),
    }

    # 自动 vs 人工一致性：仅统计已给出明确通过/驳回结论且自动评分无异常的行
    comparable_reviews = [
        r
        for r in row_results
        if r.manual_status in ("pass", "fail") and r.is_pass is not None and r.error is None
    ]
    agreed = sum(
        1 for r in comparable_reviews if (r.manual_status == "pass") == (r.is_pass is True)
    )
    manual_auto_agreement_rate = (
        round(agreed / len(comparable_reviews), 4) if comparable_reviews else None
    )
    manual_auto_disagreement_count = len(comparable_reviews) - agreed

    # Cohen's kappa：人工 vs 自动二分类，修正偶然一致后的真实一致程度。
    # 公式与 bootstrap 都在 core/agreement.py，校准脚本共用同一份实现——
    # 报告接口和离线脚本各算一遍 kappa 迟早会漂移成两个口径。
    manual_auto_kappa = None
    manual_auto_kappa_ci = None
    calibration_suggestion = None
    if comparable_reviews:
        pairs = [
            (r.manual_status == "pass", r.is_pass is True)
            for r in comparable_reviews
        ]
        manual_auto_kappa = cohens_kappa(pairs)["kappa"]
        # 点估计单独看不出抽样不确定性：n=25 时 0.82 可能对应下界 0.55，
        # 区间必须和点估计一起返回，否则前端只能展示一个伪装成定论的数字。
        manual_auto_kappa_ci = bootstrap_kappa_ci(pairs)
        calibration_suggestion = build_calibration_suggestion(
            manual_auto_kappa, manual_auto_kappa_ci, len(comparable_reviews)
        )

    # 人-人一致性：上面那个 manual_auto_kappa 现在测的是「judge vs 生效标签」
    # （生效标签 = 单标注 / 多标注一致 / 仲裁结果，见 core/annotation.py）。
    # 但 0.82 这个数算高还是算离谱，取决于两位人类之间能到多少——判断本身主观的
    # 任务上人类可能只有 0.6。所以上界必须和 judge kappa 一起返回，不能让前端
    # 拿一个没有参照系的分数去下结论。
    annotator_labels = labels_by_annotator(row_results)
    annotator_agreement = pairwise_annotator_kappa(annotator_labels)
    judge_labels = {
        r.id: (r.is_pass is True)
        for r in row_results
        if r.is_pass is not None and r.error is None
    }
    judge_ceiling_check = judge_ceiling_comparison(annotator_labels, judge_labels)
    annotation_disagreements = disagreement_rows(row_results)

    # 生效标签的来源分布：单人标注 25 条和两人一致 25 条是完全不同的证据强度，
    # 报告必须能区分，否则「25 条人工标注」这句话在两种情况下听起来一样强。
    label_basis_summary: dict[str, int] = {}
    for row in row_results:
        basis = effective_annotation(list(row.annotations or []))["basis"]
        if basis == BASIS_NONE:
            continue
        label_basis_summary[basis] = label_basis_summary.get(basis, 0) + 1

    return ReportSummary(
        eval_task=EvalTaskResponse.model_validate(task),
        total_count=total_count,
        pass_count=pass_count,
        fail_count=fail_count,
        error_count=error_count,
        pass_rate=round(pass_rate, 4),
        metric_summary=metric_summary,
        manual_review_summary=manual_review_summary,
        weighted_total_score=weighted_total_score,
        cost=cost,
        judge_reliability=judge_reliability,
        manual_auto_agreement_rate=manual_auto_agreement_rate,
        manual_auto_disagreement_count=manual_auto_disagreement_count,
        manual_auto_kappa=manual_auto_kappa,
        manual_auto_kappa_ci=manual_auto_kappa_ci,
        calibration_suggestion=calibration_suggestion,
        annotator_agreement=annotator_agreement,
        judge_ceiling_check=judge_ceiling_check,
        annotation_disagreements=annotation_disagreements,
        label_basis_summary=label_basis_summary or None,
    )


def _normalize_metric_summary(metric_summary: dict, row_results: list[EvalRowResult]) -> dict:
    """Fill display pass rates for old reports whose metrics had no threshold."""

    from app.core.evaluation_engine import RESERVED_SUMMARY_KEYS

    normalized = {
        name: dict(info or {})
        for name, info in (metric_summary or {}).items()
        if name not in RESERVED_SUMMARY_KEYS
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
        .options(
            joinedload(EvalRowResult.dataset_row),
            # 响应里带 annotations，不预取就是每页 20 行 20 条额外查询
            selectinload(EvalRowResult.annotations),
        )
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
        .options(
            joinedload(EvalRowResult.dataset_row),
            selectinload(EvalRowResult.annotations),
        )
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
        .options(
            joinedload(EvalRowResult.dataset_row),
            selectinload(EvalRowResult.annotations),
        )
        .filter(
            EvalRowResult.eval_task_id == eval_id,
            EvalRowResult.id == row_id,
        )
        .first()
    )
    if not row_result:
        raise HTTPException(status_code=404, detail="Row result not found")

    data = payload.model_dump(exclude_unset=True)
    annotator = data.pop("annotator", None) or DEFAULT_ANNOTATOR
    is_adjudication = bool(data.pop("is_adjudication", False))
    # 只有这四个键算标注内容；annotator / is_adjudication 是"写给谁"的路由信息，
    # 不能计入"这次请求有没有给出标注"的判断——否则带 annotator 的空请求会被
    # 误判成一次有效标注。
    annotation_fields = {k: v for k, v in data.items() if k in ANNOTATION_PAYLOAD_FIELDS}

    if annotation_fields:
        # 注意：全为 None 的 annotation_fields（前端"清空复核"就是这样）仍走 upsert，
        # 得到一条状态为空的标注 + reviewed_at=now。这与重构前逐字段一致：
        # 旧代码在 data 非空时同样会盖上 reviewed_at。
        upsert_annotation(
            db,
            row_result,
            annotation_fields,
            annotator=annotator,
            is_adjudication=is_adjudication,
        )
    else:
        # 空请求体 = 撤回该标注者的标注。旧代码此时只把 reviewed_at 清成 None
        # 却把 manual_* 留在原处（状态自相矛盾：有标注但显示未复核）。
        # 前端从不发空请求体，也没有测试依赖旧行为，所以这里改成一致地整条撤回。
        delete_annotation(
            db, row_result, annotator=annotator, is_adjudication=is_adjudication
        )

    db.commit()
    db.refresh(row_result)
    return row_result


@router.get("/{eval_id}/compare", response_model=ReportCompareResponse)
def compare_report(
    eval_id: int,
    baseline_eval_id: int = Query(...),
    db: Session = Depends(get_db),
):
    current = _load_eval_task_for_compare(db, eval_id)
    baseline = _load_eval_task_for_compare(db, baseline_eval_id)
    _validate_comparable_tasks(current, baseline)

    current_rows = _load_compare_rows(db, current.id)
    baseline_rows = _load_compare_rows(db, baseline.id)
    current_by_dataset_row = {row.dataset_row_id: row for row in current_rows}
    baseline_by_dataset_row = {row.dataset_row_id: row for row in baseline_rows}

    row_changes: dict[str, list[dict]] = {
        "new_failures": [],
        "fixed": [],
        "still_failing": [],
        "still_passing": [],
        "new_errors": [],
        "missing_in_current": [],
        "missing_in_baseline": [],
    }

    for dataset_row_id in sorted(set(current_by_dataset_row) | set(baseline_by_dataset_row)):
        current_row = current_by_dataset_row.get(dataset_row_id)
        baseline_row = baseline_by_dataset_row.get(dataset_row_id)
        item = _build_row_compare_item(current_row, baseline_row)

        if current_row is None:
            row_changes["missing_in_current"].append(item)
            continue
        if baseline_row is None:
            row_changes["missing_in_baseline"].append(item)
            continue
        if current_row.error is not None and baseline_row.error is None:
            row_changes["new_errors"].append(item)
        elif baseline_row.is_pass is True and current_row.is_pass is False:
            row_changes["new_failures"].append(item)
        elif baseline_row.is_pass is False and current_row.is_pass is True:
            row_changes["fixed"].append(item)
        elif baseline_row.is_pass is False and current_row.is_pass is False:
            row_changes["still_failing"].append(item)
        elif baseline_row.is_pass is True and current_row.is_pass is True:
            row_changes["still_passing"].append(item)

    current_total = len(current_rows)
    baseline_total = len(baseline_rows)
    current_pass = sum(1 for row in current_rows if row.is_pass is True)
    baseline_pass = sum(1 for row in baseline_rows if row.is_pass is True)
    current_fail = sum(1 for row in current_rows if row.is_pass is False)
    baseline_fail = sum(1 for row in baseline_rows if row.is_pass is False)
    current_errors = sum(1 for row in current_rows if row.error is not None)
    baseline_errors = sum(1 for row in baseline_rows if row.error is not None)
    current_pass_rate = current_pass / current_total if current_total else 0.0
    baseline_pass_rate = baseline_pass / baseline_total if baseline_total else 0.0

    current_summary = _normalize_metric_summary(current.summary_scores or {}, current_rows)
    baseline_summary = _normalize_metric_summary(baseline.summary_scores or {}, baseline_rows)
    metric_names = _metric_names_from_snapshot(current.scenario_snapshot)

    return {
        "current_eval": EvalTaskResponse.model_validate(current),
        "baseline_eval": EvalTaskResponse.model_validate(baseline),
        "summary_delta": {
            "current_pass_rate": round(current_pass_rate, 4),
            "baseline_pass_rate": round(baseline_pass_rate, 4),
            "pass_rate_delta": round(current_pass_rate - baseline_pass_rate, 4),
            "current_fail_count": current_fail,
            "baseline_fail_count": baseline_fail,
            "fail_count_delta": current_fail - baseline_fail,
            "current_error_count": current_errors,
            "baseline_error_count": baseline_errors,
            "error_count_delta": current_errors - baseline_errors,
        },
        "metric_deltas": [
            _build_metric_delta(name, current_summary.get(name) or {}, baseline_summary.get(name) or {})
            for name in metric_names
        ],
        "row_changes": row_changes,
        "comparability": _build_comparability(current, baseline),
    }


def _build_comparability(current: EvalTask, baseline: EvalTask) -> dict:
    """State whether the two tasks were graded by the same ruler.

    对比接口原来只校验数据集/场景/指标集一致，于是"换了判定标准后分数变好"
    和"系统真的变好"在结果里长得一模一样。跨口径对比本身是合理需求（就是要
    看严一点的尺子会怎样），所以这里不拦，只把变化的维度说清楚：
    identical 才能把 delta 直接归因于被测系统。
    """
    changed = describe_fingerprint_diff(
        {
            "dataset_version": current.dataset_version,
            "judge_snapshot": current.judge_snapshot,
            "scenario_snapshot": current.scenario_snapshot,
        },
        {
            "dataset_version": baseline.dataset_version,
            "judge_snapshot": baseline.judge_snapshot,
            "scenario_snapshot": baseline.scenario_snapshot,
        },
    )
    current_fp = current.eval_fingerprint
    baseline_fp = baseline.eval_fingerprint

    if not current_fp or not baseline_fp:
        # 指纹上线前创建的任务无法追溯口径，只能说"不确定"，不能假装一致。
        status = "unknown"
        warning = (
            "其中一个任务创建于口径指纹上线前，无法确认两次评测是否用了同一把尺子；"
            "差值不能直接归因于被测系统。"
        )
    elif current_fp == baseline_fp:
        status = "identical"
        warning = None
    else:
        status = "changed"
        detail = "、".join(changed) if changed else "未能定位到具体维度"
        warning = f"两个任务的评测口径不一致（变化维度：{detail}），差值包含尺子变化带来的部分。"

    return {
        "status": status,
        "current_fingerprint": current_fp,
        "baseline_fingerprint": baseline_fp,
        "changed_dimensions": changed,
        "attribution_safe": status == "identical",
        "warning": warning,
    }


def _load_eval_task_for_compare(db: Session, eval_id: int) -> EvalTask:
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
    if task is None:
        raise HTTPException(status_code=404, detail="Evaluation task not found")
    return task


def _load_compare_rows(db: Session, eval_id: int) -> list[EvalRowResult]:
    return (
        db.query(EvalRowResult)
        .options(joinedload(EvalRowResult.dataset_row))
        .filter(EvalRowResult.eval_task_id == eval_id)
        .all()
    )


def _metric_names_from_snapshot(snapshot: dict | None) -> list[str]:
    names = []
    for item in (snapshot or {}).get("metrics") or []:
        metric_definition = item.get("metric_definition") or {}
        name = metric_definition.get("name")
        if name:
            names.append(name)
    return names


def _validate_comparable_tasks(current: EvalTask, baseline: EvalTask) -> None:
    if current.dataset_id != baseline.dataset_id:
        raise HTTPException(status_code=422, detail="只能对比同一数据集下的评测任务")
    if current.scenario_id != baseline.scenario_id:
        raise HTTPException(status_code=422, detail="只能对比同一评测场景下的任务")
    current_metrics = set(_metric_names_from_snapshot(current.scenario_snapshot))
    baseline_metrics = set(_metric_names_from_snapshot(baseline.scenario_snapshot))
    if current_metrics != baseline_metrics:
        raise HTTPException(status_code=422, detail="只能对比指标集一致的评测任务")


def _row_status(row: EvalRowResult | None) -> str | None:
    if row is None:
        return None
    if row.error is not None:
        return "error"
    if row.is_pass is True:
        return "pass"
    if row.is_pass is False:
        return "fail"
    return "unknown"


def _score_value(row: EvalRowResult | None, metric: str):
    if row is None:
        return None
    raw = (row.metric_scores or {}).get(metric)
    return raw.get("score") if isinstance(raw, dict) else raw


def _numeric_score(value):
    if value is None:
        return None
    if isinstance(value, str):
        return 1.0 if value.lower() in ("pass", "yes", "true", "1") else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_row_compare_item(current_row: EvalRowResult | None, baseline_row: EvalRowResult | None) -> dict:
    row = current_row or baseline_row
    metric_names = set((current_row.metric_scores or {}).keys() if current_row else [])
    metric_names.update((baseline_row.metric_scores or {}).keys() if baseline_row else [])
    metric_deltas = {}
    for metric in sorted(metric_names):
        current_score = _score_value(current_row, metric)
        baseline_score = _score_value(baseline_row, metric)
        current_numeric = _numeric_score(current_score)
        baseline_numeric = _numeric_score(baseline_score)
        metric_deltas[metric] = {
            "current_score": current_score,
            "baseline_score": baseline_score,
            "delta": (
                round(current_numeric - baseline_numeric, 4)
                if current_numeric is not None and baseline_numeric is not None
                else None
            ),
        }
    return {
        "dataset_row_id": row.dataset_row_id,
        "row_index": row.row_index,
        "current_result_id": current_row.id if current_row else None,
        "baseline_result_id": baseline_row.id if baseline_row else None,
        "current_status": _row_status(current_row),
        "baseline_status": _row_status(baseline_row),
        "metric_deltas": metric_deltas,
        "dataset_row": row.dataset_row,
    }


def _delta(current, baseline):
    if current is None or baseline is None:
        return None
    return round(current - baseline, 4)


def _build_metric_delta(metric: str, current: dict, baseline: dict) -> dict:
    current_mean = current.get("mean")
    baseline_mean = baseline.get("mean")
    current_pass_rate = current.get("pass_rate")
    baseline_pass_rate = baseline.get("pass_rate")
    return {
        "metric": metric,
        "current_mean": current_mean,
        "baseline_mean": baseline_mean,
        "mean_delta": _delta(current_mean, baseline_mean),
        "current_pass_rate": current_pass_rate,
        "baseline_pass_rate": baseline_pass_rate,
        "pass_rate_delta": _delta(current_pass_rate, baseline_pass_rate),
        "current_error_count": int(current.get("error_count") or 0),
        "baseline_error_count": int(baseline.get("error_count") or 0),
    }
