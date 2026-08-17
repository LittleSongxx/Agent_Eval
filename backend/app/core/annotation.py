"""多标注者标注的存储真值与 manual_* 投影。

## 为什么会有这个模块

原先 `eval_row_results` 上挂着一套 `manual_*` 列，物理上只能存下**一个**
标注者的判断。这个限制的后果不是"少存了点数据"，而是：

    人-人 kappa 无法计算，而人-人 kappa 正是 judge-人 kappa 的上界。

judge 与某位标注者达到 kappa=0.8175，如果两位人类之间只有 0.65，那 0.8175
测的是"judge 学会了这一个人的偏好"，不是"judge 接近事实"——而且此时 0.8175
比人类天花板还高，这个数越大越可疑。缺了上界，0.8175 是个没有参照系的数字。

所以标注真值搬到 `row_annotations`（一行多条），`manual_*` 降级为**投影**：
只保留"当前生效的那一条"。

## 投影为什么不直接删掉

前端 46 处引用、3 个离线脚本、已发布报告里的全部数字都读 `manual_*`。
把重构和这些调用点的迁移绑在同一步，等于让一次数据模型变更去赌五个文件同时
改对。投影保留后，单标注者场景下逐字段与旧行为一致（见
`test_annotation.py::test_single_annotator_projection_matches_legacy_semantics`），
25 条已发布标注的口径不变。

代价是多了一份派生数据，会漂移。控制手段：投影只允许经
`project_annotations_onto_row` 单点重算，任何写入路径改完 annotations 必须调它；
`test_projection_has_no_drift_on_real_labels` 会拿真库的 25 行验证不动点。
"""

from __future__ import annotations

import typing as t
from datetime import datetime, timezone

# 回填与默认写入使用的标注者身份。
# 平台还没有登录态（API 鉴权是 P2 项），未指定标注者的写入全部归到这个身份下——
# 这样旧前端不带 annotator 字段的请求行为与重构前完全一致。
DEFAULT_ANNOTATOR = "reviewer_1"

# 参与二分类一致性统计的两个终态。needs_fix / needs_review 是过程态：
# 它们表示"还没给出通过与否的结论"，计入 kappa 会把"未决"误当成"判负"。
BINARY_STATUSES = ("pass", "fail")

# 请求体里属于"标注内容"的字段名（沿用旧前端在发的 manual_* 名字）。
# annotator / is_adjudication 是元数据，不算内容——这个区分决定了
# "空 payload = 撤回标注"的判定：把元数据算进内容里，一个只带 annotator
# 的请求就会被误判成一次有效标注，从而凭空造出一条空标注记录。
ANNOTATION_PAYLOAD_FIELDS = ("manual_status", "manual_score", "manual_tags", "manual_note")

# 投影口径：说明 manual_* 当前这一份值是怎么来的。
# 报告里必须能区分"一个人说的"和"两个人一致同意的"——两者证据强度差一个量级。
BASIS_NONE = "none"
BASIS_SINGLE = "single_annotator"
BASIS_UNANIMOUS = "unanimous"
BASIS_ADJUDICATED = "adjudicated"
BASIS_UNRESOLVED = "unresolved_disagreement"


def binary_label(status: str | None) -> bool | None:
    """把标注状态映射为二分类布尔；非终态返回 None（不参与 kappa）。"""
    if status == "pass":
        return True
    if status == "fail":
        return False
    return None


def _as_naive_utc(value: t.Any) -> t.Any:
    """把 datetime 归一成 naive UTC，供排序/取 max 使用。

    为什么需要这个：SQLite 的 DATETIME 不存时区，写进去的 aware datetime 读回来
    一定是 naive。于是同一行里可能同时存在两种 datetime——刚写入内存的 aware
    和从库里加载的 naive——`sorted()` 比较它们会直接抛
    `TypeError: can't compare offset-naive and offset-aware datetimes`。

    这个坑只在**多标注者**时才炸：单条标注排序不做任何比较，所以重构后第一次
    写入看起来完全正常，直到第二位标注者进来。归一化放在比较入口而不是只修写入端，
    是因为写入端不止一处（回填 SQL、API、测试构造），漏一处就复发。
    """
    if value is None:
        return None
    tzinfo = getattr(value, "tzinfo", None)
    if tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _sort_key(annotation: t.Any) -> tuple:
    """标注排序键：时间优先，同时刻按标注者名兜底，保证投影结果可复现。"""
    created = _as_naive_utc(getattr(annotation, "created_at", None))
    annotator = getattr(annotation, "annotator", "") or ""
    # created_at 可能为 None（对象还没 flush），排在最后而不是抛异常
    return (created is None, created, annotator)


def _merge_tags(annotations: t.Sequence[t.Any]) -> list[str] | None:
    """多标注者标签取并集，保持首次出现顺序（去重但不排序，便于人读）。"""
    merged: list[str] = []
    seen: set[str] = set()
    saw_list = False
    for item in annotations:
        tags = getattr(item, "tags", None)
        if tags is None:
            continue
        saw_list = True
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                merged.append(tag)
    if not saw_list:
        return None
    return merged


def _merge_notes(annotations: t.Sequence[t.Any]) -> str | None:
    """多标注者备注拼接，带标注者前缀——否则合并后无法追溯是谁写的。"""
    parts = [
        f"[{getattr(item, 'annotator', '?')}] {item.note}"
        for item in annotations
        if getattr(item, "note", None)
    ]
    if not parts:
        return None
    return "\n".join(parts)


def _mean_score(annotations: t.Sequence[t.Any]) -> float | None:
    scores = [
        float(item.score)
        for item in annotations
        if getattr(item, "score", None) is not None
    ]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def effective_annotation(annotations: t.Sequence[t.Any]) -> dict[str, t.Any]:
    """从一行的全部标注里算出"当前生效"的那一份。纯函数，不碰数据库。

    优先级与理由：

    1. **有仲裁 → 用仲裁**。仲裁是看过分歧后的最终判断，定义上覆盖独立标注。
    2. **单个独立标注 → 直接用**。这一支必须逐字段等于重构前的行为，
       否则 25 条已发布标注的口径会变。
    3. **多个独立标注且状态一致 → 合并**。分数取均值、标签取并集、备注带前缀拼接。
    4. **多个独立标注且状态冲突且无仲裁 → `needs_review`**。

    第 4 条是有意的：冲突未仲裁时投影**不能**替人类选一个答案。写成
    `needs_review` 有两个后果，都是想要的——前端已有该状态的展示，且它不在
    `BINARY_STATUSES` 里，于是这一行自动被排除在 kappa 之外。用未仲裁的分歧行
    算 judge 一致性，等于拿一个人类自己都没定论的标签当事实。
    """
    independent = sorted(
        [a for a in annotations if not getattr(a, "is_adjudication", False)],
        key=_sort_key,
    )
    adjudications = sorted(
        [a for a in annotations if getattr(a, "is_adjudication", False)],
        key=_sort_key,
    )
    all_sorted = sorted(annotations, key=_sort_key)

    reviewed_at = None
    # 同样要归一化：这里是 max() 而不是 sorted()，但比较仍然发生，
    # 混着 aware 和 naive 一样会抛 TypeError。
    timestamps = [
        _as_naive_utc(getattr(a, "created_at", None))
        for a in all_sorted
        if getattr(a, "created_at", None) is not None
    ]
    if timestamps:
        reviewed_at = max(timestamps)

    result: dict[str, t.Any] = {
        "manual_status": None,
        "manual_score": None,
        "manual_tags": None,
        "manual_note": None,
        "reviewed_at": reviewed_at,
        "basis": BASIS_NONE,
        "annotator_count": len(independent),
        "annotators": [a.annotator for a in independent],
        "has_adjudication": bool(adjudications),
        "is_disagreement": False,
    }

    if adjudications:
        # 多条仲裁时取最后一条（后仲裁覆盖前仲裁）
        final = adjudications[-1]
        statuses = {a.status for a in independent}
        result.update(
            {
                "manual_status": final.status,
                "manual_score": final.score,
                "manual_tags": final.tags,
                "manual_note": final.note,
                "basis": BASIS_ADJUDICATED,
                "adjudicator": final.annotator,
                "is_disagreement": len(statuses) > 1,
            }
        )
        return result

    if not independent:
        return result

    if len(independent) == 1:
        only = independent[0]
        result.update(
            {
                "manual_status": only.status,
                "manual_score": only.score,
                "manual_tags": only.tags,
                "manual_note": only.note,
                "basis": BASIS_SINGLE,
            }
        )
        return result

    statuses = {a.status for a in independent}
    if len(statuses) == 1:
        result.update(
            {
                "manual_status": independent[0].status,
                "manual_score": _mean_score(independent),
                "manual_tags": _merge_tags(independent),
                "manual_note": _merge_notes(independent),
                "basis": BASIS_UNANIMOUS,
            }
        )
        return result

    result.update(
        {
            "manual_status": "needs_review",
            "manual_score": None,
            "manual_tags": _merge_tags(independent),
            "manual_note": _merge_notes(independent),
            "basis": BASIS_UNRESOLVED,
            "is_disagreement": True,
            "conflicting_statuses": sorted(s for s in statuses if s is not None),
        }
    )
    return result


def project_annotations_onto_row(row_result: t.Any) -> dict[str, t.Any]:
    """把标注投影写回 row_result 的 manual_* 五列。**唯一**允许写这五列的地方。"""
    projection = effective_annotation(list(row_result.annotations or []))
    row_result.manual_status = projection["manual_status"]
    row_result.manual_score = projection["manual_score"]
    row_result.manual_tags = projection["manual_tags"]
    row_result.manual_note = projection["manual_note"]
    row_result.reviewed_at = projection["reviewed_at"]
    return projection


def upsert_annotation(
    db: t.Any,
    row_result: t.Any,
    fields: dict[str, t.Any],
    annotator: str = DEFAULT_ANNOTATOR,
    is_adjudication: bool = False,
) -> t.Any:
    """写入/更新某标注者对某行的标注，并同步重算投影。

    fields 使用 `manual_status` / `manual_score` / `manual_tags` / `manual_note`
    这套旧字段名（前端仍在发这套名字），内部映射到 RowAnnotation 的短名。
    只更新显式给出的键——沿用旧接口 `exclude_unset` 的语义。
    """
    from app.models.evaluation import RowAnnotation

    existing = next(
        (
            a
            for a in (row_result.annotations or [])
            if a.annotator == annotator
            and bool(a.is_adjudication) == bool(is_adjudication)
        ),
        None,
    )
    if existing is None:
        existing = RowAnnotation(
            row_result_id=row_result.id,
            annotator=annotator,
            is_adjudication=is_adjudication,
        )
        db.add(existing)
        row_result.annotations.append(existing)

    mapping = {
        "manual_status": "status",
        "manual_score": "score",
        "manual_tags": "tags",
        "manual_note": "note",
    }
    for payload_key, column in mapping.items():
        if payload_key in fields:
            setattr(existing, column, fields[payload_key])

    # created_at 由 DB default 填；此处显式兜底，保证同一请求内投影就能算出
    # reviewed_at（否则 flush 前 created_at 为 None，投影会把时间判成"未复核"）。
    #
    # 写 naive UTC 而不是 aware：这一列在 SQLite 里读回来必然是 naive，
    # 写入端存 aware 会让「刚写完的内存对象」和「重新加载的对象」时区属性不同，
    # 投影结果因此依赖于对象是否被 refresh 过——这种不确定性比时区本身更难查。
    if existing.created_at is None:
        existing.created_at = datetime.now(timezone.utc).replace(tzinfo=None)

    db.flush()
    project_annotations_onto_row(row_result)
    return existing


def delete_annotation(
    db: t.Any,
    row_result: t.Any,
    annotator: str = DEFAULT_ANNOTATOR,
    is_adjudication: bool = False,
) -> bool:
    """撤回某标注者的标注，并同步重算投影。返回是否真的删掉了一条。"""
    target = next(
        (
            a
            for a in (row_result.annotations or [])
            if a.annotator == annotator
            and bool(a.is_adjudication) == bool(is_adjudication)
        ),
        None,
    )
    if target is None:
        project_annotations_onto_row(row_result)
        return False

    row_result.annotations.remove(target)
    db.delete(target)
    db.flush()
    project_annotations_onto_row(row_result)
    return True


def labels_by_annotator(row_results: t.Sequence[t.Any]) -> dict[str, dict[int, bool]]:
    """按标注者收集二分类标签：{标注者: {row_result_id: 是否通过}}。

    只收独立标注（`is_adjudication=False`）。仲裁者看过双方答案，把仲裁结果
    塞进人-人一致性里会把一致性抬高——那是循环论证，不是测量。
    """
    collected: dict[str, dict[int, bool]] = {}
    for row in row_results:
        for annotation in row.annotations or []:
            if annotation.is_adjudication:
                continue
            label = binary_label(annotation.status)
            if label is None:
                continue
            collected.setdefault(annotation.annotator, {})[row.id] = label
    return collected


def disagreement_rows(row_results: t.Sequence[t.Any]) -> list[dict[str, t.Any]]:
    """列出标注者之间给出不同终态结论的行，供人工仲裁。

    只报"两人都给了终态且结论相反"的行。一方 pass 另一方 needs_review 不算
    分歧——那是进度差异，不是判断冲突。
    """
    rows: list[dict[str, t.Any]] = []
    for row in row_results:
        labels = {
            a.annotator: binary_label(a.status)
            for a in (row.annotations or [])
            if not a.is_adjudication and binary_label(a.status) is not None
        }
        if len(set(labels.values())) <= 1:
            continue
        adjudication = next(
            (a for a in (row.annotations or []) if a.is_adjudication), None
        )
        rows.append(
            {
                "row_result_id": row.id,
                "row_index": row.row_index,
                "labels": {name: ("pass" if value else "fail") for name, value in labels.items()},
                "auto_is_pass": row.is_pass,
                "adjudicated_status": adjudication.status if adjudication else None,
                "adjudicator": adjudication.annotator if adjudication else None,
                "resolved": adjudication is not None,
            }
        )
    return rows
