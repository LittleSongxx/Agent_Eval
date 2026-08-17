"""多标注者标注：投影正确性 + 人-人 kappa 上界。

## 这个文件锁什么

P1-1 重构把 `manual_*` 从"标注的存储位置"降级成"标注的投影"，真值搬进
`row_annotations`。这次重构的动机不是数据模型洁癖，而是一个测量上的硬缺陷：

    一行只有一套 manual_* 列 → 物理上只能存一个标注者 → 人-人 kappa 无法计算。

而人-人 kappa 正是 judge-人 kappa 的**上界参照**。项目对外写着
"judge 与人工的 kappa=0.8175"，如果两位人类之间只有 0.65，那 0.8175 测的是
"judge 拟合了这一个标注者"，不是"judge 接近事实"——这个数越高反而越可疑。

所以测试分三组：

1. **投影不能改旧行为**（`TestProjection`）。25 条已发布标注全部由单个标注者
   产生，投影在单标注者分支必须逐字段等于重构前的语义，否则所有对外数字的
   口径被静默改掉。
2. **上界要能算出来，且算不出来时必须显式说算不出来**（`TestAnnotatorAgreement`）。
   "只有 1 位标注者"和"一致性很好"是两回事，缺省值最容易被读成后者。
3. **仲裁不得污染人-人一致性**（`TestAdjudication`）。仲裁者看过双方答案，
   把仲裁结果计入人-人 kappa 是循环论证，会把上界抬高——而上界被抬高之后，
   judge 越过上界这件事就测不出来了。
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import text as sa_text

import pytest

from app.core.agreement import (
    judge_ceiling_comparison,
    pairwise_annotator_kappa,
)
from app.core.annotation import (
    BASIS_ADJUDICATED,
    BASIS_SINGLE,
    BASIS_UNANIMOUS,
    BASIS_UNRESOLVED,
    DEFAULT_ANNOTATOR,
    binary_label,
    disagreement_rows,
    effective_annotation,
    labels_by_annotator,
)



def _parse_dt(value):
    """把 sqlite 取出的 datetime 字符串解析成 naive datetime。

    sqlite 的 DATETIME 不存时区，取出来是字符串。解析成 datetime 而不是直接
    比字符串，是为了让"投影算出的 datetime"和"库里的值"在同一类型上比较——
    比字符串会让 '2026-08-06 14:03:38.579189' 和等价的其他格式化被判成不等。
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


class FakeAnnotation:
    """标注的最小替身：投影与统计函数只读属性，不需要真的 ORM 实例。

    用替身而不是真表，是为了让投影逻辑的测试与数据库解耦——投影是纯函数，
    它的正确性不应该依赖 session、建表顺序或 SQLite 的时区行为。
    真表路径由 TestApiRoundtrip 覆盖。
    """

    def __init__(
        self,
        annotator,
        status=None,
        score=None,
        tags=None,
        note=None,
        is_adjudication=False,
        created_at=None,
    ):
        self.annotator = annotator
        self.status = status
        self.score = score
        self.tags = tags
        self.note = note
        self.is_adjudication = is_adjudication
        self.created_at = created_at


class FakeRow:
    def __init__(self, row_id, row_index, is_pass, annotations):
        self.id = row_id
        self.row_index = row_index
        self.is_pass = is_pass
        self.error = None
        self.annotations = annotations
        self.manual_status = None
        self.manual_score = None
        self.manual_tags = None
        self.manual_note = None
        self.reviewed_at = None


BASE_TIME = datetime(2026, 8, 6, 14, 0, 0)


class TestProjection:
    def test_single_annotator_projection_matches_legacy_semantics(self):
        """单标注者：投影必须逐字段等于该标注者写的内容。

        这是整个重构的兼容性契约。25 条已发布标注全部走这一支，
        任何字段级偏差都会改掉已公开的 0.8175 / 0.4582 的输入数据。
        """
        annotation = FakeAnnotation(
            "reviewer_1",
            status="fail",
            score=0.3,
            tags=["回答不完整"],
            note="业务要求没满足",
            created_at=BASE_TIME,
        )
        projection = effective_annotation([annotation])

        assert projection["manual_status"] == "fail"
        assert projection["manual_score"] == 0.3
        assert projection["manual_tags"] == ["回答不完整"]
        assert projection["manual_note"] == "业务要求没满足"
        assert projection["reviewed_at"] == BASE_TIME
        assert projection["basis"] == BASIS_SINGLE
        assert projection["annotator_count"] == 1
        assert projection["is_disagreement"] is False

    def test_single_annotator_note_carries_no_prefix(self):
        """单标注者的备注**不能**被加上 `[annotator]` 前缀。

        多标注者合并时前缀是必要的（否则无法追溯是谁写的），但单标注者场景下
        加前缀就会改掉 25 条已发布标注的备注文本——同一个字段在两种情况下
        必须有不同处理，所以单独钉一条。
        """
        annotation = FakeAnnotation(
            "reviewer_1", status="pass", note="裸备注", created_at=BASE_TIME
        )
        assert effective_annotation([annotation])["manual_note"] == "裸备注"

    def test_no_annotations_yields_empty_projection(self):
        projection = effective_annotation([])
        assert projection["manual_status"] is None
        assert projection["reviewed_at"] is None
        assert projection["basis"] == "none"
        assert projection["annotator_count"] == 0

    def test_unanimous_annotators_merge_score_tags_and_notes(self):
        """两人结论一致：分数取均值、标签取并集、备注带标注者前缀拼接。"""
        annotations = [
            FakeAnnotation(
                "reviewer_1",
                status="pass",
                score=0.8,
                tags=["流畅"],
                note="可以接受",
                created_at=BASE_TIME,
            ),
            FakeAnnotation(
                "reviewer_2",
                status="pass",
                score=1.0,
                tags=["流畅", "有引用"],
                note="引用完整",
                created_at=BASE_TIME + timedelta(minutes=5),
            ),
        ]
        projection = effective_annotation(annotations)

        assert projection["manual_status"] == "pass"
        assert projection["manual_score"] == 0.9
        assert projection["manual_tags"] == ["流畅", "有引用"]
        assert "[reviewer_1] 可以接受" in projection["manual_note"]
        assert "[reviewer_2] 引用完整" in projection["manual_note"]
        assert projection["basis"] == BASIS_UNANIMOUS
        assert projection["annotator_count"] == 2
        assert projection["is_disagreement"] is False

    def test_unresolved_disagreement_becomes_needs_review_not_a_guess(self):
        """结论冲突且无仲裁：投影给 needs_review，**不能**替人类选一个答案。

        这一条是有意设计的，两个后果都是想要的：
          1. needs_review 不在 BINARY_STATUSES 里，于是这一行自动被排除在
             judge kappa 之外——用一个人类自己都没定论的标签去算 judge 一致性，
             等于把未决当成事实；
          2. 前端已有 needs_review 的展示，分歧行会自然浮现到待办里。
        """
        annotations = [
            FakeAnnotation("reviewer_1", status="pass", created_at=BASE_TIME),
            FakeAnnotation(
                "reviewer_2", status="fail", created_at=BASE_TIME + timedelta(minutes=1)
            ),
        ]
        projection = effective_annotation(annotations)

        assert projection["manual_status"] == "needs_review"
        assert projection["manual_score"] is None
        assert projection["basis"] == BASIS_UNRESOLVED
        assert projection["is_disagreement"] is True
        assert projection["conflicting_statuses"] == ["fail", "pass"]

    def test_reviewed_at_takes_latest_timestamp(self):
        annotations = [
            FakeAnnotation("reviewer_1", status="pass", created_at=BASE_TIME),
            FakeAnnotation(
                "reviewer_2", status="pass", created_at=BASE_TIME + timedelta(hours=3)
            ),
        ]
        assert effective_annotation(annotations)["reviewed_at"] == BASE_TIME + timedelta(hours=3)

    def test_mixed_tz_awareness_does_not_raise(self):
        """naive 与 aware datetime 混在同一行时不能抛异常。

        这是重构中真实炸过的一个坑：`upsert_annotation` 当时写入 aware datetime，
        而 SQLite 的 DATETIME 不存时区、读回来一定是 naive。同一行里两种 datetime
        共存，`sorted()` 直接抛
        `TypeError: can't compare offset-naive and offset-aware datetimes`。

        它只在**第二位标注者**进来时才炸：单条标注排序不做任何比较，所以重构后
        第一次写入看起来完全正常。这正是"多标注者"这条路径必须有独立测试的理由。
        """
        annotations = [
            FakeAnnotation("reviewer_1", status="pass", created_at=BASE_TIME),
            FakeAnnotation(
                "reviewer_2",
                status="pass",
                created_at=(BASE_TIME + timedelta(hours=1)).replace(tzinfo=timezone.utc),
            ),
        ]
        projection = effective_annotation(annotations)
        assert projection["manual_status"] == "pass"
        assert projection["basis"] == BASIS_UNANIMOUS

    def test_binary_label_treats_process_states_as_undecided(self):
        """needs_fix / needs_review 必须返回 None，不能塌成 False。

        塌成 False 等于把"还没结论"记成"判负"，会凭空造出 judge 与人工的分歧。
        """
        assert binary_label("pass") is True
        assert binary_label("fail") is False
        assert binary_label("needs_fix") is None
        assert binary_label("needs_review") is None
        assert binary_label(None) is None


class TestAdjudication:
    def test_adjudication_overrides_independent_labels(self):
        annotations = [
            FakeAnnotation("reviewer_1", status="pass", created_at=BASE_TIME),
            FakeAnnotation(
                "reviewer_2", status="fail", created_at=BASE_TIME + timedelta(minutes=1)
            ),
            FakeAnnotation(
                "lead",
                status="fail",
                note="以业务口径为准",
                is_adjudication=True,
                created_at=BASE_TIME + timedelta(minutes=9),
            ),
        ]
        projection = effective_annotation(annotations)

        assert projection["manual_status"] == "fail"
        assert projection["basis"] == BASIS_ADJUDICATED
        assert projection["adjudicator"] == "lead"
        assert projection["is_disagreement"] is True
        # 仲裁者不计入独立标注者数量
        assert projection["annotator_count"] == 2
        assert projection["annotators"] == ["reviewer_1", "reviewer_2"]

    def test_adjudication_excluded_from_human_human_labels(self):
        """仲裁记录**不得**进入人-人一致性统计。

        仲裁者已经看过双方答案，与其算一致性是循环论证。后果很具体：
        把仲裁计入会抬高人-人 kappa 上界，而上界一旦被抬高，
        "judge 越过人类天花板"这个信号就再也测不出来了。
        """
        rows = [
            FakeRow(
                1,
                0,
                True,
                [
                    FakeAnnotation("reviewer_1", status="pass", created_at=BASE_TIME),
                    FakeAnnotation("reviewer_2", status="fail", created_at=BASE_TIME),
                    FakeAnnotation(
                        "lead", status="fail", is_adjudication=True, created_at=BASE_TIME
                    ),
                ],
            )
        ]
        labels = labels_by_annotator(rows)
        assert sorted(labels.keys()) == ["reviewer_1", "reviewer_2"]
        assert "lead" not in labels

    def test_later_adjudication_supersedes_earlier(self):
        annotations = [
            FakeAnnotation("reviewer_1", status="pass", created_at=BASE_TIME),
            FakeAnnotation("reviewer_2", status="fail", created_at=BASE_TIME),
            FakeAnnotation(
                "lead", status="pass", is_adjudication=True, created_at=BASE_TIME + timedelta(minutes=5)
            ),
            FakeAnnotation(
                "lead2", status="fail", is_adjudication=True, created_at=BASE_TIME + timedelta(minutes=30)
            ),
        ]
        projection = effective_annotation(annotations)
        assert projection["manual_status"] == "fail"
        assert projection["adjudicator"] == "lead2"


class TestDisagreementRows:
    def test_lists_only_opposing_terminal_conclusions(self):
        """只报"双方都给了终态且结论相反"的行。

        一方 pass 另一方 needs_review 不算分歧——那是标注进度差异，
        不是判断冲突。把进度差异混进分歧列表会让仲裁队列充满噪声。
        """
        rows = [
            FakeRow(1, 0, True, [
                FakeAnnotation("reviewer_1", status="pass", created_at=BASE_TIME),
                FakeAnnotation("reviewer_2", status="fail", created_at=BASE_TIME),
            ]),
            FakeRow(2, 1, True, [
                FakeAnnotation("reviewer_1", status="pass", created_at=BASE_TIME),
                FakeAnnotation("reviewer_2", status="needs_review", created_at=BASE_TIME),
            ]),
            FakeRow(3, 2, False, [
                FakeAnnotation("reviewer_1", status="fail", created_at=BASE_TIME),
                FakeAnnotation("reviewer_2", status="fail", created_at=BASE_TIME),
            ]),
        ]
        found = disagreement_rows(rows)
        assert [r["row_index"] for r in found] == [0]
        assert found[0]["labels"] == {"reviewer_1": "pass", "reviewer_2": "fail"}
        assert found[0]["resolved"] is False

    def test_marks_resolved_when_adjudicated(self):
        rows = [
            FakeRow(1, 0, True, [
                FakeAnnotation("reviewer_1", status="pass", created_at=BASE_TIME),
                FakeAnnotation("reviewer_2", status="fail", created_at=BASE_TIME),
                FakeAnnotation("lead", status="fail", is_adjudication=True, created_at=BASE_TIME),
            ]),
        ]
        found = disagreement_rows(rows)
        assert len(found) == 1
        assert found[0]["resolved"] is True
        assert found[0]["adjudicated_status"] == "fail"
        assert found[0]["adjudicator"] == "lead"


class TestAnnotatorAgreement:
    def test_single_annotator_reports_ceiling_as_unmeasurable(self):
        """1 位标注者：必须显式返回 insufficient_annotators=True。

        这是当前真库的状态（25 条标注全部来自一个人）。此时人-人 kappa
        不存在，判断的正确表述是"judge 的 kappa 没有上界参照"，
        而不是任何形式的缺省值——缺省值会被读成"没问题"。
        """
        labels = {"reviewer_1": {1: True, 2: False, 3: True}}
        result = pairwise_annotator_kappa(labels)

        assert result["insufficient_annotators"] is True
        assert result["annotator_count"] == 1
        assert result["pairs"] == []
        assert result["ceiling_kappa"] is None
        assert "无法计算" in result["note"]

    def test_two_annotators_produce_kappa_with_interval(self):
        # 20 行完全一致 + 5 行冲突，n=25 与真库标注量一致
        labels_a = {}
        labels_b = {}
        for i in range(10):
            labels_a[i] = True
            labels_b[i] = True
        for i in range(10, 20):
            labels_a[i] = False
            labels_b[i] = False
        for i in range(20, 25):
            labels_a[i] = True
            labels_b[i] = False

        result = pairwise_annotator_kappa({"a": labels_a, "b": labels_b})

        assert result["insufficient_annotators"] is False
        assert result["annotator_count"] == 2
        assert len(result["pairs"]) == 1
        pair = result["pairs"][0]
        assert pair["overlap_n"] == 25
        assert pair["disagreement_count"] == 5
        assert pair["kappa"] is not None
        assert pair["kappa_ci"] is not None
        assert pair["kappa_ci"]["ci_low"] < pair["kappa"] < pair["kappa_ci"]["ci_high"]
        assert result["ceiling_kappa"] == pair["kappa"]

    def test_only_shared_rows_are_compared(self):
        """每对标注者只在双方都标过的行上比较。

        把缺失的行当"不一致"会凭空造出分歧，所以 overlap_n 必须是交集大小，
        而不是任一方的标注总数。
        """
        labels_a = {i: True for i in range(15)}
        labels_b = {i: True for i in range(10, 30)}
        result = pairwise_annotator_kappa({"a": labels_a, "b": labels_b})
        assert result["pairs"][0]["overlap_n"] == 5

    def test_ceiling_uses_min_not_mean_across_pairs(self):
        """三位标注者时上界取两两 kappa 的**最小值**。

        judge 只要超过任意一对人类的一致性，"它拟合了某个人"的嫌疑就已成立。
        用均值会把这个信号平均掉——一对很低、一对很高，均值看着还行。
        """
        # a 与 b 高度一致；c 与两者都差
        labels_a = {i: (i % 2 == 0) for i in range(20)}
        labels_b = dict(labels_a)
        labels_c = {i: (i % 3 == 0) for i in range(20)}

        result = pairwise_annotator_kappa({"a": labels_a, "b": labels_b, "c": labels_c})
        kappas = [p["kappa"] for p in result["pairs"] if p["kappa"] is not None]

        assert len(result["pairs"]) == 3
        assert result["ceiling_kappa"] == min(kappas)
        assert result["min_kappa"] <= result["mean_kappa"]

    def test_no_overlap_pairs_are_skipped(self):
        labels_a = {1: True, 2: False}
        labels_b = {8: True, 9: False}
        result = pairwise_annotator_kappa({"a": labels_a, "b": labels_b})
        assert result["pairs"] == []
        assert result["ceiling_kappa"] is None


class TestJudgeCeilingComparison:
    def test_single_annotator_status_is_ceiling_unknown(self):
        """1 位标注者时状态必须是 ceiling_unknown，且措辞不能像"通过"。

        这是真库当前的状态。报告在这一支必须说清"测不出上界"是数据限制，
        否则读者会把"没有告警"理解成"judge 已验证"。
        """
        labels = {"reviewer_1": {i: i % 2 == 0 for i in range(25)}}
        judge = {i: i % 2 == 0 for i in range(25)}
        result = judge_ceiling_comparison(labels, judge)

        assert result["status"] == "ceiling_unknown"
        assert result["comparison_count"] == 0
        assert "无法计算" in result["note"]

    def test_judge_matching_reference_exactly_is_flagged_above_ceiling(self):
        """judge 与参照标注者完全一致、而两位人类有分歧 → 必须告警。

        这是过拟合的典型形态：judge 复刻了 A 的每一个判断（包括 B 不同意的），
        配对 Δ 下界 > 0。它不是好消息——常见解释是 judge prompt 就是拿 A 的
        标注调出来的，而不是 judge 更接近事实。
        """
        labels_a = {i: (i < 12) for i in range(25)}
        labels_b = dict(labels_a)
        for i in range(0, 8):  # b 与 a 在 8 行上冲突
            labels_b[i] = not labels_a[i]
        judge = dict(labels_a)  # judge 与 a 逐行相同

        result = judge_ceiling_comparison({"a": labels_a, "b": labels_b}, judge)

        assert result["status"] == "judge_above_ceiling"
        assert result["comparison_count"] == 2
        flagged = [c for c in result["comparisons"] if c["judge_exceeds_human"]]
        assert flagged
        top = flagged[0]
        assert top["judge_human_kappa"] > top["human_human_kappa"]
        assert top["delta_ci_low"] > 0
        assert "拟合" in result["note"]

    def test_judge_within_human_noise_is_not_flagged(self):
        """judge 的分歧量级与人类彼此的分歧相当 → 不告警。

        这是"正常"的那一支：区间含 0，结论是 judge 处在人类一致性的量级内。
        """
        labels_a = {i: (i < 13) for i in range(25)}
        labels_b = dict(labels_a)
        for i in (1, 3, 5, 20, 22):
            labels_b[i] = not labels_a[i]
        judge = dict(labels_a)
        for i in (2, 4, 6, 21, 23):  # 与 a 的分歧数量和 b 相当，但位置不同
            judge[i] = not labels_a[i]

        result = judge_ceiling_comparison({"a": labels_a, "b": labels_b}, judge)

        assert result["status"] == "within_ceiling"
        assert result["comparison_count"] == 2
        assert all(not c["judge_exceeds_human"] for c in result["comparisons"])

    def test_insufficient_overlap_is_distinguished_from_within_ceiling(self):
        """三方共同标注行不足 → insufficient_overlap，不能混同为"没超上界"。

        样本不够导致算不出区间，和算出来区间含 0，是完全不同的两件事。
        """
        labels_a = {i: True for i in range(5)}
        labels_b = {i: True for i in range(5)}
        judge = {i: True for i in range(5)}
        result = judge_ceiling_comparison({"a": labels_a, "b": labels_b}, judge)

        assert result["status"] == "insufficient_overlap"
        assert result["comparison_count"] == 0

    def test_multiplicity_is_disclosed_not_corrected(self):
        """比较次数必须原样报出。

        k 位标注者产生 k*(k-1) 个有序对。这里不做 Bonferroni 校正——n=25
        量级下校正后几乎不会有任何一项显著，用校正把信号抹平比把多重性摆出来
        更糟。但"不校正"的前提是次数必须可见，否则读者无法自己折算。
        """
        labels_a = {i: (i < 13) for i in range(25)}
        labels_b = dict(labels_a)
        labels_b[1] = not labels_b[1]
        labels_c = dict(labels_a)
        labels_c[2] = not labels_c[2]
        judge = dict(labels_a)

        result = judge_ceiling_comparison(
            {"a": labels_a, "b": labels_b, "c": labels_c}, judge
        )
        # 3 位标注者 → 3*2 = 6 个有序对
        assert result["comparison_count"] == 6
        assert "6" in result["multiplicity_note"]
        assert "未做多重比较校正" in result["multiplicity_note"]


class TestApiRoundtrip:
    """真表路径：ORM + SQLite，覆盖投影落库与旧接口兼容。"""

    def _make_row(self, db):
        from app.models.dataset import Dataset, DatasetRow
        from app.models.evaluation import EvalRowResult

        dataset = Dataset(name="ds-annotation", description="", sample_type="qa")
        db.add(dataset)
        db.flush()
        dataset_row = DatasetRow(dataset_id=dataset.id, row_index=0, data={"q": "x"})
        db.add(dataset_row)
        db.flush()
        row = EvalRowResult(
            eval_task_id=999,
            dataset_row_id=dataset_row.id,
            row_index=0,
            is_pass=True,
        )
        db.add(row)
        db.commit()
        return row

    def _make_task(self, db, tag: str):
        """建一个可用的 completed 任务（scene_type NOT NULL，必须给值）。"""
        from app.models.dataset import Dataset
        from app.models.evaluation import EvalTask
        from app.models.llm_config import LLMConfig
        from app.models.scenario import EvalScenario

        dataset = Dataset(name=f"ds-{tag}", description="", sample_type="single_turn")
        scenario = EvalScenario(
            name=f"sc-{tag}", description="", scene_type="qa", sample_type="single_turn"
        )
        llm = LLMConfig(
            name=f"llm-{tag}",
            api_base_url="https://api.example.com/v1",
            api_key="k",
            model_name="m",
        )
        db.add_all([dataset, scenario, llm])
        db.flush()
        task = EvalTask(
            name=f"t-{tag}",
            dataset_id=dataset.id,
            scenario_id=scenario.id,
            llm_config_id=llm.id,
            status="completed",
        )
        db.add(task)
        db.flush()
        return task, dataset

    def _add_row(self, db, task, dataset, row_index: int, is_pass: bool):
        from app.models.dataset import DatasetRow
        from app.models.evaluation import EvalRowResult

        dr = DatasetRow(dataset_id=dataset.id, row_index=row_index, data={"q": f"q{row_index}"})
        db.add(dr)
        db.flush()
        row = EvalRowResult(
            eval_task_id=task.id,
            dataset_row_id=dr.id,
            row_index=row_index,
            is_pass=is_pass,
        )
        db.add(row)
        db.flush()
        return row

    def test_upsert_is_idempotent_per_annotator(self, db):
        """同一标注者重复写入只更新那一条，不叠加。

        唯一键把 (row, annotator, is_adjudication) 锁住。没有这条约束，
        一个人反复保存就会产生 N 条标注，人-人 kappa 会把同一个人算成 N 个人
        并得到虚高的一致性。
        """
        from app.core.annotation import upsert_annotation

        row = self._make_row(db)
        upsert_annotation(db, row, {"manual_status": "pass"}, annotator="reviewer_1")
        upsert_annotation(db, row, {"manual_status": "fail"}, annotator="reviewer_1")
        db.commit()
        db.refresh(row)

        assert len(row.annotations) == 1
        assert row.manual_status == "fail"

    def test_same_person_can_annotate_and_adjudicate(self, db):
        from app.core.annotation import upsert_annotation

        row = self._make_row(db)
        upsert_annotation(db, row, {"manual_status": "pass"}, annotator="lead")
        upsert_annotation(
            db, row, {"manual_status": "fail"}, annotator="lead", is_adjudication=True
        )
        db.commit()
        db.refresh(row)

        assert len(row.annotations) == 2
        assert row.manual_status == "fail"  # 仲裁优先

    def test_delete_annotation_recomputes_projection(self, db):
        from app.core.annotation import delete_annotation, upsert_annotation

        row = self._make_row(db)
        upsert_annotation(db, row, {"manual_status": "pass"}, annotator="reviewer_1")
        upsert_annotation(db, row, {"manual_status": "fail"}, annotator="reviewer_2")
        db.commit()
        db.refresh(row)
        assert row.manual_status == "needs_review"  # 冲突未仲裁

        assert delete_annotation(db, row, annotator="reviewer_2") is True
        db.commit()
        db.refresh(row)

        # 撤回后只剩一位标注者，投影回到该标注者的判断
        assert row.manual_status == "pass"
        assert len(row.annotations) == 1

    def test_delete_missing_annotation_returns_false(self, db):
        from app.core.annotation import delete_annotation

        row = self._make_row(db)
        assert delete_annotation(db, row, annotator="nobody") is False

    def test_legacy_review_endpoint_still_writes_default_annotator(self, client, db):
        """不带 annotator 的旧请求必须落到 DEFAULT_ANNOTATOR 名下。

        前端 46 处引用一个字段都没改就得继续工作，这是重构的兼容性前提。
        """
        row = self._make_row(db)
        resp = client.patch(
            f"/api/reports/999/rows/{row.id}/review",
            json={
                "manual_status": "fail",
                "manual_score": 0.3,
                "manual_tags": ["不完整"],
                "manual_note": "业务要求没满足",
            },
        )
        assert resp.status_code == 200
        payload = resp.json()

        # 旧字段逐个保持原样
        assert payload["manual_status"] == "fail"
        assert payload["manual_score"] == 0.3
        assert payload["manual_tags"] == ["不完整"]
        assert payload["reviewed_at"] is not None
        # 新增：真值以标注记录形式存在
        assert len(payload["annotations"]) == 1
        assert payload["annotations"][0]["annotator"] == DEFAULT_ANNOTATOR
        assert payload["annotations"][0]["is_adjudication"] is False

    def test_two_annotators_via_api_surface_ceiling_in_summary(self, client, db):
        """两位标注者经 API 写入后，summary 必须报出人-人 kappa 而不再是"测不出"。

        这就是 P1-1 想要的能力：重构前这个断言在物理上无法成立——
        一行只有一套 manual_* 列，第二位标注者的标注无处可存。
        """
        task, dataset = self._make_task(db, "ceiling")

        # 12 行：两位标注者在 3 行上冲突，judge 与 reviewer_1 完全一致
        judge_flags = [True] * 6 + [False] * 6
        for i, is_pass in enumerate(judge_flags):
            row = self._add_row(db, task, dataset, i, is_pass)

            a_status = "pass" if is_pass else "fail"
            b_status = a_status
            if i in (0, 1, 6):
                b_status = "fail" if a_status == "pass" else "pass"

            for annotator, status in (("reviewer_1", a_status), ("reviewer_2", b_status)):
                resp = client.patch(
                    f"/api/reports/{task.id}/rows/{row.id}/review",
                    json={"manual_status": status, "annotator": annotator},
                )
                assert resp.status_code == 200

        summary = client.get(f"/api/reports/{task.id}/summary").json()
        agreement = summary["annotator_agreement"]

        assert agreement["insufficient_annotators"] is False
        assert agreement["annotator_count"] == 2
        assert agreement["ceiling_kappa"] is not None
        assert agreement["pairs"][0]["disagreement_count"] == 3

        # 冲突未仲裁的 3 行落到 needs_review，被排除在 judge kappa 之外
        assert summary["label_basis_summary"]["unresolved_disagreement"] == 3
        assert summary["label_basis_summary"]["unanimous"] == 9
        assert len(summary["annotation_disagreements"]) == 3
        assert all(not d["resolved"] for d in summary["annotation_disagreements"])

    def test_adjudication_via_api_resolves_disagreement(self, client, db):
        task, dataset = self._make_task(db, "adj")
        row = self._add_row(db, task, dataset, 0, True)
        db.commit()

        for annotator, status in (("reviewer_1", "pass"), ("reviewer_2", "fail")):
            client.patch(
                f"/api/reports/{task.id}/rows/{row.id}/review",
                json={"manual_status": status, "annotator": annotator},
            )

        summary = client.get(f"/api/reports/{task.id}/summary").json()
        assert len(summary["annotation_disagreements"]) == 1
        assert summary["annotation_disagreements"][0]["resolved"] is False

        resp = client.patch(
            f"/api/reports/{task.id}/rows/{row.id}/review",
            json={
                "manual_status": "fail",
                "manual_note": "以业务口径为准",
                "annotator": "lead",
                "is_adjudication": True,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["manual_status"] == "fail"

        summary = client.get(f"/api/reports/{task.id}/summary").json()
        assert summary["annotation_disagreements"][0]["resolved"] is True
        assert summary["annotation_disagreements"][0]["adjudicator"] == "lead"
        assert summary["label_basis_summary"]["adjudicated"] == 1
        # 仲裁者不进入人-人统计
        assert summary["annotator_agreement"]["annotators"] == ["reviewer_1", "reviewer_2"]


class TestRealDatabaseInvariants:
    """拿真库的 25 条已发布标注验证迁移不动点。

    这一组是 `core/annotation.py` 和 `core/database.py` 两处文档注释里点名引用的
    证据。迁移刻意**不重算投影**（只读 manual_*，只插 row_annotations），理由是
    迁移期间那五列同时也是唯一数据源，回填有 bug 会就地改坏原始标注。代价是
    "投影确实等于原值"这件事没有被迁移本身保证，必须事后验证——就是这里。

    库不存在时跳过（全新 clone / CI 没有这份 dev 库），而不是失败：
    这个测试断言的是一次历史迁移的结果，不是代码的固有性质。
    """

    DB_PATH = Path(__file__).resolve().parents[1] / "eval_platform.db"

    def _load_rows(self):
        """只读打开真库，把 25 行标注读成投影函数能吃的替身对象。

        用 sqlite3 + mode=ro 而不是 ORM：这个测试要验证的是"库里躺着的字节"
        与投影的一致性，走 ORM 会把待验证的模型层塞进证据链里。
        """
        uri = f"file:{self.DB_PATH}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            annotated = conn.execute(
                """
                SELECT id, row_index, is_pass, manual_status, manual_score,
                       manual_tags, manual_note, reviewed_at
                FROM eval_row_results
                WHERE reviewed_at IS NOT NULL
                ORDER BY id
                """
            ).fetchall()
            rows = []
            for (
                row_id, row_index, is_pass, m_status, m_score,
                m_tags, m_note, reviewed_at,
            ) in annotated:
                anns = conn.execute(
                    """
                    SELECT annotator, status, score, tags, note,
                           is_adjudication, created_at
                    FROM row_annotations WHERE row_result_id = ? ORDER BY id
                    """,
                    (row_id,),
                ).fetchall()
                rows.append(
                    {
                        "id": row_id,
                        "row_index": row_index,
                        "is_pass": is_pass,
                        "stored": {
                            "manual_status": m_status,
                            "manual_score": m_score,
                            "manual_tags": json.loads(m_tags) if m_tags is not None else None,
                            "manual_note": m_note,
                            "reviewed_at": _parse_dt(reviewed_at),
                        },
                        "annotations": [
                            FakeAnnotation(
                                annotator=a[0],
                                status=a[1],
                                score=a[2],
                                tags=json.loads(a[3]) if a[3] is not None else None,
                                note=a[4],
                                is_adjudication=bool(a[5]),
                                created_at=_parse_dt(a[6]),
                            )
                            for a in anns
                        ],
                    }
                )
            return rows
        finally:
            conn.close()

    def test_projection_has_no_drift_on_real_labels(self):
        """25 条真实标注：投影五个字段逐行等于库里原本的值。

        这是整个 P1-1 重构的正确性底线。这五列此后由投影单点重算，任何一次
        重算（改一次标注就会触发）都会覆盖它们。如果投影与原值不等，那么
        重构后第一次有人点"保存人工复核"，已发布的 kappa=0.8175 就会静默变成
        另一个数——而报告上不会有任何痕迹说数字变过。
        """
        if not self.DB_PATH.exists():
            pytest.skip(f"dev 库不存在：{self.DB_PATH}")

        rows = self._load_rows()
        assert rows, "真库里应有已复核的行；若为 0 说明迁移或夹具前提已变"

        drift = []
        for row in rows:
            projection = effective_annotation(row["annotations"])
            for field, stored_value in row["stored"].items():
                if projection[field] != stored_value:
                    drift.append((row["row_index"], field, stored_value, projection[field]))

        assert drift == [], f"投影与原始 manual_* 不一致：{drift[:5]}"

    def test_every_annotated_row_has_exactly_one_backfilled_annotation(self):
        """回填后每行恰好一条标注，且归到默认标注者名下、非仲裁。

        "恰好一条"是幂等性的可观察后果：迁移在每次进程启动时都会跑，
        多插一条就会让同一个人被人-人 kappa 当成两个人算。
        """
        if not self.DB_PATH.exists():
            pytest.skip(f"dev 库不存在：{self.DB_PATH}")

        rows = self._load_rows()
        for row in rows:
            assert len(row["annotations"]) == 1, (
                f"row_index={row['row_index']} 有 {len(row['annotations'])} 条标注，"
                "回填不幂等"
            )
            only = row["annotations"][0]
            assert only.annotator == DEFAULT_ANNOTATOR
            assert only.is_adjudication is False

    def test_backfill_preserved_original_review_timestamps(self):
        """回填的 created_at 必须等于原 reviewed_at，不能是迁移那一刻。

        投影用 max(created_at) 反推 reviewed_at。若回填让 created_at 走 DB
        默认值（当前时间），下一次投影重算会把 25 条历史标注的复核时间全部改成
        迁移时刻——已发布报告的时间线被静默篡改，且没有任何报错。
        """
        if not self.DB_PATH.exists():
            pytest.skip(f"dev 库不存在：{self.DB_PATH}")

        rows = self._load_rows()
        for row in rows:
            assert row["annotations"][0].created_at == row["stored"]["reviewed_at"], (
                f"row_index={row['row_index']} 的标注时间与原 reviewed_at 不符"
            )

    def test_single_annotator_means_ceiling_is_still_unmeasurable_on_real_data(self):
        """真库当前只有 1 位标注者：上界必须报"测不出"，不能报一个数。

        这条是对项目自身结论的约束。已发布的 kappa=0.8175 目前**没有**上界参照，
        P1-1 只是让上界变得可测，并没有把它测出来——那需要第二个人真的去标 25 行。
        测试把这个事实钉住，避免文档里出现"人-人 kappa 已验证"这类没有数据支撑的话。
        """
        if not self.DB_PATH.exists():
            pytest.skip(f"dev 库不存在：{self.DB_PATH}")

        rows = self._load_rows()
        fake_rows = [
            FakeRow(r["id"], r["row_index"], bool(r["is_pass"]), r["annotations"])
            for r in rows
        ]
        agreement = pairwise_annotator_kappa(labels_by_annotator(fake_rows))

        assert agreement["annotator_count"] == 1
        assert agreement["insufficient_annotators"] is True
        assert agreement["ceiling_kappa"] is None
        assert agreement["pairs"] == []

        judge_labels = {
            r.id: (r.is_pass is True) for r in fake_rows if r.is_pass is not None
        }
        check = judge_ceiling_comparison(labels_by_annotator(fake_rows), judge_labels)
        assert check["status"] == "ceiling_unknown"
        assert check["comparison_count"] == 0


class TestBackfillCodePath:
    """在临时库上跑真正的回填函数。

    上面 `TestRealDatabaseInvariants` 断言的是一次**已经发生**的迁移的结果：
    真库 25 行都已有标注，回填的 `NOT EXISTS` 守卫会让它整段跳过。也就是说那组
    测试**测不到回填代码本身**——把回填改坏（比如让 created_at 走当前时间），
    那组测试照样全绿。

    这正是"测试通过"与"测试有效"的区别。所以这里另建一个临时库，让回填真的执行。
    """

    def _fresh_db(self, tmp_path, monkeypatch):
        """建一个只含必要两张表的临时库，并把 database 模块的 engine 指过去。"""
        from sqlalchemy import create_engine

        db_file = tmp_path / "backfill.db"
        engine = create_engine(f"sqlite:///{db_file}")
        with engine.begin() as conn:
            conn.execute(sa_text(
                """
                CREATE TABLE eval_row_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    row_index INTEGER,
                    manual_status VARCHAR(50),
                    manual_score FLOAT,
                    manual_tags JSON,
                    manual_note TEXT,
                    reviewed_at DATETIME
                )
                """
            ))
            conn.execute(sa_text(
                """
                CREATE TABLE row_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    row_result_id INTEGER NOT NULL,
                    annotator VARCHAR(100) NOT NULL,
                    status VARCHAR(50),
                    score FLOAT,
                    tags JSON,
                    note TEXT,
                    is_adjudication BOOLEAN NOT NULL DEFAULT 0,
                    created_at DATETIME,
                    updated_at DATETIME,
                    UNIQUE (row_result_id, annotator, is_adjudication)
                )
                """
            ))

        import app.core.database as database_module

        monkeypatch.setattr(database_module, "engine", engine)
        return engine, database_module

    def test_backfill_copies_reviewed_at_into_created_at(self, tmp_path, monkeypatch):
        """回填必须把 reviewed_at 抄进 created_at，而不是用当前时间。

        投影用 max(created_at) 反推 reviewed_at。若这里走 DB 默认值，
        下一次投影重算会把历史标注的复核时间集体改成迁移时刻。
        """
        engine, database_module = self._fresh_db(tmp_path, monkeypatch)
        with engine.begin() as conn:
            conn.execute(sa_text(
                "INSERT INTO eval_row_results "
                "(row_index, manual_status, manual_tags, reviewed_at) "
                "VALUES (0, 'pass', '[]', '2026-08-06 14:03:38.579189')"
            ))

        database_module._backfill_row_annotations(
            ["eval_row_results", "row_annotations"]
        )

        with engine.begin() as conn:
            got = conn.execute(sa_text(
                "SELECT annotator, status, is_adjudication, created_at "
                "FROM row_annotations"
            )).fetchall()

        assert len(got) == 1
        annotator, status, is_adj, created_at = got[0]
        assert annotator == DEFAULT_ANNOTATOR
        assert status == "pass"
        assert not is_adj
        assert str(created_at).startswith("2026-08-06 14:03:38")

    def test_backfill_is_idempotent_across_repeated_runs(self, tmp_path, monkeypatch):
        """回填在每次进程启动时都会跑，多跑几次不能多插标注。

        不幂等的后果不是"多几行垃圾数据"：同一个人被当成 N 个标注者，
        人-人 kappa 会算出一个虚高的完美一致性（同一个人当然与自己完全一致），
        于是 judge 的上界被抬到 1.0，"judge 越过上界"这件事永远测不出来。
        """
        engine, database_module = self._fresh_db(tmp_path, monkeypatch)
        with engine.begin() as conn:
            conn.execute(sa_text(
                "INSERT INTO eval_row_results "
                "(row_index, manual_status, reviewed_at) "
                "VALUES (0, 'fail', '2026-08-06 14:03:38')"
            ))

        for _ in range(3):
            database_module._backfill_row_annotations(
                ["eval_row_results", "row_annotations"]
            )

        with engine.begin() as conn:
            count = conn.execute(
                sa_text("SELECT COUNT(*) FROM row_annotations")
            ).scalar()
        assert count == 1

    def test_backfill_skips_rows_without_review_timestamp(self, tmp_path, monkeypatch):
        """reviewed_at 为 NULL 的行不搬：给不出可信时间就不迁移。

        若强行搬迁并让 created_at 取当前时间，投影会把这一行的 reviewed_at
        从 NULL 变成迁移时刻——一行从未被人复核过的数据，会凭空长出复核记录。
        """
        engine, database_module = self._fresh_db(tmp_path, monkeypatch)
        with engine.begin() as conn:
            conn.execute(sa_text(
                "INSERT INTO eval_row_results (row_index, manual_status, reviewed_at) "
                "VALUES (0, 'pass', NULL)"
            ))

        database_module._backfill_row_annotations(
            ["eval_row_results", "row_annotations"]
        )

        with engine.begin() as conn:
            count = conn.execute(
                sa_text("SELECT COUNT(*) FROM row_annotations")
            ).scalar()
        assert count == 0

    def test_backfill_does_not_touch_manual_columns(self, tmp_path, monkeypatch):
        """回填只读 manual_*，不写。

        迁移期间这五列同时也是**唯一**数据源。回填顺手重算投影的话，
        一旦逻辑有 bug 就会把原始标注就地改坏，且没有第二份可对照。
        """
        engine, database_module = self._fresh_db(tmp_path, monkeypatch)
        with engine.begin() as conn:
            conn.execute(sa_text(
                "INSERT INTO eval_row_results "
                "(row_index, manual_status, manual_score, manual_tags, manual_note, reviewed_at) "
                "VALUES (0, 'pass', 0.9, '[\"t\"]', 'n', '2026-08-06 14:03:38')"
            ))
            before = conn.execute(sa_text(
                "SELECT manual_status, manual_score, manual_tags, manual_note, reviewed_at "
                "FROM eval_row_results"
            )).fetchall()

        database_module._backfill_row_annotations(
            ["eval_row_results", "row_annotations"]
        )

        with engine.begin() as conn:
            after = conn.execute(sa_text(
                "SELECT manual_status, manual_score, manual_tags, manual_note, reviewed_at "
                "FROM eval_row_results"
            )).fetchall()
        assert before == after
