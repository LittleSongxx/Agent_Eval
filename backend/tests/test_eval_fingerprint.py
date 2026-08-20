"""评测口径指纹（eval fingerprint）回归测试。

背景（P0-1 缺陷）：判定标准 `criteria` 原先只以模块级 Python 常量存在于
`prompt_manager`，既不落库也不进任务快照。运行时 `NativeBuiltinLLMMetric`
直接读当前代码里的常量，因此改一次常量就会让历史任务的口径静默漂移——任务
本身完全看不出变化，`dataset_version` 冻结了数据却没冻结尺子。

真实事故：commit 13112ee 改了 answer_relevancy 的 criteria，正是 kappa
0.4582 → 0.8175 的那次修订。任务 10 与任务 13 用的是两把不同的尺子，而库里
没有任何记录能区分它们。

一句话：rubric = 数据 + 尺子 + 裁判。这些测试锁住后两者。
"""
from types import SimpleNamespace

import pytest

from app.core.prompt_manager import BUILTIN_LLM_METRIC_SPECS
from app.core.scenario_snapshot import (
    build_judge_snapshot,
    build_scenario_snapshot,
    compute_eval_fingerprint,
    describe_fingerprint_diff,
    engine_prompt_digest,
    resolve_effective_criteria,
    snapshot_to_scenario_metrics,
)

RELEVANCY_TYPE = "builtin_answer_relevancy"


def _fake_scenario(
    metric_type: str = RELEVANCY_TYPE,
    metric_name: str = "answer_relevancy",
    prompt_override: str | None = None,
    pass_threshold: float | None = 0.7,
    weight: float | None = 1.0,
):
    """Minimal duck-typed scenario — build_scenario_snapshot never touches the DB."""
    metric_definition = SimpleNamespace(
        id=1,
        name=metric_name,
        display_name="Answer Relevancy",
        metric_type=metric_type,
        config={},
        category="rag",
        is_builtin=True,
    )
    scenario_metric = SimpleNamespace(
        id=1,
        metric_definition_id=1,
        weight=weight,
        pass_threshold=pass_threshold,
        prompt_override=prompt_override,
        metric_definition=metric_definition,
    )
    return SimpleNamespace(
        id=10,
        name="RAG",
        description="",
        scene_type="rag",
        sample_type="single_turn",
        is_preset=False,
        metrics=[scenario_metric],
    )


def _build_metric_from_snapshot(snapshot):
    from app.core.evaluation_engine import build_metric

    hydrated = snapshot_to_scenario_metrics(snapshot)
    scenario_metric = hydrated[0]
    _kind, metric = build_metric(
        scenario_metric.metric_definition,
        prompt_override=scenario_metric.prompt_override,
        scenario_metric=scenario_metric,
    )
    return metric


# --------------------------------------------------------------------------
# criteria 冻结
# --------------------------------------------------------------------------

def test_snapshot_freezes_effective_criteria():
    snapshot = build_scenario_snapshot(_fake_scenario())
    frozen = snapshot["metrics"][0]["effective_criteria"]

    assert frozen, "内建 LLM 指标必须把判定标准落进快照"
    assert frozen == BUILTIN_LLM_METRIC_SPECS[RELEVANCY_TYPE]["criteria"]


def test_task_criteria_survives_prompt_manager_edit(monkeypatch):
    """P0-1 核心回归：改 prompt_manager 的常量不得改动既有任务的判定标准。

    这是整个修复的意义所在——历史任务必须可复现。
    """
    snapshot = build_scenario_snapshot(_fake_scenario())
    original = snapshot["metrics"][0]["effective_criteria"]

    monkeypatch.setitem(
        BUILTIN_LLM_METRIC_SPECS[RELEVANCY_TYPE], "criteria", "改过的判定标准"
    )

    # 守卫断言：先证明 patch 真的生效了。少了这句，本测试就可能因为"patch 根本
    # 没起作用"而通过——那正是本项目反复强调的同义反复实验。
    assert (
        build_scenario_snapshot(_fake_scenario())["metrics"][0]["effective_criteria"]
        == "改过的判定标准"
    ), "新建任务应当采用改动后的标准，否则本测试无法证伪"

    # 真正要断言的：已存在的任务仍然按当初冻结的标准打分。
    metric = _build_metric_from_snapshot(snapshot)
    assert metric.criteria == original
    assert metric.criteria != "改过的判定标准"


def test_prompt_override_beats_spec_constant():
    snapshot = build_scenario_snapshot(_fake_scenario(prompt_override="  只看是否跑题  "))

    assert snapshot["metrics"][0]["effective_criteria"] == "只看是否跑题"
    assert _build_metric_from_snapshot(snapshot).criteria == "只看是否跑题"


def test_deterministic_metric_has_no_criteria():
    """检索类指标是纯确定性计算，没有判定标准可言。"""
    snapshot = build_scenario_snapshot(
        _fake_scenario(metric_type="code_retrieval_hit_rate", metric_name="hit_rate")
    )

    assert snapshot["metrics"][0]["effective_criteria"] is None
    assert resolve_effective_criteria("hit_rate", "code_retrieval_hit_rate", None) is None


def test_legacy_snapshot_without_criteria_still_scores(monkeypatch):
    """快照机制上线前创建的任务没有 effective_criteria，必须仍能回落执行。"""
    snapshot = build_scenario_snapshot(_fake_scenario())
    snapshot["metrics"][0].pop("effective_criteria")

    metric = _build_metric_from_snapshot(snapshot)
    assert metric.criteria == BUILTIN_LLM_METRIC_SPECS[RELEVANCY_TYPE]["criteria"]


# --------------------------------------------------------------------------
# 引擎自带 prompt
# --------------------------------------------------------------------------

def test_engine_prompt_digest_separates_composite_metrics():
    """复合指标不读 criteria，它们的口径在模块级 prompt 常量里。"""
    generic = engine_prompt_digest(RELEVANCY_TYPE)
    claim = engine_prompt_digest("builtin_faithfulness_claim")
    generative = engine_prompt_digest("builtin_answer_relevancy_generative")

    assert len({generic, claim, generative}) == 3, "三条执行路径的 prompt 集合不同，摘要必须可区分"


def test_engine_prompt_digest_tracks_judge_cot_mode(monkeypatch):
    """裁判系统提示词换成 CoT 版会改变所有 LLM 指标的行为，指纹要能反映。"""
    from app.core.config import settings

    monkeypatch.setattr(settings, "JUDGE_COT_MODE", False)
    plain = engine_prompt_digest(RELEVANCY_TYPE)
    monkeypatch.setattr(settings, "JUDGE_COT_MODE", True)
    cot = engine_prompt_digest(RELEVANCY_TYPE)

    assert plain != cot


# --------------------------------------------------------------------------
# 裁判快照
# --------------------------------------------------------------------------

def test_judge_snapshot_never_contains_api_key():
    llm_config = SimpleNamespace(
        id=3,
        name="Qwen Judge",
        provider="dashscope",
        api_base_url="https://example.com/v1",
        api_key="sk-must-not-leak",
        model_name="qwen-plus",
        temperature=0.01,
        max_tokens=1024,
    )

    snapshot = build_judge_snapshot(llm_config, judge_panel=[4, 5])

    assert "api_key" not in snapshot
    assert "sk-must-not-leak" not in str(snapshot)
    assert snapshot["model_name"] == "qwen-plus"
    assert snapshot["judge_panel"] == [4, 5]


def test_judge_panel_snapshot_records_model_without_secret():
    primary = SimpleNamespace(
        id=1, name="Primary", provider="p", api_base_url="u", api_key="primary-key",
        model_name="qwen-plus", temperature=0.01, max_tokens=1024,
    )
    panel = SimpleNamespace(
        id=4, name="Panel", provider="p", api_base_url="u2", api_key="panel-key",
        model_name="qwen-max", temperature=0.2, max_tokens=2048,
    )
    snapshot = build_judge_snapshot(primary, [4], [panel])
    assert snapshot["judge_panel_snapshots"][0]["model_name"] == "qwen-max"
    assert "api_key" not in snapshot["judge_panel_snapshots"][0]


# --------------------------------------------------------------------------
# 指纹
# --------------------------------------------------------------------------

def _fingerprint(scenario=None, judge=None, dataset_version=1, judge_samples=1):
    return compute_eval_fingerprint(
        build_scenario_snapshot(scenario or _fake_scenario()),
        judge
        or build_judge_snapshot(
            SimpleNamespace(
                id=1,
                name="Judge",
                provider="dashscope",
                api_base_url="https://example.com/v1",
                api_key="k",
                model_name="qwen-plus",
                temperature=0.01,
                max_tokens=1024,
            )
        ),
        dataset_id=1,
        dataset_version=dataset_version,
        judge_samples=judge_samples,
    )


def test_fingerprint_is_stable_for_identical_config():
    assert _fingerprint() == _fingerprint()


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"dataset_version": 2}, id="dataset_version"),
        pytest.param({"judge_samples": 3}, id="judge_samples"),
        pytest.param({"scenario": _fake_scenario(pass_threshold=0.8)}, id="pass_threshold"),
        pytest.param({"scenario": _fake_scenario(weight=2.0)}, id="weight"),
        pytest.param({"scenario": _fake_scenario(prompt_override="别的标准")}, id="criteria"),
    ],
)
def test_fingerprint_changes_when_any_rubric_dimension_moves(kwargs):
    assert _fingerprint(**kwargs) != _fingerprint()


def test_fingerprint_changes_when_judge_model_changes():
    other_judge = build_judge_snapshot(
        SimpleNamespace(
            id=1,
            name="Judge",
            provider="dashscope",
            api_base_url="https://example.com/v1",
            api_key="k",
            model_name="qwen-max",  # 换了裁判
            temperature=0.01,
            max_tokens=1024,
        )
    )
    assert _fingerprint(judge=other_judge) != _fingerprint()


def test_fingerprint_changes_when_panel_model_changes():
    primary = SimpleNamespace(
        id=1, name="Primary", provider="p", api_base_url="u", api_key="k",
        model_name="qwen-plus", temperature=0.01, max_tokens=1024,
    )
    panel_a = SimpleNamespace(
        id=4, name="Panel", provider="p", api_base_url="u", api_key="k",
        model_name="qwen-plus", temperature=0.01, max_tokens=1024,
    )
    panel_b = SimpleNamespace(
        id=4, name="Panel", provider="p", api_base_url="u", api_key="k",
        model_name="qwen-max", temperature=0.01, max_tokens=1024,
    )
    scenario = build_scenario_snapshot(_fake_scenario())
    assert compute_eval_fingerprint(
        scenario, build_judge_snapshot(primary, [4], [panel_a]), 1, 1
    ) != compute_eval_fingerprint(
        scenario, build_judge_snapshot(primary, [4], [panel_b]), 1, 1
    )


def test_fingerprint_changes_when_same_version_row_content_moves():
    """A direct JSON edit must not hide behind an unchanged Dataset.version."""
    from app.core.scenario_snapshot import dataset_snapshot_digest

    judge = build_judge_snapshot(
        SimpleNamespace(
            id=1, name="Judge", provider="p", api_base_url="u", api_key="k",
            model_name="qwen-plus", temperature=0.01, max_tokens=1024,
        )
    )
    rows_a = [{"id": 1, "row_index": 0, "data": {"response": "old"}}]
    rows_b = [{"id": 1, "row_index": 0, "data": {"response": "new"}}]
    assert dataset_snapshot_digest(rows_a) != dataset_snapshot_digest(rows_b)
    assert compute_eval_fingerprint(
        build_scenario_snapshot(_fake_scenario()), judge, 1, 1, dataset_snapshot=rows_a
    ) != compute_eval_fingerprint(
        build_scenario_snapshot(_fake_scenario()), judge, 1, 1, dataset_snapshot=rows_b
    )


def test_fingerprint_ignores_judge_config_rename():
    """改配置显示名不影响打分，不该让历史任务显示为不可比。"""
    renamed = build_judge_snapshot(
        SimpleNamespace(
            id=1,
            name="Renamed Judge",
            provider="dashscope",
            api_base_url="https://example.com/v1",
            api_key="k",
            model_name="qwen-plus",
            temperature=0.01,
            max_tokens=1024,
        )
    )
    assert _fingerprint(judge=renamed) == _fingerprint()


def test_fingerprint_reacts_to_prompt_manager_edit(monkeypatch):
    """常量漂移必须体现在指纹上——否则两次不同口径的评测会被当作可比。"""
    before = _fingerprint()
    monkeypatch.setitem(BUILTIN_LLM_METRIC_SPECS[RELEVANCY_TYPE], "criteria", "改过的判定标准")
    assert _fingerprint() != before


# --------------------------------------------------------------------------
# 可比性差异说明
# --------------------------------------------------------------------------

def test_describe_fingerprint_diff_names_changed_dimensions():
    baseline = {
        "dataset_version": 1,
        "judge_samples": 1,
        "judge_snapshot": build_judge_snapshot(
            SimpleNamespace(
                id=1, name="J", provider="p", api_base_url="u",
                api_key="k", model_name="qwen-plus", temperature=0.01, max_tokens=1024,
            )
        ),
        "scenario_snapshot": build_scenario_snapshot(_fake_scenario()),
    }
    current = {
        "dataset_version": 2,
        "judge_samples": 3,
        "judge_snapshot": build_judge_snapshot(
            SimpleNamespace(
                id=1, name="J", provider="p", api_base_url="u",
                api_key="k", model_name="qwen-max", temperature=0.01, max_tokens=1024,
            )
        ),
        "scenario_snapshot": build_scenario_snapshot(
            _fake_scenario(prompt_override="新标准", pass_threshold=0.9)
        ),
    }

    changed = describe_fingerprint_diff(current, baseline)

    assert "dataset_version" in changed
    assert "judge.samples" in changed
    assert "judge.model_name" in changed
    assert "criteria.answer_relevancy" in changed
    assert "threshold.answer_relevancy" in changed


def test_describe_fingerprint_diff_empty_when_comparable():
    snapshot = {
        "dataset_version": 1,
        "judge_snapshot": build_judge_snapshot(
            SimpleNamespace(
                id=1, name="J", provider="p", api_base_url="u",
                api_key="k", model_name="qwen-plus", temperature=0.01, max_tokens=1024,
            )
        ),
        "scenario_snapshot": build_scenario_snapshot(_fake_scenario()),
    }
    assert describe_fingerprint_diff(snapshot, dict(snapshot)) == []


# --------------------------------------------------------------------------
# 端到端：任务创建落库
# --------------------------------------------------------------------------

def test_create_evaluation_persists_fingerprint_and_judge_snapshot(client, db, test_llm_payload):
    from app.models.evaluation import EvalTask

    llm = client.post("/api/llm-configs", json=test_llm_payload).json()
    metric = client.post(
        "/api/metrics",
        json={
            "name": "answer_relevancy",
            "display_name": "Answer Relevancy",
            "metric_type": RELEVANCY_TYPE,
            "config": {},
            "category": "rag",
        },
    ).json()
    scenario = client.post(
        "/api/scenarios",
        json={
            "name": "Fingerprint Scenario",
            "description": "",
            "scene_type": "rag",
            "sample_type": "single_turn",
            "metrics": [
                {"metric_definition_id": metric["id"], "weight": 1.0, "pass_threshold": 0.7}
            ],
        },
    ).json()
    dataset = client.post(
        "/api/datasets",
        json={
            "name": "Fingerprint Dataset",
            "sample_type": "single_turn",
            "field_schema": [
                {"name": "user_input", "type": "text", "required": True, "description": ""},
                {"name": "response", "type": "text", "required": False, "description": ""},
            ],
        },
    ).json()
    client.post(
        f"/api/datasets/{dataset['id']}/rows",
        json={"data": {"user_input": "Q", "response": "A"}},
    )

    resp = client.post(
        "/api/evaluations",
        json={
            "name": "Fingerprint Eval",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
        },
    )
    assert resp.status_code == 201

    task = db.query(EvalTask).filter(EvalTask.id == resp.json()["id"]).first()

    assert task.eval_fingerprint and len(task.eval_fingerprint) == 64
    assert task.judge_snapshot["model_name"] == test_llm_payload["model_name"]
    assert "api_key" not in task.judge_snapshot
    assert task.scenario_snapshot["metrics"][0]["effective_criteria"]


# ---------------------------------------------------------------------------
# 对比接口的口径可比性
# ---------------------------------------------------------------------------


def _seed_compare_pair(db, *, baseline_criteria: str | None = None):
    """Two completed tasks over the same rows; baseline optionally graded by another ruler."""
    from app.models.dataset import Dataset, DatasetRow
    from app.models.evaluation import EvalRowResult, EvalTask
    from app.models.llm_config import LLMConfig
    from app.models.metric_definition import MetricDefinition
    from app.models.scenario import EvalScenario, ScenarioMetric

    llm = LLMConfig(
        name="Judge",
        api_base_url="https://example.com/v1",
        api_key="secret-key",
        model_name="qwen-plus",
        temperature=0.01,
        max_tokens=1024,
    )
    metric = MetricDefinition(
        name="answer_relevancy",
        display_name="Answer Relevancy",
        metric_type=RELEVANCY_TYPE,
        config={},
        category="rag",
        is_builtin=True,
    )
    scenario = EvalScenario(name="RAG", scene_type="rag", sample_type="single_turn", is_preset=False)
    dataset = Dataset(name="Rows", sample_type="single_turn", field_schema=[], row_count=2, version=1)
    db.add_all([llm, metric, scenario, dataset])
    db.flush()
    scenario.metrics.append(
        ScenarioMetric(metric_definition_id=metric.id, pass_threshold=0.7, metric_definition=metric)
    )
    db.flush()

    snapshot = build_scenario_snapshot(scenario)
    judge_snapshot = build_judge_snapshot(llm, [])
    fingerprint = compute_eval_fingerprint(snapshot, judge_snapshot, dataset.id, dataset.version)

    baseline_snapshot = snapshot
    baseline_fingerprint = fingerprint
    if baseline_criteria is not None:
        # 模拟"基线跑的是另一把尺子"：只改判定标准，其余维度全同。
        import copy

        baseline_snapshot = copy.deepcopy(snapshot)
        baseline_snapshot["metrics"][0]["effective_criteria"] = baseline_criteria
        baseline_fingerprint = compute_eval_fingerprint(
            baseline_snapshot, judge_snapshot, dataset.id, dataset.version
        )

    rows = [
        DatasetRow(dataset_id=dataset.id, row_index=0, data={"user_input": "Q1"}),
        DatasetRow(dataset_id=dataset.id, row_index=1, data={"user_input": "Q2"}),
    ]
    db.add_all(rows)
    db.flush()

    def _task(name, snap, fp):
        return EvalTask(
            name=name,
            dataset_id=dataset.id,
            scenario_id=scenario.id,
            llm_config_id=llm.id,
            status="completed",
            total_rows=2,
            completed_rows=2,
            dataset_version=dataset.version,
            scenario_snapshot=snap,
            judge_snapshot=judge_snapshot,
            eval_fingerprint=fp,
            summary_scores={"answer_relevancy": {"mean": 0.8, "pass_rate": 0.5, "error_count": 0}},
        )

    baseline = _task("Baseline", baseline_snapshot, baseline_fingerprint)
    current = _task("Current", snapshot, fingerprint)
    db.add_all([baseline, current])
    db.flush()
    for task in (baseline, current):
        db.add_all(
            [
                EvalRowResult(
                    eval_task_id=task.id,
                    dataset_row_id=rows[0].id,
                    row_index=0,
                    metric_scores={"answer_relevancy": {"score": 0.9}},
                    is_pass=True,
                ),
                EvalRowResult(
                    eval_task_id=task.id,
                    dataset_row_id=rows[1].id,
                    row_index=1,
                    metric_scores={"answer_relevancy": {"score": 0.4}},
                    is_pass=False,
                ),
            ]
        )
    db.commit()
    return current, baseline


def test_compare_marks_same_ruler_as_attribution_safe(client, db):
    current, baseline = _seed_compare_pair(db)

    resp = client.get(f"/api/reports/{current.id}/compare?baseline_eval_id={baseline.id}")

    assert resp.status_code == 200
    comparability = resp.json()["comparability"]
    assert comparability["status"] == "identical"
    assert comparability["attribution_safe"] is True
    assert comparability["changed_dimensions"] == []
    assert comparability["warning"] is None


def test_compare_flags_criteria_change_and_refuses_attribution(client, db):
    """核心断言：换了尺子时，delta 不得被默认当成系统变好。"""
    current, baseline = _seed_compare_pair(db, baseline_criteria="旧版判定标准")

    resp = client.get(f"/api/reports/{current.id}/compare?baseline_eval_id={baseline.id}")

    assert resp.status_code == 200
    comparability = resp.json()["comparability"]
    assert comparability["status"] == "changed"
    assert comparability["attribution_safe"] is False
    assert "criteria.answer_relevancy" in comparability["changed_dimensions"]
    assert comparability["warning"]
    # 对比本身仍然可用——跨口径对比是合理需求，只是必须交底。
    assert resp.json()["summary_delta"]["pass_rate_delta"] == 0.0


def test_compare_reports_unknown_for_pre_fingerprint_tasks(client, db):
    """指纹上线前的历史任务无法追溯口径，只能说不确定，不能假装一致。"""
    current, baseline = _seed_compare_pair(db)
    baseline.eval_fingerprint = None
    db.commit()

    resp = client.get(f"/api/reports/{current.id}/compare?baseline_eval_id={baseline.id}")

    assert resp.status_code == 200
    comparability = resp.json()["comparability"]
    assert comparability["status"] == "unknown"
    assert comparability["attribution_safe"] is False
    assert comparability["warning"]


def test_compare_never_leaks_judge_api_key(client, db):
    """judge_snapshot 会经对比接口出网，必须确认里面没有密钥。"""
    current, baseline = _seed_compare_pair(db)

    resp = client.get(f"/api/reports/{current.id}/compare?baseline_eval_id={baseline.id}")

    assert resp.status_code == 200
    assert "secret-key" not in resp.text
