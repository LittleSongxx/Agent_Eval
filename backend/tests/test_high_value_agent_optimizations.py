from types import SimpleNamespace

from app.core.badcase import classify_badcase
from app.core.trace_lint import lint_agent_trace


def test_trace_lint_catches_registry_and_duplicate_failures():
    registry = [{
        "name": "create_return",
        "parameters_schema": {"required": ["order_id"]},
        "has_side_effect": True,
        "risk_level": "high",
    }]
    trajectory = [
        {"step": 1, "tool": "unknown", "tool_input": {}, "tool_output": "ok"},
        {"step": 2, "tool": "create_return", "tool_input": {}, "tool_output": "failed", "is_error": True},
        {"step": 3, "tool": "create_return", "tool_input": {}, "tool_output": "failed", "is_error": True},
    ]
    result = lint_agent_trace(trajectory, {"create_return": "创建退货"}, registry)
    rules = {item["rule"] for item in result["violations"]}
    assert {"unregistered_tool", "missing_required_arguments", "duplicate_call", "repeated_failed_call", "high_risk_without_confirmation"} <= rules
    assert result["passed"] is False


def test_badcase_classification_prefers_trace_signal():
    info = classify_badcase({
        "answer_relevancy": {"score": 0.2},
        "trace_lint": {"score": 0.4},
    })
    assert info["category"] == "trace_lint"


def test_quality_gate_includes_cost_and_latency_delta(client, db):
    from app.models.dataset import Dataset, DatasetRow
    from app.models.evaluation import EvalRowResult, EvalTask
    from app.models.llm_config import LLMConfig
    from app.models.metric_definition import MetricDefinition
    from app.models.scenario import EvalScenario, ScenarioMetric
    from app.core.scenario_snapshot import build_judge_snapshot, build_scenario_snapshot, compute_eval_fingerprint

    llm = LLMConfig(name="Judge", api_base_url="https://example.com/v1", api_key="key", model_name="model")
    metric = MetricDefinition(name="trace_lint", display_name="Trace Lint", metric_type="builtin_trace_lint", config={}, category="agent", is_builtin=True)
    scenario = EvalScenario(name="Agent", scene_type="agent", sample_type="multi_turn", is_preset=False)
    dataset = Dataset(name="Traces", sample_type="multi_turn", field_schema=[], row_count=1, version=1)
    db.add_all([llm, metric, scenario, dataset]); db.flush()
    scenario.metrics.append(ScenarioMetric(metric_definition_id=metric.id, pass_threshold=0.8, metric_definition=metric)); db.flush()
    snapshot = build_scenario_snapshot(scenario); judge_snapshot = build_judge_snapshot(llm, [])
    fingerprint = compute_eval_fingerprint(snapshot, judge_snapshot, dataset.id, dataset.version)
    row = DatasetRow(dataset_id=dataset.id, row_index=0, data={"user_input": "q"}); db.add(row); db.flush()
    common = dict(dataset_id=dataset.id, scenario_id=scenario.id, llm_config_id=llm.id, status="completed", total_rows=1, completed_rows=1, dataset_version=1, scenario_snapshot=snapshot, judge_snapshot=judge_snapshot, eval_fingerprint=fingerprint)
    baseline = EvalTask(name="base", **common, summary_scores={"trace_lint": {"mean": 1.0, "pass_rate": 1.0}, "cost": {"estimated_cost": 1.0}, "latency": {"row_p95_ms": 100}})
    current = EvalTask(name="current", **common, summary_scores={"trace_lint": {"mean": 1.0, "pass_rate": 1.0}, "cost": {"estimated_cost": 1.4}, "latency": {"row_p95_ms": 180}})
    db.add_all([baseline, current]); db.flush()
    db.add_all([EvalRowResult(eval_task_id=baseline.id, dataset_row_id=row.id, row_index=0, metric_scores={"trace_lint": {"score": 1.0}}, is_pass=True), EvalRowResult(eval_task_id=current.id, dataset_row_id=row.id, row_index=0, metric_scores={"trace_lint": {"score": 1.0}}, is_pass=True)]); db.commit()
    response = client.post(f"/api/reports/{current.id}/quality-gate", json={"baseline_eval_id": baseline.id, "maximum_cost_increase_cny": 0.2, "maximum_latency_p95_increase_ms": 50})
    assert response.status_code == 200
    rules = {item["rule"] for item in response.json()["violations"]}
    assert {"maximum_cost_increase_cny", "maximum_latency_p95_increase_ms"} <= rules
