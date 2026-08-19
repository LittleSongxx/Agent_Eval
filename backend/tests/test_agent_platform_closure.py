"""Focused tests for the Agent trace -> evaluation -> quality-gate closure."""

import asyncio
from types import SimpleNamespace

from app.core.agent_trace import normalize_agent_trace
from app.core.endpoint_eval import extract_eval_fields
from app.core.scenario_snapshot import (
    build_judge_snapshot,
    build_scenario_snapshot,
    compute_eval_fingerprint,
)


def _trace(task="查询订单"):
    return {
        "input": task,
        "output": "订单已发货",
        "available_tools": [{"name": "query_order", "description": "查询订单"}],
        "events": [
            {
                "type": "assistant",
                "content": "我先查询订单。",
                "tool_calls": [
                    {"id": "call-1", "name": "query_order", "arguments": {"order_id": "1"}}
                ],
            },
            {
                "type": "tool",
                "tool_call_id": "call-1",
                "name": "query_order",
                "content": '{"status":"shipped"}',
            },
            {"type": "assistant", "content": "订单已发货"},
        ],
        "reference": "查询并告知订单状态",
        "reference_tool_calls": [{"name": "query_order", "args": {"order_id": "1"}}],
    }


def test_normalize_agent_trace_builds_conversation_and_trajectory():
    row = normalize_agent_trace(_trace())

    assert row["user_input"][0] == {"type": "human", "content": "查询订单"}
    assert row["response"] == "订单已发货"
    assert row["tool_calls"][0]["name"] == "query_order"
    assert row["agent_trajectory"][0]["tool_output"] == '{"status":"shipped"}'
    assert row["available_tools"] == {"query_order": "查询订单"}
    assert row["trace_id"].startswith("trace-")


def test_endpoint_mapping_normalizes_events_to_agent_trajectory():
    parsed = {
        "answer": "订单已发货",
        "trace": {
            "events": _trace()["events"],
            "available_tools": _trace()["available_tools"],
        },
    }
    fields, errors = extract_eval_fields(
        {"parsed_response": parsed, "raw_response": ""},
        {
            "response_path": "answer",
            "agent_trajectory_path": "trace.events",
            "available_tools_path": "trace.available_tools",
        },
    )

    assert errors == {}
    assert fields["agent_trajectory"][0]["tool"] == "query_order"
    assert fields["available_tools"]["query_order"] == "查询订单"


def test_trajectory_metrics_are_buildable_without_generic_fallback():
    from app.core.evaluation_engine import ErrorRecoveryMetric, ToolSelectionRationalityMetric, TrajectoryFaithfulnessMetric, build_metric

    for name, metric_type, expected_class in (
        ("trajectory_faithfulness", "builtin_trajectory_faithfulness", TrajectoryFaithfulnessMetric),
        ("error_recovery", "builtin_error_recovery", ErrorRecoveryMetric),
        ("tool_selection_rationality", "builtin_tool_selection_rationality", ToolSelectionRationalityMetric),
    ):
        definition = SimpleNamespace(name=name, display_name=name, metric_type=metric_type, config={})
        kind, metric = build_metric(definition)
        assert kind in {"llm", "simple"}
        assert isinstance(metric, expected_class)

    recovery = ErrorRecoveryMetric(
        SimpleNamespace(name="error_recovery", display_name="recovery", metric_type="builtin_error_recovery", config={})
    )
    result = asyncio.run(recovery.ascore({"agent_trajectory": [{"step": 1, "tool": "q", "tool_output": "ok"}]}, None))
    assert result.value == 1.0

    error_trace = _trace()
    error_trace["events"][1]["content"] = "temporarily unavailable"
    error_trace["events"][1]["is_error"] = True
    normalized = normalize_agent_trace(error_trace)
    assert normalized["agent_trajectory"][0]["is_error"] is True


def test_agent_trace_import_is_atomic_and_deduplicates(client):
    dataset = client.post(
        "/api/datasets",
        json={
            "name": "Agent traces",
            "description": "",
            "sample_type": "multi_turn",
            "field_schema": [
                {"name": "user_input", "type": "conversation", "required": True},
                {"name": "agent_trajectory", "type": "json", "required": True},
                {"name": "available_tools", "type": "json", "required": True},
            ],
        },
    ).json()
    dataset_id = dataset["id"]

    invalid = {"input": "bad", "events": [{"type": "unknown", "content": "x"}]}
    response = client.post(
        f"/api/datasets/{dataset_id}/agent-traces/import",
        json={"traces": [_trace(), invalid]},
    )
    assert response.status_code == 422
    assert client.get(f"/api/datasets/{dataset_id}/rows").json()["total"] == 0

    response = client.post(
        f"/api/datasets/{dataset_id}/agent-traces/import",
        json={"traces": [_trace(), _trace()]},
    )
    assert response.status_code == 200
    assert response.json()["imported_count"] == 1
    assert response.json()["skipped_duplicates"] == 1
    assert response.json()["dataset_version"] == 2


def test_quality_gate_blocks_new_failure(client, db):
    from app.models.dataset import Dataset, DatasetRow
    from app.models.evaluation import EvalRowResult, EvalTask
    from app.models.llm_config import LLMConfig
    from app.models.metric_definition import MetricDefinition
    from app.models.scenario import EvalScenario, ScenarioMetric

    llm = LLMConfig(name="Judge", api_base_url="https://example.com/v1", api_key="key", model_name="model")
    metric = MetricDefinition(
        name="trajectory_faithfulness",
        display_name="Trajectory Faithfulness",
        metric_type="builtin_trajectory_faithfulness",
        config={"required_fields": ["agent_trajectory"]},
        category="agent",
        is_builtin=True,
    )
    scenario = EvalScenario(name="Agent", scene_type="agent", sample_type="multi_turn", is_preset=False)
    dataset = Dataset(name="Traces", sample_type="multi_turn", field_schema=[], row_count=2, version=1)
    db.add_all([llm, metric, scenario, dataset])
    db.flush()
    scenario.metrics.append(ScenarioMetric(metric_definition_id=metric.id, pass_threshold=0.8, metric_definition=metric))
    db.flush()
    snapshot = build_scenario_snapshot(scenario)
    judge_snapshot = build_judge_snapshot(llm, [])
    fingerprint = compute_eval_fingerprint(snapshot, judge_snapshot, dataset.id, dataset.version)
    rows = [
        DatasetRow(dataset_id=dataset.id, row_index=0, data={"user_input": "q1"}),
        DatasetRow(dataset_id=dataset.id, row_index=1, data={"user_input": "q2"}),
    ]
    db.add_all(rows)
    db.flush()
    baseline = EvalTask(
        name="baseline", dataset_id=dataset.id, scenario_id=scenario.id, llm_config_id=llm.id,
        status="completed", total_rows=2, completed_rows=2, dataset_version=1,
        scenario_snapshot=snapshot, judge_snapshot=judge_snapshot, eval_fingerprint=fingerprint,
        summary_scores={"trajectory_faithfulness": {"mean": 1.0, "pass_rate": 1.0}},
    )
    current = EvalTask(
        name="current", dataset_id=dataset.id, scenario_id=scenario.id, llm_config_id=llm.id,
        status="completed", total_rows=2, completed_rows=2, dataset_version=1,
        scenario_snapshot=snapshot, judge_snapshot=judge_snapshot, eval_fingerprint=fingerprint,
        summary_scores={"trajectory_faithfulness": {"mean": 0.5, "pass_rate": 0.5}},
    )
    db.add_all([baseline, current])
    db.flush()
    db.add_all([
        EvalRowResult(eval_task_id=baseline.id, dataset_row_id=rows[0].id, row_index=0, metric_scores={"trajectory_faithfulness": {"score": 1.0}}, is_pass=True),
        EvalRowResult(eval_task_id=baseline.id, dataset_row_id=rows[1].id, row_index=1, metric_scores={"trajectory_faithfulness": {"score": 1.0}}, is_pass=True),
        EvalRowResult(eval_task_id=current.id, dataset_row_id=rows[0].id, row_index=0, metric_scores={"trajectory_faithfulness": {"score": 0.5}}, is_pass=False),
        EvalRowResult(eval_task_id=current.id, dataset_row_id=rows[1].id, row_index=1, metric_scores={"trajectory_faithfulness": {"score": 0.5}}, is_pass=False),
    ])
    db.commit()

    response = client.post(
        f"/api/reports/{current.id}/quality-gate",
        json={"baseline_eval_id": baseline.id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["passed"] is False
    assert body["status"] == "blocked"
    assert any(item["rule"] == "maximum_new_failures" for item in body["violations"])
