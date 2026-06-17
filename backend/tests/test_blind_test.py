import asyncio

from app.core.blind_test_engine import run_blind_test
from app.models.evaluation import BlindTestRowResult, BlindTestTask, EvalRowResult, EvalTask
from app.models.llm_config import LLMConfig
from app.models.metric_definition import MetricDefinition
from app.models.scenario import EvalScenario, ScenarioMetric

from .conftest import TestingSessionLocal
from .test_dataset import _create_dataset


def _create_llm_config(client, name="Judge LLM"):
    resp = client.post(
        "/api/llm-configs",
        json={
            "name": name,
            "api_base_url": "https://example.com/v1",
            "api_key": "xxx",
            "model_name": "test-model",
        },
    )
    assert resp.status_code == 201
    return resp.json()


def test_blind_test_flow_with_model_and_endpoint(client, monkeypatch):
    ds_id = _create_dataset(client, name="Blind DS").json()["id"]
    for idx in range(2):
        client.post(
            f"/api/datasets/{ds_id}/rows",
            json={"data": {"user_input": f"问题 {idx + 1}"}},
        )
    llm = _create_llm_config(client, name="Baseline")

    monkeypatch.setattr("app.api.blind_test._launch_blind_test", lambda task_id: None)

    resp = client.post(
        "/api/blind-tests",
        json={
            "name": "A/B 盲测",
            "dataset_id": ds_id,
            "target_a": {
                "target_type": "llm_config",
                "name": "模型 A",
                "llm_config_id": llm["id"],
            },
            "target_b": {
                "target_type": "endpoint",
                "name": "接口 B",
                "endpoint_url": "https://example.com/chat",
                "transport_mode": "json",
                "authorization": "Bearer xxx",
                "request_body_template": '{"question":"{{question}}"}',
            },
        },
    )
    assert resp.status_code == 201
    task = resp.json()
    assert task["total_rows"] == 2
    assert task["target_b"]["authorization"] is None
    assert task["target_b"]["authorization_masked"] == "****"

    async def fake_llm_answer(self, row_data):
        return f"LLM::{row_data['user_input']}"

    async def fake_endpoint_answer(self, row_data):
        return f"ENDPOINT::{row_data['user_input']}"

    monkeypatch.setattr("app.core.blind_test_engine.OpenAIResponseClient.answer", fake_llm_answer)
    monkeypatch.setattr("app.core.blind_test_engine.EndpointResponseClient.answer", fake_endpoint_answer)

    asyncio.run(run_blind_test(task["id"], TestingSessionLocal))

    summary_resp = client.get(f"/api/blind-tests/{task['id']}/summary")
    assert summary_resp.status_code == 200
    summary = summary_resp.json()
    assert summary["completed_count"] == 2
    assert summary["total_count"] == 2

    rows_resp = client.get(f"/api/blind-tests/{task['id']}/rows")
    rows = rows_resp.json()["items"]
    assert rows[0]["answer_a"].startswith("LLM::")
    assert rows[0]["answer_b"].startswith("ENDPOINT::")

    vote_resp = client.post(
        f"/api/blind-tests/{task['id']}/rows/{rows[0]['id']}/vote",
        json={"vote": "left", "vote_note": "回答更完整"},
    )
    assert vote_resp.status_code == 200

    after_vote_summary = client.get(f"/api/blind-tests/{task['id']}/summary").json()
    assert after_vote_summary["voted_count"] == 1
    assert after_vote_summary["model_a_wins"] + after_vote_summary["model_b_wins"] == 1


def test_report_row_manual_review_roundtrip(client, db):
    ds_id = _create_dataset(client, name="Review DS").json()["id"]
    row_resp = client.post(
        f"/api/datasets/{ds_id}/rows",
        json={"data": {"user_input": "你好", "response": "你好", "reference": "你好"}},
    )
    dataset_row_id = row_resp.json()["id"]

    llm = LLMConfig(
        name="Judge Config",
        api_base_url="https://example.com/v1",
        api_key="key",
        model_name="test-model",
    )
    metric = MetricDefinition(
        name="answer_relevancy",
        display_name="Answer Relevancy",
        metric_type="llm_judge",
        category="answer_quality",
        is_builtin=True,
    )
    scenario = EvalScenario(
        name="测试场景",
        description="",
        scene_type="qa",
        sample_type="single_turn",
        is_preset=False,
    )
    db.add_all([llm, metric, scenario])
    db.flush()
    db.add(ScenarioMetric(scenario_id=scenario.id, metric_definition_id=metric.id, weight=1.0, pass_threshold=0.6))
    task = EvalTask(
        name="评测任务",
        dataset_id=ds_id,
        scenario_id=scenario.id,
        llm_config_id=llm.id,
        status="completed",
        total_rows=1,
        completed_rows=1,
    )
    db.add(task)
    db.flush()
    row = EvalRowResult(
        eval_task_id=task.id,
        dataset_row_id=dataset_row_id,
        row_index=0,
        metric_scores={"answer_relevancy": {"score": 0.92, "reason": "回答相关"}},
        is_pass=True,
        execution_time_ms=120,
    )
    db.add(row)
    db.commit()

    review_resp = client.patch(
        f"/api/reports/{task.id}/rows/{row.id}/review",
        json={
            "manual_status": "fail",
            "manual_score": 0.3,
            "manual_tags": ["回答不完整", "需要人工介入"],
            "manual_note": "自动分数偏高，但业务要求没有满足。",
        },
    )
    assert review_resp.status_code == 200
    payload = review_resp.json()
    assert payload["manual_status"] == "fail"
    assert payload["manual_tags"] == ["回答不完整", "需要人工介入"]

    summary = client.get(f"/api/reports/{task.id}/summary").json()
    assert summary["manual_review_summary"]["reviewed_count"] == 1
    assert summary["manual_review_summary"]["manual_fail_count"] == 1


def test_blind_test_target_smoke_endpoint(client, monkeypatch):
    async def fake_endpoint_answer(self, row_data):
        return f"echo::{row_data['user_input']}"

    monkeypatch.setattr("app.core.blind_test_engine.EndpointResponseClient.answer", fake_endpoint_answer)

    resp = client.post(
        "/api/blind-tests/test-target",
        json={
            "target": {
                "target_type": "endpoint",
                "name": "接口测试",
                "endpoint_url": "https://example.com/chat",
                "transport_mode": "json",
                "request_body_template": '{"question":"{{question}}"}',
            },
            "test_question": "你好，请介绍一下自己",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["answer_preview"] == "echo::你好，请介绍一下自己"
