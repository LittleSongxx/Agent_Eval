"""Regression tests for the immutable evaluation input contract."""

import asyncio

from sqlalchemy.orm import sessionmaker

from app.core.evaluation_engine import _MetricResult, run_evaluation
from app.models.dataset import Dataset, DatasetRow
from app.models.evaluation import EvalTask
from app.models.llm_config import LLMConfig


def _prerequisites(client, payload):
    llm = client.post("/api/llm-configs", json=payload).json()
    metric = client.post(
        "/api/metrics",
        json={
            "name": "snapshot_metric",
            "display_name": "Snapshot Metric",
            "metric_type": "aspect_critic",
            "config": {"definition": "判断回答是否合格"},
            "category": "custom",
        },
    ).json()
    scenario = client.post(
        "/api/scenarios",
        json={
            "name": "Snapshot Scenario",
            "scene_type": "general",
            "sample_type": "single_turn",
            "metrics": [{"metric_definition_id": metric["id"], "pass_threshold": 0.5, "weight": 1}],
        },
    ).json()
    dataset = client.post(
        "/api/datasets",
        json={"name": "Snapshot Dataset", "sample_type": "single_turn", "field_schema": []},
    ).json()
    client.post(f"/api/datasets/{dataset['id']}/rows", json={"data": {"user_input": "old question", "response": "old answer"}})
    return llm, scenario, dataset


def test_task_uses_frozen_rows_and_judge_after_mutation(client, db, test_llm_payload, monkeypatch):
    llm, scenario, dataset = _prerequisites(client, test_llm_payload)
    task_response = client.post(
        "/api/evaluations",
        json={
            "name": "Immutable task",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
        },
    )
    assert task_response.status_code == 201
    task_body = task_response.json()
    assert task_body["dataset_snapshot_digest"]
    assert "api_key" not in str(task_body.get("judge_snapshot"))

    row = db.query(DatasetRow).filter(DatasetRow.dataset_id == dataset["id"]).one()
    row.data = {"user_input": "new question", "response": "new answer"}
    config = db.query(LLMConfig).filter(LLMConfig.id == llm["id"]).one()
    config.model_name = "mutated-after-create"
    config.api_key = "mutated-key"
    db.commit()

    seen_rows = []
    seen_judges = []

    class StubJudge:
        def __init__(self, runtime_config):
            seen_judges.append(runtime_config)
            self.row_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def reset_row_usage(self):
            self.row_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def take_row_usage(self):
            return dict(self.row_usage)

    async def fake_score(self, row_data, _judge):
        seen_rows.append(dict(row_data))
        return _MetricResult(1.0, "ok")

    monkeypatch.setattr("app.core.evaluation_engine.OpenAIJudgeClient", StubJudge)
    monkeypatch.setattr("app.core.evaluation_engine.NativePromptMetric.ascore", fake_score)
    TestingSession = sessionmaker(bind=db.get_bind())
    asyncio.run(run_evaluation(task_body["id"], TestingSession))

    assert seen_rows and seen_rows[0]["user_input"] == "old question"
    assert seen_rows[0]["response"] == "old answer"
    assert seen_judges and seen_judges[0].model_name == test_llm_payload["model_name"]
    assert seen_judges[0].api_key == test_llm_payload["api_key"]
    task = db.query(EvalTask).filter(EvalTask.id == task_body["id"]).one()
    assert task.status == "completed"


def test_dataset_version_bumps_when_endpoint_write_back_changes_data(client, db, test_llm_payload, monkeypatch):
    # Reuse the normal endpoint setup pattern with a deterministic metric stub.
    llm, scenario, dataset = _prerequisites(client, test_llm_payload)
    row = db.query(DatasetRow).filter(DatasetRow.dataset_id == dataset["id"]).one()
    async def fake_invoke(_row_data, _target_config):
        return {
            "status_code": 200,
            "latency_ms": 1,
            "request_body": {},
            "raw_response": "{}",
            "parsed_response": {"answer": "generated"},
        }

    class StubJudge:
        def __init__(self, _config):
            self.row_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def reset_row_usage(self):
            pass

        def take_row_usage(self):
            return self.row_usage

    async def fake_score(self, row_data, _judge):
        return _MetricResult(1.0, "ok")

    monkeypatch.setattr("app.core.evaluation_engine.invoke_endpoint", fake_invoke)
    monkeypatch.setattr("app.core.evaluation_engine.OpenAIJudgeClient", StubJudge)
    monkeypatch.setattr("app.core.evaluation_engine.NativePromptMetric.ascore", fake_score)
    task = client.post(
        "/api/evaluations",
        json={
            "name": "writeback version",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
            "evaluation_mode": "endpoint",
            "target_config": {"endpoint_url": "https://example.com/chat"},
            "response_mapping": {"response_path": "answer"},
            "result_save_mode": "write_back",
        },
    ).json()
    TestingSession = sessionmaker(bind=db.get_bind())
    asyncio.run(run_evaluation(task["id"], TestingSession))

    refreshed = db.query(Dataset).filter(Dataset.id == dataset["id"]).one()
    assert refreshed.version == task["dataset_version"] + 1
    assert db.query(DatasetRow).filter(DatasetRow.id == row.id).one().data["response"] == "generated"
