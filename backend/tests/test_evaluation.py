import pytest


def _setup_eval_prerequisites(client, test_llm_payload):
    """Create LLM config, metric, scenario, and dataset with one row.

    Returns (llm_config, dataset, scenario) response dicts.
    """
    llm = client.post(
        "/api/llm-configs",
        json=test_llm_payload,
    ).json()

    metric = client.post(
        "/api/metrics",
        json={
            "name": "test_metric",
            "display_name": "Test Metric",
            "metric_type": "aspect_critic",
            "config": {"definition": "Is the response correct?"},
            "category": "custom",
        },
    ).json()

    scenario = client.post(
        "/api/scenarios",
        json={
            "name": "Test Scenario",
            "description": "test",
            "scene_type": "rag",
            "sample_type": "single_turn",
            "metrics": [
                {
                    "metric_definition_id": metric["id"],
                    "weight": 1.0,
                    "pass_threshold": 0.5,
                }
            ],
        },
    ).json()

    dataset = client.post(
        "/api/datasets",
        json={
            "name": "Test Dataset",
            "sample_type": "single_turn",
            "field_schema": [
                {"name": "user_input", "type": "text", "required": True, "description": ""},
                {"name": "response", "type": "text", "required": False, "description": ""},
            ],
        },
    ).json()

    client.post(
        f"/api/datasets/{dataset['id']}/rows",
        json={"data": {"user_input": "What is Python?", "response": "A programming language"}},
    )

    return llm, dataset, scenario


def test_create_evaluation(client, test_llm_payload):
    llm, dataset, scenario = _setup_eval_prerequisites(client, test_llm_payload)
    resp = client.post(
        "/api/evaluations",
        json={
            "name": "Test Eval",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Test Eval"
    assert body["status"] in ("pending", "running")
    assert body["dataset_id"] == dataset["id"]
    assert body["scenario_id"] == scenario["id"]
    assert body["llm_config_id"] == llm["id"]


def test_prompt_renderer_supports_namespaced_variables():
    from types import SimpleNamespace

    from app.core.evaluation_engine import build_template_context
    from app.core.prompt_manager import render_prompt

    context = build_template_context(
        {
            "user_input": "问题",
            "response": "旧回答",
            "_dataset_data": {"user_input": "问题", "reference": "标准答案"},
            "_endpoint_trace": {
                "status_code": 200,
                "latency_ms": 12,
                "raw_response": '{"answer":"接口回答"}',
                "extracted_fields": {"response": "接口回答"},
            },
        },
        {
            "status_code": 200,
            "latency_ms": 12,
            "raw_response": '{"answer":"接口回答"}',
            "extracted_fields": {"response": "接口回答"},
        },
        SimpleNamespace(
            id=1,
            name="answer_quality",
            display_name="回答质量",
            metric_type="numeric",
            config={"allowed_values": [0, 1], "description": "质量"},
        ),
        SimpleNamespace(pass_threshold=0.7, weight=1.0),
    )

    rendered = render_prompt(
        "{dataset.user_input}|{dataset.reference}|{endpoint.response}|"
        "{endpoint.status_code}|{metric.display_name}|{metric.pass_threshold}|{response}",
        context,
    )

    assert rendered == "问题|标准答案|接口回答|200|回答质量|0.7|接口回答"


def test_create_evaluation_freezes_metric_overrides(client, test_llm_payload):
    llm, dataset, scenario = _setup_eval_prerequisites(client, test_llm_payload)
    metric_definition_id = scenario["metrics"][0]["metric_definition_id"]

    resp = client.post(
        "/api/evaluations",
        json={
            "name": "Override Eval",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
            "metric_overrides": [
                {
                    "metric_definition_id": metric_definition_id,
                    "prompt_override": "本次实验规则：{dataset.user_input} / {response}",
                    "pass_threshold": 0.8,
                    "weight": 2.0,
                }
            ],
        },
    )

    assert resp.status_code == 201
    metric_snapshot = resp.json()["scenario_snapshot"]["metrics"][0]
    assert metric_snapshot["prompt_override"] == "本次实验规则：{dataset.user_input} / {response}"
    assert metric_snapshot["pass_threshold"] == 0.8
    assert metric_snapshot["weight"] == 2.0


def test_debug_uses_metric_override_prompt(client, test_llm_payload, monkeypatch):
    import json

    from app.core import evaluation_engine

    class FakeJudgeClient:
        def __init__(self, _llm_config):
            self.last_messages = None
            self.last_raw_response = None

        async def judge_json(self, payload):
            self.last_messages = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
            self.last_raw_response = '{"score": 1, "reason": "ok"}'
            return {"score": 1, "reason": "ok"}

    monkeypatch.setattr(evaluation_engine, "OpenAIJudgeClient", FakeJudgeClient)

    llm, dataset, scenario = _setup_eval_prerequisites(client, test_llm_payload)
    metric_definition_id = scenario["metrics"][0]["metric_definition_id"]
    resp = client.post(
        "/api/evaluations/debug",
        json={
            "name": "Debug Override",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
            "metric_overrides": [
                {
                    "metric_definition_id": metric_definition_id,
                    "prompt_override": "覆盖规则：{dataset.user_input} / {response} / {metric.display_name}",
                }
            ],
        },
    )

    body = resp.json()
    assert resp.status_code == 200
    prompt_content = body["judge_traces"][0]["prompt_messages"][0]["content"]
    assert "覆盖规则" in prompt_content
    assert "What is Python?" in prompt_content
    assert "A programming language" in prompt_content
    assert "Test Metric" in prompt_content


def test_debug_warns_when_aspect_prompt_uses_numeric_ranges(client, test_llm_payload, monkeypatch):
    from app.core import evaluation_engine

    class FakeJudgeClient:
        last_messages = [{"role": "user", "content": "fake"}]
        last_raw_response = '{"score": 1, "reason": "ok"}'

        def __init__(self, _llm_config):
            pass

        async def judge_json(self, _payload):
            return {"score": 1, "reason": "ok"}

    monkeypatch.setattr(evaluation_engine, "OpenAIJudgeClient", FakeJudgeClient)

    llm = client.post("/api/llm-configs", json=test_llm_payload).json()
    metric = client.post(
        "/api/metrics",
        json={
            "name": "debug_aspect_conflict",
            "display_name": "调试冲突指标",
            "metric_type": "aspect_critic",
            "category": "custom",
            "config": {
                "definition": "如果回答准确给 0.8 到 1.0 分，否则给 0 到 0.49 分。回答：{response}",
                "description": "conflict",
            },
        },
    ).json()
    scenario = client.post(
        "/api/scenarios",
        json={
            "name": "Debug Conflict",
            "scene_type": "general",
            "sample_type": "single_turn",
            "metrics": [{"metric_definition_id": metric["id"], "weight": 1, "pass_threshold": 0.7}],
        },
    ).json()
    dataset = client.post(
        "/api/datasets",
        json={
            "name": "Debug DS",
            "description": "",
            "sample_type": "single_turn",
            "field_schema": [
                {"name": "user_input", "type": "text", "required": True, "description": ""},
            ],
        },
    ).json()
    client.post(
        f"/api/datasets/{dataset['id']}/rows",
        json={"data": {"user_input": "Q"}},
    )

    resp = client.post(
        "/api/evaluations/debug",
        json={
            "name": "debug",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
            "evaluation_mode": "offline",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert any("0/1 判断" in warning for warning in body["warnings"])
    assert any("{response}" in warning for warning in body["warnings"])


def test_list_evaluations(client, test_llm_payload):
    llm, dataset, scenario = _setup_eval_prerequisites(client, test_llm_payload)
    for i in range(2):
        client.post(
            "/api/evaluations",
            json={
                "name": f"Eval {i}",
                "dataset_id": dataset["id"],
                "scenario_id": scenario["id"],
                "llm_config_id": llm["id"],
            },
        )
    resp = client.get("/api/evaluations")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) >= 2


def test_create_evaluation_uses_actual_dataset_row_count(client, db, test_llm_payload):
    """Task total_rows should use actual DatasetRow count, not stale Dataset.row_count."""
    llm, dataset, scenario = _setup_eval_prerequisites(client, test_llm_payload)
    client.post(
        f"/api/datasets/{dataset['id']}/rows",
        json={"data": {"user_input": "Second row", "response": "Another answer"}},
    )

    from app.models.dataset import Dataset

    db_dataset = db.query(Dataset).filter(Dataset.id == dataset["id"]).first()
    db_dataset.row_count = 1
    db.commit()

    resp = client.post(
        "/api/evaluations",
        json={
            "name": "Count Sync Eval",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["total_rows"] == 2


def test_create_endpoint_evaluation_requires_endpoint_url(client, test_llm_payload):
    llm, dataset, scenario = _setup_eval_prerequisites(client, test_llm_payload)
    resp = client.post(
        "/api/evaluations",
        json={
            "name": "Endpoint Eval",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
            "evaluation_mode": "endpoint",
            "target_config": {"transport_mode": "json"},
        },
    )
    assert resp.status_code == 400


def test_endpoint_target_crud_and_evaluation_snapshot(client, test_llm_payload):
    llm, dataset, scenario = _setup_eval_prerequisites(client, test_llm_payload)
    target_resp = client.post(
        "/api/endpoint-targets",
        json={
            "name": "DeepSeek Chat",
            "endpoint_url": "https://api.deepseek.com/v1/chat/completions",
            "transport_mode": "json",
            "authorization": "Bearer xxx",
            "extra_headers": "{}",
            "request_body_template": '{"model":"deepseek-chat","messages":[{"role":"user","content":"{{user_input}}"}]}',
            "response_mapping": {"response_path": "choices.0.message.content"},
            "default_test_input": "介绍 DeepSeek",
        },
    )
    assert target_resp.status_code == 201
    target = target_resp.json()
    assert target["authorization"] is None
    assert target["authorization_masked"] == "****"

    task_resp = client.post(
        "/api/evaluations",
        json={
            "name": "Endpoint Target Eval",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
            "evaluation_mode": "endpoint",
            "endpoint_target_id": target["id"],
            "result_save_mode": "task_only",
        },
    )
    assert task_resp.status_code == 201
    task = task_resp.json()
    assert task["endpoint_target_id"] == target["id"]
    assert task["target_config"]["endpoint_url"] == "https://api.deepseek.com/v1/chat/completions"
    assert task["target_config"]["authorization"] is None
    assert task["target_config"]["authorization_masked"] == "****"
    assert task["response_mapping"]["response_path"] == "choices.0.message.content"

    client.put(
        f"/api/endpoint-targets/{target['id']}",
        json={"endpoint_url": "https://example.com/changed"},
    )
    existing_task = client.get(f"/api/evaluations/{task['id']}").json()
    assert existing_task["target_config"]["endpoint_url"] == "https://api.deepseek.com/v1/chat/completions"
    assert existing_task["target_config"]["authorization"] is None


def test_endpoint_target_update_keeps_authorization_when_blank(client, db, monkeypatch):
    target_resp = client.post(
        "/api/endpoint-targets",
        json={
            "name": "Private API",
            "endpoint_url": "https://example.com/chat",
            "transport_mode": "json",
            "authorization": "Bearer xxx",
            "extra_headers": "{}",
            "request_body_template": '{"question":"{{user_input}}"}',
            "response_mapping": {"response_path": "data.answer"},
        },
    )
    assert target_resp.status_code == 201

    update_resp = client.put(
        f"/api/endpoint-targets/{target_resp.json()['id']}",
        json={"description": "updated", "authorization": ""},
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["authorization"] is None
    assert update_resp.json()["authorization_masked"] == "****"

    async def fake_invoke_endpoint(row_data, target_config):
        assert target_config["authorization"] == "Bearer xxx"
        return {
            "status_code": 200,
            "latency_ms": 1,
            "request_body": {"question": row_data["user_input"]},
            "raw_response": '{"data":{"answer":"ok"}}',
            "parsed_response": {"data": {"answer": "ok"}},
        }

    monkeypatch.setattr("app.api.endpoint_target.invoke_endpoint", fake_invoke_endpoint)
    test_resp = client.post(
        f"/api/endpoint-targets/{target_resp.json()['id']}/test",
        json={"row_data": {"user_input": "hello"}},
    )
    assert test_resp.status_code == 200
    assert test_resp.json()["success"] is True


def test_evaluation_endpoint_draft_test_uses_saved_authorization(client, monkeypatch):
    target_resp = client.post(
        "/api/endpoint-targets",
        json={
            "name": "Private API",
            "endpoint_url": "https://example.com/chat",
            "transport_mode": "json",
            "authorization": "Bearer xxx",
            "extra_headers": "{}",
            "request_body_template": '{"question":"{{user_input}}"}',
            "response_mapping": {"response_path": "data.answer"},
        },
    )
    assert target_resp.status_code == 201

    async def fake_invoke_endpoint(row_data, target_config):
        assert target_config["authorization"] == "Bearer xxx"
        return {
            "status_code": 200,
            "latency_ms": 1,
            "request_body": {"question": row_data["user_input"]},
            "raw_response": '{"data":{"answer":"ok"}}',
            "parsed_response": {"data": {"answer": "ok"}},
        }

    monkeypatch.setattr("app.core.endpoint_eval.invoke_endpoint", fake_invoke_endpoint)
    resp = client.post(
        "/api/evaluations/test-endpoint",
        json={
            "endpoint_target_id": target_resp.json()["id"],
            "target_config": {
                "endpoint_url": "https://example.com/chat",
                "transport_mode": "json",
                "authorization": "",
                "request_body_template": '{"question":"{{user_input}}"}',
            },
            "response_mapping": {"response_path": "data.answer"},
            "row_data": {"user_input": "hello"},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert resp.json()["extracted_fields"]["response"] == "ok"


def test_endpoint_evaluation_merges_extracted_fields_without_writeback(
    client, db, monkeypatch, test_llm_payload
):
    import asyncio
    from sqlalchemy.orm import sessionmaker

    from app.core.evaluation_engine import run_evaluation
    from app.models.dataset import DatasetRow
    from app.models.evaluation import EvalRowResult, EvalTask

    llm = client.post("/api/llm-configs", json=test_llm_payload).json()
    metric = client.post(
        "/api/metrics",
        json={
            "name": "retrieval_hit_rate",
            "display_name": "HitRate",
            "metric_type": "code_retrieval_hit_rate",
            "config": {"k": 3},
            "category": "retrieval",
        },
    ).json()
    scenario = client.post(
        "/api/scenarios",
        json={
            "name": "接口检索评测",
            "description": "",
            "scene_type": "rag",
            "sample_type": "single_turn",
            "metrics": [
                {
                    "metric_definition_id": metric["id"],
                    "weight": 1.0,
                    "pass_threshold": 0.7,
                }
            ],
        },
    ).json()
    dataset = client.post(
        "/api/datasets",
        json={
            "name": "输入集",
            "sample_type": "single_turn",
            "field_schema": [
                {"name": "user_input", "type": "text", "required": True, "description": ""},
                {"name": "reference_context_ids", "type": "array", "required": True, "description": ""},
            ],
        },
    ).json()
    client.post(
        f"/api/datasets/{dataset['id']}/rows",
        json={"data": {"user_input": "查保修", "reference_context_ids": ["doc-1"]}},
    )

    async def fake_invoke_endpoint(row_data, target_config):
        return {
            "status_code": 200,
            "latency_ms": 12,
            "request_body": {"question": row_data["user_input"]},
            "raw_response": '{"data":{"ids":["doc-1","doc-2"]}}',
            "parsed_response": {"data": {"ids": ["doc-1", "doc-2"]}},
        }

    monkeypatch.setattr("app.core.evaluation_engine.invoke_endpoint", fake_invoke_endpoint)
    task = client.post(
        "/api/evaluations",
        json={
            "name": "接口评测",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
            "evaluation_mode": "endpoint",
            "target_config": {
                "endpoint_url": "https://example.com/chat",
                "transport_mode": "json",
                "request_body_template": '{"question":"{{user_input}}"}',
            },
            "response_mapping": {"retrieved_context_ids_path": "data.ids"},
            "result_save_mode": "task_only",
        },
    ).json()

    TestingSession = sessionmaker(bind=db.get_bind())
    asyncio.run(run_evaluation(task["id"], TestingSession))

    result = db.query(EvalRowResult).filter(EvalRowResult.eval_task_id == task["id"]).one()
    assert result.is_pass is True
    assert result.metric_scores["retrieval_hit_rate"]["score"] == 1.0
    assert result.endpoint_trace["extracted_fields"]["retrieved_context_ids"] == ["doc-1", "doc-2"]

    row = db.query(DatasetRow).filter(DatasetRow.dataset_id == dataset["id"]).one()
    assert "retrieved_context_ids" not in row.data


def test_endpoint_evaluation_can_write_back_extracted_fields(
    client, db, monkeypatch, test_llm_payload
):
    import asyncio
    from sqlalchemy.orm import sessionmaker

    from app.core.evaluation_engine import run_evaluation
    from app.models.dataset import Dataset, DatasetRow

    llm, dataset, scenario = _setup_eval_prerequisites(client, test_llm_payload)
    row = db.query(DatasetRow).filter(DatasetRow.dataset_id == dataset["id"]).first()
    row.data = {"user_input": "What is Python?", "response": "old"}
    db.commit()

    async def fake_invoke_endpoint(row_data, target_config):
        return {
            "status_code": 200,
            "latency_ms": 10,
            "request_body": {"question": row_data["user_input"]},
            "raw_response": '{"data":{"answer":"new answer"}}',
            "parsed_response": {"data": {"answer": "new answer"}},
        }

    monkeypatch.setattr("app.core.evaluation_engine.invoke_endpoint", fake_invoke_endpoint)

    class _StubJudge:
        """最小可用裁判替身：必须带 token 核算接口。

        原来这里是 `lambda _llm_config: object()`。裸 object 没有
        `reset_row_usage` / `take_row_usage`，评测引擎调到就抛 AttributeError，
        任务被标记为 failed——而这个测试当时仍然是绿的，因为它断言的是回写
        字段，而回写在指标循环崩溃**之前**就已经 commit 了。也就是说这个用例
        从来没有真正验证过"评分跑通"，只验证了"回写落库"。并发化把回写移到
        行评测完成之后，这个洞才暴露出来。
        """

        def __init__(self, _llm_config):
            self.row_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def reset_row_usage(self):
            self.row_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def take_row_usage(self):
            usage = dict(self.row_usage)
            self.reset_row_usage()
            return usage

    monkeypatch.setattr("app.core.evaluation_engine.OpenAIJudgeClient", _StubJudge)

    async def fake_ascore(self, row_data, judge):
        from app.core.evaluation_engine import _MetricResult

        return _MetricResult(1.0, f"response={row_data.get('response')}")

    monkeypatch.setattr("app.core.evaluation_engine.NativePromptMetric.ascore", fake_ascore)

    task = client.post(
        "/api/evaluations",
        json={
            "name": "接口评测回写",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
            "evaluation_mode": "endpoint",
            "target_config": {"endpoint_url": "https://example.com/chat"},
            "response_mapping": {"response_path": "data.answer"},
            "result_save_mode": "write_back",
        },
    ).json()
    TestingSession = sessionmaker(bind=db.get_bind())
    asyncio.run(run_evaluation(task["id"], TestingSession))

    db.expire_all()
    updated_row = db.query(DatasetRow).filter(DatasetRow.dataset_id == dataset["id"]).one()
    assert updated_row.data["response"] == "new answer"
    updated_dataset = db.query(Dataset).filter(Dataset.id == dataset["id"]).one()
    assert any(field["name"] == "response" for field in updated_dataset.field_schema)


def test_update_custom_scenario_does_not_mutate_existing_task_snapshot(client, test_llm_payload):
    llm, dataset, scenario = _setup_eval_prerequisites(client, test_llm_payload)
    replacement_metric = client.post(
        "/api/metrics",
        json={
            "name": "replacement_metric",
            "display_name": "Replacement Metric",
            "metric_type": "aspect_critic",
            "config": {"definition": "Replacement"},
            "category": "custom",
        },
    ).json()

    task_resp = client.post(
        "/api/evaluations",
        json={
            "name": "Snapshot Eval",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
        },
    )
    assert task_resp.status_code == 201
    task = task_resp.json()
    assert task["scenario_snapshot"]["name"] == "Test Scenario"
    assert task["scenario_snapshot"]["metrics"][0]["metric_definition"]["name"] == "test_metric"

    update_resp = client.put(
        f"/api/scenarios/{scenario['id']}",
        json={
            "name": "Edited Scenario",
            "description": "edited",
            "scene_type": "rag",
            "sample_type": "single_turn",
            "metrics": [
                {
                    "metric_definition_id": replacement_metric["id"],
                    "weight": 1.0,
                    "pass_threshold": 0.7,
                }
            ],
        },
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["name"] == "Edited Scenario"

    existing_task = client.get(f"/api/evaluations/{task['id']}").json()
    assert existing_task["scenario_snapshot"]["name"] == "Test Scenario"
    assert existing_task["scenario_snapshot"]["metrics"][0]["metric_definition"]["name"] == "test_metric"


def test_scenario_metric_prompt_override_is_frozen_in_task_snapshot(client, test_llm_payload):
    llm = client.post("/api/llm-configs", json=test_llm_payload).json()
    metric = client.post(
        "/api/metrics",
        json={
            "name": "business_completeness",
            "display_name": "业务完整性",
            "metric_type": "numeric",
            "config": {"prompt": "默认完整性标准", "allowed_values": [0, 1]},
            "category": "custom",
        },
    ).json()
    scenario = client.post(
        "/api/scenarios",
        json={
            "name": "业务场景",
            "description": "",
            "scene_type": "rag",
            "sample_type": "single_turn",
            "metrics": [
                {
                    "metric_definition_id": metric["id"],
                    "weight": 1.0,
                    "pass_threshold": 0.7,
                    "prompt_override": "请按电商客服业务规则评估回答完整性。",
                }
            ],
        },
    ).json()
    dataset = client.post(
        "/api/datasets",
        json={
            "name": "业务数据集",
            "sample_type": "single_turn",
            "field_schema": [
                {"name": "user_input", "type": "text", "required": True, "description": ""},
                {"name": "response", "type": "text", "required": True, "description": ""},
            ],
        },
    ).json()
    client.post(
        f"/api/datasets/{dataset['id']}/rows",
        json={"data": {"user_input": "怎么退货？", "response": "可申请退货。"}},
    )

    task = client.post(
        "/api/evaluations",
        json={
            "name": "业务评测",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
        },
    ).json()
    assert task["scenario_snapshot"]["metrics"][0]["prompt_override"] == "请按电商客服业务规则评估回答完整性。"

    client.put(
        f"/api/scenarios/{scenario['id']}",
        json={
            "name": "业务场景编辑",
            "description": "",
            "scene_type": "rag",
            "sample_type": "single_turn",
            "metrics": [
                {
                    "metric_definition_id": metric["id"],
                    "weight": 1.0,
                    "pass_threshold": 0.7,
                    "prompt_override": "新的评分口径",
                }
            ],
        },
    )
    existing_task = client.get(f"/api/evaluations/{task['id']}").json()
    assert existing_task["scenario_snapshot"]["metrics"][0]["prompt_override"] == "请按电商客服业务规则评估回答完整性。"


def test_run_evaluation_uses_scenario_snapshot_after_scenario_edit(client, db, test_llm_payload):
    import asyncio
    from sqlalchemy.orm import sessionmaker

    from app.core.evaluation_engine import run_evaluation
    from app.models.evaluation import EvalRowResult, EvalTask

    llm = client.post("/api/llm-configs", json=test_llm_payload).json()
    hit_rate_metric = client.post(
        "/api/metrics",
        json={
            "name": "snapshot_hit_rate",
            "display_name": "Snapshot Hit Rate",
            "metric_type": "code_retrieval_hit_rate",
            "config": {"k": 3},
            "category": "rag",
        },
    ).json()
    mrr_metric = client.post(
        "/api/metrics",
        json={
            "name": "snapshot_mrr",
            "display_name": "Snapshot MRR",
            "metric_type": "code_retrieval_mrr",
            "config": {},
            "category": "rag",
        },
    ).json()
    scenario = client.post(
        "/api/scenarios",
        json={
            "name": "Snapshot Scenario",
            "description": "snapshot",
            "scene_type": "rag",
            "sample_type": "single_turn",
            "metrics": [
                {
                    "metric_definition_id": hit_rate_metric["id"],
                    "weight": 1.0,
                    "pass_threshold": 0.5,
                }
            ],
        },
    ).json()
    dataset = client.post(
        "/api/datasets",
        json={
            "name": "Retrieval Dataset",
            "sample_type": "single_turn",
            "field_schema": [
                {"name": "user_input", "type": "text", "required": True, "description": ""},
                {"name": "retrieved_context_ids", "type": "text_list", "required": True, "description": ""},
                {"name": "reference_context_ids", "type": "text_list", "required": True, "description": ""},
            ],
        },
    ).json()
    client.post(
        f"/api/datasets/{dataset['id']}/rows",
        json={
            "data": {
                "user_input": "Which doc answers the question?",
                "retrieved_context_ids": ["doc-1", "doc-2"],
                "reference_context_ids": ["doc-1"],
            }
        },
    )
    task = client.post(
        "/api/evaluations",
        json={
            "name": "Snapshot Run",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
        },
    ).json()

    client.put(
        f"/api/scenarios/{scenario['id']}",
        json={
            "name": "Edited Snapshot Scenario",
            "description": "edited",
            "scene_type": "rag",
            "sample_type": "single_turn",
            "metrics": [
                {
                    "metric_definition_id": mrr_metric["id"],
                    "weight": 1.0,
                    "pass_threshold": 0.5,
                }
            ],
        },
    )

    SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=db.bind)
    asyncio.run(run_evaluation(task["id"], SessionFactory))

    db.expire_all()
    db_task = db.query(EvalTask).filter(EvalTask.id == task["id"]).first()
    row_result = db.query(EvalRowResult).filter(EvalRowResult.eval_task_id == task["id"]).first()
    assert db_task.status == "completed"
    assert "snapshot_hit_rate" in row_result.metric_scores
    assert "snapshot_mrr" not in row_result.metric_scores


def test_report_summary_backfills_metric_pass_rate_without_threshold(client, db):
    """Old reports without per-metric thresholds should still display pass rates."""
    from app.models.dataset import Dataset, DatasetRow
    from app.models.evaluation import EvalTask, EvalRowResult
    from app.models.llm_config import LLMConfig
    from app.models.metric_definition import MetricDefinition
    from app.models.scenario import EvalScenario, ScenarioMetric

    llm = LLMConfig(
        name="Judge",
        api_base_url="https://example.com/v1",
        api_key="key",
        model_name="model",
    )
    metric = MetricDefinition(
        name="turn_relevancy",
        display_name="轮次相关性",
        metric_type="builtin_turn_relevancy",
        config={},
        category="multi_turn",
        is_builtin=True,
    )
    scenario = EvalScenario(
        name="No Threshold Scenario",
        scene_type="multi_turn",
        sample_type="multi_turn",
        is_preset=False,
    )
    dataset = Dataset(name="Rows", sample_type="multi_turn", field_schema=[], row_count=2)
    db.add_all([llm, metric, scenario, dataset])
    db.flush()
    db.add(ScenarioMetric(scenario_id=scenario.id, metric_definition_id=metric.id, pass_threshold=None))
    rows = [
        DatasetRow(dataset_id=dataset.id, row_index=0, data={"user_input": []}),
        DatasetRow(dataset_id=dataset.id, row_index=1, data={"user_input": []}),
    ]
    db.add_all(rows)
    db.flush()
    task = EvalTask(
        name="Old Report",
        dataset_id=dataset.id,
        scenario_id=scenario.id,
        llm_config_id=llm.id,
        status="completed",
        total_rows=2,
        completed_rows=2,
        summary_scores={
            "turn_relevancy": {
                "mean": 0.75,
                "min": 0.6,
                "max": 0.9,
                "pass_rate": None,
                "count": 2,
                "error_count": 0,
            }
        },
    )
    db.add(task)
    db.flush()
    db.add_all(
        [
            EvalRowResult(
                eval_task_id=task.id,
                dataset_row_id=rows[0].id,
                row_index=0,
                metric_scores={"turn_relevancy": {"score": 0.6, "reason": "低于默认阈值"}},
                is_pass=True,
            ),
            EvalRowResult(
                eval_task_id=task.id,
                dataset_row_id=rows[1].id,
                row_index=1,
                metric_scores={"turn_relevancy": {"score": 0.9, "reason": "高于默认阈值"}},
                is_pass=True,
            ),
        ]
    )
    db.commit()

    resp = client.get(f"/api/reports/{task.id}/summary")
    assert resp.status_code == 200
    metric_summary = resp.json()["metric_summary"]["turn_relevancy"]
    assert metric_summary["pass_rate"] == 0.5
    assert metric_summary["effective_pass_threshold"] == 0.7


def test_compare_reports_groups_regressions_and_fixes(client, db):
    from app.models.dataset import Dataset, DatasetRow
    from app.models.evaluation import EvalTask, EvalRowResult
    from app.models.llm_config import LLMConfig
    from app.models.metric_definition import MetricDefinition
    from app.models.scenario import EvalScenario, ScenarioMetric
    from app.core.scenario_snapshot import build_scenario_snapshot

    llm = LLMConfig(name="Judge", api_base_url="https://example.com/v1", api_key="key", model_name="model")
    metric = MetricDefinition(
        name="answer_relevancy",
        display_name="Answer Relevancy",
        metric_type="builtin_answer_relevancy",
        config={},
        category="rag",
        is_builtin=True,
    )
    scenario = EvalScenario(name="RAG", scene_type="rag", sample_type="single_turn", is_preset=False)
    dataset = Dataset(name="Rows", sample_type="single_turn", field_schema=[], row_count=3)
    db.add_all([llm, metric, scenario, dataset])
    db.flush()
    scenario.metrics.append(ScenarioMetric(metric_definition_id=metric.id, pass_threshold=0.7, metric_definition=metric))
    db.flush()
    scenario_snapshot = build_scenario_snapshot(scenario)
    rows = [
        DatasetRow(dataset_id=dataset.id, row_index=0, data={"user_input": "Q1"}),
        DatasetRow(dataset_id=dataset.id, row_index=1, data={"user_input": "Q2"}),
        DatasetRow(dataset_id=dataset.id, row_index=2, data={"user_input": "Q3"}),
    ]
    db.add_all(rows)
    db.flush()

    baseline = EvalTask(
        name="Baseline",
        dataset_id=dataset.id,
        scenario_id=scenario.id,
        llm_config_id=llm.id,
        status="completed",
        total_rows=3,
        completed_rows=3,
        scenario_snapshot=scenario_snapshot,
        summary_scores={"answer_relevancy": {"mean": 0.7, "pass_rate": 0.6667, "error_count": 0}},
    )
    current = EvalTask(
        name="Current",
        dataset_id=dataset.id,
        scenario_id=scenario.id,
        llm_config_id=llm.id,
        status="completed",
        total_rows=3,
        completed_rows=3,
        scenario_snapshot=scenario_snapshot,
        summary_scores={"answer_relevancy": {"mean": 0.75, "pass_rate": 0.6667, "error_count": 0}},
    )
    db.add_all([baseline, current])
    db.flush()
    db.add_all(
        [
            EvalRowResult(eval_task_id=baseline.id, dataset_row_id=rows[0].id, row_index=0, metric_scores={"answer_relevancy": {"score": 0.9}}, is_pass=True),
            EvalRowResult(eval_task_id=baseline.id, dataset_row_id=rows[1].id, row_index=1, metric_scores={"answer_relevancy": {"score": 0.4}}, is_pass=False),
            EvalRowResult(eval_task_id=baseline.id, dataset_row_id=rows[2].id, row_index=2, metric_scores={"answer_relevancy": {"score": 0.8}}, is_pass=True),
            EvalRowResult(eval_task_id=current.id, dataset_row_id=rows[0].id, row_index=0, metric_scores={"answer_relevancy": {"score": 0.5}}, is_pass=False),
            EvalRowResult(eval_task_id=current.id, dataset_row_id=rows[1].id, row_index=1, metric_scores={"answer_relevancy": {"score": 0.85}}, is_pass=True),
            EvalRowResult(eval_task_id=current.id, dataset_row_id=rows[2].id, row_index=2, metric_scores={"answer_relevancy": {"score": 0.9}}, is_pass=True),
        ]
    )
    db.commit()

    resp = client.get(f"/api/reports/{current.id}/compare?baseline_eval_id={baseline.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary_delta"]["current_fail_count"] == 1
    assert body["summary_delta"]["baseline_fail_count"] == 1
    assert len(body["row_changes"]["new_failures"]) == 1
    assert body["row_changes"]["new_failures"][0]["dataset_row_id"] == rows[0].id
    assert len(body["row_changes"]["fixed"]) == 1
    assert body["row_changes"]["fixed"][0]["dataset_row_id"] == rows[1].id
    assert len(body["row_changes"]["still_passing"]) == 1
    assert body["metric_deltas"][0]["mean_delta"] == 0.05


def test_compare_reports_rejects_different_dataset(client, db):
    from app.models.dataset import Dataset
    from app.models.evaluation import EvalTask
    from app.models.llm_config import LLMConfig
    from app.models.scenario import EvalScenario

    llm = LLMConfig(name="Judge", api_base_url="https://example.com/v1", api_key="key", model_name="model")
    scenario = EvalScenario(name="RAG", scene_type="rag", sample_type="single_turn", is_preset=False)
    ds1 = Dataset(name="A", sample_type="single_turn", field_schema=[], row_count=0)
    ds2 = Dataset(name="B", sample_type="single_turn", field_schema=[], row_count=0)
    db.add_all([llm, scenario, ds1, ds2])
    db.flush()
    snapshot = {"metrics": []}
    t1 = EvalTask(name="A", dataset_id=ds1.id, scenario_id=scenario.id, llm_config_id=llm.id, status="completed", scenario_snapshot=snapshot)
    t2 = EvalTask(name="B", dataset_id=ds2.id, scenario_id=scenario.id, llm_config_id=llm.id, status="completed", scenario_snapshot=snapshot)
    db.add_all([t1, t2])
    db.commit()

    resp = client.get(f"/api/reports/{t1.id}/compare?baseline_eval_id={t2.id}")
    assert resp.status_code == 422


def test_get_report_summary(client, db):
    """Test report summary using seed data inserted directly into the test DB."""
    from app.seed import run_seed

    run_seed(db)
    db.commit()

    # The seed creates an eval task with id=1
    resp = client.get("/api/reports/1/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_count"] == 5
    assert body["pass_count"] == 4
    assert body["fail_count"] == 1
    assert body["error_count"] == 0
    assert body["pass_rate"] == 0.8
    assert "faithfulness" in body["metric_summary"]
    assert "context_recall" in body["metric_summary"]
    assert "context_precision" in body["metric_summary"]
    assert "answer_relevancy" in body["metric_summary"]
    assert "factual_correctness" in body["metric_summary"]
    assert "answer_completeness" in body["metric_summary"]
    assert body["eval_task"]["status"] == "completed"


def test_list_reports_returns_report_summaries(client, db):
    from app.models.dataset import Dataset, DatasetRow
    from app.models.evaluation import EvalRowResult, EvalTask
    from app.models.llm_config import LLMConfig
    from app.models.metric_definition import MetricDefinition
    from app.models.scenario import EvalScenario, ScenarioMetric

    dataset = Dataset(name="报告数据集", sample_type="single_turn", row_count=2)
    db.add(dataset)
    db.flush()
    row_a = DatasetRow(dataset_id=dataset.id, row_index=0, data={"user_input": "a"})
    row_b = DatasetRow(dataset_id=dataset.id, row_index=1, data={"user_input": "b"})
    metric = MetricDefinition(
        name="report_metric",
        display_name="报告指标",
        metric_type="numeric",
        config={"prompt": "score"},
        category="custom",
    )
    scenario = EvalScenario(name="报告场景", scene_type="general", sample_type="single_turn")
    llm = LLMConfig(name="Judge", api_base_url="https://api.example.com/v1", api_key="k", model_name="m")
    db.add_all([row_a, row_b, metric, scenario, llm])
    db.flush()
    db.add(ScenarioMetric(scenario_id=scenario.id, metric_definition_id=metric.id, pass_threshold=0.7))
    task = EvalTask(
        name="报告列表任务",
        dataset_id=dataset.id,
        scenario_id=scenario.id,
        llm_config_id=llm.id,
        status="completed",
        progress=1.0,
        total_rows=2,
        completed_rows=2,
        evaluation_mode="offline",
        summary_scores={"report_metric": {"mean": 0.75}},
    )
    db.add(task)
    db.flush()
    db.add_all([
        EvalRowResult(eval_task_id=task.id, dataset_row_id=row_a.id, row_index=0, metric_scores={}, is_pass=True),
        EvalRowResult(eval_task_id=task.id, dataset_row_id=row_b.id, row_index=1, metric_scores={}, is_pass=False, error="bad"),
    ])
    db.commit()

    resp = client.get("/api/reports")
    assert resp.status_code == 200
    body = resp.json()
    item = next(item for item in body["items"] if item["eval_id"] == task.id)
    assert item["task_name"] == "报告列表任务"
    assert item["dataset_name"] == "报告数据集"
    assert item["scenario_name"] == "报告场景"
    assert item["total_count"] == 2
    assert item["pass_count"] == 1
    assert item["fail_count"] == 1
    assert item["error_count"] == 1
    assert item["pass_rate"] == 0.5


def test_seed_rag_template_has_p0_metric_layer(client, db):
    """The preset RAG scenario should include the full P0 metric layer."""
    from app.models.dataset import Dataset
    from app.models.scenario import EvalScenario
    from app.seed import run_seed

    run_seed(db)
    db.commit()

    scenario = db.query(EvalScenario).filter(EvalScenario.scene_type == "rag").first()
    metric_names = {item.metric_definition.name for item in scenario.metrics}
    assert {
        "faithfulness",
        "context_recall",
        "context_precision",
        "answer_relevancy",
        "factual_correctness",
        "answer_completeness",
    }.issubset(metric_names)

    dataset = db.query(Dataset).filter(Dataset.name == "RAG 示例数据集").first()
    field_names = {field["name"] for field in dataset.field_schema}
    assert "retrieved_context_ids" in field_names
    assert "reference_context_ids" in field_names


def test_seed_agent_and_multi_turn_templates_have_core_metric_layers(client, db):
    """Preset Agent and multi-turn scenarios should include the new core metrics."""
    from app.models.dataset import Dataset
    from app.models.scenario import EvalScenario
    from app.seed import run_seed

    run_seed(db)
    db.commit()

    agent = db.query(EvalScenario).filter(EvalScenario.scene_type == "agent").first()
    agent_metric_names = {item.metric_definition.name for item in agent.metrics}
    assert {
        "task_completion",
        "tool_call_accuracy",
        "argument_correctness",
        "step_efficiency",
        "agent_goal_accuracy",
    }.issubset(agent_metric_names)

    multi_turn = db.query(EvalScenario).filter(EvalScenario.scene_type == "multi_turn").first()
    multi_turn_metric_names = {item.metric_definition.name for item in multi_turn.metrics}
    assert {
        "topic_adherence",
        "turn_relevancy",
        "conversation_completeness",
        "knowledge_retention",
        "role_adherence",
    }.issubset(multi_turn_metric_names)

    dataset = db.query(Dataset).filter(Dataset.name == "多轮对话示例数据集").first()
    field_names = {field["name"] for field in dataset.field_schema}
    assert "reference_role" in field_names
    assert "retrieved_contexts" in field_names
    assert all(row.data.get("reference_role") for row in dataset.rows)
    assert all(row.data.get("retrieved_contexts") for row in dataset.rows)

    agent_dataset = db.query(Dataset).filter(Dataset.name == "Agent 工具调用示例数据集").first()
    agent_field_names = {field["name"] for field in agent_dataset.field_schema}
    assert "retrieved_contexts" in agent_field_names
    assert "reference_role" in agent_field_names
    assert all(row.data.get("retrieved_contexts") for row in agent_dataset.rows)


def test_deterministic_retrieval_metrics_score_without_llm():
    """ID-based retrieval metrics should close the loop when document IDs exist."""
    from types import SimpleNamespace

    from app.core.evaluation_engine import build_metric

    metric_def = SimpleNamespace(
        name="retrieval_hit_rate",
        metric_type="code_retrieval_hit_rate",
        config={"k": 2},
    )
    kind, metric = build_metric(metric_def, llm=None)
    assert kind == "simple"
    result = metric.score(
        retrieved_context_ids=["doc-a", "doc-b"],
        reference_context_ids=["doc-b"],
    )
    assert result.value == 1.0

    metric_def = SimpleNamespace(
        name="retrieval_mrr",
        metric_type="code_retrieval_mrr",
        config={},
    )
    kind, metric = build_metric(metric_def, llm=None)
    result = metric.score(
        retrieved_context_ids=["doc-a", "doc-b", "doc-c"],
        reference_context_ids=["doc-c"],
    )
    assert kind == "simple"
    assert result.value == 0.3333


def test_agent_deterministic_metrics_score_without_llm():
    """Agent tool/argument/step metrics should produce explainable deterministic scores."""
    import asyncio
    from types import SimpleNamespace

    from app.core.evaluation_engine import build_metric

    row_data = {
        "user_input": [
            {"type": "human", "content": "查订单并提交退货"},
            {"type": "ai", "content": "", "tool_calls": [{"name": "query_order", "args": {"order_id": "ORD-1"}}]},
            {"type": "tool", "content": "{}"},
            {"type": "ai", "content": "", "tool_calls": [{"name": "create_return", "args": {"order_id": "ORD-2", "item": "手机壳"}}]},
            {"type": "ai", "content": "", "tool_calls": [{"name": "query_order", "args": {"order_id": "ORD-1"}}]},
        ],
        "reference_tool_calls": [
            {"name": "query_order", "args": {"order_id": "ORD-1"}},
            {"name": "create_return", "args": {"order_id": "ORD-1", "item": "手机壳"}},
        ],
    }

    metric_def = SimpleNamespace(
        name="tool_call_accuracy",
        metric_type="builtin_tool_call_accuracy",
        config={},
    )
    kind, metric = build_metric(metric_def, llm=None)
    assert kind == "simple"
    result = asyncio.run(metric.ascore(row_data, None))
    assert result.value == 0.5
    assert "完全匹配 1 个" in result.reason

    metric_def = SimpleNamespace(
        name="argument_correctness",
        metric_type="builtin_argument_correctness",
        config={},
    )
    kind, metric = build_metric(metric_def, llm=None)
    result = asyncio.run(metric.ascore(row_data, None))
    assert kind == "simple"
    assert result.value == 0.75
    assert "参数平均匹配度" in result.reason

    metric_def = SimpleNamespace(
        name="step_efficiency",
        metric_type="builtin_step_efficiency",
        config={},
    )
    kind, metric = build_metric(metric_def, llm=None)
    result = asyncio.run(metric.ascore(row_data, None))
    assert kind == "simple"
    assert result.value == 0.6667
    assert "实际工具步骤 3 个" in result.reason


def test_new_llm_metrics_validate_required_fields_and_use_judge_payload():
    """New Judge metrics should validate required fields and send concrete payloads."""
    import asyncio
    from types import SimpleNamespace

    from app.core.evaluation_engine import build_metric

    class FakeJudge:
        def __init__(self):
            self.payloads = []

        async def judge_json(self, payload):
            self.payloads.append(payload)
            return {"score": 0.82, "reason": "根据样本字段，回复满足该指标的大部分要求。"}

    metric_def = SimpleNamespace(
        name="role_adherence",
        display_name="角色遵守度 (Role Adherence)",
        metric_type="builtin_role_adherence",
        config={},
    )
    kind, metric = build_metric(metric_def, llm=None)
    assert kind == "llm"

    missing_result = asyncio.run(metric.ascore({"user_input": []}, FakeJudge()))
    assert missing_result.value is None
    assert "reference_role" in missing_result.reason

    judge = FakeJudge()
    result = asyncio.run(
        metric.ascore(
            {
                "user_input": [{"type": "human", "content": "能帮我直接付款吗？"}],
                "reference_role": "客服助手，不能替用户付款。",
            },
            judge,
        )
    )
    assert result.value == 0.82
    assert result.reason
    assert judge.payloads[0]["metric"]["required_fields"] == ["user_input", "reference_role"]
    assert "reference_role" in judge.payloads[0]["sample"]

    metric_def = SimpleNamespace(
        name="contextual_relevancy",
        display_name="上下文相关性 (Contextual Relevancy)",
        metric_type="builtin_contextual_relevancy",
        config={},
    )
    kind, metric = build_metric(metric_def, llm=None)
    assert kind == "llm"
    judge = FakeJudge()
    result = asyncio.run(
        metric.ascore(
            {"user_input": "退货政策是什么？", "retrieved_contexts": ["7 天内可退货。"]},
            judge,
        )
    )
    assert result.value == 0.82
    assert judge.payloads[0]["metric"]["name"] == "contextual_relevancy"


def test_aspect_critic_metric_renders_definition_variables():
    """Custom aspect metrics should render row variables before judging."""
    import asyncio
    from types import SimpleNamespace

    from app.core.evaluation_engine import build_metric

    class FakeJudge:
        def __init__(self):
            self.payloads = []

        async def judge_json(self, payload):
            self.payloads.append(payload)
            return {"score": 1, "reason": "回答覆盖参考答案并满足评估标准。"}

    metric_def = SimpleNamespace(
        name="reference_answer_quality_is_ok",
        display_name="参考答案符合度",
        metric_type="aspect_critic",
        config={
            "definition": (
                "用户输入：{user_input}\n"
                "参考答案：{reference}\n"
                "评估标准：{rubrics}\n"
                "被测接口回答：{response}"
            ),
            "description": "根据参考答案和业务评估标准判断被测回答质量",
        },
    )
    kind, metric = build_metric(metric_def, llm=None)
    assert kind == "llm"

    judge = FakeJudge()
    result = asyncio.run(
        metric.ascore(
            {
                "user_input": "请介绍 DeepSeek",
                "reference": "应说明它是 AI/大模型相关能力。",
                "rubrics": "不得声称完全免费。",
                "response": "DeepSeek 是大模型能力提供方。",
            },
            judge,
        )
    )

    definition = judge.payloads[0]["metric"]["definition"]
    sample = judge.payloads[0]["sample"]
    assert result.value == 1.0
    assert "{user_input}" not in definition
    assert "{reference}" not in definition
    assert "请介绍 DeepSeek" in definition
    assert "不得声称完全免费" in definition
    assert "user_input" not in sample
    assert "reference" not in sample
    assert "rubrics" not in sample
    assert "response" not in sample


def test_prompt_renderer_keeps_json_braces_and_deduplicates_sample_fields():
    """Business prompts may contain JSON examples plus row variables."""
    from app.core.evaluation_engine import _render_custom_prompt_and_sample

    prompt, sample = _render_custom_prompt_and_sample(
        '请判断回答：{response}\n输出 JSON 示例：{"score": 1, "reason": "ok"}',
        {
            "user_input": "问题",
            "response": "实际回答",
            "reference": "标准答案",
        },
    )

    assert "实际回答" in prompt
    assert '{"score": 1, "reason": "ok"}' in prompt
    assert "response" not in sample
    assert sample["user_input"] == "问题"
    assert sample["reference"] == "标准答案"


# ======================== 平台级聚合：加权总分 / 采样稳定性 / 成本 / 一致性 ========================


def test_weighted_total_score_in_summary():
    """加权总分应按场景快照冻结的权重聚合数值型指标均值。"""
    from types import SimpleNamespace

    from app.core.evaluation_engine import _compute_summary_scores

    metrics = [
        ("faithfulness", object(), SimpleNamespace(pass_threshold=0.5, weight=1.0)),
        ("answer_relevancy", object(), SimpleNamespace(pass_threshold=0.5, weight=3.0)),
    ]
    all_row_scores = [
        {
            "faithfulness": {"score": 0.8, "reason": "a"},
            "answer_relevancy": {"score": 0.6, "reason": "b"},
        },
        {
            "faithfulness": {"score": 1.0, "reason": "a"},
            "answer_relevancy": {"score": 0.4, "reason": "b"},
        },
    ]
    summary = _compute_summary_scores(all_row_scores, metrics)

    assert summary["faithfulness"]["mean"] == 0.9
    assert summary["answer_relevancy"]["mean"] == 0.5
    weighted = summary["weighted_total_score"]
    # (0.9 * 1 + 0.5 * 3) / (1 + 3) = 0.6
    assert weighted["weighted_mean"] == pytest.approx(0.6)
    assert weighted["weights"] == {"faithfulness": 1.0, "answer_relevancy": 3.0}
    assert weighted["metric_count"] == 2


def test_weighted_total_score_skips_metrics_without_numeric_mean():
    """无数值均值（全部评分失败）的指标不应进入加权总分。"""
    from types import SimpleNamespace

    from app.core.evaluation_engine import _compute_summary_scores

    metrics = [
        ("ok_metric", object(), SimpleNamespace(pass_threshold=0.5, weight=2.0)),
        ("broken_metric", object(), SimpleNamespace(pass_threshold=0.5, weight=1.0)),
    ]
    all_row_scores = [
        {"ok_metric": {"score": 1.0, "reason": "a"}, "broken_metric": {"score": None, "reason": "judge failed"}},
        {"ok_metric": {"score": 0.8, "reason": "a"}, "broken_metric": {"score": None, "reason": "judge failed"}},
    ]
    summary = _compute_summary_scores(all_row_scores, metrics)

    assert summary["broken_metric"]["mean"] is None
    weighted = summary["weighted_total_score"]
    assert weighted["weighted_mean"] == pytest.approx(0.9)
    assert weighted["metric_count"] == 1


def test_judge_multi_sampling_aggregates_mean_and_std():
    """EVAL_JUDGE_SAMPLES > 1 时应对同一行多次采样，取均值并报告标准差。"""
    import asyncio
    from types import SimpleNamespace

    from app.core.config import settings
    from app.core.evaluation_engine import _score_metric_with_sampling, build_metric

    class FakeJudge:
        def __init__(self):
            self.calls = 0

        async def judge_json(self, payload):
            self.calls += 1
            scores = [0.8, 0.6, 0.7]
            idx = (self.calls - 1) % len(scores)
            return {"score": scores[idx], "reason": f"第 {self.calls} 次采样理由"}

    metric_def = SimpleNamespace(
        name="answer_relevancy",
        display_name="回答相关性",
        metric_type="builtin_answer_relevancy",
        config={},
    )
    _kind, metric = build_metric(metric_def, llm=None)

    original = settings.EVAL_JUDGE_SAMPLES
    settings.EVAL_JUDGE_SAMPLES = 3
    try:
        result, stats = asyncio.run(
            _score_metric_with_sampling(metric, {"user_input": "q", "response": "a"}, FakeJudge())
        )
    finally:
        settings.EVAL_JUDGE_SAMPLES = original

    assert stats["sample_count"] == 3
    assert result.value == pytest.approx(0.7)  # mean(0.8, 0.6, 0.7)
    assert stats["score_std"] == pytest.approx(0.0816, abs=1e-3)  # pstdev
    assert len(stats["sample_scores"]) == 3


def test_judge_multi_sampling_keeps_single_call_by_default():
    """默认 EVAL_JUDGE_SAMPLES=1 时保持单次采样，不产生额外统计。"""
    import asyncio
    from types import SimpleNamespace

    from app.core.config import settings
    from app.core.evaluation_engine import _score_metric_with_sampling, build_metric

    class FakeJudge:
        def __init__(self):
            self.calls = 0

        async def judge_json(self, payload):
            self.calls += 1
            return {"score": 0.9, "reason": "单次采样"}

    metric_def = SimpleNamespace(
        name="answer_relevancy",
        display_name="回答相关性",
        metric_type="builtin_answer_relevancy",
        config={},
    )
    _kind, metric = build_metric(metric_def, llm=None)

    judge = FakeJudge()
    original = settings.EVAL_JUDGE_SAMPLES
    settings.EVAL_JUDGE_SAMPLES = 1
    try:
        result, stats = asyncio.run(
            _score_metric_with_sampling(metric, {"user_input": "q", "response": "a"}, judge)
        )
    finally:
        settings.EVAL_JUDGE_SAMPLES = original

    assert judge.calls == 1
    assert stats == {}
    assert result.value == 0.9


def test_judge_json_parse_recovers_embedded_object_and_coerce_float():
    """Judge 返回带前后缀文本时仍能解析 JSON；字符串分数可提取数值。"""
    from app.core.evaluation_engine import _coerce_float, _parse_json_object

    parsed = _parse_json_object('好的，评分如下：{"score": 0.85, "reason": "回复完整"}。')
    assert parsed["score"] == 0.85
    assert parsed["reason"] == "回复完整"

    assert _coerce_float("得分 0.82 分") == pytest.approx(0.82)
    assert _coerce_float("pass") is None
    assert _coerce_float(1) == 1.0

    with pytest.raises(ValueError):
        _parse_json_object("没有 JSON 的纯文本")


def test_report_summary_exposes_platform_aggregates_and_manual_agreement(client, db, test_llm_payload):
    """报告摘要应剥离平台级保留键并暴露加权总分/成本/Judge 稳定性/人工一致性。"""
    from app.models.dataset import DatasetRow
    from app.models.evaluation import EvalRowResult, EvalTask

    llm, dataset, scenario = _setup_eval_prerequisites(client, test_llm_payload)
    resp = client.post(
        "/api/evaluations",
        json={
            "name": "Aggregate Eval",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
        },
    )
    eval_id = resp.json()["id"]

    dataset_row = db.query(DatasetRow).filter(DatasetRow.dataset_id == dataset["id"]).first()
    task = db.query(EvalTask).filter(EvalTask.id == eval_id).first()
    task.summary_scores = {
        "test_metric": {"mean": 0.8, "pass_rate": 0.9, "count": 1},
        "weighted_total_score": {"weighted_mean": 0.78, "metric_count": 1},
        "cost": {"total_tokens": 1000, "estimated_cost": 0.003, "currency": "CNY"},
        "judge_reliability": {"mean_std": 0.02, "low_confidence_row_count": 0},
    }
    db.add(
        EvalRowResult(
            eval_task_id=eval_id,
            dataset_row_id=dataset_row.id,
            row_index=1,
            metric_scores={"test_metric": {"score": 0.8, "reason": "ok"}},
            is_pass=True,
            manual_status="pass",
        )
    )
    db.commit()

    summary = client.get(f"/api/reports/{eval_id}/summary").json()

    assert summary["weighted_total_score"]["weighted_mean"] == 0.78
    assert summary["cost"]["total_tokens"] == 1000
    assert summary["judge_reliability"]["mean_std"] == 0.02
    assert "weighted_total_score" not in summary["metric_summary"]
    assert "cost" not in summary["metric_summary"]
    assert summary["manual_auto_agreement_rate"] == 1.0
    assert summary["manual_auto_disagreement_count"] == 0


def test_manual_auto_disagreement_counted_in_summary(client, db, test_llm_payload):
    """人工判定与自动判定不一致的行应被计入分歧数。"""
    from app.models.dataset import DatasetRow
    from app.models.evaluation import EvalRowResult, EvalTask

    llm, dataset, scenario = _setup_eval_prerequisites(client, test_llm_payload)
    resp = client.post(
        "/api/evaluations",
        json={
            "name": "Disagree Eval",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
        },
    )
    eval_id = resp.json()["id"]

    dataset_row = db.query(DatasetRow).filter(DatasetRow.dataset_id == dataset["id"]).first()
    db.add(
        EvalRowResult(
            eval_task_id=eval_id,
            dataset_row_id=dataset_row.id,
            row_index=1,
            metric_scores={},
            is_pass=True,
            manual_status="fail",
        )
    )
    db.commit()

    summary = client.get(f"/api/reports/{eval_id}/summary").json()
    assert summary["manual_auto_agreement_rate"] == 0.0
    assert summary["manual_auto_disagreement_count"] == 1


def test_recover_stale_eval_tasks_uses_heartbeat(db, test_llm_payload):
    """陈旧回收应参考进程内心跳，避免误杀仍在运行的长任务。"""
    import time as time_module
    from datetime import datetime, timedelta, timezone

    from app.api.evaluation import recover_stale_eval_tasks
    from app.core.evaluation_engine import task_heartbeats
    from app.models.dataset import Dataset, DatasetRow
    from app.models.evaluation import EvalTask
    from app.models.llm_config import LLMConfig
    from app.models.scenario import EvalScenario

    dataset = Dataset(name="hb-ds", sample_type="single_turn", field_schema=[], row_count=1)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    db.add(DatasetRow(dataset_id=dataset.id, row_index=1, data={"user_input": "q"}))
    llm = LLMConfig(name="hb-llm", api_base_url="http://x", api_key="k", model_name="m")
    db.add(llm)
    scenario = EvalScenario(name="hb-sc", scene_type="rag", sample_type="single_turn")
    db.add(scenario)
    db.commit()
    db.refresh(llm)
    db.refresh(scenario)

    task = EvalTask(
        name="hb-task",
        dataset_id=dataset.id,
        scenario_id=scenario.id,
        llm_config_id=llm.id,
        status="running",
        started_at=datetime.now(timezone.utc) - timedelta(minutes=60),
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    try:
        # 有心跳 → 不回收
        task_heartbeats[task.id] = time_module.time()
        assert recover_stale_eval_tasks(db) == 0
        db.refresh(task)
        assert task.status == "running"

        # 心跳过期 → 回收
        task_heartbeats.pop(task.id, None)
        assert recover_stale_eval_tasks(db) == 1
        db.refresh(task)
        assert task.status == "failed"
        assert "自动标记为失败" in task.error_message
    finally:
        task_heartbeats.clear()


def test_invoke_endpoint_retries_transient_failures(monkeypatch):
    """接口调用应针对 5xx/网络抖动做有限次退避重试。"""
    import asyncio
    import httpx

    import app.core.endpoint_eval as endpoint_eval

    calls = {"count": 0}

    class _FakeResponse:
        status_code = 200
        text = '{"answer": "ok"}'
        headers = {"content-type": "application/json"}

        def json(self):
            return {"answer": "ok"}

        def raise_for_status(self):
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] < 3:
                raise httpx.HTTPStatusError(
                    "boom",
                    request=httpx.Request("POST", "http://example.com"),
                    response=httpx.Response(500, request=httpx.Request("POST", "http://example.com")),
                )
            return _FakeResponse()

    monkeypatch.setattr(endpoint_eval.httpx, "AsyncClient", _FakeClient)

    result = asyncio.run(
        endpoint_eval.invoke_endpoint(
            {"user_input": "hi"},
            {"endpoint_url": "http://example.com/chat", "transport_mode": "json"},
        )
    )
    assert calls["count"] == 3
    assert result["answer"] == "ok"


def test_invoke_endpoint_fails_fast_on_config_error(monkeypatch):
    """配置类错误（缺少 endpoint_url）不应触发重试。"""
    import asyncio

    import app.core.endpoint_eval as endpoint_eval

    with pytest.raises(ValueError):
        asyncio.run(endpoint_eval.invoke_endpoint({"user_input": "hi"}, {"transport_mode": "json"}))


# ======================== 双通道指标 / 断言级忠实度 / 多裁判 / 换序 / kappa / 版本化 ========================


def test_claim_faithfulness_metric_decomposes_and_verifies():
    """断言级忠实度：拆解回答为原子断言，逐条核验上下文支持，按占比计分。"""
    import asyncio
    from types import SimpleNamespace

    from app.core.evaluation_engine import ClaimFaithfulnessMetric

    class FakeJudge:
        api_base_url = "http://x"
        api_key = "k"
        model = "m"

        async def chat_json(self, system_prompt, user_prompt):
            if "拆解" in user_prompt:
                return {"claims": ["元组是不可变的", "元组可以用 append 添加元素"]}
            if "append" in user_prompt:
                return {"verdict": "unsupported", "reason": "上下文只说明元组不可变"}
            return {"verdict": "supported", "reason": "上下文明确支持"}

    metric = ClaimFaithfulnessMetric(
        SimpleNamespace(
            name="faithfulness_claim",
            display_name="断言级忠实度",
            metric_type="builtin_faithfulness_claim",
            config={},
        )
    )
    result = asyncio.run(
        metric.ascore(
            {
                "response": "元组是不可变的，可以用 append 修改。",
                "retrieved_contexts": ["元组(tuple)是不可变序列，一旦创建就不能修改。"],
            },
            FakeJudge(),
        )
    )
    assert result.value == 0.5  # 只有第一条断言被支持
    assert "2 条断言" in result.reason


def test_claim_faithfulness_missing_context_returns_none():
    import asyncio
    from types import SimpleNamespace

    from app.core.evaluation_engine import ClaimFaithfulnessMetric

    metric = ClaimFaithfulnessMetric(
        SimpleNamespace(
            name="faithfulness_claim",
            display_name="断言级忠实度",
            metric_type="builtin_faithfulness_claim",
            config={},
        )
    )
    result = asyncio.run(metric.ascore({"response": "x"}, None))
    assert result.value is None
    assert "retrieved_contexts" in result.reason


def test_generative_answer_relevancy_uses_embeddings(monkeypatch):
    """生成式相关性：反推问题 + embedding 相似度，取最大相似度。"""
    import asyncio
    from types import SimpleNamespace

    from app.core.evaluation_engine import GenerativeAnswerRelevancyMetric

    class FakeJudge:
        api_base_url = "http://x"
        api_key = "k"
        model = "m"

        async def chat_json(self, system_prompt, user_prompt):
            return {"questions": ["列表和元组有什么区别？", "元组可以修改吗？"]}

    class FakeEmbeddingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def embed_texts(self, texts):
            vectors = []
            for idx, _text in enumerate(texts):
                vectors.append([1.0, 0.0, 0.0] if idx in (0, 1) else [0.5, 0.5, 0.0])
            return vectors

        @staticmethod
        def cosine_similarity(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    monkeypatch.setattr(
        "app.core.evaluation_engine.OpenAICompatibleEmbeddingClient", FakeEmbeddingClient
    )

    metric = GenerativeAnswerRelevancyMetric(
        SimpleNamespace(
            name="answer_relevancy_generative",
            display_name="生成式相关性",
            metric_type="builtin_answer_relevancy_generative",
            config={},
        )
    )
    result = asyncio.run(
        metric.ascore(
            {"user_input": "列表和元组有什么区别？", "response": "列表可变，元组不可变。"},
            FakeJudge(),
        )
    )
    assert result.value == 1.0
    assert "最大语义相似度" in result.reason


def test_generative_answer_relevancy_noncommittal_scores_zero():
    """反推问题全部为回避式时记 0 分（回答未正面回应问题）。"""
    import asyncio
    from types import SimpleNamespace

    from app.core.evaluation_engine import GenerativeAnswerRelevancyMetric

    class FakeJudge:
        api_base_url = "http://x"
        api_key = "k"
        model = "m"

        async def chat_json(self, system_prompt, user_prompt):
            return {"questions": ["这个问题我不清楚", "不知道，无法回答"]}

    metric = GenerativeAnswerRelevancyMetric(
        SimpleNamespace(
            name="answer_relevancy_generative",
            display_name="生成式相关性",
            metric_type="builtin_answer_relevancy_generative",
            config={},
        )
    )
    result = asyncio.run(
        metric.ascore({"user_input": "q", "response": "敷衍回答"}, FakeJudge())
    )
    assert result.value == 0.0
    assert "回避式" in result.reason


def test_aggregate_judge_results_numeric_mean_and_discrete_majority():
    """多裁判聚合：数值取均值 + 裁判间 MAD；离散标签取多数票。"""
    from app.core.evaluation_engine import _MetricResult, _aggregate_judge_results

    numeric = [
        ("judge-a", _MetricResult(0.8, "r1"), {}),
        ("judge-b", _MetricResult(0.6, "r2"), {}),
        ("judge-c", _MetricResult(0.7, "r3"), {}),
    ]
    value, reason, stats = _aggregate_judge_results(numeric)
    assert value == pytest.approx(0.7)
    assert stats["judge_count"] == 3
    assert stats["judge_mad"] == pytest.approx(0.0667, abs=1e-3)
    assert stats["judge_scores"] == {"judge-a": 0.8, "judge-b": 0.6, "judge-c": 0.7}

    discrete = [
        ("judge-a", _MetricResult("pass", "r1"), {}),
        ("judge-b", _MetricResult("fail", "r2"), {}),
        ("judge-c", _MetricResult("pass", "r3"), {}),
    ]
    value, reason, stats = _aggregate_judge_results(discrete)
    assert value == "pass"
    assert stats["judge_mad"] is None

    all_none = [("judge-a", _MetricResult(None, "x"), {}), ("judge-b", _MetricResult(None, "x"), {})]
    value, reason, stats = _aggregate_judge_results(all_none)
    assert value is None
    assert "未返回有效分数" in reason


def test_swap_row_data_reverses_list_fields():
    """换序互评：仅反转列表字段，标量字段保持不变。"""
    from app.core.evaluation_engine import _swap_row_data

    swapped = _swap_row_data(
        {"user_input": "q", "retrieved_contexts": ["a", "b"], "reference": "r"}
    )
    assert swapped["retrieved_contexts"] == ["b", "a"]
    assert swapped["user_input"] == "q"
    assert swapped["reference"] == "r"


def test_swap_consistency_aggregated_in_summary():
    """换序一致性在指标摘要中聚合，低于阈值的行计数。"""
    from types import SimpleNamespace

    from app.core.evaluation_engine import _compute_summary_scores

    metrics = [("m1", object(), SimpleNamespace(pass_threshold=None, weight=1.0))]
    rows = [
        {"m1": {"score": 0.8, "reason": "a", "swap_consistency": 0.95}},
        {"m1": {"score": 0.6, "reason": "b", "swap_consistency": 0.85}},
    ]
    summary = _compute_summary_scores(rows, metrics)
    assert summary["m1"]["swap_consistency_mean"] == pytest.approx(0.9)
    assert summary["m1"]["swap_inconsistent_count"] == 1


def test_dataset_version_bumps_on_row_operations(client):
    """数据集行变更（增/删/导入）后版本自增。"""
    ds = client.post(
        "/api/datasets",
        json={"name": "Version DS", "sample_type": "single_turn", "field_schema": []},
    ).json()
    assert ds["version"] == 1

    client.post(f"/api/datasets/{ds['id']}/rows", json={"data": {"user_input": "q1"}})
    assert client.get(f"/api/datasets/{ds['id']}").json()["version"] == 2

    client.post(f"/api/datasets/{ds['id']}/rows", json={"data": {"user_input": "q2"}})
    rows = client.get(f"/api/datasets/{ds['id']}/rows").json()["items"]
    assert client.get(f"/api/datasets/{ds['id']}").json()["version"] == 3

    client.delete(f"/api/datasets/{ds['id']}/rows/{rows[0]['id']}")
    assert client.get(f"/api/datasets/{ds['id']}").json()["version"] == 4


def test_eval_task_records_dataset_version_and_judge_panel(client, test_llm_payload):
    """任务创建时冻结数据集版本与多裁判面板。"""
    llm, dataset, scenario = _setup_eval_prerequisites(client, test_llm_payload)
    llm2 = client.post(
        "/api/llm-configs", json={**test_llm_payload, "name": "Panel Judge 2"}
    ).json()

    resp = client.post(
        "/api/evaluations",
        json={
            "name": "Panel Eval",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
            "judge_llm_config_ids": [llm2["id"]],
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["judge_panel"] == [llm2["id"]]
    assert body["dataset_version"] == 2  # 预置数据已加一行，版本自增到 2


def test_eval_task_rejects_missing_panel_judge(client, test_llm_payload):
    """面板裁判配置不存在时返回 404。"""
    llm, dataset, scenario = _setup_eval_prerequisites(client, test_llm_payload)
    resp = client.post(
        "/api/evaluations",
        json={
            "name": "Bad Panel",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
            "judge_llm_config_ids": [99999],
        },
    )
    assert resp.status_code == 404


def test_report_kappa_and_calibration_suggestion(client, db, test_llm_payload):
    """报告计算人工 vs 自动二分类 Cohen's kappa，低 kappa 给出校准提示。"""
    from app.models.dataset import DatasetRow
    from app.models.evaluation import EvalRowResult, EvalTask

    llm, dataset, scenario = _setup_eval_prerequisites(client, test_llm_payload)
    resp = client.post(
        "/api/evaluations",
        json={
            "name": "Kappa Eval",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
        },
    )
    eval_id = resp.json()["id"]
    dataset_row = db.query(DatasetRow).filter(DatasetRow.dataset_id == dataset["id"]).first()

    # 10 条可比较复核：4 双通过 / 4 双失败 / 1 人工过自动挂 / 1 人工挂自动过
    pattern = [("pass", True)] * 4 + [("fail", False)] * 4 + [("pass", False), ("fail", True)]
    for idx, (manual, auto_pass) in enumerate(pattern):
        db.add(
            EvalRowResult(
                eval_task_id=eval_id,
                dataset_row_id=dataset_row.id,
                row_index=idx,
                metric_scores={},
                is_pass=auto_pass,
                manual_status=manual,
            )
        )
    db.commit()

    summary = client.get(f"/api/reports/{eval_id}/summary").json()
    # a=4, b=1, c=1, d=4 → observed=0.8, expected=0.5, kappa=0.6
    assert summary["manual_auto_kappa"] == pytest.approx(0.6, abs=1e-3)
    assert summary["manual_auto_agreement_rate"] == pytest.approx(0.8)
    assert summary["calibration_suggestion"] is not None
    assert "0.7" in summary["calibration_suggestion"]


def test_build_metric_routes_dual_channel_metrics_to_dedicated_executors():
    """双通道指标必须走专用执行器，而非被 spec 分支遮蔽成通用 Judge。"""
    from types import SimpleNamespace

    from app.core.evaluation_engine import (
        ClaimFaithfulnessMetric,
        GenerativeAnswerRelevancyMetric,
        build_metric,
    )

    claim_def = SimpleNamespace(
        name="faithfulness_claim",
        display_name="断言级忠实度",
        metric_type="builtin_faithfulness_claim",
        config={},
        required_fields=[],
        category="rag",
        is_builtin=True,
    )
    kind, metric = build_metric(claim_def, llm=None)
    assert kind == "llm"
    assert isinstance(metric, ClaimFaithfulnessMetric)

    gen_def = SimpleNamespace(
        name="answer_relevancy_generative",
        display_name="生成式相关性",
        metric_type="builtin_answer_relevancy_generative",
        config={},
        required_fields=[],
        category="rag",
        is_builtin=True,
    )
    kind, metric = build_metric(gen_def, llm=None)
    assert kind == "llm"
    assert isinstance(metric, GenerativeAnswerRelevancyMetric)


def test_is_llm_metric_covers_dual_channel_metrics():
    """双通道指标必须被识别为 LLM 指标（参与多裁判面板与换序互评）。"""
    from types import SimpleNamespace

    from app.core.evaluation_engine import (
        ClaimFaithfulnessMetric,
        GenerativeAnswerRelevancyMetric,
        _is_llm_metric,
    )

    claim = ClaimFaithfulnessMetric(
        SimpleNamespace(name="faithfulness_claim", display_name="x", metric_type="builtin_faithfulness_claim", config={})
    )
    generative = GenerativeAnswerRelevancyMetric(
        SimpleNamespace(name="answer_relevancy_generative", display_name="x", metric_type="builtin_answer_relevancy_generative", config={})
    )
    assert _is_llm_metric(claim) is True
    assert _is_llm_metric(generative) is True


def test_claim_faithfulness_empty_claims_scores_zero():
    """回答未包含可核验断言时（答非所问/未基于上下文），忠实性按 0 处理。"""
    import asyncio
    from types import SimpleNamespace

    from app.core.evaluation_engine import ClaimFaithfulnessMetric

    class EmptyDecompositionJudge:
        api_base_url = "http://x"
        api_key = "k"
        model = "m"

        async def chat_json(self, system_prompt, user_prompt):
            return {"claims": []}

    metric = ClaimFaithfulnessMetric(
        SimpleNamespace(name="faithfulness_claim", display_name="x", metric_type="builtin_faithfulness_claim", config={})
    )
    result = asyncio.run(
        metric.ascore({"response": "今天天气不错。", "retrieved_contexts": ["Python 列表是可变序列。"]}, EmptyDecompositionJudge())
    )
    assert result.value == 0.0
    assert "未包含可核验断言" in result.reason


def test_sample_payload_hides_generation_meta_from_judge():
    """数据集生成元数据不得进入 Judge 可见的 sample。

    generation_meta 里带有样本的构造标签（如 kind=hallucinated），
    等于把答案泄露给裁判，会虚高指标的区分度，使评测结果失去效力。
    """
    import json

    from app.core.evaluation_engine import _sample_payload

    payload = _sample_payload(
        {
            "user_input": "保修期多长？",
            "response": "保修期为 1 年。",
            "reference": "保修期一般为 1 年。",
            "retrieved_contexts": ["保修期一般为 1 年，自签收日计算。"],
            "generation_meta": {"kind": "hallucinated", "sample_id": 7},
        }
    )

    assert "generation_meta" not in payload
    # 键不在还不够：标签字符串不能出现在 payload 任何角落
    assert "hallucinated" not in json.dumps(payload, ensure_ascii=False)
    # 正常字段不受影响
    assert payload["user_input"] == "保修期多长？"
    assert payload["response"] == "保修期为 1 年。"
    assert payload["retrieved_contexts"] == ["保修期一般为 1 年，自签收日计算。"]


def test_builtin_llm_metric_judge_prompt_excludes_generation_meta():
    """端到端：内置 LLM 指标下发给裁判的 payload 不含构造标签。"""
    import asyncio
    import json
    from types import SimpleNamespace

    from app.core.evaluation_engine import build_metric

    captured = {}

    class CapturingJudge:
        api_base_url = "http://x"
        api_key = "k"
        model = "m"

        async def judge_json(self, payload):
            captured["payload"] = payload
            return {"score": 1.0, "reason": "上下文覆盖了参考答案要点。"}

    _kind, metric = build_metric(
        SimpleNamespace(
            id=0,
            name="context_recall",
            display_name="context_recall",
            metric_type="builtin_context_recall",
            config={},
            required_fields=[],
            category="rag",
            is_builtin=True,
        ),
        llm=None,
    )
    asyncio.run(
        metric.ascore(
            {
                "user_input": "保修期多长？",
                "response": "保修期为 10 年。",
                "reference": "保修期一般为 1 年。",
                "retrieved_contexts": ["保修期一般为 1 年，自签收日计算。"],
                "generation_meta": {"kind": "hallucinated"},
            },
            CapturingJudge(),
        )
    )

    serialized = json.dumps(captured["payload"], ensure_ascii=False)
    assert "generation_meta" not in serialized
    assert "hallucinated" not in serialized


def _capture_judge_sample(metric_type: str, row_data: dict, metric_name: str | None = None) -> dict:
    """构建内置指标、跑一次打分，返回它实际下发给裁判的 sample 块。

    缺陷 B 的回归测试都依赖"实际下发了什么"，而不是"声明了什么"——
    口径越界的本质就是两者不一致，所以只能抓真实 payload。
    """
    import asyncio
    from types import SimpleNamespace

    from app.core.evaluation_engine import build_metric

    captured: dict = {}

    class CapturingJudge:
        api_base_url = "http://x"
        api_key = "k"
        model = "m"

        async def judge_json(self, payload):
            captured["payload"] = payload
            return {"score": 1.0, "reason": "占位理由。"}

    name = metric_name or metric_type.replace("builtin_", "")
    _kind, metric = build_metric(
        SimpleNamespace(
            id=0,
            name=name,
            display_name=name,
            metric_type=metric_type,
            config={},
            required_fields=[],
            category="rag",
            is_builtin=True,
        ),
        llm=None,
    )
    asyncio.run(metric.ascore(dict(row_data), CapturingJudge()))
    return captured["payload"]["sample"]


_FULL_ROW = {
    "user_input": "保修期多长？",
    "response": "保修期为 10 年。",
    "reference": "保修期一般为 1 年。",
    "retrieved_contexts": ["保修期一般为 1 年，自签收日计算。"],
    "reference_topics": ["保修政策"],
    "reference_role": "售后客服",
    "reference_tool_calls": [{"name": "query_policy"}],
    "tool_calls": [{"name": "query_policy"}],
    "generation_meta": {"kind": "hallucinated"},
}


def test_sample_payload_allow_fields_is_strict_whitelist():
    """allow_fields 给定时必须严格白名单，兜底循环整段不生效。"""
    from app.core.evaluation_engine import _sample_payload

    payload = _sample_payload(
        {
            "user_input": "问题",
            "response": "回答",
            "reference": "参考",
            "retrieved_contexts": ["上下文"],
            "source_document_names": ["policy.md"],
        },
        allow_fields=["reference", "retrieved_contexts"],
    )

    assert set(payload) == {"reference", "retrieved_contexts"}
    # 白名单不给的字段一律不下发，哪怕它在 important_fields 里
    assert "response" not in payload
    assert "user_input" not in payload
    # 兜底循环失效：口径外的普通字段也不会被补进来
    assert "source_document_names" not in payload


def test_retrieval_side_metrics_do_not_see_response():
    """缺陷 B 核心：检索侧指标不得看到 response。

    检索侧指标衡量的是"检索到的上下文够不够好"，与回答写得对不对无关。
    裁判一旦看到错误回答，会把生成质量算进检索分——A/B 实验已证实：
    屏蔽 response 后 context_recall / context_precision 在幻觉样本上
    从 0.79/0.81 回到 1.0，与 RAGAS 完全一致。
    """
    import json

    for metric_type in (
        "builtin_context_recall",
        "builtin_context_precision",
        "builtin_contextual_relevancy",
    ):
        sample = _capture_judge_sample(metric_type, _FULL_ROW)
        assert "response" not in sample, f"{metric_type} 仍能看到 response"
        # 键不在还不够：回答内容不能以任何形式出现在 sample 里
        assert "10 年" not in json.dumps(sample, ensure_ascii=False), (
            f"{metric_type} 的 sample 里出现了回答内容"
        )


def test_answer_relevancy_does_not_see_reference():
    """缺陷 B 生成侧那一半：只判"是否回应问题"的指标不得看到参考答案。

    看到 reference 会让裁判把事实错误也算进扣分，指标口径从"相关性"
    悄悄漂移成"正确性"——这是它与 RAGAS answer_relevancy 只有 0.44
    相关的根因（RAGAS 侧不惩罚事实错误）。
    """
    sample = _capture_judge_sample("builtin_answer_relevancy", _FULL_ROW)

    assert "reference" not in sample
    assert "retrieved_contexts" not in sample
    # 它该看的两个字段必须在
    assert sample["user_input"] == "保修期多长？"
    assert sample["response"] == "保修期为 10 年。"


def test_agent_and_multi_turn_metrics_still_see_response():
    """防过度修正：判断"AI 回复好不好"的指标必须看得到 response。

    这些指标不把 response 列入 required_fields，是因为 Agent/多轮评测跑真实
    接口、回复在运行时才注入。如果有人把可见范围直接等同于 required_fields
    来"修"缺陷 B，这 8 个指标会瞎判——本测试就是拦这个的。
    """
    for metric_type in (
        "builtin_agent_goal_accuracy",
        "builtin_task_completion",
        "builtin_topic_adherence",
        "builtin_turn_relevancy",
        "builtin_conversation_completeness",
        "builtin_knowledge_retention",
        "builtin_role_adherence",
        "builtin_turn_faithfulness",
    ):
        sample = _capture_judge_sample(metric_type, _FULL_ROW)
        assert "response" in sample, f"{metric_type} 看不到 response，无法判断回复质量"
        assert sample["response"] == "保修期为 10 年。"


def test_every_builtin_spec_declares_judge_fields_covering_required():
    """结构不变量：每个内置指标都必须显式声明可见范围，且不能校验后瞎判。

    新增指标若忘记声明 judge_fields，会退回 required_fields——对检索侧是安全的，
    但对"要看 response 却不把它列为必需"的指标就是静默瞎判。因此这里强制显式声明。
    """
    from app.core.prompt_manager import (
        BUILTIN_LLM_METRIC_SPECS,
        NAMED_LLM_METRIC_SPECS,
    )

    all_specs = {**BUILTIN_LLM_METRIC_SPECS, **NAMED_LLM_METRIC_SPECS}
    assert all_specs, "指标 spec 注册表为空，测试失去意义"

    for key, spec in all_specs.items():
        judge_fields = spec.get("judge_fields")
        assert judge_fields, f"{key} 未声明 judge_fields"
        # 校验门槛内的字段必须对裁判可见，否则等于"要求存在却不给看"
        missing = set(spec.get("required_fields") or []) - set(judge_fields)
        assert not missing, f"{key} 的必需字段对裁判不可见: {sorted(missing)}"
