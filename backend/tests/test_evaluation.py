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
            "authorization": "Bearer test-key",
            "extra_headers": "{}",
            "request_body_template": '{"model":"deepseek-chat","messages":[{"role":"user","content":"{{user_input}}"}]}',
            "response_mapping": {"response_path": "choices.0.message.content"},
            "default_test_input": "介绍 DeepSeek",
        },
    )
    assert target_resp.status_code == 201
    target = target_resp.json()

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
    assert task["response_mapping"]["response_path"] == "choices.0.message.content"

    client.put(
        f"/api/endpoint-targets/{target['id']}",
        json={"endpoint_url": "https://example.com/changed"},
    )
    existing_task = client.get(f"/api/evaluations/{task['id']}").json()
    assert existing_task["target_config"]["endpoint_url"] == "https://api.deepseek.com/v1/chat/completions"


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
    monkeypatch.setattr(
        "app.core.evaluation_engine.OpenAIJudgeClient",
        lambda _llm_config: object(),
    )

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
