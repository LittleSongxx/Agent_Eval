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
