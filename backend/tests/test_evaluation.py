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
