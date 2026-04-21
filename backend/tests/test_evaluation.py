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
    assert "factual_correctness" in body["metric_summary"]
    assert body["eval_task"]["status"] == "completed"
