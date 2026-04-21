def _create(client, test_llm_payload, **overrides):
    """Helper to create an LLM config via the API."""
    data = dict(test_llm_payload)
    data.update(overrides)
    return client.post("/api/llm-configs", json=data)


def test_create_llm_config(client, test_llm_payload):
    resp = _create(client, test_llm_payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Test LLM"
    assert body["api_base_url"] == test_llm_payload["api_base_url"]
    assert body["api_key"] == test_llm_payload["api_key"]
    assert body["model_name"] == test_llm_payload["model_name"]
    assert body["provider"] == "openai"
    assert body["temperature"] == 0.01
    assert body["max_tokens"] == 1024
    assert body["id"] is not None
    assert body["created_at"] is not None


def test_list_llm_configs(client, test_llm_payload):
    _create(client, test_llm_payload, name="LLM A")
    _create(client, test_llm_payload, name="LLM B")
    resp = client.get("/api/llm-configs")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2
    names = {item["name"] for item in items}
    assert names == {"LLM A", "LLM B"}


def test_get_llm_config(client, test_llm_payload):
    create_resp = _create(client, test_llm_payload)
    config_id = create_resp.json()["id"]
    resp = client.get(f"/api/llm-configs/{config_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == config_id
    assert body["name"] == "Test LLM"
    assert body["model_name"] == test_llm_payload["model_name"]
    assert body["api_base_url"] == test_llm_payload["api_base_url"]


def test_update_llm_config(client, test_llm_payload):
    config_id = _create(client, test_llm_payload).json()["id"]
    resp = client.put(
        f"/api/llm-configs/{config_id}",
        json={"name": "Updated LLM", "temperature": 0.5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Updated LLM"
    assert body["temperature"] == 0.5
    # Unchanged fields should remain the same
    assert body["model_name"] == test_llm_payload["model_name"]


def test_delete_llm_config(client, test_llm_payload):
    config_id = _create(client, test_llm_payload).json()["id"]
    # Delete
    del_resp = client.delete(f"/api/llm-configs/{config_id}")
    assert del_resp.status_code == 204
    # Verify gone
    get_resp = client.get(f"/api/llm-configs/{config_id}")
    assert get_resp.status_code == 404
