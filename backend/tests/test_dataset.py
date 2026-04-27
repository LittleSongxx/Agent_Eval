import io
import json


def _create_dataset(client, name="Test DS", sample_type="single_turn"):
    """Helper to create a dataset via the API."""
    return client.post(
        "/api/datasets",
        json={
            "name": name,
            "description": "test dataset",
            "sample_type": sample_type,
            "field_schema": [
                {"name": "user_input", "type": "text", "required": True, "description": ""},
                {"name": "response", "type": "text", "required": False, "description": ""},
                {"name": "reference", "type": "text", "required": False, "description": ""},
            ],
        },
    )


def test_create_dataset(client):
    resp = _create_dataset(client)
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Test DS"
    assert body["sample_type"] == "single_turn"
    assert body["row_count"] == 0
    assert body["id"] is not None
    assert len(body["field_schema"]) == 3
    assert body["field_schema"][0]["name"] == "user_input"


def test_add_row(client):
    ds_id = _create_dataset(client).json()["id"]
    resp = client.post(
        f"/api/datasets/{ds_id}/rows",
        json={"data": {"user_input": "Hello", "response": "Hi", "reference": "Hi there"}},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["data"]["user_input"] == "Hello"
    assert body["data"]["response"] == "Hi"
    assert body["dataset_id"] == ds_id
    assert body["row_index"] == 0


def test_list_rows(client):
    ds_id = _create_dataset(client).json()["id"]
    for i in range(3):
        client.post(
            f"/api/datasets/{ds_id}/rows",
            json={"data": {"user_input": f"Q{i}", "response": f"A{i}"}},
        )
    # Request page 1 with page_size=2
    resp = client.get(f"/api/datasets/{ds_id}/rows?page=1&page_size=2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["page"] == 1
    assert body["page_size"] == 2

    # Request page 2 to get the remaining row
    resp2 = client.get(f"/api/datasets/{ds_id}/rows?page=2&page_size=2")
    body2 = resp2.json()
    assert len(body2["items"]) == 1


def test_delete_row(client):
    ds_id = _create_dataset(client).json()["id"]
    row_id = client.post(
        f"/api/datasets/{ds_id}/rows",
        json={"data": {"user_input": "Q", "response": "A"}},
    ).json()["id"]

    # Delete the row
    del_resp = client.delete(f"/api/datasets/{ds_id}/rows/{row_id}")
    assert del_resp.status_code == 204

    # Verify row is gone via listing
    rows_resp = client.get(f"/api/datasets/{ds_id}/rows")
    assert rows_resp.json()["total"] == 0

    # Verify row_count updated on the dataset
    ds_resp = client.get(f"/api/datasets/{ds_id}")
    assert ds_resp.json()["row_count"] == 0


def test_import_csv(client):
    ds_id = _create_dataset(client).json()["id"]
    csv_content = "user_input,response,reference\nWhat is Python?,A language,A programming language\nWhat is JS?,A language,JavaScript\n"
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    resp = client.post(f"/api/datasets/{ds_id}/import", files=files)
    assert resp.status_code == 200
    assert resp.json()["imported_count"] == 2
    assert resp.json()["skipped_duplicates"] == 0

    # Verify rows were actually stored
    rows_resp = client.get(f"/api/datasets/{ds_id}/rows")
    assert rows_resp.json()["total"] == 2


def test_import_skips_duplicate_rows(client):
    ds_id = _create_dataset(client).json()["id"]
    client.post(
        f"/api/datasets/{ds_id}/rows",
        json={"data": {"user_input": "What is Python?", "response": "A language", "reference": "A programming language"}},
    )
    csv_content = (
        "user_input,response,reference\n"
        "What is Python?,A language,A programming language\n"
        "What is Python?,A language,A programming language\n"
        "What is JS?,A language,JavaScript\n"
    )
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    resp = client.post(f"/api/datasets/{ds_id}/import", files=files)
    assert resp.status_code == 200
    assert resp.json()["imported_count"] == 1
    assert resp.json()["skipped_duplicates"] == 2

    rows_resp = client.get(f"/api/datasets/{ds_id}/rows")
    assert rows_resp.json()["total"] == 2


def test_export_dataset_as_json(client):
    ds_id = _create_dataset(client).json()["id"]
    client.post(
        f"/api/datasets/{ds_id}/rows",
        json={"data": {"user_input": "Hello", "response": "Hi", "reference": ["A", "B"]}},
    )

    resp = client.get(f"/api/datasets/{ds_id}/export?format=json")
    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]
    payload = json.loads(resp.content.decode("utf-8"))
    assert payload == [{"user_input": "Hello", "response": "Hi", "reference": ["A", "B"]}]


def test_export_dataset_as_csv(client):
    ds_id = _create_dataset(client).json()["id"]
    client.post(
        f"/api/datasets/{ds_id}/rows",
        json={"data": {"user_input": "Hello", "response": "Hi", "reference": ["A", "B"]}},
    )

    resp = client.get(f"/api/datasets/{ds_id}/export?format=csv")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
    content = resp.content.decode("utf-8-sig")
    assert "user_input,response,reference" in content
    assert 'Hello,Hi,"[""A"", ""B""]"' in content


def test_export_dataset_with_chinese_name_as_json(client):
    ds_id = _create_dataset(client, name="多轮对话示例数据集", sample_type="multi_turn").json()["id"]
    client.post(
        f"/api/datasets/{ds_id}/rows",
        json={"data": {"user_input": [{"type": "human", "content": "你好"}], "response": "你好", "reference": "你好"}},
    )

    resp = client.get(f"/api/datasets/{ds_id}/export?format=json")
    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]
    assert "filename*=UTF-8''" in resp.headers["content-disposition"]
    payload = json.loads(resp.content.decode("utf-8"))
    assert payload[0]["response"] == "你好"
