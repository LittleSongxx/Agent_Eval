import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx

from app.core import rag_dataset_generator as generator
from app.models.dataset import DatasetRow
from app.models.rag_dataset_job import RagDatasetChunk, RagDatasetJob, RagDatasetSample

from .conftest import TestingSessionLocal


def _create_llm(client, test_llm_payload, name):
    payload = dict(test_llm_payload)
    payload["name"] = name
    response = client.post("/api/llm-configs", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def _create_job(client, question_llm_id, **overrides):
    payload = {
        "name": "RAG Builder Test",
        "description": "",
        "question_llm_config_id": question_llm_id,
        "target_endpoint_url": "https://api.example.com/chat/send-stream",
        "target_transport_mode": "sse",
        "target_authorization": "AT-test-token",
        "target_extra_headers": json.dumps({"origin": "https://coreagent.indusmind.me", "pfb": "pfb19"}),
        "target_request_body_template": json.dumps(
            {"question": "{{question}}", "kb_codes": [], "payload": {"files": []}}
        ),
        "target_response_mode": "answer_with_contexts",
        "question_count_mode": "custom",
        "requested_question_count": 1,
    }
    payload.update(overrides)
    response = client.post("/api/rag-dataset-jobs", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def _upload_text_document(client, job_id, filename="manual.txt", content=None):
    raw = content or "员工年假政策：入职满一年可享受 5 天年假，需至少提前 3 天提交审批。"
    response = client.post(
        f"/api/rag-dataset-jobs/{job_id}/documents",
        files=[("files", (filename, raw.encode("utf-8"), "text/plain"))],
    )
    assert response.status_code == 200, response.text
    return response.json()


class _FakeClient:
    def __init__(self, llm_config):
        self.llm_config = llm_config


def test_create_rag_dataset_job_validates_custom_question_count(client, test_llm_payload):
    question_llm = _create_llm(client, test_llm_payload, "Question LLM")

    response = client.post(
        "/api/rag-dataset-jobs",
        json={
            "name": "Invalid Job",
            "question_llm_config_id": question_llm["id"],
            "target_endpoint_url": "https://api.example.com/chat/send-stream",
            "question_count_mode": "custom",
        },
    )

    assert response.status_code == 422
    assert "requested_question_count" in response.text


def test_run_rag_dataset_job_creates_full_rag_dataset(client, monkeypatch, test_llm_payload):
    question_llm = _create_llm(client, test_llm_payload, "Question LLM Full")
    job = _create_job(
        client,
        question_llm["id"],
        name="员工手册自动生成",
    )
    _upload_text_document(client, job["id"])

    monkeypatch.setattr(generator, "OpenAICompatibleClient", _FakeClient)

    async def fake_generate_chunk_samples(question_client, chunk, filename, question_count):
        assert question_count == 1
        return [{"question": "员工年假需要提前多久申请？", "reference": "至少提前 3 天提交审批。"}]

    async def fake_fetch_target_answer(target_client, question, response_mode, target_system_prompt=None):
        assert response_mode == "answer_with_contexts"
        return {
            "answer": "需要至少提前 3 天提交审批。",
            "retrieved_contexts": ["入职满一年可享受 5 天年假，需至少提前 3 天提交审批。"],
            "retrieved_context_ids": ["doc-1-chunk-0"],
        }

    monkeypatch.setattr(generator, "generate_chunk_samples", fake_generate_chunk_samples)
    monkeypatch.setattr(generator, "fetch_target_answer", fake_fetch_target_answer)

    asyncio.run(generator.run_rag_dataset_job(job["id"], TestingSessionLocal, scope="full"))

    session = TestingSessionLocal()
    try:
        stored_job = session.query(RagDatasetJob).filter(RagDatasetJob.id == job["id"]).first()
        sample = session.query(RagDatasetSample).filter(RagDatasetSample.job_id == job["id"]).first()
        row = session.query(DatasetRow).filter(DatasetRow.id == sample.dataset_row_id).first()

        assert stored_job.status == "completed"
        assert stored_job.dataset_id is not None
        assert stored_job.completed_samples == 1
        assert stored_job.failed_samples == 0
        assert sample.status == "completed"
        assert row.data["user_input"] == "员工年假需要提前多久申请？"
        assert row.data["reference_context_ids"] == ["doc-1-chunk-0"]
        assert row.data["retrieved_context_ids"] == ["doc-1-chunk-0"]
        assert "retrieved_contexts" in row.data
        assert stored_job.dataset.field_schema[-2]["name"] == "retrieved_contexts"
        assert stored_job.dataset.row_count == 1
    finally:
        session.close()

    detail = client.get(f"/api/rag-dataset-jobs/{job['id']}")
    assert detail.status_code == 200
    summary = detail.json()["generation_summary"]
    assert "retrieval_hit_rate" in summary["supported_metrics"]
    assert summary["unsupported_metrics"] == []


def test_answer_only_mode_marks_retrieval_metrics_unsupported(client, monkeypatch, test_llm_payload):
    question_llm = _create_llm(client, test_llm_payload, "Question LLM Answer Only")
    job = _create_job(
        client,
        question_llm["id"],
        name="FAQ answer-only 自动生成",
        target_response_mode="answer_only",
    )
    _upload_text_document(client, job["id"], filename="faq.txt", content="退货政策：签收后 7 天内可申请退货。")

    monkeypatch.setattr(generator, "OpenAICompatibleClient", _FakeClient)

    async def fake_generate_chunk_samples(question_client, chunk, filename, question_count):
        return [{"question": "多久内可以退货？", "reference": "签收后 7 天内可申请退货。"}]

    async def fake_fetch_target_answer(target_client, question, response_mode, target_system_prompt=None):
        assert response_mode == "answer_only"
        return {
            "answer": "签收后 7 天内可以申请退货。",
            "retrieved_contexts": None,
            "retrieved_context_ids": None,
        }

    monkeypatch.setattr(generator, "generate_chunk_samples", fake_generate_chunk_samples)
    monkeypatch.setattr(generator, "fetch_target_answer", fake_fetch_target_answer)

    asyncio.run(generator.run_rag_dataset_job(job["id"], TestingSessionLocal, scope="full"))

    session = TestingSessionLocal()
    try:
        stored_job = session.query(RagDatasetJob).filter(RagDatasetJob.id == job["id"]).first()
        sample = session.query(RagDatasetSample).filter(RagDatasetSample.job_id == job["id"]).first()
        row = session.query(DatasetRow).filter(DatasetRow.id == sample.dataset_row_id).first()
        schema_names = [field["name"] for field in stored_job.dataset.field_schema]

        assert stored_job.status == "completed"
        assert "retrieved_contexts" not in schema_names
        assert "retrieved_context_ids" not in schema_names
        assert "retrieved_contexts" not in row.data
        assert "retrieval_hit_rate" in stored_job.dataset.description
        assert "不应拿来评真实检索质量" in stored_job.dataset.description
    finally:
        session.close()

    detail = client.get(f"/api/rag-dataset-jobs/{job['id']}")
    summary = detail.json()["generation_summary"]
    assert "answer_relevancy" in summary["supported_metrics"]
    assert "retrieval_hit_rate" in summary["unsupported_metrics"]
    assert summary["notes"]


def test_retry_failed_and_rerun_selected_update_existing_dataset_rows(
    client, monkeypatch, test_llm_payload
):
    question_llm = _create_llm(client, test_llm_payload, "Question LLM Retry")
    job = _create_job(
        client,
        question_llm["id"],
        name="重试与局部重跑",
        requested_question_count=2,
    )
    _upload_text_document(client, job["id"], content="报销流程：先提交申请，再上传票据，最后等待财务审核。")

    monkeypatch.setattr(generator, "OpenAICompatibleClient", _FakeClient)

    async def fake_generate_chunk_samples(question_client, chunk, filename, question_count):
        return [
            {"question": "报销流程第一步是什么？", "reference": "先提交申请。"},
            {"question": "报销完成前还需要做什么？", "reference": "上传票据并等待财务审核。"},
        ]

    attempts = {}

    async def fake_fetch_target_answer(target_client, question, response_mode, target_system_prompt=None):
        attempts[question] = attempts.get(question, 0) + 1
        if question == "报销流程第一步是什么？" and attempts[question] == 1:
            raise ValueError("temporary upstream error")
        if question == "报销完成前还需要做什么？":
            suffix = "v2" if attempts[question] >= 2 else "v1"
            return {
                "answer": f"需要上传票据并等待财务审核（{suffix}）。",
                "retrieved_contexts": ["先提交申请，再上传票据，最后等待财务审核。"],
                "retrieved_context_ids": ["doc-1-chunk-0"],
            }
        return {
            "answer": "第一步是先提交申请。",
            "retrieved_contexts": ["先提交申请，再上传票据，最后等待财务审核。"],
            "retrieved_context_ids": ["doc-1-chunk-0"],
        }

    monkeypatch.setattr(generator, "generate_chunk_samples", fake_generate_chunk_samples)
    monkeypatch.setattr(generator, "fetch_target_answer", fake_fetch_target_answer)

    asyncio.run(generator.run_rag_dataset_job(job["id"], TestingSessionLocal, scope="full"))

    session = TestingSessionLocal()
    try:
        stored_job = session.query(RagDatasetJob).filter(RagDatasetJob.id == job["id"]).first()
        samples = (
            session.query(RagDatasetSample)
            .filter(RagDatasetSample.job_id == job["id"])
            .order_by(RagDatasetSample.id)
            .all()
        )
        completed_sample = next(item for item in samples if item.status == "completed")
        failed_sample = next(item for item in samples if item.status == "failed")
        row_count_after_first_run = stored_job.dataset.row_count

        assert stored_job.status == "partial"
        assert row_count_after_first_run == 1
        assert failed_sample.retry_count == 1
        rerun_sample_id = completed_sample.id
        rerun_row_id = completed_sample.dataset_row_id
    finally:
        session.close()

    asyncio.run(generator.run_rag_dataset_job(job["id"], TestingSessionLocal, scope="retry_failed"))

    session = TestingSessionLocal()
    try:
        stored_job = session.query(RagDatasetJob).filter(RagDatasetJob.id == job["id"]).first()
        samples = (
            session.query(RagDatasetSample)
            .filter(RagDatasetSample.job_id == job["id"])
            .order_by(RagDatasetSample.id)
            .all()
        )
        assert stored_job.status == "completed"
        assert stored_job.dataset.row_count == 2
        assert all(item.status == "completed" for item in samples)
    finally:
        session.close()

    asyncio.run(
        generator.run_rag_dataset_job(
            job["id"],
            TestingSessionLocal,
            scope="rerun_samples",
            sample_ids=[rerun_sample_id],
        )
    )

    session = TestingSessionLocal()
    try:
        rerun_sample = (
            session.query(RagDatasetSample)
            .filter(RagDatasetSample.id == rerun_sample_id)
            .first()
        )
        rerun_row = session.query(DatasetRow).filter(DatasetRow.id == rerun_row_id).first()
        assert rerun_sample.dataset_row_id == rerun_row_id
        assert rerun_row.data["response"].endswith("（v2）。")
        assert session.query(DatasetRow).filter(DatasetRow.dataset_id == rerun_row.dataset_id).count() == 2
    finally:
        session.close()


def test_target_chat_requests_run_with_max_three_concurrency(client, monkeypatch, test_llm_payload):
    question_llm = _create_llm(client, test_llm_payload, "Question LLM Concurrency")
    job = _create_job(
        client,
        question_llm["id"],
        name="并发拉取测试",
        requested_question_count=5,
    )
    _upload_text_document(client, job["id"], content="报销流程：先提交申请，再上传票据，最后等待财务审核。")

    monkeypatch.setattr(generator, "OpenAICompatibleClient", _FakeClient)

    async def fake_generate_chunk_samples(question_client, chunk, filename, question_count):
        return [
            {"question": f"问题{i}", "reference": f"答案{i}"}
            for i in range(1, 6)
        ]

    active = 0
    peak = 0

    async def fake_fetch_target_answer(target_client, question, response_mode, target_system_prompt=None):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return {
            "answer": f"{question}的回答",
            "retrieved_contexts": ["先提交申请，再上传票据，最后等待财务审核。"],
            "retrieved_context_ids": ["doc-1-chunk-0"],
        }

    monkeypatch.setattr(generator, "generate_chunk_samples", fake_generate_chunk_samples)
    monkeypatch.setattr(generator, "fetch_target_answer", fake_fetch_target_answer)

    asyncio.run(generator.run_rag_dataset_job(job["id"], TestingSessionLocal, scope="full"))

    session = TestingSessionLocal()
    try:
        stored_job = session.query(RagDatasetJob).filter(RagDatasetJob.id == job["id"]).first()
        assert stored_job.status == "completed"
        assert peak == 3
        assert stored_job.completed_samples == 5
    finally:
        session.close()


def test_running_job_blocks_duplicate_operations(client, test_llm_payload):
    question_llm = _create_llm(client, test_llm_payload, "Question LLM Running")
    job = _create_job(client, question_llm["id"], name="运行中保护")
    _upload_text_document(client, job["id"])

    session = TestingSessionLocal()
    try:
        stored_job = session.query(RagDatasetJob).filter(RagDatasetJob.id == job["id"]).first()
        stored_job.status = "running"
        chunk = session.query(RagDatasetChunk).filter(RagDatasetChunk.document_id == stored_job.documents[0].id).first()
        sample = RagDatasetSample(
            job_id=stored_job.id,
            document_id=stored_job.documents[0].id,
            chunk_id=chunk.id,
            question="示例问题",
            reference="示例答案",
            reference_context_ids=[chunk.chunk_key],
            source_chunk_ids=[chunk.chunk_key],
            status="completed",
        )
        session.add(sample)
        session.commit()
        sample_id = sample.id
    finally:
        session.close()

    start_response = client.post(f"/api/rag-dataset-jobs/{job['id']}/start")
    retry_response = client.post(f"/api/rag-dataset-jobs/{job['id']}/retry-failed")
    rerun_response = client.post(
        f"/api/rag-dataset-jobs/{job['id']}/rerun-samples",
        json={"sample_ids": [sample_id]},
    )

    assert start_response.status_code == 409
    assert retry_response.status_code == 409
    assert rerun_response.status_code == 409


def test_stale_running_job_is_auto_recovered(client, test_llm_payload):
    question_llm = _create_llm(client, test_llm_payload, "Question LLM Stale")
    job = _create_job(client, question_llm["id"], name="陈旧运行任务")
    _upload_text_document(client, job["id"])

    session = TestingSessionLocal()
    try:
        stored_job = session.query(RagDatasetJob).filter(RagDatasetJob.id == job["id"]).first()
        stored_job.status = "running"
        stored_job.started_at = datetime.now(timezone.utc) - timedelta(minutes=20)
        chunk = session.query(RagDatasetChunk).filter(RagDatasetChunk.document_id == stored_job.documents[0].id).first()
        chunk.generation_status = "running"
        sample = RagDatasetSample(
            job_id=stored_job.id,
            document_id=stored_job.documents[0].id,
            chunk_id=chunk.id,
            question="示例问题",
            reference="示例答案",
            reference_context_ids=[chunk.chunk_key],
            source_chunk_ids=[chunk.chunk_key],
            status="running",
        )
        session.add(sample)
        session.commit()
    finally:
        session.close()

    detail = client.get(f"/api/rag-dataset-jobs/{job['id']}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "failed"

    session = TestingSessionLocal()
    try:
        stored_job = session.query(RagDatasetJob).filter(RagDatasetJob.id == job["id"]).first()
        chunk = session.query(RagDatasetChunk).filter(RagDatasetChunk.document_id == stored_job.documents[0].id).first()
        sample = session.query(RagDatasetSample).filter(RagDatasetSample.job_id == stored_job.id).first()
        assert stored_job.status == "failed"
        assert "自动标记为失败" in (stored_job.error_message or "")
        assert chunk.generation_status == "failed"
        assert sample.status == "failed"
    finally:
        session.close()


def test_run_rag_dataset_job_fails_after_timeout(client, monkeypatch, test_llm_payload):
    question_llm = _create_llm(client, test_llm_payload, "Question LLM Timeout")
    job = _create_job(client, question_llm["id"], name="超时失败")
    _upload_text_document(client, job["id"])

    monkeypatch.setattr(generator, "OpenAICompatibleClient", _FakeClient)
    monkeypatch.setattr(generator, "RAG_JOB_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(generator, "RAG_QUESTION_GENERATION_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(generator, "RAG_TARGET_REQUEST_TIMEOUT_SECONDS", 1)

    async def slow_generate_chunk_samples(question_client, chunk, filename, question_count):
        await asyncio.sleep(1.2)
        return [{"question": "员工年假需要提前多久申请？", "reference": "至少提前 3 天提交审批。"}]

    monkeypatch.setattr(generator, "generate_chunk_samples", slow_generate_chunk_samples)

    asyncio.run(generator.run_rag_dataset_job(job["id"], TestingSessionLocal, scope="full"))

    session = TestingSessionLocal()
    try:
        stored_job = session.query(RagDatasetJob).filter(RagDatasetJob.id == job["id"]).first()
        assert stored_job.status == "failed"
        assert "超时" in (stored_job.error_message or "")
    finally:
        session.close()


def test_target_endpoint_client_renders_body_and_parses_sse_stream():
    job = RagDatasetJob(
        name="SSE Test",
        question_llm_config_id=1,
        target_llm_config_id=1,
        target_endpoint_url="https://api.example.com/chat/send-stream",
        target_transport_mode="sse",
        target_authorization="AT-secret-token",
        target_extra_headers='{"origin":"https://coreagent.indusmind.me","pfb":"pfb19"}',
        target_request_body_template='{"question":"{{question}}","kb_codes":[],"payload":{"files":[]}}',
    )
    client = generator.TargetEndpointClient(job)
    headers = client.build_headers()
    body = client.build_body("五一去哪玩？")

    assert headers["authorization"] == "AT-secret-token"
    assert headers["accept"] == "text/event-stream"
    assert body["question"] == "五一去哪玩？"

    class FakeStreamResponse:
        headers = {"content-type": "text/event-stream"}

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            for line in [
                'data: {"content":"建议优先考虑重庆。"}',
                "",
                'data: {"content":"气候更稳定，城市游更集中。"}',
                "",
                "data: [DONE]",
                "",
            ]:
                yield line

    class FakeStreamContext:
        async def __aenter__(self):
            return FakeStreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeAsyncClient:
        def stream(self, method, url, headers=None, json=None):
            assert method == "POST"
            assert url == "https://api.example.com/chat/send-stream"
            assert json["question"] == "五一去哪玩？"
            return FakeStreamContext()

    result = asyncio.run(client._request_sse(FakeAsyncClient(), headers, body))
    assert result["answer"] == "建议优先考虑重庆。气候更稳定，城市游更集中。"


def test_target_endpoint_client_retries_retryable_sse_disconnect(monkeypatch):
    job = RagDatasetJob(
        name="SSE Retry Test",
        question_llm_config_id=1,
        target_llm_config_id=1,
        target_endpoint_url="https://api.example.com/chat/send-stream",
        target_transport_mode="sse",
        target_authorization="AT-secret-token",
        target_request_body_template='{"question":"{{question}}","kb_codes":[],"payload":{"files":[]}}',
    )
    client = generator.TargetEndpointClient(job)

    class RetryableStreamContext:
        def __init__(self, attempt):
            self.attempt = attempt

        async def __aenter__(self):
            if self.attempt == 1:
                raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            for line in [
                'data: {"content":"建议优先考虑重庆。"}',
                "",
                'data: {"content":"城市节奏更集中。"}',
                "",
                "data: [DONE]",
                "",
            ]:
                yield line

    class FakeAsyncClient:
        attempts = 0

        def __init__(self, *args, **kwargs):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, headers=None, json=None):
            FakeAsyncClient.attempts += 1
            assert headers["connection"] == "close"
            return RetryableStreamContext(FakeAsyncClient.attempts)

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(generator.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(generator.asyncio, "sleep", fake_sleep)

    result = asyncio.run(client.request("五一去哪玩？"))
    assert FakeAsyncClient.attempts == 2
    assert result["answer"] == "建议优先考虑重庆。城市节奏更集中。"


class _FakeQuestionClient:
    def __init__(self, responses):
        self.responses = list(responses)

    async def chat(self, messages, as_json=False):
        assert as_json is True
        return self.responses.pop(0)


def test_pdf_cleanup_filters_repeated_headers_and_footer_noise(monkeypatch):
    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakePdf:
        def __init__(self, pages):
            self.pages = pages

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_open(_stream):
        return FakePdf(
            [
                FakePage(
                    "第1页 4.2.3 施少峰 孙薇 缪汉根 2024年7月30日\n报销管理制度\n员工在出差结束后 5 个工作日内提交报销申请。"
                ),
                FakePage(
                    "第2页 4.2.3 施少峰 孙薇 缪汉根 2024年7月30日\n报销管理制度\n发票需与出差行程一致，并上传审批单。"
                ),
            ]
        )

    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=fake_open))

    cleaned = generator.parse_uploaded_file("rules.pdf", b"fake-pdf")
    assert "施少峰" not in cleaned
    assert "第2页" not in cleaned
    assert "5 个工作日内提交报销申请" in cleaned
    assert "发票需与出差行程一致" in cleaned


def test_fitz_pdf_extraction_prefers_body_blocks(monkeypatch):
    class FakePage:
        def __init__(self, height, blocks):
            self.rect = SimpleNamespace(height=height)
            self._blocks = blocks

        def get_text(self, mode, sort=False):
            assert mode == "blocks"
            assert sort is True
            return self._blocks

    class FakeDoc:
        def __init__(self, pages):
            self._pages = pages

        def __iter__(self):
            return iter(self._pages)

        def __len__(self):
            return len(self._pages)

        def close(self):
            return None

    def fake_open(stream, filetype):
        assert filetype == "pdf"
        return FakeDoc(
            [
                FakePage(
                    1000,
                    [
                        (0, 20, 200, 40, "第1页 4.2.3 施少峰 2024年7月30日", 0, 0),
                        (0, 180, 400, 260, "员工考勤管理规定\n员工需在上班前完成打卡。", 1, 0),
                        (0, 950, 100, 970, "1", 2, 0),
                    ],
                ),
                FakePage(
                    1000,
                    [
                        (0, 18, 200, 42, "第2页 4.2.3 施少峰 2024年7月30日", 0, 0),
                        (0, 180, 400, 260, "迟到超过 30 分钟需补充说明。", 1, 0),
                    ],
                ),
            ]
        )

    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace(open=fake_open))
    extracted = generator._extract_pdf_with_fitz(b"fake-pdf")
    assert "施少峰" not in extracted
    assert "第2页" not in extracted
    assert "员工需在上班前完成打卡" in extracted
    assert "迟到超过 30 分钟需补充说明" in extracted


def test_noise_chunk_gets_zero_question_allocation():
    noise = "第2页 4.2.3 施少峰 孙薇 缪汉根 2024年7月30日"
    revision_noise = "编制：人事行政部 文件编号：DFSH-ICP-3.4 版本号 更改页码 更改条款号 新建/修订人"
    form_noise = "海创小学春季一日研学告家长书回执 家长签名 学生姓名"
    garble_noise = "Cita Te fneo"
    useful = "员工在出差结束后 5 个工作日内提交报销申请，并上传审批单。"

    assert generator.suggest_question_count_for_chunk(noise) == 0
    assert generator.suggest_question_count_for_chunk(revision_noise) == 0
    assert generator.suggest_question_count_for_chunk(form_noise) == 0
    assert generator.suggest_question_count_for_chunk(garble_noise) == 0
    assert generator.suggest_question_count_for_chunk(useful) >= 1


def test_prune_form_tail_stops_at_receipt_section():
    lines = [
        "学校定于周四组织学生参加一日研学活动。",
        "请家长阅读活动须知并准备相关物品。",
        "活动费用总计 138 元。",
        "海创小学春季一日研学告家长书回执",
        "学生姓名",
        "家长签名",
    ]
    pruned = generator._prune_form_tail(lines)
    assert pruned == lines[:3]


def test_structure_aware_chunking_prefers_regulation_headings():
    text = """
总则说明：本制度用于规范员工考勤与请假管理。
4.2 迟到与早退
4.2.1 迟到：超过规定时间上班，1 小时以内未到岗视为迟到。
4.2.2 早退：早于规定时间下班，离岗 1 小时以内视为早退。
4.2.3 一个月内累计迟到、早退 2 次及以上者，将视情况予以警告处分。
4.3 旷工管理
4.3.1 未办理请假手续且迟到或早退超过 1 小时的，视为旷工。
4.3.2 连续旷工 3 天及以上或一年累计旷工 5 天及以上，公司可解除劳动关系。
""".strip()

    chunks = generator.split_into_chunks(text, chunk_size=420, overlap=80)
    assert len(chunks) >= 2
    assert any("4.2.1 迟到" in chunk and "4.2.2 早退" in chunk for chunk in chunks)
    assert any("4.3.1 未办理请假手续" in chunk for chunk in chunks)
    assert all("4.2.1 迟到" not in chunk or "4.3.1 未办理请假手续" not in chunk for chunk in chunks)


def test_generate_chunk_samples_filters_meta_and_duplicate_questions():
    generation_response = json.dumps(
        {
            "items": [
                {"question": "文档提到了什么？", "reference": "员工需在上班前完成打卡。"},
                {"question": "员工需要在什么时间前完成打卡？", "reference": "上班前完成打卡。"},
                {"question": "员工需要在什么时间前完成打卡？", "reference": "上班前完成打卡。"},
                {"question": "迟到超过 30 分钟后需要做什么？", "reference": "需要补充说明。"},
            ]
        },
        ensure_ascii=False,
    )
    validation_response = json.dumps(
        {
            "items": [
                {"question": "员工需要在什么时间前完成打卡？", "keep": True, "score": 0.95, "reason": "问题明确。"},
                {"question": "迟到超过 30 分钟后需要做什么？", "keep": True, "score": 0.9, "reason": "问题明确。"},
            ]
        },
        ensure_ascii=False,
    )

    client = _FakeQuestionClient([generation_response, validation_response])
    chunk = SimpleNamespace(chunk_key="doc-1-chunk-0", content="员工需在上班前完成打卡。迟到超过 30 分钟需补充说明。")
    items = asyncio.run(generator.generate_chunk_samples(client, chunk, "考勤制度.pdf", 2))
    assert len(items) == 2
    assert all("文档提到了什么" not in item["question"] for item in items)
