import asyncio
import threading
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session, joinedload

from app.core.database import SessionLocal, get_db
from app.core.rag_dataset_generator import (
    RAG_JOB_TIMEOUT_SECONDS,
    evaluate_chunk_quality,
    parse_uploaded_file,
    persist_uploaded_document,
    recover_stale_rag_dataset_job,
    run_rag_dataset_job,
    summarize_supported_metrics,
)
from app.models.llm_config import LLMConfig
from app.models.rag_dataset_job import (
    RagDatasetChunk,
    RagDatasetDocument,
    RagDatasetJob,
    RagDatasetSample,
)
from app.schemas.llm_config import LLMConfigResponse
from app.schemas.dataset import DatasetResponse
from app.schemas.rag_dataset_job import (
    RagDatasetChunkListResponse,
    RagDatasetChunkResponse,
    RagDatasetDocumentResponse,
    RagDatasetJobCreate,
    RagDatasetJobDetailResponse,
    RagDatasetJobResponse,
    RagDatasetJobRunRequest,
    RagDatasetJobRunResponse,
    RagDatasetSampleResponse,
    RagDatasetSamplesResponse,
)

router = APIRouter(prefix="/rag-dataset-jobs", tags=["RAG Dataset Jobs"])


def _launch_rag_job(job_id: int, scope: str, sample_ids: list[int] | None = None) -> str:
    task_id = f"rag-job-{job_id}-{scope}"

    def runner() -> None:
        asyncio.run(
            run_rag_dataset_job(
                job_id=job_id,
                session_factory=SessionLocal,
                scope=scope,
                sample_ids=sample_ids,
            )
        )

    thread = threading.Thread(
        target=runner,
        name=task_id,
        daemon=True,
    )
    thread.start()
    return task_id


def _load_job_query(db: Session):
    return db.query(RagDatasetJob).options(
        joinedload(RagDatasetJob.documents),
        joinedload(RagDatasetJob.dataset),
        joinedload(RagDatasetJob.question_llm_config),
        joinedload(RagDatasetJob.target_llm_config),
    )


def _ensure_job_not_running(job: RagDatasetJob) -> None:
    if job.status == "running":
        raise HTTPException(status_code=409, detail="任务运行中，请等待当前执行完成")


def _recover_stale_job(db: Session, job_id: int) -> RagDatasetJob | None:
    return recover_stale_rag_dataset_job(db, job_id, timeout_seconds=RAG_JOB_TIMEOUT_SECONDS)


def _serialize_llm_config(config: Optional[LLMConfig]) -> Optional[dict]:
    if config is None:
        return None
    return LLMConfigResponse.from_orm_with_mask(config).model_dump()


def _mask_secret(value: Optional[str]) -> str:
    secret = (value or "").strip()
    if not secret:
        return ""
    return secret[:3] + "****" + secret[-4:] if len(secret) > 8 else "****"


def _serialize_job(job: RagDatasetJob, detail: bool = False) -> dict:
    payload = {
        "id": job.id,
        "name": job.name,
        "description": job.description,
        "status": job.status,
        "target_endpoint_url": job.target_endpoint_url,
        "target_transport_mode": job.target_transport_mode or "sse",
        "target_authorization_masked": _mask_secret(job.target_authorization),
        "target_extra_headers": job.target_extra_headers or "",
        "target_request_body_template": job.target_request_body_template
        or '{"question":"{{question}}","kb_codes":[],"payload":{"files":[]}}',
        "target_response_mode": job.target_response_mode,
        "target_system_prompt": job.target_system_prompt,
        "question_count_mode": job.question_count_mode,
        "requested_question_count": job.requested_question_count,
        "suggested_question_count": job.suggested_question_count,
        "total_documents": job.total_documents,
        "total_chunks": job.total_chunks,
        "total_samples": job.total_samples,
        "completed_samples": job.completed_samples,
        "failed_samples": job.failed_samples,
        "error_message": job.error_message,
        "logs": job.logs or "",
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "dataset_id": job.dataset_id,
        "dataset": DatasetResponse.model_validate(job.dataset).model_dump() if job.dataset else None,
        "documents": [
            RagDatasetDocumentResponse.model_validate(document).model_dump()
            for document in (job.documents or [])
        ],
    }
    payload["question_llm_config"] = _serialize_llm_config(job.question_llm_config)
    payload["target_llm_config"] = _serialize_llm_config(job.target_llm_config)
    if detail:
        supported, unsupported, notes = summarize_supported_metrics(job.target_response_mode)
        payload["generation_summary"] = {
            "supported_metrics": supported,
            "unsupported_metrics": unsupported,
            "notes": notes,
        }
    return payload


def _serialize_chunk(item: RagDatasetChunk) -> dict:
    quality = evaluate_chunk_quality(item.content or "")
    return RagDatasetChunkResponse(
        id=item.id,
        document_id=item.document_id,
        chunk_index=item.chunk_index,
        chunk_key=item.chunk_key,
        content=item.content,
        char_count=item.char_count,
        suggested_question_count=item.suggested_question_count,
        allocated_question_count=item.allocated_question_count,
        quality_label=quality["label"],
        quality_score=quality["score"],
        quality_reasons=quality["reasons"],
        generation_status=item.generation_status,
        generation_error=item.generation_error,
        created_at=item.created_at,
    ).model_dump()


@router.get("", response_model=List[RagDatasetJobResponse])
def list_rag_dataset_jobs(db: Session = Depends(get_db)):
    jobs = _load_job_query(db).order_by(RagDatasetJob.created_at.desc()).all()
    recovered = False
    for job in jobs:
        if job.status != "running":
            continue
        refreshed = _recover_stale_job(db, job.id)
        if refreshed is not None and refreshed.status != "running":
            recovered = True
    if recovered:
        jobs = _load_job_query(db).order_by(RagDatasetJob.created_at.desc()).all()
    return [_serialize_job(job) for job in jobs]


@router.post("", response_model=RagDatasetJobResponse, status_code=201)
def create_rag_dataset_job(payload: RagDatasetJobCreate, db: Session = Depends(get_db)):
    question_llm = db.query(LLMConfig).filter(LLMConfig.id == payload.question_llm_config_id).first()
    if not question_llm:
        raise HTTPException(status_code=404, detail="Question LLM config not found")

    target_llm_id = payload.target_llm_config_id or payload.question_llm_config_id
    target_llm = db.query(LLMConfig).filter(LLMConfig.id == target_llm_id).first()
    if not target_llm:
        raise HTTPException(status_code=404, detail="Fallback target LLM config not found")

    job = RagDatasetJob(**payload.model_dump(exclude={"target_llm_config_id"}))
    job.target_llm_config_id = target_llm_id
    db.add(job)
    db.commit()
    db.refresh(job)
    job = _load_job_query(db).filter(RagDatasetJob.id == job.id).first()
    return _serialize_job(job)


@router.get("/{job_id}", response_model=RagDatasetJobDetailResponse)
def get_rag_dataset_job(job_id: int, db: Session = Depends(get_db)):
    _recover_stale_job(db, job_id)
    job = _load_job_query(db).filter(RagDatasetJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="RAG dataset job not found")
    return _serialize_job(job, detail=True)


@router.post("/{job_id}/documents", response_model=List[RagDatasetDocumentResponse])
def upload_rag_documents(
    job_id: int,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    _recover_stale_job(db, job_id)
    job = _load_job_query(db).filter(RagDatasetJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="RAG dataset job not found")
    if job.status == "running":
        raise HTTPException(status_code=409, detail="任务运行中，暂不能修改文档")

    created: list[RagDatasetDocument] = []
    for upload in files:
        raw_bytes = upload.file.read()
        if not raw_bytes:
            continue
        try:
            raw_text = parse_uploaded_file(upload.filename or "unknown.txt", raw_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        document = persist_uploaded_document(
            db,
            job,
            upload.filename or "unknown.txt",
            raw_bytes,
            raw_text,
        )
        created.append(document)

    job.total_documents = (
        db.query(RagDatasetDocument).filter(RagDatasetDocument.job_id == job.id).count()
    )
    job.total_chunks = (
        db.query(RagDatasetChunk)
        .join(RagDatasetDocument, RagDatasetChunk.document_id == RagDatasetDocument.id)
        .filter(RagDatasetDocument.job_id == job.id)
        .count()
    )
    db.commit()
    return [RagDatasetDocumentResponse.model_validate(item) for item in created]


@router.get("/{job_id}/documents", response_model=List[RagDatasetDocumentResponse])
def list_rag_documents(job_id: int, db: Session = Depends(get_db)):
    documents = (
        db.query(RagDatasetDocument)
        .filter(RagDatasetDocument.job_id == job_id)
        .order_by(RagDatasetDocument.id.desc())
        .all()
    )
    return [RagDatasetDocumentResponse.model_validate(item) for item in documents]


@router.get("/{job_id}/chunks", response_model=RagDatasetChunkListResponse)
def list_rag_chunks(
    job_id: int,
    document_id: int | None = Query(None),
    db: Session = Depends(get_db),
):
    query = (
        db.query(RagDatasetChunk)
        .join(RagDatasetDocument, RagDatasetChunk.document_id == RagDatasetDocument.id)
        .filter(RagDatasetDocument.job_id == job_id)
        .order_by(RagDatasetChunk.document_id, RagDatasetChunk.chunk_index)
    )
    if document_id is not None:
        query = query.filter(RagDatasetChunk.document_id == document_id)
    items = query.all()
    return RagDatasetChunkListResponse(
        items=[RagDatasetChunkResponse.model_validate(_serialize_chunk(item)) for item in items]
    )


@router.get("/{job_id}/samples", response_model=RagDatasetSamplesResponse)
def list_rag_samples(
    job_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = Query(None),
    db: Session = Depends(get_db),
):
    query = db.query(RagDatasetSample).filter(RagDatasetSample.job_id == job_id)
    if status:
        query = query.filter(RagDatasetSample.status == status)
    total = query.count()
    items = (
        query.order_by(RagDatasetSample.id)
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return RagDatasetSamplesResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[RagDatasetSampleResponse.model_validate(item) for item in items],
    )


@router.post("/{job_id}/start", response_model=RagDatasetJobRunResponse)
def start_rag_dataset_job(job_id: int, db: Session = Depends(get_db)):
    _recover_stale_job(db, job_id)
    job = db.query(RagDatasetJob).filter(RagDatasetJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="RAG dataset job not found")
    _ensure_job_not_running(job)
    if not job.documents:
        raise HTTPException(status_code=422, detail="请先上传至少一份知识库文档")
    task_id = _launch_rag_job(job_id, scope="full")
    return RagDatasetJobRunResponse(
        task_id=task_id,
        job_id=job_id,
        status="running",
        scope="full",
        message="已启动 RAG 数据集生成任务",
    )


@router.post("/{job_id}/retry-failed", response_model=RagDatasetJobRunResponse)
def retry_failed_rag_dataset_job(job_id: int, db: Session = Depends(get_db)):
    _recover_stale_job(db, job_id)
    job = db.query(RagDatasetJob).filter(RagDatasetJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="RAG dataset job not found")
    _ensure_job_not_running(job)
    task_id = _launch_rag_job(job_id, scope="retry_failed")
    return RagDatasetJobRunResponse(
        task_id=task_id,
        job_id=job_id,
        status="running",
        scope="retry_failed",
        message="已启动失败样本重试",
    )


@router.post("/{job_id}/rerun-samples", response_model=RagDatasetJobRunResponse)
def rerun_selected_rag_samples(
    job_id: int,
    payload: RagDatasetJobRunRequest,
    db: Session = Depends(get_db),
):
    _recover_stale_job(db, job_id)
    job = db.query(RagDatasetJob).filter(RagDatasetJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="RAG dataset job not found")
    _ensure_job_not_running(job)
    sample_ids = payload.sample_ids or []
    if not sample_ids:
        raise HTTPException(status_code=422, detail="请至少选择一条样本")
    existing = (
        db.query(RagDatasetSample.id)
        .filter(RagDatasetSample.job_id == job_id, RagDatasetSample.id.in_(sample_ids))
        .all()
    )
    if len(existing) != len(sample_ids):
        raise HTTPException(status_code=404, detail="部分样本不存在")
    task_id = _launch_rag_job(job_id, scope="rerun_samples", sample_ids=sample_ids)
    return RagDatasetJobRunResponse(
        task_id=task_id,
        job_id=job_id,
        status="running",
        scope="rerun_samples",
        sample_ids=sample_ids,
        message="已启动局部重跑",
    )
