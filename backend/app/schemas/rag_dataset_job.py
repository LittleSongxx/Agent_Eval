from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from app.schemas.dataset import DatasetResponse
from app.schemas.llm_config import LLMConfigResponse


class RagDatasetJobCreate(BaseModel):
    name: str
    description: str = ""
    question_llm_config_id: int
    target_llm_config_id: Optional[int] = None
    target_endpoint_url: str
    target_transport_mode: Literal["json", "sse"] = "sse"
    target_authorization: str = ""
    target_extra_headers: str = ""
    target_request_body_template: str = '{"question":"{{question}}","kb_codes":[],"payload":{"files":[]}}'
    target_response_mode: Literal["answer_only", "answer_with_contexts"] = "answer_with_contexts"
    target_system_prompt: str = ""
    question_count_mode: Literal["auto", "custom"] = "auto"
    requested_question_count: Optional[int] = None

    @model_validator(mode="after")
    def validate_question_count(self):
        if self.question_count_mode == "custom" and not self.requested_question_count:
            raise ValueError("自定义总题数模式下必须提供 requested_question_count")
        if not (self.target_endpoint_url or "").strip():
            raise ValueError("必须提供 target_endpoint_url")
        return self


class RagDatasetDocumentResponse(BaseModel):
    id: int
    job_id: int
    filename: str
    file_path: Optional[str] = None
    char_count: int
    chunk_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RagDatasetChunkResponse(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    chunk_key: str
    content: str
    char_count: int
    suggested_question_count: int
    allocated_question_count: int
    quality_label: str = "medium"
    quality_score: float = 0.0
    quality_reasons: List[str] = []
    generation_status: str
    generation_error: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RagDatasetSampleResponse(BaseModel):
    id: int
    job_id: int
    document_id: int
    chunk_id: Optional[int] = None
    dataset_row_id: Optional[int] = None
    question: str
    reference: str
    reference_context_ids: List[str] = []
    source_chunk_ids: List[str] = []
    response: Optional[str] = None
    retrieved_contexts: Optional[List[str]] = None
    retrieved_context_ids: Optional[List[str]] = None
    status: str
    error_message: Optional[str] = None
    retry_count: int
    selected: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class RagDatasetSamplesResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[RagDatasetSampleResponse]


class RagDatasetChunkListResponse(BaseModel):
    items: List[RagDatasetChunkResponse]


class RagDatasetJobResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    status: str
    target_endpoint_url: Optional[str] = None
    target_transport_mode: str
    target_authorization_masked: str = ""
    target_extra_headers: str = ""
    target_request_body_template: str = ""
    target_response_mode: str
    target_system_prompt: Optional[str] = None
    question_count_mode: str
    requested_question_count: Optional[int] = None
    suggested_question_count: Optional[int] = None
    total_documents: int
    total_chunks: int
    total_samples: int
    completed_samples: int
    failed_samples: int
    error_message: Optional[str] = None
    logs: str = ""
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    dataset_id: Optional[int] = None
    dataset: Optional[DatasetResponse] = None
    question_llm_config: Optional[LLMConfigResponse] = None
    target_llm_config: Optional[LLMConfigResponse] = None
    documents: List[RagDatasetDocumentResponse] = []

    model_config = ConfigDict(from_attributes=True)


class RagDatasetJobRunRequest(BaseModel):
    chunk_ids: Optional[List[int]] = None
    sample_ids: Optional[List[int]] = None


class RagDatasetJobRunResponse(BaseModel):
    task_id: str
    job_id: int
    status: str
    scope: str
    sample_ids: List[int] = []
    chunk_ids: List[int] = []
    message: str = ""


class RagGeneratedDatasetResult(BaseModel):
    dataset_id: int
    dataset_name: str
    row_count: int
    supported_metrics: List[str]
    unsupported_metrics: List[str]
    notes: List[str] = []


class RagDatasetJobDetailResponse(RagDatasetJobResponse):
    generation_summary: Optional[dict[str, Any]] = None
