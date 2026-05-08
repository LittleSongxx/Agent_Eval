from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class RagDatasetJob(Base):
    __tablename__ = "rag_dataset_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(50), default="draft", nullable=False)
    question_llm_config_id = Column(Integer, ForeignKey("llm_configs.id"), nullable=False)
    target_llm_config_id = Column(Integer, ForeignKey("llm_configs.id"), nullable=False)
    target_endpoint_url = Column(String(1000), nullable=True)
    target_transport_mode = Column(String(50), default="sse", nullable=False)
    target_authorization = Column(Text, nullable=True)
    target_extra_headers = Column(Text, nullable=True)
    target_request_body_template = Column(Text, nullable=True)
    target_response_mode = Column(String(50), default="answer_with_contexts", nullable=False)
    target_system_prompt = Column(Text, nullable=True)
    question_count_mode = Column(String(50), default="auto", nullable=False)
    requested_question_count = Column(Integer, nullable=True)
    suggested_question_count = Column(Integer, nullable=True)
    total_documents = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    total_samples = Column(Integer, default=0)
    completed_samples = Column(Integer, default=0)
    failed_samples = Column(Integer, default=0)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    error_message = Column(Text, nullable=True)
    logs = Column(Text, default="")
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    question_llm_config = relationship("LLMConfig", foreign_keys=[question_llm_config_id])
    target_llm_config = relationship("LLMConfig", foreign_keys=[target_llm_config_id])
    dataset = relationship("Dataset")
    documents = relationship(
        "RagDatasetDocument",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="RagDatasetDocument.id",
    )
    samples = relationship(
        "RagDatasetSample",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="RagDatasetSample.id",
    )


class RagDatasetDocument(Base):
    __tablename__ = "rag_dataset_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("rag_dataset_jobs.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=True)
    content_hash = Column(String(64), nullable=False)
    raw_content = Column(Text, nullable=False)
    char_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())

    job = relationship("RagDatasetJob", back_populates="documents")
    chunks = relationship(
        "RagDatasetChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="RagDatasetChunk.chunk_index",
    )


class RagDatasetChunk(Base):
    __tablename__ = "rag_dataset_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("rag_dataset_documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_key = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    char_count = Column(Integer, default=0)
    suggested_question_count = Column(Integer, default=0)
    allocated_question_count = Column(Integer, default=0)
    generation_status = Column(String(50), default="pending", nullable=False)
    generation_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())

    document = relationship("RagDatasetDocument", back_populates="chunks")
    samples = relationship(
        "RagDatasetSample",
        back_populates="chunk",
        order_by="RagDatasetSample.id",
    )


class RagDatasetSample(Base):
    __tablename__ = "rag_dataset_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("rag_dataset_jobs.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(Integer, ForeignKey("rag_dataset_documents.id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(Integer, ForeignKey("rag_dataset_chunks.id", ondelete="SET NULL"), nullable=True)
    dataset_row_id = Column(Integer, ForeignKey("dataset_rows.id"), nullable=True)
    question = Column(Text, nullable=False)
    reference = Column(Text, nullable=False)
    reference_context_ids = Column(JSON, nullable=False)
    source_chunk_ids = Column(JSON, nullable=False)
    response = Column(Text, nullable=True)
    retrieved_contexts = Column(JSON, nullable=True)
    retrieved_context_ids = Column(JSON, nullable=True)
    status = Column(String(50), default="pending", nullable=False)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    selected = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    job = relationship("RagDatasetJob", back_populates="samples")
    document = relationship("RagDatasetDocument")
    chunk = relationship("RagDatasetChunk", back_populates="samples")
    dataset_row = relationship("DatasetRow")
