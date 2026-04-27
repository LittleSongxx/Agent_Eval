from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class EvalTask(Base):
    __tablename__ = "eval_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    scenario_id = Column(Integer, ForeignKey("eval_scenarios.id"), nullable=False)
    llm_config_id = Column(Integer, ForeignKey("llm_configs.id"), nullable=False)
    status = Column(String(50), default="pending")
    progress = Column(Float, default=0.0)
    total_rows = Column(Integer, nullable=True)
    completed_rows = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())
    summary_scores = Column(JSON, nullable=True)
    scenario_snapshot = Column(JSON, nullable=True)
    logs = Column(Text, default="")

    dataset = relationship("Dataset")
    scenario = relationship("EvalScenario")
    llm_config = relationship("LLMConfig")
    row_results = relationship("EvalRowResult", back_populates="eval_task", cascade="all, delete-orphan")


class EvalRowResult(Base):
    __tablename__ = "eval_row_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    eval_task_id = Column(Integer, ForeignKey("eval_tasks.id", ondelete="CASCADE"), nullable=False)
    dataset_row_id = Column(Integer, ForeignKey("dataset_rows.id"), nullable=False)
    row_index = Column(Integer, nullable=False)
    metric_scores = Column(JSON, nullable=True)
    is_pass = Column(Boolean, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)
    manual_status = Column(String(50), nullable=True)
    manual_score = Column(Float, nullable=True)
    manual_tags = Column(JSON, nullable=True)
    manual_note = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())

    eval_task = relationship("EvalTask", back_populates="row_results")
    dataset_row = relationship("DatasetRow")


class BlindTestTask(Base):
    __tablename__ = "blind_test_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    status = Column(String(50), default="pending")
    progress = Column(Float, default=0.0)
    total_rows = Column(Integer, nullable=True)
    completed_rows = Column(Integer, default=0)
    voted_rows = Column(Integer, default=0)
    sample_limit = Column(Integer, nullable=True)
    target_a = Column(JSON, nullable=False)
    target_b = Column(JSON, nullable=False)
    error_message = Column(Text, nullable=True)
    logs = Column(Text, default="")
    summary = Column(JSON, nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())

    dataset = relationship("Dataset")
    row_results = relationship("BlindTestRowResult", back_populates="blind_test_task", cascade="all, delete-orphan")


class BlindTestRowResult(Base):
    __tablename__ = "blind_test_row_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    blind_test_task_id = Column(Integer, ForeignKey("blind_test_tasks.id", ondelete="CASCADE"), nullable=False)
    dataset_row_id = Column(Integer, ForeignKey("dataset_rows.id"), nullable=False)
    row_index = Column(Integer, nullable=False)
    answer_a = Column(Text, nullable=True)
    answer_b = Column(Text, nullable=True)
    answer_a_error = Column(Text, nullable=True)
    answer_b_error = Column(Text, nullable=True)
    display_order = Column(JSON, nullable=False)
    vote = Column(String(50), nullable=True)
    vote_note = Column(Text, nullable=True)
    voted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())

    blind_test_task = relationship("BlindTestTask", back_populates="row_results")
    dataset_row = relationship("DatasetRow")
