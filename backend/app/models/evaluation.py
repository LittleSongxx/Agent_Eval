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
    created_at = Column(DateTime, default=func.now())

    eval_task = relationship("EvalTask", back_populates="row_results")
    dataset_row = relationship("DatasetRow")
