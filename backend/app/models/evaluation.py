from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    UniqueConstraint,
)
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
    evaluation_mode = Column(String(50), default="offline")
    endpoint_target_id = Column(Integer, ForeignKey("endpoint_targets.id"), nullable=True)
    target_config = Column(JSON, nullable=True)
    response_mapping = Column(JSON, nullable=True)
    result_save_mode = Column(String(50), default="task_only")
    logs = Column(Text, default="")
    # 多裁判面板：附加裁判的 LLM 配置 ID 列表（与主裁判独立打分后聚合）
    judge_panel = Column(JSON, nullable=True)
    # 创建任务时冻结的数据集版本（数据集变更后版本自增，用于追溯评测口径）
    dataset_version = Column(Integer, nullable=True)
    # 创建任务时冻结的裁判身份（模型名/温度/base_url，不含 api_key）。
    # LLMConfig 是原地可改且无版本的，只留 llm_config_id 事后查不出真正打分的模型。
    judge_snapshot = Column(JSON, nullable=True)
    # 评测口径指纹 = 数据 + 尺子 + 裁判。两个任务指纹相同才是严格可比的；
    # 不同则对比接口会列出具体变化维度，而不是把差异都算作被测系统的改进。
    eval_fingerprint = Column(String(64), nullable=True)
    # 执行评测的后端进程 PID：启动 recovery 时据此跳过仍在存活进程里运行的任务，
    # 避免 TestClient / 误启动的 lifespan 把运行中的任务误标失败
    worker_pid = Column(Integer, nullable=True)

    dataset = relationship("Dataset")
    scenario = relationship("EvalScenario")
    llm_config = relationship("LLMConfig")
    endpoint_target = relationship("EndpointTarget")
    row_results = relationship("EvalRowResult", back_populates="eval_task", cascade="all, delete-orphan")


class EvalRowResult(Base):
    __tablename__ = "eval_row_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    eval_task_id = Column(Integer, ForeignKey("eval_tasks.id", ondelete="CASCADE"), nullable=False)
    dataset_row_id = Column(Integer, ForeignKey("dataset_rows.id"), nullable=False)
    row_index = Column(Integer, nullable=False)
    metric_scores = Column(JSON, nullable=True)
    endpoint_trace = Column(JSON, nullable=True)
    is_pass = Column(Boolean, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)
    # 以下 manual_* 五列是 row_annotations 的**投影**，不是标注的存储位置。
    # 真值在 RowAnnotation：一行可以有多个标注者，投影只保留"当前生效的那一个"。
    # 保留投影的原因是向后兼容——前端 46 处引用与 3 个离线脚本都读这几列，
    # 一次性改完等于把重构和迁移风险绑在一起。投影由
    # core/annotation.py::project_annotations_onto_row 单点重算，禁止其他地方直接赋值。
    manual_status = Column(String(50), nullable=True)
    manual_score = Column(Float, nullable=True)
    manual_tags = Column(JSON, nullable=True)
    manual_note = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())

    eval_task = relationship("EvalTask", back_populates="row_results")
    dataset_row = relationship("DatasetRow")
    annotations = relationship(
        "RowAnnotation",
        back_populates="row_result",
        cascade="all, delete-orphan",
        order_by="RowAnnotation.created_at",
    )


class RowAnnotation(Base):
    """单个标注者对单行结果的一次标注。

    为什么要把 manual_* 从 eval_row_results 里拆出来：
      原先一行只有一套 manual_* 列，物理上只能存下一个标注者的判断。
      后果不是"少存了点数据"，而是**人-人 kappa 无法计算**——而人-人 kappa
      正是 judge-人 kappa 的上界。judge 与某位标注者达到 0.82，若两位人类
      之间只有 0.65，那 0.82 measure 的是"judge 学会了这一个人的偏好"，
      不是"judge 接近事实"。少了这个上界，0.82 是没有参照系的数字。

    is_adjudication 区分两种记录：
      - False：一位标注者的独立判断，可参与人-人一致性统计；
      - True ：看过分歧之后做的仲裁，**不得**参与人-人一致性统计
               （仲裁者已知双方答案，与其算一致性是循环论证）。
    """

    __tablename__ = "row_annotations"
    __table_args__ = (
        # 同一标注者对同一行只保留一条独立标注 + 最多一条仲裁记录。
        # 把 is_adjudication 纳入唯一键：同一个人既标注又仲裁是合法的。
        UniqueConstraint(
            "row_result_id", "annotator", "is_adjudication", name="uq_row_annotation_author"
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    row_result_id = Column(
        Integer, ForeignKey("eval_row_results.id", ondelete="CASCADE"), nullable=False, index=True
    )
    annotator = Column(String(100), nullable=False)
    status = Column(String(50), nullable=True)
    score = Column(Float, nullable=True)
    tags = Column(JSON, nullable=True)
    note = Column(Text, nullable=True)
    is_adjudication = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    row_result = relationship("EvalRowResult", back_populates="annotations")


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
