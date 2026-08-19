from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import settings

database_url = make_url(settings.DATABASE_URL)
engine_options = {"pool_pre_ping": True}
if database_url.drivername.startswith("sqlite"):
    engine_options["connect_args"] = {"check_same_thread": False}

engine = create_engine(settings.DATABASE_URL, **engine_options)


if database_url.drivername.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def ensure_runtime_schema() -> None:
    """Apply tiny SQLite-compatible schema additions for existing local DBs."""

    if not engine.dialect.name.startswith("sqlite"):
        return

    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if "eval_tasks" not in table_names:
        return

    eval_task_columns = {column["name"] for column in inspector.get_columns("eval_tasks")}
    eval_task_additions = {
        "scenario_snapshot": "ALTER TABLE eval_tasks ADD COLUMN scenario_snapshot JSON",
        "evaluation_mode": "ALTER TABLE eval_tasks ADD COLUMN evaluation_mode VARCHAR(50) DEFAULT 'offline'",
        "endpoint_target_id": "ALTER TABLE eval_tasks ADD COLUMN endpoint_target_id INTEGER",
        "target_config": "ALTER TABLE eval_tasks ADD COLUMN target_config JSON",
        "response_mapping": "ALTER TABLE eval_tasks ADD COLUMN response_mapping JSON",
        "result_save_mode": "ALTER TABLE eval_tasks ADD COLUMN result_save_mode VARCHAR(50) DEFAULT 'task_only'",
        "judge_panel": "ALTER TABLE eval_tasks ADD COLUMN judge_panel JSON",
        "dataset_version": "ALTER TABLE eval_tasks ADD COLUMN dataset_version INTEGER",
        "worker_pid": "ALTER TABLE eval_tasks ADD COLUMN worker_pid INTEGER",
        "judge_snapshot": "ALTER TABLE eval_tasks ADD COLUMN judge_snapshot JSON",
        "eval_fingerprint": "ALTER TABLE eval_tasks ADD COLUMN eval_fingerprint VARCHAR(64)",
        "tool_registry_snapshot": "ALTER TABLE eval_tasks ADD COLUMN tool_registry_snapshot JSON",
    }
    missing_eval_task_sql = [
        sql for name, sql in eval_task_additions.items() if name not in eval_task_columns
    ]
    if missing_eval_task_sql:
        with engine.begin() as connection:
            for sql in missing_eval_task_sql:
                connection.execute(text(sql))

    if "datasets" in table_names:
        dataset_columns = {column["name"] for column in inspector.get_columns("datasets")}
        if "version" not in dataset_columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE datasets ADD COLUMN version INTEGER DEFAULT 1 NOT NULL"))

    if "rag_dataset_jobs" in table_names:
        rag_job_columns = {column["name"] for column in inspector.get_columns("rag_dataset_jobs")}
        additions = {
            "target_endpoint_url": "ALTER TABLE rag_dataset_jobs ADD COLUMN target_endpoint_url VARCHAR(1000)",
            "target_transport_mode": "ALTER TABLE rag_dataset_jobs ADD COLUMN target_transport_mode VARCHAR(50) DEFAULT 'sse'",
            "target_authorization": "ALTER TABLE rag_dataset_jobs ADD COLUMN target_authorization TEXT",
            "target_extra_headers": "ALTER TABLE rag_dataset_jobs ADD COLUMN target_extra_headers TEXT",
            "target_request_body_template": "ALTER TABLE rag_dataset_jobs ADD COLUMN target_request_body_template TEXT",
        }
        missing_sql = [sql for name, sql in additions.items() if name not in rag_job_columns]
        if missing_sql:
            with engine.begin() as connection:
                for sql in missing_sql:
                    connection.execute(text(sql))

    if "eval_row_results" in table_names:
        row_result_columns = {column["name"] for column in inspector.get_columns("eval_row_results")}
        additions = {
            "manual_status": "ALTER TABLE eval_row_results ADD COLUMN manual_status VARCHAR(50)",
            "manual_score": "ALTER TABLE eval_row_results ADD COLUMN manual_score FLOAT",
            "manual_tags": "ALTER TABLE eval_row_results ADD COLUMN manual_tags JSON",
            "manual_note": "ALTER TABLE eval_row_results ADD COLUMN manual_note TEXT",
            "reviewed_at": "ALTER TABLE eval_row_results ADD COLUMN reviewed_at DATETIME",
            "endpoint_trace": "ALTER TABLE eval_row_results ADD COLUMN endpoint_trace JSON",
            "badcase_category": "ALTER TABLE eval_row_results ADD COLUMN badcase_category VARCHAR(50)",
            "badcase_confidence": "ALTER TABLE eval_row_results ADD COLUMN badcase_confidence FLOAT",
            "badcase_source": "ALTER TABLE eval_row_results ADD COLUMN badcase_source VARCHAR(100)",
        }
        missing_sql = [sql for name, sql in additions.items() if name not in row_result_columns]
        if missing_sql:
            with engine.begin() as connection:
                for sql in missing_sql:
                    connection.execute(text(sql))

    if "scenario_metrics" in table_names:
        scenario_metric_columns = {column["name"] for column in inspector.get_columns("scenario_metrics")}
        if "prompt_override" not in scenario_metric_columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE scenario_metrics ADD COLUMN prompt_override TEXT"))

    _backfill_row_annotations(table_names)


def _backfill_row_annotations(table_names: list[str]) -> None:
    """把历史 manual_* 标注搬进 row_annotations，归到默认标注者名下。

    这是本文件里**唯一**一处写数据（而非改表结构）的迁移，所以三条性质都要单独保证：

    1. **幂等**。只处理"还没有任何标注记录"的行。重复调用（每次进程启动都会调）
       不会产生第二条标注，也不会覆盖此后真人改过的标注。
    2. **不碰 manual_\\***。这五列此后是投影，但迁移期间它们同时也是**唯一的数据源**。
       迁移里若顺手重算投影，一旦回填逻辑有 bug 就会把原始标注就地改坏，
       且没有第二份副本可对照。所以这里只读不写；投影的正确性由
       `test_projection_has_no_drift_on_real_labels` 事后验证。
    3. **created_at 取 reviewed_at**。投影用 max(created_at) 反推 reviewed_at，
       若这里让 created_at 走 DB 默认值（当前时间），那么下一次投影重算会把
       25 条历史标注的复核时间全部改成迁移那一刻——已发布报告的时间线会被静默篡改。
       reviewed_at 为 NULL 的行按同样理由跳过：给不出可信时间就不搬。
    """

    if "eval_row_results" not in table_names or "row_annotations" not in table_names:
        return

    from app.core.annotation import DEFAULT_ANNOTATOR

    with engine.begin() as connection:
        pending = connection.execute(
            text(
                """
                SELECT COUNT(*) FROM eval_row_results r
                WHERE r.reviewed_at IS NOT NULL
                  AND (
                    r.manual_status IS NOT NULL OR r.manual_score IS NOT NULL
                    OR r.manual_tags IS NOT NULL OR r.manual_note IS NOT NULL
                  )
                  AND NOT EXISTS (SELECT 1 FROM row_annotations a WHERE a.row_result_id = r.id)
                """
            )
        ).scalar()
        if not pending:
            return

        connection.execute(
            text(
                """
                INSERT INTO row_annotations
                    (row_result_id, annotator, status, score, tags, note,
                     is_adjudication, created_at, updated_at)
                SELECT r.id, :annotator, r.manual_status, r.manual_score,
                       r.manual_tags, r.manual_note, 0, r.reviewed_at, r.reviewed_at
                FROM eval_row_results r
                WHERE r.reviewed_at IS NOT NULL
                  AND (
                    r.manual_status IS NOT NULL OR r.manual_score IS NOT NULL
                    OR r.manual_tags IS NOT NULL OR r.manual_note IS NOT NULL
                  )
                  AND NOT EXISTS (SELECT 1 FROM row_annotations a WHERE a.row_result_id = r.id)
                """
            ),
            {"annotator": DEFAULT_ANNOTATOR},
        )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
