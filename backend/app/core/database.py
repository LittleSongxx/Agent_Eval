from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def ensure_runtime_schema() -> None:
    """Apply tiny SQLite-compatible schema additions for existing local DBs."""

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
    }
    missing_eval_task_sql = [
        sql for name, sql in eval_task_additions.items() if name not in eval_task_columns
    ]
    if missing_eval_task_sql:
        with engine.begin() as connection:
            for sql in missing_eval_task_sql:
                connection.execute(text(sql))

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


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
