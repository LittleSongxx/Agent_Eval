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
    if "eval_tasks" not in inspector.get_table_names():
        return

    eval_task_columns = {column["name"] for column in inspector.get_columns("eval_tasks")}
    if "scenario_snapshot" not in eval_task_columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE eval_tasks ADD COLUMN scenario_snapshot JSON"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
