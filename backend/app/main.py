import os
import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import engine, Base, ensure_runtime_schema
from app.core.ws_manager import manager


def _configure_app_logging() -> None:
    """Send project logger output to the console when running with uvicorn.

    Uvicorn configures its own access/error loggers, but application loggers
    such as ``app.core.evaluation_engine`` may not have a console handler in
    local ``uv run ./run.py`` startup mode. This keeps debug traces like the
    full Judge Prompt visible without changing every module's logger setup.
    """

    level_name = os.getenv("APP_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    app_logger = logging.getLogger("app")
    app_logger.setLevel(level)

    if not app_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        )
        app_logger.addHandler(handler)

    app_logger.propagate = False


_configure_app_logging()

# Import all models so they register with Base before create_all
import app.models  # noqa: F401


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Startup: create all tables
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema()

    # Bind the server event loop for cross-thread WebSocket progress push
    manager.bind_loop(asyncio.get_running_loop())

    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    if settings.SEED_ON_STARTUP:
        # Run seed data (import here to avoid circular imports)
        from app.seed import run_seed
        from app.core.database import SessionLocal
        db = SessionLocal()
        try:
            run_seed(db)
        finally:
            db.close()

    _backfill_eval_task_scenario_snapshots()
    _recover_interrupted_eval_tasks()

    yield

    # Shutdown: nothing to clean up for now


def _backfill_eval_task_scenario_snapshots() -> None:
    """Freeze current scenario configs for historical tasks created before snapshots."""

    from sqlalchemy.orm import joinedload

    from app.core.database import SessionLocal
    from app.core.scenario_snapshot import build_scenario_snapshot
    from app.models.evaluation import EvalTask
    from app.models.scenario import EvalScenario, ScenarioMetric

    db = SessionLocal()
    try:
        tasks = (
            db.query(EvalTask)
            .filter(EvalTask.scenario_snapshot.is_(None))
            .all()
        )
        if not tasks:
            return

        scenario_ids = {task.scenario_id for task in tasks}
        scenarios = (
            db.query(EvalScenario)
            .options(
                joinedload(EvalScenario.metrics).joinedload(ScenarioMetric.metric_definition)
            )
            .filter(EvalScenario.id.in_(scenario_ids))
            .all()
        )
        scenario_by_id = {scenario.id: scenario for scenario in scenarios}
        for task in tasks:
            scenario = scenario_by_id.get(task.scenario_id)
            if scenario is not None:
                task.scenario_snapshot = build_scenario_snapshot(scenario)
        db.commit()
    finally:
        db.close()


def _recover_interrupted_eval_tasks() -> int:
    """Mark tasks left in pending/running by a previous process as failed.

    评测任务由 daemon 线程执行，服务重启后这些线程随之消失；启动时统一
    回收，避免任务永久悬挂在运行中状态。
    """
    from datetime import datetime, timezone

    from app.core.database import SessionLocal
    from app.models.evaluation import EvalTask

    db = SessionLocal()
    try:
        tasks = (
            db.query(EvalTask)
            .filter(EvalTask.status.in_(["pending", "running"]))
            .all()
        )
        for task in tasks:
            task.status = "failed"
            task.error_message = "服务重启导致评测任务中断，请重新创建评测任务"
            task.finished_at = datetime.now(timezone.utc)
        db.commit()
        return len(tasks)
    finally:
        db.close()


app = FastAPI(
    title="AI Evaluation Platform",
    description="Backend API for the AI Evaluation Platform",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from app.api.llm_config import router as llm_config_router
from app.api.dataset import router as dataset_router
from app.api.metric import router as metric_router
from app.api.endpoint_target import router as endpoint_target_router
from app.api.scenario import router as scenario_router
from app.api.evaluation import router as evaluation_router
from app.api.report import router as report_router
from app.api.rag_dataset_job import router as rag_dataset_job_router
from app.api.blind_test import router as blind_test_router
from app.api.ws import router as ws_router

app.include_router(llm_config_router, prefix="/api")
app.include_router(dataset_router, prefix="/api")
app.include_router(metric_router, prefix="/api")
app.include_router(endpoint_target_router, prefix="/api")
app.include_router(scenario_router, prefix="/api")
app.include_router(evaluation_router, prefix="/api")
app.include_router(report_router, prefix="/api")
app.include_router(rag_dataset_job_router, prefix="/api")
app.include_router(blind_test_router, prefix="/api")
# WebSocket 路由走独立的 /ws 前缀（与 vite 代理 /ws 对齐）
app.include_router(ws_router)

# Static file serving for uploads
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
