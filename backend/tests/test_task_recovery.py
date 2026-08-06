"""Regression tests for lifespan recovery: must not reap tasks whose worker is alive.

背景（2026-08-06 线上事故）：任务 11/12 被 pytest 的 TestClient 误杀——conftest 的
client fixture 进入 `with TestClient(app)` 时执行 lifespan startup →
_recover_interrupted_eval_tasks() 无条件把所有 running 任务标 failed。
修复：任务启动时记录 worker_pid（后端进程 pid），recovery 只回收 worker 已死的任务。
"""
import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.evaluation import EvalTask

# 与 conftest 相同的测试库（recovery 函数会自建/自关 session，不能共用 fixture session）
TEST_ENGINE = create_engine(
    "sqlite:///./test_recovery.db",
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)
Base.metadata.create_all(bind=TEST_ENGINE)


@pytest.fixture(autouse=True)
def _clean_tasks():
    """每个测试独立：清空上次运行残留的任务，避免 recovery 计数被污染。"""
    session = TestingSessionLocal()
    try:
        session.query(EvalTask).delete()
        session.commit()
    finally:
        session.close()


def _seed_task(name: str, worker_pid) -> int:
    session = TestingSessionLocal()
    try:
        task = EvalTask(
            name=name,
            dataset_id=1,
            scenario_id=1,
            llm_config_id=1,
            status="running",
            progress=0.1,
            total_rows=25,
            completed_rows=2,
            worker_pid=worker_pid,
        )
        session.add(task)
        session.commit()
        return task.id
    finally:
        session.close()


def _task_status(task_id: int) -> tuple[str, str | None]:
    session = TestingSessionLocal()
    try:
        task = session.query(EvalTask).filter(EvalTask.id == task_id).first()
        return task.status, task.error_message
    finally:
        session.close()


def _run_recovery(monkeypatch) -> int:
    import app.core.database as db_mod
    from app.main import _recover_interrupted_eval_tasks

    monkeypatch.setattr(db_mod, "SessionLocal", TestingSessionLocal)
    return _recover_interrupted_eval_tasks()


def test_recovery_skips_task_with_alive_worker(monkeypatch):
    task_id = _seed_task("存活worker任务", os.getpid())

    recovered = _run_recovery(monkeypatch)

    assert recovered == 0
    status, _ = _task_status(task_id)
    assert status == "running"


def test_recovery_reaps_task_with_dead_worker(monkeypatch):
    task_id = _seed_task("死worker任务", 999999)

    recovered = _run_recovery(monkeypatch)

    assert recovered == 1
    status, error = _task_status(task_id)
    assert status == "failed"
    assert "服务重启导致评测任务中断" in error


def test_recovery_reaps_legacy_task_without_worker_pid(monkeypatch):
    """旧任务没有 worker_pid（列后加），仍按原行为回收。"""
    task_id = _seed_task("无worker_pid旧任务", None)

    recovered = _run_recovery(monkeypatch)

    assert recovered == 1
    status, _ = _task_status(task_id)
    assert status == "failed"


def test_recovery_mixed_alive_and_dead_workers(monkeypatch):
    alive_id = _seed_task("存活worker任务2", os.getpid())
    dead_id = _seed_task("死worker任务2", 999999)

    recovered = _run_recovery(monkeypatch)

    assert recovered == 1
    alive_status, _ = _task_status(alive_id)
    dead_status, _ = _task_status(dead_id)
    assert alive_status == "running"
    assert dead_status == "failed"
