"""取消评测（P1-4 配套）：取消信号的检测、传播与落库语义。

为什么必须单独补这一份：
  取消机制在并发化里被整体换掉了。串行版的做法是"每个指标开始前
  `db.refresh(task)` 看一眼状态"——25 行 × 6 指标就是 150 次刷库，而且
  refresh 用的是调度侧那个正在写事务的 session。并发化之后改成了两段结构：

    1. 一个独立 session 的轮询协程 `_watch_for_cancel()`，每
       CANCEL_POLL_INTERVAL_SECONDS 秒读一次状态，读到 cancelled 就 set 一个
       进程内的 `asyncio.Event`；
    2. 所有行共享这个 Event，在三个位置检查它：借裁判客户端之前、行开始
       之前、以及行内每个指标边界。

  换掉之后整套机制的测试覆盖是 0——`grep -c cancel tests/` 在这份文件之前
  返回 0。也就是说"点了取消到底还停不停"这件事，完全没有任何断言保护。

关于这份文件的分工：
  - 行内语义（预闸门、指标边界 break）用**直接调用 `_evaluate_single_row`**
    来测。这类测试完全确定，不依赖任何时序。
  - 端到端语义（轮询协程真的读到了别的 session 提交的状态、取消后不写
    summary、状态不被 completed 覆盖）必须走 `run_evaluation`，这类测试
    天然带时序，所以断言写成区间而不是精确值——被测的性质是"提前停下来"，
    不是"正好停在第 4 行"。
"""

import asyncio

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from .test_eval_concurrency import _setup_multirow_prerequisites, _StubJudge


class _FakeMetric:
    """最小指标替身，只提供 `ascore`。

    可以传一个 `on_score` 钩子，用来在"第 N 个指标正在打分"这个精确时刻
    触发取消——这是测试指标边界 break 唯一可靠的注入点。
    """

    def __init__(self, name, on_score=None):
        self.name = name
        self.on_score = on_score
        self.calls = 0

    async def ascore(self, row_data, judge):
        from app.core.evaluation_engine import _MetricResult

        self.calls += 1
        if self.on_score is not None:
            self.on_score(self.name)
        return _MetricResult(1.0, f"{self.name} scored")


class _FakeScenarioMetric:
    """`metrics` 三元组里的第三项，只有阈值判定会用到。"""

    def __init__(self, weight=1.0, pass_threshold=0.5):
        self.weight = weight
        self.pass_threshold = pass_threshold


def _row_kwargs(metrics, cancel_event):
    """`_evaluate_single_row` 的关键字参数——它是全 keyword-only 的。"""
    return {
        "task_id": 1,
        "row_index": 7,
        "row_position": 1,
        "total_rows": 1,
        "base_row_data": {"user_input": "q", "response": "a"},
        "metrics": metrics,
        "judge_client": _StubJudge(None),
        "panel_judges": [],
        "panel_judge_names": [],
        "primary_judge_name": "test-model",
        "evaluation_mode": "dataset",
        "target_config": {},
        "response_mapping": {},
        "cancel_event": cancel_event,
    }


# ---------------------------------------------------------------------------
# 行内语义：确定性测试，不依赖时序
# ---------------------------------------------------------------------------


def test_row_returns_cancelled_without_scoring_when_event_already_set():
    """取消信号已置位时，这一行一个指标都不该打。

    对应 `_run_row` 里"借客户端前就发现已取消"之后进入本函数的情形，以及
    并发闸门排队的行——闸门上可能积着十几行，取消之后它们必须直接返回，
    而不是排到自己了再去调一次 Judge。每一次多调都是真金白银。
    """
    from app.core.evaluation_engine import _evaluate_single_row

    cancel_event = asyncio.Event()
    cancel_event.set()
    metric = _FakeMetric("m1")
    metrics = [("m1", metric, _FakeScenarioMetric())]

    result = asyncio.run(_evaluate_single_row(**_row_kwargs(metrics, cancel_event)))

    assert result["cancelled"] is True
    assert result["row_index"] == 7, "取消返回值也必须带 row_index，否则调度侧没法对账"
    assert metric.calls == 0, "已取消却仍然调用了 Judge"
    assert "metric_scores" not in result, "取消的行不该产生分数，否则会被当成有效结果落库"


def test_row_stops_at_next_metric_boundary_when_cancelled_midway():
    """取消信号在行内到达时，停在下一个指标边界，不打断正在进行的指标。

    行内指标是串行的（为了让 token 核算按"行 × 指标"归属正确），所以取消的
    粒度只能到指标边界。这里验证的是：第 1 个指标打完，第 2、3 个不再打。

    刻意不做"立刻中断当前指标"：Judge 调用已经发出去了，token 已经花了，
    半路扔掉只会得到一个既花了钱又没有结果的空洞。
    """
    from app.core.evaluation_engine import _evaluate_single_row

    cancel_event = asyncio.Event()

    def cancel_after_first(name):
        if name == "m1":
            cancel_event.set()

    m1 = _FakeMetric("m1", on_score=cancel_after_first)
    m2 = _FakeMetric("m2")
    m3 = _FakeMetric("m3")
    metrics = [
        ("m1", m1, _FakeScenarioMetric()),
        ("m2", m2, _FakeScenarioMetric()),
        ("m3", m3, _FakeScenarioMetric()),
    ]

    result = asyncio.run(_evaluate_single_row(**_row_kwargs(metrics, cancel_event)))

    assert m1.calls == 1, "第一个指标应正常完成"
    assert m2.calls == 0 and m3.calls == 0, "取消后仍在继续打后续指标"
    assert result["cancelled"] is True
    boundary_logs = [line for line in result["logs"] if "用户取消评测" in line]
    assert boundary_logs, f"缺少指标边界取消日志，实际日志: {result['logs']}"
    assert "m2" in boundary_logs[0], "日志里应指明停在哪个指标上，否则排查时不知道停在哪"


def test_uncancelled_row_returns_full_result():
    """对照组：没有取消时必须走完整路径。

    没有这个对照，上面两个测试无法排除"函数在任何情况下都返回 cancelled"
    这种假绿。
    """
    from app.core.evaluation_engine import _evaluate_single_row

    m1 = _FakeMetric("m1")
    m2 = _FakeMetric("m2")
    metrics = [("m1", m1, _FakeScenarioMetric()), ("m2", m2, _FakeScenarioMetric())]

    result = asyncio.run(_evaluate_single_row(**_row_kwargs(metrics, asyncio.Event())))

    assert result["cancelled"] is False
    assert m1.calls == 1 and m2.calls == 1
    assert set(result["metric_scores"]) == {"m1", "m2"}
    assert result["execution_time_ms"] >= 0


# ---------------------------------------------------------------------------
# 端到端语义：轮询协程 + 调度侧
# ---------------------------------------------------------------------------


def _cancel_in_background_thread(engine, task_id):
    """在**另一个线程**里用独立 session 提交取消状态。

    为什么必须放到线程里（这是这份文件最容易踩的坑）：
      调度侧 session 每完成一行 commit 一次，紧接着 `_broadcast_progress(task)`
      会读 task 属性；因为 `expire_on_commit=True`，这次读会立刻发起一条
      SELECT，也就是**又开了一个读事务**，SQLite 下对应一个 SHARED 锁，且
      一直挂到下一次 commit/rollback。

      此时如果在协程里同步写库，写方 commit 需要 EXCLUSIVE 锁，必须等那个
      SHARED 锁释放；而 SHARED 锁要等事件循环继续跑到下一次 commit 才释放；
      但事件循环正被这次同步写阻塞着——死锁，直到 busy_timeout 超时报
      "database is locked"。

      放到线程里就没这个问题：事件循环继续跑，下一行 commit 后锁自然释放，
      写入随即成功。这同时也更贴近真实拓扑——线上 cancel 接口跑在 FastAPI
      的线程池里，评测跑在事件循环里，本来就是两个执行体。
    """
    from app.models.evaluation import EvalTask

    def _write():
        session = sessionmaker(bind=engine)()
        try:
            # 抬高 busy_timeout：要等调度侧下一次 commit 释放读锁
            session.execute(text("PRAGMA busy_timeout = 30000"))
            session.query(EvalTask).filter(EvalTask.id == task_id).update(
                {"status": "cancelled"}
            )
            session.commit()
        finally:
            session.close()

    return asyncio.to_thread(_write)


def _run_with_midrun_cancel(
    client,
    db,
    monkeypatch,
    test_llm_payload,
    poll_interval,
    row_count=16,
    concurrency=4,
    expire_on_commit=True,
    delay=0.05,
):
    """跑一个任务，并在第一次指标调用时从另一个线程提交取消。

    `expire_on_commit` 可配是为了钉住调度侧那句 `db.refresh(task)`：默认
    True 时 commit 会让 task 的属性全部过期，之后读 `task.status` 本身就会
    重新查库，refresh 因此是冗余的；改成 False 之后属性不再过期，refresh
    就是唯一能看到"别的 session 刚提交的 cancelled"的途径。
    """
    from app.core.config import settings
    from app.core.evaluation_engine import _MetricResult, run_evaluation

    _StubJudge.instances = []
    monkeypatch.setattr(settings, "EVAL_ROW_CONCURRENCY", concurrency)
    monkeypatch.setattr(
        "app.core.evaluation_engine.CANCEL_POLL_INTERVAL_SECONDS", poll_interval
    )
    monkeypatch.setattr("app.core.evaluation_engine.OpenAIJudgeClient", _StubJudge)

    llm, dataset, scenario = _setup_multirow_prerequisites(
        client, test_llm_payload, row_count=row_count
    )
    task = client.post(
        "/api/evaluations",
        json={
            "name": "Cancel Run",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
        },
    ).json()

    engine = db.get_bind()
    state = {"rows_started": 0, "cancel_sent": False}

    async def fake_ascore(_self, row_data, _judge):
        state["rows_started"] += 1
        if not state["cancel_sent"]:
            state["cancel_sent"] = True
            await _cancel_in_background_thread(engine, task["id"])
        await asyncio.sleep(delay)
        return _MetricResult(1.0, f"scored {row_data.get('user_input')}")

    monkeypatch.setattr("app.core.evaluation_engine.NativePromptMetric.ascore", fake_ascore)

    TestingSession = sessionmaker(bind=engine, expire_on_commit=expire_on_commit)
    asyncio.run(run_evaluation(task["id"], TestingSession))
    return task, state


def test_cancel_mid_run_stops_remaining_rows(client, db, monkeypatch, test_llm_payload):
    """运行中提交取消，剩余行必须停下来，任务落到 cancelled。

    这条同时是"轮询协程真的读到了别的 session 的提交"的守卫。

    关于轮询里那句 `watch_db.rollback()`：写它的时候我以为它是承重的——
    以为不 rollback 就会一直读到旧快照。实测把它换成 pass，9 个测试全绿。
    原因是 pysqlite 默认 `isolation_level=''`，驱动只在 DML 前隐式开事务，
    纯 SELECT 根本没发出过 BEGIN，所以每次读都能看到最新提交。
    （SQLAlchemy 侧 `in_transaction()` 返回 True 只是它自己的记账。）
    所以那行是防御性的，不是承重的，注释已按实测改写。

    断言写成区间而不是精确行数：轮询是按真实时间走的，"取消正好落在第几行"
    不可能稳定。被测的性质是提前停止，16 行里停在 8 行以内即可证明。
    """
    from app.models.evaluation import EvalRowResult, EvalTask

    task, state = _run_with_midrun_cancel(
        client, db, monkeypatch, test_llm_payload, poll_interval=0.005
    )

    db.expire_all()
    stored = db.query(EvalTask).filter(EvalTask.id == task["id"]).one()
    persisted = (
        db.query(EvalRowResult).filter(EvalRowResult.eval_task_id == task["id"]).count()
    )

    assert stored.status == "cancelled", f"取消后状态应为 cancelled，实际 {stored.status}"
    assert persisted < 16, f"取消未生效：16 行全部落库（{persisted}）"
    assert state["rows_started"] <= 8, (
        f"取消信号未能拦住后续批次：已开始 {state['rows_started']} 行"
    )
    assert stored.finished_at is not None, "取消也是终态，必须写 finished_at"
    assert "评测已取消" in (stored.logs or ""), "日志里应留下取消记录"


def test_cancelled_task_does_not_get_summary_scores(
    client, db, monkeypatch, test_llm_payload
):
    """取消的任务不能生成汇总统计。

    半个数据集算出来的均值、通过率、p95 会被报告页当成完整结果展示，比没有
    数字更危险——它看起来像一次正常评测的结论。
    """
    from app.models.evaluation import EvalTask

    task, _state = _run_with_midrun_cancel(
        client, db, monkeypatch, test_llm_payload, poll_interval=0.005
    )

    db.expire_all()
    stored = db.query(EvalTask).filter(EvalTask.id == task["id"]).one()

    assert stored.status == "cancelled"
    assert not (stored.summary_scores or {}), (
        f"取消任务却写了汇总统计: {stored.summary_scores}"
    )
    assert (stored.progress or 0) < 1.0, "取消任务的进度不该被写成 100%"


def test_poll_interval_constant_governs_detection_latency(
    client, db, monkeypatch, test_llm_payload
):
    """轮询间隔常量必须真的被轮询协程使用。

    这条是给常量抽取本身上的锁，而这个锁是有讲究的。间隔原先硬编码成 1.0
    写在 `wait_for(...)` 里；抽成 CANCEL_POLL_INTERVAL_SECONDS 时，"声明了
    常量但 wait_for 里仍写着 1.0" 是一种极常见的半途编辑——本次就真的发生
    过一次（回滚踩到旧备份，常量声明和引用一起丢了）。

    这里踩到的坑值得记下来：这个测试的第一版**杀不掉那个变异**。
    原因是量纲——16 行 ÷ 并发 4 × 50ms ≈ 200ms，整个任务比 1 秒短得多，
    所以硬编码 1.0 的轮询协程在任务结束前根本没醒过。间隔是 60 还是 1，
    结果完全一样，测试照绿。也就是说：一个"对照组"只有在对照量足够大时
    才是对照组，否则它只是把两个都没触发的分支放在一起比。

    修法是把单行耗时抬到 0.4s，总时长约 1.6s，跨过 1 秒这条线：
      - 常量接上了（=60s）：轮询在任务结束前一次都不触发 → 16 行全跑完
      - 常量没接上（硬编码 1.0）：约 1s 处触发 → 最后一批行被闸门拦掉
    两种结果的差异此时只可能来自这个常量。
    """
    from app.models.evaluation import EvalRowResult, EvalTask

    # delay 必须让总时长跨过 1 秒，否则硬编码 1.0 的变异体测不出来（见 docstring）
    task, state = _run_with_midrun_cancel(
        client, db, monkeypatch, test_llm_payload, poll_interval=60.0, delay=0.4
    )

    db.expire_all()
    stored = db.query(EvalTask).filter(EvalTask.id == task["id"]).one()
    persisted = (
        db.query(EvalRowResult).filter(EvalRowResult.eval_task_id == task["id"]).count()
    )

    assert state["rows_started"] == 16, (
        "轮询间隔被调到 60s，任务全程不该检测到取消，16 行应全部执行；"
        f"实际只跑了 {state['rows_started']} 行——说明 wait_for 没走常量"
    )
    assert persisted == 16
    # 全部跑完，但状态仍是 cancelled——见下一个测试
    assert stored.status == "cancelled"


def test_completed_run_does_not_overwrite_a_cancel_written_late(
    client, db, monkeypatch, test_llm_payload
):
    """所有行都跑完了，也不能把用户的 cancelled 覆盖成 completed。

    这是取消语义里最容易被忽略的一条：取消可能在最后一行完成之后、写终态
    之前才落库。此时所有行都已正常评完，调度侧如果按"跑完了就是 completed"
    写状态，用户点的取消就凭空消失了。

    关于 `db.refresh(task)`：写代码时我以为是它在承重。实测把那行删掉，本
    测试仍然绿——因为 session 是 `expire_on_commit=True`，逐行 commit 已经
    把 `task` 的属性全部过期了，接下来读 `task.status` 本身就会重新查库。
    也就是说当前配置下 refresh 是冗余的。

    但它值得留着，而且这条测试真正钉住的是**行为**而不是那一行代码：
    "跑满也不能覆盖 cancelled" 这个性质，在 `expire_on_commit=False`（一个
    很合理的后续优化，正好能省掉我为并发特意 hoist 的那些 frozen_* 取值）
    下就完全依赖显式 refresh 了。断言写在行为上，两种配置都能守住。
    """
    from app.models.evaluation import EvalRowResult, EvalTask

    task, state = _run_with_midrun_cancel(
        client, db, monkeypatch, test_llm_payload, poll_interval=60.0
    )

    db.expire_all()
    stored = db.query(EvalTask).filter(EvalTask.id == task["id"]).one()

    assert state["rows_started"] == 16, "前置条件：这一轮所有行都应跑完"
    assert (
        db.query(EvalRowResult).filter(EvalRowResult.eval_task_id == task["id"]).count()
        == 16
    )
    assert stored.status == "cancelled", (
        f"cancelled 被覆盖成 {stored.status}——跑满的任务把用户的取消吃掉了"
    )
    assert not (stored.summary_scores or {}), "被取消的任务即使跑满也不该出汇总"


def test_cancel_survives_even_when_session_does_not_expire_on_commit(
    client, db, monkeypatch, test_llm_payload
):
    """`expire_on_commit=False` 下，取消状态同样不能被覆盖成 completed。

    上一条测试说明了在默认的 `expire_on_commit=True` 下，逐行 commit 顺手把
    `task` 的属性过期掉了，所以 `db.refresh(task)` 是冗余的——删掉也全绿。
    这条测试补的正是那个缺口：把 session 换成 `expire_on_commit=False`，
    commit 不再过期属性，session 里的 `task.status` 会一直停留在引擎自己
    写进去的 "running"。此时唯一能看见用户取消的途径就是那次显式 refresh。

    为什么要专门覆盖这个配置：`expire_on_commit=False` 不是假想情况，而是
    这个引擎接下来最可能做的一次优化。并发化时我为了避免 commit 后的隐式
    懒加载，专门把 target_config / response_mapping 等 hoist 成了 frozen_*
    纯 dict；把 expire_on_commit 关掉是同一个问题更彻底的解法。真去关的时候，
    取消语义不该跟着一起悄悄坏掉。

    删掉引擎里的 `db.refresh(task)`，这条会红，上一条依然绿。
    """
    from app.models.evaluation import EvalRowResult, EvalTask

    task, state = _run_with_midrun_cancel(
        client,
        db,
        monkeypatch,
        test_llm_payload,
        poll_interval=60.0,
        expire_on_commit=False,
    )

    db.expire_all()
    stored = db.query(EvalTask).filter(EvalTask.id == task["id"]).one()

    assert state["rows_started"] == 16, "前置条件：这一轮所有行都应跑完"
    assert (
        db.query(EvalRowResult).filter(EvalRowResult.eval_task_id == task["id"]).count()
        == 16
    )
    assert stored.status == "cancelled", (
        f"expire_on_commit=False 下 cancelled 被覆盖成 {stored.status}"
        "——写终态前少了 db.refresh(task)"
    )
    assert not (stored.summary_scores or {}), "被取消的任务即使跑满也不该出汇总"


def test_cancel_endpoint_marks_task_cancelled(client, db, test_llm_payload):
    """接口契约：cancel 端点只负责把状态写成 cancelled。

    引擎侧完全依赖这一个字段来判定取消，所以这个契约要单独钉住——端点如果
    改成写别的字段（比如加一个 cancel_requested 标记），引擎不会报错，只会
    再也停不下来。
    """
    from app.models.evaluation import EvalTask

    llm, dataset, scenario = _setup_multirow_prerequisites(
        client, test_llm_payload, row_count=2
    )
    task = client.post(
        "/api/evaluations",
        json={
            "name": "Cancel Contract",
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
        },
    ).json()

    resp = client.post(f"/api/evaluations/{task['id']}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"

    db.expire_all()
    stored = db.query(EvalTask).filter(EvalTask.id == task["id"]).one()
    assert stored.status == "cancelled"


def test_cancel_endpoint_404_for_unknown_task(client):
    """取消一个不存在的任务应当 404，而不是 500。"""
    assert client.post("/api/evaluations/999999/cancel").status_code == 404
