"""并发评测（P1-4）：并发真的生效 + 并发不改变结果 + 延迟分位数落库。

为什么需要独立一份测试：
  评测主循环原先是严格串行的 `for row in rows`，每写一行日志就 `db.commit()`
  一次。25 行 × 6 指标的任务里绝大多数墙钟时间都花在等 Judge 返回上，CPU
  基本空转。并发化之后有三类回归风险，每一类都必须被钉住：

  1. **并发没生效**。写了 `asyncio.gather` 不等于真的并行——如果每行之间
     仍隔着一次同步查库（ORM 属性在 commit 后过期会触发懒加载），实际执行
     依然是串行的，而且从结果上完全看不出来：分数一模一样，只是慢。所以
     这里直接测"同时在飞的行数"，不测墙钟时间。
  2. **并发改变了结果**。共用可变状态（`OpenAIJudgeClient.row_usage`）会让
     token 核算互相清零；完成顺序影响汇总会让同一份数据每次跑出不同的
     summary。这两件事都会安静地发生。
  3. **并发数没被遵守**。池子漏了或者闸门写错，25 行会一次全部打到 Judge
     上，线上直接限流。

  注意现有 164 个测试全部用单行数据集，`min(concurrency, total_rows)` 恒等于
  1——也就是说在这份文件之前，并发路径一行代码都没被执行过。
"""

import asyncio

from sqlalchemy.orm import sessionmaker


def metric_name_for(index):
    """第 1 个指标沿用 concurrency_metric，后续加序号。

    保持第一个名字不变，是为了让并发测试里那些按名字断言 per_metric /
    metric_scores 的用例不受"支持多指标"这个改动影响。
    """
    return "concurrency_metric" if index == 0 else f"concurrency_metric_{index + 1}"


def _setup_multirow_prerequisites(client, test_llm_payload, row_count=8, metric_count=1):
    """建一个多行数据集 + N 指标场景。

    行数必须 > 并发数，否则 `min(concurrency, total_rows)` 会把并发压回行数，
    测不到"第 N+1 行在闸门上等待"这个分支。

    metric_count > 1 用于取消测试：行内指标是串行的，只有多指标才存在
    "取消信号到达时这一行停在某个指标边界上"这个分支。
    """
    llm = client.post("/api/llm-configs", json=test_llm_payload).json()
    scenario_metrics = []
    for i in range(metric_count):
        metric = client.post(
            "/api/metrics",
            json={
                "name": metric_name_for(i),
                "display_name": f"Concurrency Metric {i + 1}",
                "metric_type": "aspect_critic",
                "config": {"definition": "Is the response correct?"},
                "category": "custom",
            },
        ).json()
        scenario_metrics.append(
            {
                "metric_definition_id": metric["id"],
                "weight": 1.0,
                "pass_threshold": 0.5,
            }
        )
    scenario = client.post(
        "/api/scenarios",
        json={
            "name": "Concurrency Scenario",
            "description": "concurrency",
            "scene_type": "rag",
            "sample_type": "single_turn",
            "metrics": scenario_metrics,
        },
    ).json()
    dataset = client.post(
        "/api/datasets",
        json={
            "name": "Concurrency Dataset",
            "sample_type": "single_turn",
            "field_schema": [
                {"name": "user_input", "type": "text", "required": True, "description": ""},
                {"name": "response", "type": "text", "required": False, "description": ""},
            ],
        },
    ).json()
    for i in range(row_count):
        client.post(
            f"/api/datasets/{dataset['id']}/rows",
            json={"data": {"user_input": f"question-{i}", "response": f"answer-{i}"}},
        )
    return llm, dataset, scenario


class _StubJudge:
    """最小裁判替身，实现引擎真正依赖的那部分接口。

    刻意实现 reset/take_row_usage 而不是用 `object()`：引擎会调它们做 token
    核算，用不完整的替身会让测试在"未被测到的代码路径"上假绿。
    """

    instances: list["_StubJudge"] = []

    def __init__(self, _llm_config):
        self.row_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        _StubJudge.instances.append(self)

    def reset_row_usage(self):
        self.row_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def take_row_usage(self):
        usage = dict(self.row_usage)
        self.reset_row_usage()
        return usage


def _install_tracking_metric(monkeypatch, delay=0.05, tracker=None):
    """把指标打分替换成"睡一会儿"，并记录并发在飞数量。

    用 sleep 而不是真实调用：需要的是一个可控的 await 点。真实 Judge 的
    延迟抖动会让"在飞数量"的断言变得不稳定。
    """
    state = tracker if tracker is not None else {}
    state.setdefault("in_flight", 0)
    state.setdefault("max_in_flight", 0)
    state.setdefault("judges_seen", [])
    state.setdefault("rows_started", 0)

    async def fake_ascore(self, row_data, judge):
        from app.core.evaluation_engine import _MetricResult

        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        state["rows_started"] += 1
        state["judges_seen"].append(id(judge))
        try:
            await asyncio.sleep(delay)
            return _MetricResult(1.0, f"scored {row_data.get('user_input')}")
        finally:
            state["in_flight"] -= 1

    monkeypatch.setattr("app.core.evaluation_engine.NativePromptMetric.ascore", fake_ascore)
    return state


def _run_task(client, db, llm, dataset, scenario, name="Concurrency Run"):
    task = client.post(
        "/api/evaluations",
        json={
            "name": name,
            "dataset_id": dataset["id"],
            "scenario_id": scenario["id"],
            "llm_config_id": llm["id"],
        },
    ).json()
    from app.core.evaluation_engine import run_evaluation

    TestingSession = sessionmaker(bind=db.get_bind())
    asyncio.run(run_evaluation(task["id"], TestingSession))
    return task


def test_rows_actually_run_concurrently(client, db, monkeypatch, test_llm_payload):
    """并发数 4 时，同时在飞的行数必须真的 > 1。

    这是整个 P1-4 唯一不可替代的断言。分数、行数、通过率在串行和并发下
    完全一致，所以它们都无法证明并发生效——只有"在飞数量"能。
    """
    from app.core.config import settings

    _StubJudge.instances = []
    monkeypatch.setattr(settings, "EVAL_ROW_CONCURRENCY", 4)
    monkeypatch.setattr("app.core.evaluation_engine.OpenAIJudgeClient", _StubJudge)
    state = _install_tracking_metric(monkeypatch, delay=0.05)

    llm, dataset, scenario = _setup_multirow_prerequisites(client, test_llm_payload, row_count=8)
    _run_task(client, db, llm, dataset, scenario)

    assert state["rows_started"] == 8, "8 行都应被评测"
    assert state["max_in_flight"] > 1, (
        f"并发未生效：峰值在飞行数 {state['max_in_flight']}，说明仍在串行执行"
    )


def test_concurrency_respects_configured_cap(client, db, monkeypatch, test_llm_payload):
    """在飞行数不得超过配置的并发上限——超了线上就是限流。"""
    from app.core.config import settings

    _StubJudge.instances = []
    monkeypatch.setattr(settings, "EVAL_ROW_CONCURRENCY", 3)
    monkeypatch.setattr("app.core.evaluation_engine.OpenAIJudgeClient", _StubJudge)
    state = _install_tracking_metric(monkeypatch, delay=0.05)

    llm, dataset, scenario = _setup_multirow_prerequisites(client, test_llm_payload, row_count=9)
    _run_task(client, db, llm, dataset, scenario)

    assert state["max_in_flight"] <= 3, (
        f"并发闸门失效：峰值在飞 {state['max_in_flight']} > 上限 3"
    )
    assert state["max_in_flight"] > 1, "并发数 3 却没有并发发生"


def test_each_concurrent_row_gets_its_own_judge_client(
    client, db, monkeypatch, test_llm_payload
):
    """每个并发槽位一套独立裁判客户端，且客户端总数等于并发数。

    共用一个客户端时，`row_usage` 会被其他行的 reset/take 清掉，成本统计
    静默失真——分数完全正常，只有 token 数字是错的，这是最难发现的一类 bug。
    """
    from app.core.config import settings

    _StubJudge.instances = []
    monkeypatch.setattr(settings, "EVAL_ROW_CONCURRENCY", 4)
    monkeypatch.setattr("app.core.evaluation_engine.OpenAIJudgeClient", _StubJudge)
    state = _install_tracking_metric(monkeypatch, delay=0.05)

    llm, dataset, scenario = _setup_multirow_prerequisites(client, test_llm_payload, row_count=8)
    _run_task(client, db, llm, dataset, scenario)

    # 4 个槽位 → 恰好 4 套客户端（第一套复用主裁判，另建 3 套）
    assert len(_StubJudge.instances) == 4, (
        f"应为每个并发槽位建一套裁判，实际建了 {len(_StubJudge.instances)} 套"
    )
    # 8 行跑在 4 套客户端上，说明池子在被复用而不是每行新建
    assert len(set(state["judges_seen"])) == 4


def test_latency_percentiles_land_in_summary(client, db, monkeypatch, test_llm_payload):
    """p50/p95 必须进 summary_scores。

    修复前 `execution_time_ms` 只逐行写进 eval_row_results，任务级报告里
    没有任何延迟数字——"这套评测跑一轮多久"只能靠人翻行记录自己算。
    """
    from app.core.config import settings
    from app.models.evaluation import EvalTask

    _StubJudge.instances = []
    monkeypatch.setattr(settings, "EVAL_ROW_CONCURRENCY", 4)
    monkeypatch.setattr("app.core.evaluation_engine.OpenAIJudgeClient", _StubJudge)
    _install_tracking_metric(monkeypatch, delay=0.05)

    llm, dataset, scenario = _setup_multirow_prerequisites(client, test_llm_payload, row_count=8)
    task = _run_task(client, db, llm, dataset, scenario)

    db.expire_all()
    stored = db.query(EvalTask).filter(EvalTask.id == task["id"]).one()
    latency = (stored.summary_scores or {}).get("latency")
    assert latency is not None, "summary_scores 缺少 latency 聚合"

    assert latency["row_count"] == 8
    assert latency["row_p50_ms"] is not None
    assert latency["row_p95_ms"] is not None
    assert latency["row_p95_ms"] >= latency["row_p50_ms"]
    assert latency["row_concurrency"] == 4
    assert latency["wall_clock_ms"] > 0
    # 每行都 sleep 了 50ms，逐行耗时之和应明显大于并发后的墙钟
    assert latency["speedup_estimate"] > 1.0, (
        f"加速比 {latency['speedup_estimate']} ≈ 1，说明并发没有真正压缩墙钟时间"
    )
    # 逐指标分位数：最慢指标是优化第一落点，必须有数字支撑
    assert "concurrency_metric" in latency["per_metric"]
    assert latency["slowest_metric"]["name"] == "concurrency_metric"


def test_latency_is_not_exposed_as_a_metric(client, db, monkeypatch, test_llm_payload):
    """latency 是平台级聚合键，不能在指标表里冒充成一个评测指标。

    summary_scores 同时装指标统计和平台级聚合，靠 RESERVED_SUMMARY_KEYS 区分。
    漏登记的话，报告页的指标表会多出一行叫 "latency" 的假指标。
    """
    from app.core.config import settings
    from app.core.evaluation_engine import RESERVED_SUMMARY_KEYS

    assert "latency" in RESERVED_SUMMARY_KEYS

    _StubJudge.instances = []
    monkeypatch.setattr(settings, "EVAL_ROW_CONCURRENCY", 4)
    monkeypatch.setattr("app.core.evaluation_engine.OpenAIJudgeClient", _StubJudge)
    _install_tracking_metric(monkeypatch, delay=0.02)

    llm, dataset, scenario = _setup_multirow_prerequisites(client, test_llm_payload, row_count=6)
    task = _run_task(client, db, llm, dataset, scenario)

    summary = client.get(f"/api/reports/{task['id']}/summary").json()
    assert "latency" not in (summary["metric_summary"] or {})
    assert "concurrency_metric" in summary["metric_summary"]


def test_concurrent_and_serial_produce_identical_results(
    client, db, monkeypatch, test_llm_payload
):
    """同一份数据在并发与串行下必须得到完全相同的分数与汇总。

    这是并发化最该被怀疑的地方：完成顺序不能泄漏进结果。汇总按 row_index
    排序后再算就是为了这个——`as_completed` 的返回顺序是不确定的，直接按
    完成顺序累加会让同一份数据每次跑出不同的 summary。
    """
    from app.core.config import settings
    from app.models.evaluation import EvalRowResult, EvalTask

    def _run_with_concurrency(concurrency, name):
        _StubJudge.instances = []
        monkeypatch.setattr(settings, "EVAL_ROW_CONCURRENCY", concurrency)
        monkeypatch.setattr("app.core.evaluation_engine.OpenAIJudgeClient", _StubJudge)
        _install_tracking_metric(monkeypatch, delay=0.01)
        task = _run_task(client, db, llm, dataset, scenario, name=name)
        db.expire_all()
        stored = db.query(EvalTask).filter(EvalTask.id == task["id"]).one()
        rows = (
            db.query(EvalRowResult)
            .filter(EvalRowResult.eval_task_id == task["id"])
            .order_by(EvalRowResult.row_index)
            .all()
        )
        scores = [(r.row_index, r.metric_scores["concurrency_metric"]["score"]) for r in rows]
        return stored.summary_scores, scores, stored.status

    llm, dataset, scenario = _setup_multirow_prerequisites(client, test_llm_payload, row_count=6)

    serial_summary, serial_scores, serial_status = _run_with_concurrency(1, "Serial Run")
    concurrent_summary, concurrent_scores, concurrent_status = _run_with_concurrency(
        4, "Concurrent Run"
    )

    assert serial_status == concurrent_status == "completed"
    assert serial_scores == concurrent_scores, "并发改变了逐行分数"
    # 指标维度统计必须逐字节一致；latency 天然不同（墙钟/加速比），单独排除
    serial_metrics = {k: v for k, v in serial_summary.items() if k != "latency"}
    concurrent_metrics = {k: v for k, v in concurrent_summary.items() if k != "latency"}
    assert serial_metrics == concurrent_metrics, "并发改变了汇总统计"

    # 串行路径下并发数被压到 1，不该额外建裁判客户端
    assert serial_summary["latency"]["row_concurrency"] == 1
    assert concurrent_summary["latency"]["row_concurrency"] == 4


def test_all_rows_persisted_with_correct_row_index(client, db, monkeypatch, test_llm_payload):
    """并发完成顺序不能影响落库的 row_index 归属。

    行结果是按完成顺序写库的（为了让进度实时可见），所以必须验证每一行的
    分数确实挂在自己的 row_index 上，而不是挂到先完成的那一行上。
    """
    from app.core.config import settings
    from app.models.evaluation import EvalRowResult

    _StubJudge.instances = []
    monkeypatch.setattr(settings, "EVAL_ROW_CONCURRENCY", 4)
    monkeypatch.setattr("app.core.evaluation_engine.OpenAIJudgeClient", _StubJudge)

    # 让不同行耗时不同，强制完成顺序与 row_index 顺序不一致
    async def staggered_ascore(self, row_data, judge):
        from app.core.evaluation_engine import _MetricResult

        user_input = str(row_data.get("user_input") or "")
        index = int(user_input.rsplit("-", 1)[-1]) if "-" in user_input else 0
        # 倒序延迟：row 0 最慢，row 7 最快 → 完成顺序基本反转
        await asyncio.sleep(0.01 * (8 - index))
        return _MetricResult(round(0.1 * (index + 1), 4), f"scored {user_input}")

    monkeypatch.setattr(
        "app.core.evaluation_engine.NativePromptMetric.ascore", staggered_ascore
    )

    llm, dataset, scenario = _setup_multirow_prerequisites(client, test_llm_payload, row_count=8)
    task = _run_task(client, db, llm, dataset, scenario)

    db.expire_all()
    rows = (
        db.query(EvalRowResult)
        .filter(EvalRowResult.eval_task_id == task["id"])
        .order_by(EvalRowResult.row_index)
        .all()
    )
    assert len(rows) == 8
    # reason 里带着 user_input，用它验证分数没有串行到别的行上
    for row in rows:
        entry = row.metric_scores["concurrency_metric"]
        assert f"question-{row.row_index}" in entry["reason"], (
            f"row_index {row.row_index} 的结果挂到了别的行上: {entry['reason']}"
        )
