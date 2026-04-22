"""
Evaluation engine -- bridges UI configuration to ragas metric execution.

This module is the core of the AI Evaluation Platform. It translates database
models (EvalTask, EvalScenario, MetricDefinition, LLMConfig, Dataset) into
ragas metric instances, builds ragas samples from dataset rows, and orchestrates
the full evaluation loop with progress tracking and error handling.

Called via ``asyncio.create_task(run_evaluation(task.id, SessionLocal))`` from
the evaluation API endpoint.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
import typing as t
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. build_ragas_llm
# ---------------------------------------------------------------------------

def build_ragas_llm(llm_config):
    """
    Create a ragas-compatible LLM wrapper from an LLMConfig database model.

    Tries the modern ``llm_factory`` path first (instructor-based, no langchain
    dependency required at runtime). Falls back to ``LangchainLLMWrapper`` when
    ``llm_factory`` is unavailable or raises.

    Parameters
    ----------
    llm_config : app.models.llm_config.LLMConfig
        Database row with provider, api_base_url, api_key, model_name, etc.

    Returns
    -------
    BaseRagasLLM or InstructorBaseRagasLLM
        A ragas LLM usable by both legacy and simple metric classes.
    """
    try:
        import openai as openai_lib
        from ragas.llms import llm_factory

        client = openai_lib.OpenAI(
            base_url=llm_config.api_base_url,
            api_key=llm_config.api_key,
        )
        llm = llm_factory(
            model=llm_config.model_name,
            client=client,
        )
        logger.info(
            "Built ragas LLM via llm_factory: model=%s base_url=%s",
            llm_config.model_name,
            llm_config.api_base_url,
        )
        return llm
    except Exception as exc:
        logger.warning(
            "llm_factory failed (%s), falling back to LangchainLLMWrapper", exc
        )

    # Fallback: langchain-openai + LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    from ragas.llms.base import LangchainLLMWrapper

    langchain_llm = ChatOpenAI(
        base_url=llm_config.api_base_url,
        api_key=llm_config.api_key,
        model=llm_config.model_name,
        temperature=llm_config.temperature or 0.01,
        max_tokens=llm_config.max_tokens or 1024,
    )
    wrapper = LangchainLLMWrapper(langchain_llm)
    logger.info(
        "Built ragas LLM via LangchainLLMWrapper: model=%s base_url=%s",
        llm_config.model_name,
        llm_config.api_base_url,
    )
    return wrapper


# ---------------------------------------------------------------------------
# 2. build_metric
# ---------------------------------------------------------------------------

# Registry: metric_type -> (module_path, class_name, needs_llm)
_BUILTIN_METRIC_REGISTRY: t.Dict[str, t.Tuple[str, str, bool]] = {
    "builtin_faithfulness": (
        "ragas.metrics._faithfulness",
        "Faithfulness",
        True,
    ),
    "builtin_context_recall": (
        "ragas.metrics._context_recall",
        "LLMContextRecall",
        True,
    ),
    "builtin_context_precision": (
        "ragas.metrics._context_precision",
        "LLMContextPrecisionWithReference",
        True,
    ),
    "builtin_factual_correctness": (
        "ragas.metrics._factual_correctness",
        "FactualCorrectness",
        True,
    ),
    "builtin_answer_relevancy": (
        "ragas.metrics._answer_relevance",
        "AnswerRelevancy",
        True,
    ),
    "builtin_tool_call_accuracy": (
        "ragas.metrics._tool_call_accuracy",
        "ToolCallAccuracy",
        False,
    ),
    "builtin_agent_goal_accuracy": (
        "ragas.metrics._goal_accuracy",
        "AgentGoalAccuracyWithReference",
        True,
    ),
    "builtin_topic_adherence": (
        "ragas.metrics._topic_adherence",
        "TopicAdherenceScore",
        True,
    ),
}


def build_metric(
    metric_def,
    llm,
) -> t.Tuple[str, t.Any]:
    """
    Instantiate a ragas metric from a MetricDefinition database row.

    Parameters
    ----------
    metric_def : app.models.metric_definition.MetricDefinition
        Row with ``metric_type``, ``name``, ``config`` (JSON dict), etc.
    llm
        The ragas LLM to attach to LLM-based metrics.

    Returns
    -------
    (metric_kind, metric_instance)
        ``metric_kind`` is ``"legacy"`` for built-in ragas metrics / AspectCritic
        (scored via ``single_turn_ascore`` / ``multi_turn_ascore``) or ``"simple"``
        for DiscreteMetric / NumericMetric (scored via ``.score(llm=..., **kwargs)``).
    """
    metric_type: str = metric_def.metric_type
    config: dict = metric_def.config or {}

    # --- Built-in legacy metrics ------------------------------------------------
    if metric_type in _BUILTIN_METRIC_REGISTRY:
        module_path, class_name, needs_llm = _BUILTIN_METRIC_REGISTRY[metric_type]
        mod = importlib.import_module(module_path)
        MetricClass = getattr(mod, class_name)
        metric_instance = MetricClass()
        if needs_llm:
            metric_instance.llm = llm
        return ("legacy", metric_instance)

    # --- AspectCritic (legacy, but user-configured) -----------------------------
    if metric_type == "aspect_critic":
        from ragas.metrics._aspect_critic import AspectCritic

        definition = config.get("definition", "")
        metric_instance = AspectCritic(
            name=metric_def.name,
            definition=definition,
            llm=llm,
        )
        return ("legacy", metric_instance)

    # --- DiscreteMetric (simple) ------------------------------------------------
    if metric_type == "discrete":
        from ragas.metrics.discrete import DiscreteMetric

        prompt_text = config.get("prompt", "")
        allowed_values = config.get("allowed_values", ["pass", "fail"])
        metric_instance = DiscreteMetric(
            name=metric_def.name,
            prompt=prompt_text,
            allowed_values=allowed_values,
        )
        return ("simple", metric_instance)

    # --- NumericMetric (simple) -------------------------------------------------
    if metric_type == "numeric":
        from ragas.metrics.numeric import NumericMetric

        prompt_text = config.get("prompt", "")
        raw_range = config.get("allowed_values", [0.0, 1.0])
        allowed_values = (float(raw_range[0]), float(raw_range[1]))
        metric_instance = NumericMetric(
            name=metric_def.name,
            prompt=prompt_text,
            allowed_values=allowed_values,
        )
        return ("simple", metric_instance)

    raise ValueError(f"Unknown metric_type: {metric_type!r}")


# ---------------------------------------------------------------------------
# 3. build_single_turn_sample
# ---------------------------------------------------------------------------

def build_single_turn_sample(row_data: dict):
    """
    Convert a flat dict (from ``DatasetRow.data``) into a ragas
    ``SingleTurnSample``.  Missing keys are silently omitted so the pydantic
    model uses its ``None`` defaults.

    Parameters
    ----------
    row_data : dict
        Typically a subset of: ``user_input``, ``response``, ``reference``,
        ``retrieved_contexts``, ``reference_contexts``, etc.

    Returns
    -------
    SingleTurnSample
    """
    from ragas.dataset_schema import SingleTurnSample

    accepted_fields = {
        "user_input",
        "response",
        "reference",
        "retrieved_contexts",
        "reference_contexts",
        "retrieved_context_ids",
        "reference_context_ids",
        "multi_responses",
        "rubrics",
    }

    kwargs: dict = {}
    for key in accepted_fields:
        if key in row_data and row_data[key] is not None:
            kwargs[key] = row_data[key]

    return SingleTurnSample(**kwargs)


# ---------------------------------------------------------------------------
# 4. build_multi_turn_sample
# ---------------------------------------------------------------------------

def build_multi_turn_sample(row_data: dict):
    """
    Convert a dict with conversation data into a ragas ``MultiTurnSample``.

    ``row_data["user_input"]`` should be a list of message dicts::

        [
            {"type": "human", "content": "Hello"},
            {"type": "ai", "content": "Hi!", "tool_calls": [
                {"name": "search", "args": {"q": "hello"}}
            ]},
            {"type": "tool", "content": "search result ..."},
        ]

    Each dict is converted into the corresponding ragas message type.

    Parameters
    ----------
    row_data : dict
        Must contain ``user_input`` (list[dict]).  May also contain
        ``reference``, ``reference_tool_calls``, ``reference_topics``,
        ``rubrics``.

    Returns
    -------
    MultiTurnSample
    """
    from ragas.dataset_schema import MultiTurnSample
    from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

    raw_messages = row_data.get("user_input", [])
    messages: list = []

    for msg in raw_messages:
        msg_type = msg.get("type", "human")
        content = msg.get("content", "")
        metadata = msg.get("metadata")

        if msg_type == "human":
            messages.append(HumanMessage(content=content, metadata=metadata))

        elif msg_type == "ai":
            tool_calls_raw = msg.get("tool_calls") or []
            tool_calls = [
                ToolCall(name=tc["name"], args=tc.get("args", {}))
                for tc in tool_calls_raw
            ] or None
            messages.append(
                AIMessage(content=content, tool_calls=tool_calls, metadata=metadata)
            )

        elif msg_type == "tool":
            messages.append(ToolMessage(content=content, metadata=metadata))

        else:
            logger.warning("Unknown message type %r, treating as human", msg_type)
            messages.append(HumanMessage(content=content, metadata=metadata))

    kwargs: dict = {"user_input": messages}

    if row_data.get("reference") is not None:
        kwargs["reference"] = row_data["reference"]

    if row_data.get("reference_tool_calls") is not None:
        kwargs["reference_tool_calls"] = [
            ToolCall(name=tc["name"], args=tc.get("args", {}))
            for tc in row_data["reference_tool_calls"]
        ]

    if row_data.get("reference_topics") is not None:
        kwargs["reference_topics"] = row_data["reference_topics"]

    if row_data.get("rubrics") is not None:
        kwargs["rubrics"] = row_data["rubrics"]

    return MultiTurnSample(**kwargs)


# ---------------------------------------------------------------------------
# 5. run_evaluation  (async entry-point)
# ---------------------------------------------------------------------------

async def run_evaluation(task_id: int, session_factory) -> None:
    """
    Execute a full evaluation run for the given EvalTask.

    Designed to be launched with ``asyncio.create_task(...)`` from the API
    endpoint.  ``session_factory`` is the sync ``SessionLocal`` class -- we
    open our own session so the request session can close independently.

    Parameters
    ----------
    task_id : int
        Primary key of the ``EvalTask`` to execute.
    session_factory
        Sync SQLAlchemy ``sessionmaker`` (e.g. ``SessionLocal``).
    """
    from app.models.dataset import DatasetRow
    from app.models.evaluation import EvalRowResult, EvalTask
    from app.models.metric_definition import MetricDefinition
    from app.models.scenario import EvalScenario, ScenarioMetric

    db = session_factory()

    def _log(task_obj, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        task_obj.logs = (task_obj.logs or "") + line
        logger.info(msg)

    try:
        task = db.query(EvalTask).get(task_id)
        if task is None:
            logger.error("EvalTask %d not found -- aborting", task_id)
            return

        dataset = task.dataset
        scenario = task.scenario
        llm_config = task.llm_config

        scenario_metrics: t.List[ScenarioMetric] = (
            db.query(ScenarioMetric)
            .filter(ScenarioMetric.scenario_id == scenario.id)
            .all()
        )
        for sm in scenario_metrics:
            _ = sm.metric_definition

        _log(task, f"========== 评测任务启动 ==========")
        _log(task, f"任务: {task.name} (ID={task_id})")
        _log(task, f"数据集: {dataset.name} ({dataset.row_count} 条)")
        _log(task, f"场景: {scenario.name}")
        _log(task, f"LLM: {llm_config.model_name} @ {llm_config.api_base_url}")
        _log(task, f"指标数: {len(scenario_metrics)} 个")
        db.commit()

        _log(task, f"正在构建 LLM 实例...")
        db.commit()
        ragas_llm = build_ragas_llm(llm_config)
        _log(task, f"✓ LLM 构建成功")
        db.commit()

        legacy_metrics: t.List[t.Tuple[str, t.Any, ScenarioMetric]] = []
        simple_metrics: t.List[t.Tuple[str, t.Any, ScenarioMetric]] = []

        for sm in scenario_metrics:
            metric_def: MetricDefinition = sm.metric_definition
            try:
                kind, metric_instance = build_metric(metric_def, ragas_llm)
                _log(task, f"✓ 指标 [{metric_def.display_name}] 构建成功 (类型: {kind})")
            except Exception as exc:
                _log(task, f"✗ 指标 [{metric_def.display_name}] 构建失败: {exc}")
                continue

            if kind == "legacy":
                legacy_metrics.append((metric_def.name, metric_instance, sm))
            else:
                simple_metrics.append((metric_def.name, metric_instance, sm))

        if not legacy_metrics and not simple_metrics:
            _log(task, "✗ 错误: 没有任何指标构建成功，无法执行评测")
            db.commit()
            raise RuntimeError("No metrics could be built for this scenario")

        from ragas.run_config import RunConfig
        run_config = RunConfig(timeout=180, max_retries=3, max_wait=60)
        for metric_name, metric_instance, _sm in legacy_metrics:
            try:
                metric_instance.init(run_config)
            except Exception as exc:
                _log(task, f"⚠ 指标 [{metric_name}] 初始化警告: {exc}")

        dataset_rows: t.List[DatasetRow] = (
            db.query(DatasetRow)
            .filter(DatasetRow.dataset_id == dataset.id)
            .order_by(DatasetRow.row_index)
            .all()
        )
        total_rows = len(dataset_rows)
        if total_rows == 0:
            _log(task, "✗ 数据集没有数据行")
            db.commit()
            raise RuntimeError(f"Dataset {dataset.id} has no rows")

        task.status = "running"
        task.total_rows = total_rows
        task.completed_rows = 0
        task.progress = 0.0
        task.started_at = datetime.now(timezone.utc)
        _log(task, f"========== 开始逐行评测 ({total_rows} 条) ==========")
        db.commit()

        # ------------------------------------------------------------------
        # Step 7: Process each row
        # ------------------------------------------------------------------
        sample_type = scenario.sample_type or "single_turn"
        all_row_scores: t.List[t.Dict[str, t.Any]] = []

        for idx, dataset_row in enumerate(dataset_rows):
            row_start = time.time()
            row_data: dict = dataset_row.data or {}
            metric_scores: t.Dict[str, t.Any] = {}
            row_error: t.Optional[str] = None

            # -- Build ragas sample --
            try:
                if sample_type == "multi_turn":
                    sample = build_multi_turn_sample(row_data)
                else:
                    sample = build_single_turn_sample(row_data)
            except Exception as exc:
                row_error = f"Failed to build sample: {exc}"
                _log(task, f"  ✗ 行 #{idx}: 构建样本失败 - {exc}")
                execution_time_ms = int((time.time() - row_start) * 1000)
                _persist_row_result(db, task, dataset_row, metric_scores, row_error, False, execution_time_ms)
                task.completed_rows = idx + 1
                task.progress = round((idx + 1) / total_rows, 4)
                db.commit()
                all_row_scores.append(metric_scores)
                await asyncio.sleep(0)
                continue

            db.refresh(task)
            if task.status == "cancelled":
                _log(task, f"⚠ 用户取消评测，已完成 {idx}/{total_rows} 行")
                db.commit()
                break

            user_input_preview = str(row_data.get("user_input", ""))[:60]
            _log(task, f"── 行 #{idx+1}/{total_rows}: {user_input_preview}...")
            db.commit()

            for metric_name, metric_instance, _sm in legacy_metrics:
                _log(task, f"  ▸ 评测指标 [{metric_name}]...")
                db.commit()
                try:
                    if sample_type == "multi_turn":
                        score = await metric_instance.multi_turn_ascore(sample)
                    else:
                        score = await metric_instance.single_turn_ascore(sample)
                    metric_scores[metric_name] = {"score": round(float(score), 4), "reason": ""}
                    _log(task, f"  ✓ [{metric_name}] = {round(float(score), 4)}")
                except Exception as exc:
                    metric_scores[metric_name] = {"score": None, "reason": str(exc)[:500]}
                    _log(task, f"  ✗ [{metric_name}] 失败: {str(exc)[:200]}")
                db.commit()

            for metric_name, metric_instance, _sm in simple_metrics:
                _log(task, f"  ▸ 评测指标 [{metric_name}]...")
                db.commit()
                try:
                    variables = metric_instance.get_variables()
                    score_kwargs: dict = {"llm": ragas_llm}
                    for var in variables:
                        if var in row_data:
                            score_kwargs[var] = row_data[var]
                    result = metric_instance.score(**score_kwargs)
                    val = result.value if hasattr(result, "value") else result
                    reason = str(result.reason) if hasattr(result, "reason") else ""
                    if isinstance(val, (int, float)):
                        val = round(float(val), 4)
                    metric_scores[metric_name] = {"score": val, "reason": reason[:500]}
                    _log(task, f"  ✓ [{metric_name}] = {val}")
                except Exception as exc:
                    metric_scores[metric_name] = {"score": None, "reason": str(exc)[:500]}
                    _log(task, f"  ✗ [{metric_name}] 失败: {str(exc)[:200]}")
                db.commit()

            is_pass = _determine_pass(metric_scores, legacy_metrics, simple_metrics)
            execution_time_ms = int((time.time() - row_start) * 1000)

            _persist_row_result(db, task, dataset_row, metric_scores, row_error, is_pass, execution_time_ms)

            task.completed_rows = idx + 1
            task.progress = round((idx + 1) / total_rows, 4)

            status_icon = "✓ 通过" if is_pass else "✗ 不通过"
            _log(task, f"  → 结果: {status_icon} ({execution_time_ms}ms)")
            db.commit()

            all_row_scores.append(metric_scores)
            await asyncio.sleep(0)

        db.refresh(task)
        if task.status == "cancelled":
            _log(task, f"========== 评测已取消 ==========")
            task.finished_at = datetime.now(timezone.utc)
            db.commit()
        else:
            _log(task, f"========== 计算汇总统计 ==========")
            db.commit()
            summary_scores = _compute_summary_scores(all_row_scores, legacy_metrics, simple_metrics)
            task.status = "completed"
            task.finished_at = datetime.now(timezone.utc)
            task.summary_scores = summary_scores
            task.progress = 1.0

            pass_count = sum(1 for s in all_row_scores if _determine_pass(s, legacy_metrics, simple_metrics))
            _log(task, f"通过: {pass_count}/{total_rows} ({round(pass_count/total_rows*100, 1)}%)")
            for m_name, info in summary_scores.items():
                mean = info.get("mean")
                pr = info.get("pass_rate")
                _log(task, f"  {m_name}: 均值={mean}, 通过率={pr}")
            _log(task, f"========== 评测完成 ==========")
            db.commit()

    except Exception as exc:
        logger.exception("Evaluation failed for task %d", task_id)
        try:
            db.rollback()
            task = db.query(EvalTask).get(task_id)
            if task is not None:
                task.status = "failed"
                task.error_message = str(exc)[:2000]
                task.finished_at = datetime.now(timezone.utc)
                _log(task, f"✗✗✗ 评测失败: {str(exc)[:500]}")
                _log(task, f"========== 评测异常终止 ==========")
                db.commit()
        except Exception:
            logger.exception(
                "Failed to persist error status for task %d", task_id
            )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _persist_row_result(
    db,
    task,
    dataset_row,
    metric_scores: dict,
    error: t.Optional[str],
    is_pass: bool,
    execution_time_ms: int,
) -> None:
    """Create an ``EvalRowResult`` and add it to the session (caller commits)."""
    from app.models.evaluation import EvalRowResult

    row_result = EvalRowResult(
        eval_task_id=task.id,
        dataset_row_id=dataset_row.id,
        row_index=dataset_row.row_index,
        metric_scores=metric_scores,
        is_pass=is_pass,
        execution_time_ms=execution_time_ms,
        error=error,
    )
    db.add(row_result)


def _determine_pass(
    metric_scores: dict,
    legacy_metrics: list,
    simple_metrics: list,
) -> bool:
    """
    Determine whether a row passes based on per-metric ``pass_threshold``
    from the ``ScenarioMetric`` association.

    A row passes when **all** metrics that have a defined threshold meet or
    exceed that threshold.  Metrics with errors (score=None) cause failure.
    """
    all_metrics = list(legacy_metrics) + list(simple_metrics)
    for metric_name, _metric_instance, scenario_metric in all_metrics:
        threshold = scenario_metric.pass_threshold
        if threshold is None:
            continue
        raw = metric_scores.get(metric_name)
        if raw is None:
            return False
        score = raw.get("score") if isinstance(raw, dict) else raw
        if score is None:
            return False
        if isinstance(score, str):
            numeric_score = 1.0 if score.lower() in ("pass", "yes", "true", "1") else 0.0
        else:
            numeric_score = float(score)
        if numeric_score < threshold:
            return False
    return True


def _compute_summary_scores(
    all_row_scores: t.List[t.Dict[str, t.Any]],
    legacy_metrics: list,
    simple_metrics: list,
) -> dict:
    """
    Compute per-metric aggregate statistics across all evaluated rows.

    Returns a dict like::

        {
            "faithfulness": {
                "mean": 0.85, "min": 0.6, "max": 1.0,
                "pass_rate": 0.9, "count": 10, "error_count": 0
            },
            ...
        }
    """
    all_metrics = list(legacy_metrics) + list(simple_metrics)
    thresholds = {name: sm.pass_threshold for name, _, sm in all_metrics}

    summary: dict = {}

    for metric_name, _, _ in all_metrics:
        numeric_values: t.List[float] = []
        error_count = 0

        for row_scores in all_row_scores:
            raw = row_scores.get(metric_name)
            if raw is None:
                error_count += 1
                continue
            score = raw.get("score") if isinstance(raw, dict) else raw
            if score is None:
                error_count += 1
                continue
            if isinstance(score, str):
                numeric_values.append(
                    1.0 if score.lower() in ("pass", "yes", "true", "1") else 0.0
                )
            else:
                numeric_values.append(float(score))

        if numeric_values:
            mean_val = sum(numeric_values) / len(numeric_values)
            min_val = min(numeric_values)
            max_val = max(numeric_values)

            threshold = thresholds.get(metric_name)
            if threshold is not None:
                pass_count = sum(1 for v in numeric_values if v >= threshold)
                pass_rate = pass_count / len(numeric_values)
            else:
                pass_rate = None
        else:
            mean_val = None
            min_val = None
            max_val = None
            pass_rate = None

        summary[metric_name] = {
            "mean": round(mean_val, 4) if mean_val is not None else None,
            "min": round(min_val, 4) if min_val is not None else None,
            "max": round(max_val, 4) if max_val is not None else None,
            "pass_rate": round(pass_rate, 4) if pass_rate is not None else None,
            "count": len(numeric_values),
            "error_count": error_count,
        }

    return summary
