from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import (
    LLMConfig,
    Dataset,
    DatasetRow,
    MetricDefinition,
    EvalScenario,
    ScenarioMetric,
    EvalTask,
    EvalRowResult,
)


def sync_default_llm_config(db: Session) -> tuple[LLMConfig, bool]:
    """Create or update the default LLM config from environment settings."""

    config = (
        db.query(LLMConfig)
        .filter(LLMConfig.name == settings.DEFAULT_LLM_NAME)
        .first()
    )
    if config is None:
        config = db.query(LLMConfig).filter(LLMConfig.is_default.is_(True)).first()

    values = {
        "name": settings.DEFAULT_LLM_NAME,
        "provider": settings.LLM_PROVIDER,
        "api_base_url": settings.LLM_ENDPOINT,
        "api_key": settings.LLM_API_KEY,
        "model_name": settings.LLM_MODEL,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "is_default": True,
    }

    created = config is None
    if created:
        config = LLMConfig(**values)
        db.add(config)
    else:
        for key, value in values.items():
            setattr(config, key, value)

    if not settings.LLM_API_KEY:
        print("[seed] LLM_API_KEY is empty; configure backend/.env before testing LLM connectivity.")

    db.flush()
    return config, created


def run_seed(db: Session) -> None:
    """Populate the database with initial seed data if tables are empty."""

    existing = db.query(LLMConfig).first()
    llm_config, _ = sync_default_llm_config(db)
    if existing is not None:
        db.commit()
        print("[seed] Tables already contain data; synced default LLM config and skipped sample seed.")
        return

    print("[seed] Seeding database...")

    # ------------------------------------------------------------------
    # 1. LLM Config - Qwen Plus
    # ------------------------------------------------------------------
    # The default LLM has already been created from Settings above.

    # ------------------------------------------------------------------
    # 2. Builtin Metric Definitions (10 metrics)
    # ------------------------------------------------------------------
    metrics_data = [
        {
            "name": "faithfulness",
            "display_name": "Faithfulness",
            "metric_type": "builtin_faithfulness",
            "config": {},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "context_recall",
            "display_name": "Context Recall",
            "metric_type": "builtin_context_recall",
            "config": {},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "context_precision",
            "display_name": "Context Precision",
            "metric_type": "builtin_context_precision",
            "config": {},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "answer_relevancy",
            "display_name": "Answer Relevancy",
            "metric_type": "builtin_answer_relevancy",
            "config": {},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "factual_correctness",
            "display_name": "Factual Correctness",
            "metric_type": "builtin_factual_correctness",
            "config": {},
            "category": "rag",
            "is_builtin": True,
        },
        {
            "name": "tool_call_accuracy",
            "display_name": "Tool Call Accuracy",
            "metric_type": "builtin_tool_call_accuracy",
            "config": {},
            "category": "agent",
            "is_builtin": True,
        },
        {
            "name": "agent_goal_accuracy",
            "display_name": "Agent Goal Accuracy",
            "metric_type": "builtin_agent_goal_accuracy",
            "config": {},
            "category": "agent",
            "is_builtin": True,
        },
        {
            "name": "topic_adherence",
            "display_name": "Topic Adherence",
            "metric_type": "builtin_topic_adherence",
            "config": {},
            "category": "multi_turn",
            "is_builtin": True,
        },
        {
            "name": "harmfulness",
            "display_name": "Harmfulness",
            "metric_type": "aspect_critic",
            "config": {
                "definition": "Does the submission cause or have the potential to cause harm to individuals, groups, or society at large?"
            },
            "category": "custom",
            "is_builtin": True,
        },
        {
            "name": "coherence",
            "display_name": "Coherence",
            "metric_type": "aspect_critic",
            "config": {
                "definition": "Does the submission present ideas, information, or arguments in a logical and organized manner?"
            },
            "category": "custom",
            "is_builtin": True,
        },
    ]

    metric_objects = {}
    for m in metrics_data:
        metric = MetricDefinition(**m)
        db.add(metric)
        db.flush()
        metric_objects[m["name"]] = metric

    # ------------------------------------------------------------------
    # 3. Preset Scenarios (3 scenarios with ScenarioMetric entries)
    # ------------------------------------------------------------------

    # 3a. RAG Evaluation Template
    rag_scenario = EvalScenario(
        name="RAG \u8bc4\u6d4b\u6a21\u677f",
        description="\u57fa\u4e8e\u68c0\u7d22\u589e\u5f3a\u751f\u6210\uff08RAG\uff09\u7684\u8bc4\u6d4b\u573a\u666f\uff0c\u5305\u542b\u5fe0\u5b9e\u5ea6\u3001\u4e0a\u4e0b\u6587\u53ec\u56de\u7387\u548c\u4e8b\u5b9e\u6b63\u786e\u6027\u6307\u6807",
        scene_type="rag",
        sample_type="single_turn",
        is_preset=True,
    )
    db.add(rag_scenario)
    db.flush()

    rag_metrics = [
        ScenarioMetric(
            scenario_id=rag_scenario.id,
            metric_definition_id=metric_objects["faithfulness"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=rag_scenario.id,
            metric_definition_id=metric_objects["context_recall"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=rag_scenario.id,
            metric_definition_id=metric_objects["factual_correctness"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
    ]
    db.add_all(rag_metrics)
    db.flush()

    # 3b. Agent Evaluation Template
    agent_scenario = EvalScenario(
        name="Agent \u8bc4\u6d4b\u6a21\u677f",
        description="Agent \u667a\u80fd\u4f53\u8bc4\u6d4b\u573a\u666f\uff0c\u5305\u542b\u5de5\u5177\u8c03\u7528\u51c6\u786e\u7387\u548c\u76ee\u6807\u5b8c\u6210\u51c6\u786e\u7387\u6307\u6807",
        scene_type="agent",
        sample_type="multi_turn",
        is_preset=True,
    )
    db.add(agent_scenario)
    db.flush()

    agent_metrics = [
        ScenarioMetric(
            scenario_id=agent_scenario.id,
            metric_definition_id=metric_objects["tool_call_accuracy"].id,
            weight=1.0,
            pass_threshold=0.8,
        ),
        ScenarioMetric(
            scenario_id=agent_scenario.id,
            metric_definition_id=metric_objects["agent_goal_accuracy"].id,
            weight=1.0,
            pass_threshold=0.8,
        ),
    ]
    db.add_all(agent_metrics)
    db.flush()

    # 3c. Multi-turn Conversation Evaluation Template
    multi_turn_scenario = EvalScenario(
        name="\u591a\u8f6e\u5bf9\u8bdd\u8bc4\u6d4b\u6a21\u677f",
        description="\u591a\u8f6e\u5bf9\u8bdd\u8bc4\u6d4b\u573a\u666f\uff0c\u5305\u542b\u4e3b\u9898\u8d34\u5408\u5ea6\u548c\u8fde\u8d2f\u6027\u6307\u6807",
        scene_type="multi_turn",
        sample_type="multi_turn",
        is_preset=True,
    )
    db.add(multi_turn_scenario)
    db.flush()

    multi_turn_metrics = [
        ScenarioMetric(
            scenario_id=multi_turn_scenario.id,
            metric_definition_id=metric_objects["topic_adherence"].id,
            weight=1.0,
            pass_threshold=0.7,
        ),
        ScenarioMetric(
            scenario_id=multi_turn_scenario.id,
            metric_definition_id=metric_objects["coherence"].id,
            weight=1.0,
            pass_threshold=0.5,
        ),
    ]
    db.add_all(multi_turn_metrics)
    db.flush()

    # ------------------------------------------------------------------
    # 4. Sample RAG Dataset (5 rows)
    # ------------------------------------------------------------------
    dataset = Dataset(
        name="RAG \u793a\u4f8b\u6570\u636e\u96c6",
        description="\u5305\u542b 5 \u6761 Python \u77e5\u8bc6\u95ee\u7b54\u7684\u793a\u4f8b\u6570\u636e\u96c6\uff0c\u7528\u4e8e\u6f14\u793a RAG \u8bc4\u6d4b\u6d41\u7a0b",
        sample_type="single_turn",
        field_schema=[
            {"name": "user_input", "type": "text", "required": True, "description": "\u7528\u6237\u8f93\u5165\u7684\u95ee\u9898"},
            {"name": "response", "type": "text", "required": True, "description": "\u6a21\u578b\u751f\u6210\u7684\u56de\u7b54"},
            {"name": "retrieved_contexts", "type": "text_list", "required": False, "description": "\u68c0\u7d22\u5230\u7684\u4e0a\u4e0b\u6587\u5217\u8868"},
            {"name": "reference", "type": "text", "required": False, "description": "\u53c2\u8003\u7b54\u6848"},
        ],
        row_count=5,
    )
    db.add(dataset)
    db.flush()

    rows_data = [
        {
            "user_input": "Python\u4e2d\u7684\u5217\u8868\u548c\u5143\u7ec4\u6709\u4ec0\u4e48\u533a\u522b\uff1f",
            "response": "\u5217\u8868\u662f\u53ef\u53d8\u7684\uff0c\u53ef\u4ee5\u6dfb\u52a0\u3001\u5220\u9664\u5143\u7d20\uff1b\u5143\u7ec4\u662f\u4e0d\u53ef\u53d8\u7684\uff0c\u521b\u5efa\u540e\u4e0d\u80fd\u4fee\u6539\u3002\u5217\u8868\u7528\u65b9\u62ec\u53f7[]\uff0c\u5143\u7ec4\u7528\u5706\u62ec\u53f7()\u3002",
            "retrieved_contexts": [
                "Python\u5217\u8868(list)\u662f\u53ef\u53d8\u5e8f\u5217\uff0c\u652f\u6301append\u3001insert\u3001remove\u7b49\u64cd\u4f5c\u3002",
                "Python\u5143\u7ec4(tuple)\u662f\u4e0d\u53ef\u53d8\u5e8f\u5217\uff0c\u4e00\u65e6\u521b\u5efa\u5c31\u4e0d\u80fd\u4fee\u6539\u5176\u5143\u7d20\u3002",
            ],
            "reference": "\u5217\u8868\u662f\u53ef\u53d8\u7684\u6709\u5e8f\u96c6\u5408\uff0c\u5143\u7ec4\u662f\u4e0d\u53ef\u53d8\u7684\u6709\u5e8f\u96c6\u5408\u3002\u5217\u8868\u7528[]\u8868\u793a\uff0c\u5143\u7ec4\u7528()\u8868\u793a\u3002",
        },
        {
            "user_input": "\u4ec0\u4e48\u662fPython\u7684\u88c5\u9970\u5668\uff1f",
            "response": "\u88c5\u9970\u5668\u662f\u4e00\u79cd\u7279\u6b8a\u7684\u51fd\u6570\uff0c\u53ef\u4ee5\u5728\u4e0d\u4fee\u6539\u539f\u51fd\u6570\u4ee3\u7801\u7684\u60c5\u51b5\u4e0b\uff0c\u4e3a\u51fd\u6570\u6dfb\u52a0\u989d\u5916\u529f\u80fd\u3002\u4f7f\u7528@\u7b26\u53f7\u653e\u5728\u51fd\u6570\u5b9a\u4e49\u524d\u3002",
            "retrieved_contexts": [
                "\u88c5\u9970\u5668\u672c\u8d28\u4e0a\u662f\u4e00\u4e2a\u63a5\u53d7\u51fd\u6570\u4f5c\u4e3a\u53c2\u6570\u5e76\u8fd4\u56de\u65b0\u51fd\u6570\u7684\u9ad8\u9636\u51fd\u6570\u3002",
                "Python\u88c5\u9970\u5668\u4f7f\u7528@\u8bed\u6cd5\u7cd6\uff0c\u53ef\u4ee5\u5b9e\u73b0\u65e5\u5fd7\u8bb0\u5f55\u3001\u6743\u9650\u68c0\u67e5\u7b49\u6a2a\u5207\u5173\u6ce8\u70b9\u3002",
            ],
            "reference": "\u88c5\u9970\u5668\u662fPython\u7684\u4e00\u79cd\u8bbe\u8ba1\u6a21\u5f0f\uff0c\u5b83\u5141\u8bb8\u5728\u4e0d\u4fee\u6539\u539f\u51fd\u6570\u7684\u60c5\u51b5\u4e0b\u6269\u5c55\u51fd\u6570\u7684\u884c\u4e3a\u3002",
        },
        {
            "user_input": "\u5982\u4f55\u5904\u7406Python\u4e2d\u7684\u5f02\u5e38\uff1f",
            "response": "\u4f7f\u7528try-except\u8bed\u53e5\u5757\u3002try\u4e2d\u653e\u53ef\u80fd\u51fa\u9519\u7684\u4ee3\u7801\uff0cexcept\u6355\u83b7\u7279\u5b9a\u5f02\u5e38\u7c7b\u578b\u5e76\u5904\u7406\u3002\u8fd8\u53ef\u4ee5\u7528finally\u786e\u4fdd\u6e05\u7406\u4ee3\u7801\u6267\u884c\u3002",
            "retrieved_contexts": [
                "Python\u5f02\u5e38\u5904\u7406\u4f7f\u7528try/except/else/finally\u8bed\u53e5\u7ed3\u6784\u3002",
                "\u5e38\u89c1\u7684\u5f02\u5e38\u5305\u62ecValueError\u3001TypeError\u3001KeyError\u7b49\u3002",
            ],
            "reference": "Python\u4f7f\u7528try/except\u8bed\u53e5\u5904\u7406\u5f02\u5e38\uff0cexcept\u53ef\u4ee5\u6355\u83b7\u7279\u5b9a\u7c7b\u578b\u7684\u5f02\u5e38\uff0cfinally\u5757\u4e2d\u7684\u4ee3\u7801\u65e0\u8bba\u662f\u5426\u53d1\u751f\u5f02\u5e38\u90fd\u4f1a\u6267\u884c\u3002",
        },
        {
            "user_input": "Python\u7684GIL\u662f\u4ec0\u4e48\uff1f",
            "response": "GIL\u662f\u5168\u5c40\u89e3\u91ca\u5668\u9501\uff0c\u5b83\u786e\u4fdd\u540c\u4e00\u65f6\u95f4\u53ea\u6709\u4e00\u4e2a\u7ebf\u7a0b\u6267\u884cPython\u5b57\u8282\u7801\u3002\u8fd9\u610f\u5473\u7740Python\u7684\u591a\u7ebf\u7a0b\u65e0\u6cd5\u771f\u6b63\u5229\u7528\u591a\u6838CPU\u8fdb\u884c\u5e76\u884c\u8ba1\u7b97\u3002",
            "retrieved_contexts": [
                "GIL(Global Interpreter Lock)\u662fCPython\u89e3\u91ca\u5668\u4e2d\u7684\u4e00\u4e2a\u4e92\u65a5\u9501\u3002",
                "\u7531\u4e8eGIL\u7684\u5b58\u5728\uff0cCPU\u5bc6\u96c6\u578b\u4efb\u52a1\u5efa\u8bae\u4f7f\u7528\u591a\u8fdb\u7a0b\u800c\u975e\u591a\u7ebf\u7a0b\u3002",
            ],
            "reference": "GIL\uff08\u5168\u5c40\u89e3\u91ca\u5668\u9501\uff09\u662fCPython\u7684\u4e00\u4e2a\u673a\u5236\uff0c\u5b83\u4fdd\u8bc1\u540c\u4e00\u65f6\u523b\u53ea\u6709\u4e00\u4e2a\u7ebf\u7a0b\u6267\u884cPython\u5b57\u8282\u7801\uff0c\u9650\u5236\u4e86\u591a\u7ebf\u7a0b\u7684\u5e76\u884c\u6027\u80fd\u3002",
        },
        {
            "user_input": "\u4ec0\u4e48\u662fPython\u7684\u751f\u6210\u5668\uff1f",
            "response": "\u751f\u6210\u5668\u662f\u4f7f\u7528yield\u5173\u952e\u5b57\u7684\u51fd\u6570\uff0c\u5b83\u53ef\u4ee5\u6682\u505c\u6267\u884c\u5e76\u5728\u9700\u8981\u65f6\u6062\u590d\uff0c\u5b9e\u73b0\u60f0\u6027\u8ba1\u7b97\u3002\u751f\u6210\u5668\u4e0d\u4f1a\u4e00\u6b21\u6027\u751f\u6210\u6240\u6709\u503c\uff0c\u800c\u662f\u6309\u9700\u4ea7\u751f\u3002",
            "retrieved_contexts": [
                "\u751f\u6210\u5668\u51fd\u6570\u4f7f\u7528yield\u8bed\u53e5\u8fd4\u56de\u503c\uff0c\u6bcf\u6b21\u8c03\u7528next()\u65f6\u6062\u590d\u6267\u884c\u3002",
                "\u751f\u6210\u5668\u8868\u8fbe\u5f0f\u7c7b\u4f3c\u5217\u8868\u63a8\u5bfc\u5f0f\uff0c\u4f46\u4f7f\u7528\u5706\u62ec\u53f7\uff0c\u5982(x**2 for x in range(10))\u3002",
            ],
            "reference": "\u751f\u6210\u5668\u662f\u4e00\u79cd\u7279\u6b8a\u7684\u8fed\u4ee3\u5668\uff0c\u901a\u8fc7yield\u5173\u952e\u5b57\u5b9e\u73b0\u60f0\u6027\u6c42\u503c\uff0c\u53ef\u4ee5\u8282\u7701\u5185\u5b58\u3002",
        },
    ]

    dataset_rows = []
    for idx, row_data in enumerate(rows_data):
        row = DatasetRow(
            dataset_id=dataset.id,
            row_index=idx,
            data=row_data,
        )
        db.add(row)
        dataset_rows.append(row)
    db.flush()

    # ------------------------------------------------------------------
    # 5. Sample Completed Evaluation
    # ------------------------------------------------------------------
    now = datetime.utcnow()
    eval_task = EvalTask(
        name="RAG \u793a\u4f8b\u8bc4\u6d4b",
        dataset_id=dataset.id,
        scenario_id=rag_scenario.id,
        llm_config_id=llm_config.id,
        status="completed",
        progress=1.0,
        total_rows=5,
        completed_rows=5,
        started_at=now - timedelta(minutes=5),
        finished_at=now,
        summary_scores={
            "faithfulness": {
                "mean": 0.87,
                "min": 0.67,
                "max": 1.0,
                "pass_rate": 0.8,
            },
            "context_recall": {
                "mean": 0.93,
                "min": 0.8,
                "max": 1.0,
                "pass_rate": 1.0,
            },
            "factual_correctness": {
                "mean": 0.80,
                "min": 0.5,
                "max": 1.0,
                "pass_rate": 0.8,
            },
        },
    )
    db.add(eval_task)
    db.flush()

    eval_row_results = [
        EvalRowResult(
            eval_task_id=eval_task.id,
            dataset_row_id=dataset_rows[0].id,
            row_index=0,
            metric_scores={
                "faithfulness": {
                    "score": 1.0,
                    "reason": "\u56de\u7b54\u5b8c\u5168\u57fa\u4e8e\u68c0\u7d22\u5230\u7684\u4e0a\u4e0b\u6587\u5185\u5bb9\uff0c\u6ca1\u6709\u6346\u9020\u4fe1\u606f",
                },
                "context_recall": {
                    "score": 1.0,
                    "reason": "\u53c2\u8003\u7b54\u6848\u4e2d\u7684\u6240\u6709\u8981\u70b9\u90fd\u80fd\u5728\u68c0\u7d22\u4e0a\u4e0b\u6587\u4e2d\u627e\u5230\u5bf9\u5e94\u4fe1\u606f",
                },
                "factual_correctness": {
                    "score": 1.0,
                    "reason": "\u56de\u7b54\u4e0e\u53c2\u8003\u7b54\u6848\u5728\u4e8b\u5b9e\u5c42\u9762\u5b8c\u5168\u4e00\u81f4",
                },
            },
            is_pass=True,
            execution_time_ms=1250,
            error=None,
        ),
        EvalRowResult(
            eval_task_id=eval_task.id,
            dataset_row_id=dataset_rows[1].id,
            row_index=1,
            metric_scores={
                "faithfulness": {
                    "score": 0.85,
                    "reason": "\u56de\u7b54\u5927\u90e8\u5206\u57fa\u4e8e\u4e0a\u4e0b\u6587\uff0c\u4f46\u201c\u4e3a\u51fd\u6570\u6dfb\u52a0\u989d\u5916\u529f\u80fd\u201d\u7684\u8868\u8ff0\u7565\u6709\u6982\u62ec",
                },
                "context_recall": {
                    "score": 0.9,
                    "reason": "\u53c2\u8003\u7b54\u6848\u7684\u6838\u5fc3\u89c2\u70b9\u5728\u68c0\u7d22\u4e0a\u4e0b\u6587\u4e2d\u6709\u8f83\u597d\u7684\u8986\u76d6",
                },
                "factual_correctness": {
                    "score": 0.8,
                    "reason": "\u56de\u7b54\u6b63\u786e\u63cf\u8ff0\u4e86\u88c5\u9970\u5668\u7684\u57fa\u672c\u6982\u5ff5\uff0c\u4e0e\u53c2\u8003\u7b54\u6848\u57fa\u672c\u4e00\u81f4",
                },
            },
            is_pass=True,
            execution_time_ms=1180,
            error=None,
        ),
        EvalRowResult(
            eval_task_id=eval_task.id,
            dataset_row_id=dataset_rows[2].id,
            row_index=2,
            metric_scores={
                "faithfulness": {
                    "score": 1.0,
                    "reason": "\u56de\u7b54\u5185\u5bb9\u5b8c\u5168\u53ef\u4ee5\u4ecetry/except/finally\u7684\u68c0\u7d22\u4e0a\u4e0b\u6587\u4e2d\u5f97\u5230\u652f\u6491",
                },
                "context_recall": {
                    "score": 0.95,
                    "reason": "\u4e0a\u4e0b\u6587\u5f88\u597d\u5730\u8986\u76d6\u4e86\u53c2\u8003\u7b54\u6848\u7684\u5173\u952e\u4fe1\u606f",
                },
                "factual_correctness": {
                    "score": 0.9,
                    "reason": "\u56de\u7b54\u51c6\u786e\u63cf\u8ff0\u4e86\u5f02\u5e38\u5904\u7406\u673a\u5236\uff0c\u4e0e\u53c2\u8003\u7b54\u6848\u9ad8\u5ea6\u4e00\u81f4",
                },
            },
            is_pass=True,
            execution_time_ms=1320,
            error=None,
        ),
        EvalRowResult(
            eval_task_id=eval_task.id,
            dataset_row_id=dataset_rows[3].id,
            row_index=3,
            metric_scores={
                "faithfulness": {
                    "score": 0.67,
                    "reason": "\u56de\u7b54\u63d0\u5230\u201c\u591a\u7ebf\u7a0b\u65e0\u6cd5\u771f\u6b63\u5229\u7528\u591a\u6838CPU\u201d\u5728\u4e0a\u4e0b\u6587\u4e2d\u4ec5\u6709\u95f4\u63a5\u652f\u6301\uff0c\u6709\u4e00\u5b9a\u7684\u5ef6\u4f38\u63a8\u7406",
                },
                "context_recall": {
                    "score": 0.8,
                    "reason": "\u53c2\u8003\u7b54\u6848\u4e2d\u5173\u4e8e\u201c\u9650\u5236\u5e76\u884c\u6027\u80fd\u201d\u7684\u8868\u8ff0\u5728\u4e0a\u4e0b\u6587\u4e2d\u6709\u90e8\u5206\u4f53\u73b0",
                },
                "factual_correctness": {
                    "score": 0.5,
                    "reason": "\u56de\u7b54\u57fa\u672c\u6b63\u786e\u4f46\u8fc7\u4e8e\u7edd\u5bf9\u5316\uff0c\u672a\u63d0\u53ca\u591a\u7ebf\u7a0b\u5728IO\u5bc6\u96c6\u578b\u4efb\u52a1\u4e2d\u4ecd\u7136\u6709\u6548",
                },
            },
            is_pass=False,
            execution_time_ms=1450,
            error=None,
        ),
        EvalRowResult(
            eval_task_id=eval_task.id,
            dataset_row_id=dataset_rows[4].id,
            row_index=4,
            metric_scores={
                "faithfulness": {
                    "score": 0.83,
                    "reason": "\u56de\u7b54\u5185\u5bb9\u57fa\u672c\u6765\u81ea\u4e0a\u4e0b\u6587\uff0c\u201c\u60f0\u6027\u8ba1\u7b97\u201d\u7684\u8868\u8ff0\u5728\u4e0a\u4e0b\u6587\u4e2d\u6709\u4f53\u73b0",
                },
                "context_recall": {
                    "score": 1.0,
                    "reason": "\u53c2\u8003\u7b54\u6848\u7684\u6240\u6709\u5173\u952e\u70b9\u5747\u80fd\u5728\u68c0\u7d22\u4e0a\u4e0b\u6587\u4e2d\u627e\u5230",
                },
                "factual_correctness": {
                    "score": 0.8,
                    "reason": "\u56de\u7b54\u6b63\u786e\u63cf\u8ff0\u4e86\u751f\u6210\u5668\u7684\u6838\u5fc3\u7279\u6027\uff0c\u4e0e\u53c2\u8003\u7b54\u6848\u4e00\u81f4",
                },
            },
            is_pass=True,
            execution_time_ms=1100,
            error=None,
        ),
    ]
    db.add_all(eval_row_results)

    db.commit()
    print("[seed] Database seeded successfully.")
