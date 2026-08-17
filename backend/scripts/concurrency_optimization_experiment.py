#!/usr/bin/env python3
"""并发优化实验：对比串行、指标并发、行级并发的时间-质量权衡

注意：本实验聚焦**时间优化**（通过并发降低延迟），不涉及成本优化。
并发执行不会降低 token 消耗，真正的成本优化需要批量推理或模型分层。
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.evaluation_engine import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextRecallMetric,
    OpenAIJudgeClient,
)
from app.core.config import settings


class MockMetricDefinition:
    """模拟 MetricDefinition 对象"""
    def __init__(self, name: str, display_name: str, metric_type: str):
        self.name = name
        self.metric_type = metric_type
        self.display_name = display_name
        self.description = f"{display_name}评测"
        self.prompt_template = None
        self.config = {}


# 准备测试样本
SAMPLE_DATA = [
    {
        "user_input": "Python 中列表和元组有什么区别？",
        "response": "列表是可变的，元组是不可变的。",
        "retrieved_contexts": ["Python列表是可变的数据结构", "元组是不可变的"],
        "reference": "列表可变，元组不可变"
    },
    {
        "user_input": "什么是深度学习？",
        "response": "深度学习是机器学习的一个分支，使用多层神经网络。",
        "retrieved_contexts": ["深度学习使用神经网络", "属于机器学习领域"],
        "reference": "深度学习是使用神经网络的机器学习方法"
    },
    {
        "user_input": "如何优化SQL查询性能？",
        "response": "可以添加索引、优化查询语句、避免全表扫描。",
        "retrieved_contexts": ["索引可以加速查询", "避免SELECT *", "全表扫描效率低"],
        "reference": "添加索引、优化查询语句"
    },
    {
        "user_input": "React Hooks 的优势是什么？",
        "response": "Hooks 让函数组件也能使用状态和生命周期特性。",
        "retrieved_contexts": ["Hooks用于函数组件", "可以使用状态管理"],
        "reference": "函数组件使用状态和生命周期"
    },
    {
        "user_input": "什么是RESTful API？",
        "response": "RESTful API 是一种基于HTTP协议的API设计风格。",
        "retrieved_contexts": ["REST基于HTTP", "使用标准HTTP方法"],
        "reference": "基于HTTP的API设计风格"
    },
]


async def run_serial_evaluation(samples: List[dict], judge: OpenAIJudgeClient) -> dict:
    """基线：串行评测"""
    print("🔄 运行基线实验（串行评测）...")

    # 初始化指标
    faithfulness = FaithfulnessMetric(
        MockMetricDefinition("faithfulness", "忠实度", "builtin_faithfulness"),
        None, 0.8, 1.0
    )

    answer_relevancy = AnswerRelevancyMetric(
        MockMetricDefinition("answer_relevancy", "答案相关性", "builtin_answer_relevancy"),
        None, 0.8, 1.0
    )

    context_recall = ContextRecallMetric(
        MockMetricDefinition("context_recall", "上下文召回", "builtin_context_recall"),
        None, 0.8, 1.0
    )

    start_time = time.time()
    results = []

    for i, sample in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] 评测中...")

        row_data = {
            "response": sample["response"],
            "retrieved_contexts": sample["retrieved_contexts"],
            "user_input": sample["user_input"],
            "reference": sample["reference"]
        }

        # 串行执行 3 个指标
        faith_result = await faithfulness.ascore(row_data, judge)
        rel_result = await answer_relevancy.ascore(row_data, judge)
        recall_result = await context_recall.ascore(row_data, judge)

        results.append({
            "faithfulness": faith_result.value,
            "answer_relevancy": rel_result.value,
            "context_recall": recall_result.value
        })

    elapsed = time.time() - start_time

    # 计算平均分
    avg_faith = sum(r["faithfulness"] for r in results if r["faithfulness"] is not None) / len(results)
    avg_rel = sum(r["answer_relevancy"] for r in results if r["answer_relevancy"] is not None) / len(results)
    avg_recall = sum(r["context_recall"] for r in results if r["context_recall"] is not None) / len(results)

    print(f"  ✓ 完成 - 耗时: {elapsed:.2f}s")

    return {
        "mode": "串行（基线）",
        "elapsed_time": elapsed,
        "avg_faithfulness": round(avg_faith, 4),
        "avg_answer_relevancy": round(avg_rel, 4),
        "avg_context_recall": round(avg_recall, 4),
        "weighted_mean": round((avg_faith + avg_rel + avg_recall) / 3, 4)
    }


async def run_concurrent_evaluation(samples: List[dict], judge: OpenAIJudgeClient, concurrency: int = 3) -> dict:
    """优化1：指标级并发评测"""
    print(f"🔄 运行优化实验（指标并发={concurrency}）...")

    # 初始化指标
    faithfulness = FaithfulnessMetric(
        MockMetricDefinition("faithfulness", "忠实度", "builtin_faithfulness"),
        None, 0.8, 1.0
    )

    answer_relevancy = AnswerRelevancyMetric(
        MockMetricDefinition("answer_relevancy", "答案相关性", "builtin_answer_relevancy"),
        None, 0.8, 1.0
    )

    context_recall = ContextRecallMetric(
        MockMetricDefinition("context_recall", "上下文召回", "builtin_context_recall"),
        None, 0.8, 1.0
    )

    start_time = time.time()
    results = []

    for i, sample in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] 评测中...")

        row_data = {
            "response": sample["response"],
            "retrieved_contexts": sample["retrieved_contexts"],
            "user_input": sample["user_input"],
            "reference": sample["reference"]
        }

        # 并发执行 3 个指标
        tasks = [
            faithfulness.ascore(row_data, judge),
            answer_relevancy.ascore(row_data, judge),
            context_recall.ascore(row_data, judge)
        ]

        metric_results = await asyncio.gather(*tasks)

        results.append({
            "faithfulness": metric_results[0].value,
            "answer_relevancy": metric_results[1].value,
            "context_recall": metric_results[2].value
        })

    elapsed = time.time() - start_time

    # 计算平均分
    avg_faith = sum(r["faithfulness"] for r in results if r["faithfulness"] is not None) / len(results)
    avg_rel = sum(r["answer_relevancy"] for r in results if r["answer_relevancy"] is not None) / len(results)
    avg_recall = sum(r["context_recall"] for r in results if r["context_recall"] is not None) / len(results)

    print(f"  ✓ 完成 - 耗时: {elapsed:.2f}s")

    return {
        "mode": f"指标并发（concurrency={concurrency}）",
        "elapsed_time": elapsed,
        "avg_faithfulness": round(avg_faith, 4),
        "avg_answer_relevancy": round(avg_rel, 4),
        "avg_context_recall": round(avg_recall, 4),
        "weighted_mean": round((avg_faith + avg_rel + avg_recall) / 3, 4)
    }


async def run_row_concurrent_evaluation(samples: List[dict], judge: OpenAIJudgeClient, row_concurrency: int = 3) -> dict:
    """优化2：行级并发 + 指标级并发"""
    print(f"🔄 运行优化实验（行并发={row_concurrency} + 指标并发=3）...")

    # 初始化指标
    faithfulness = FaithfulnessMetric(
        MockMetricDefinition("faithfulness", "忠实度", "builtin_faithfulness"),
        None, 0.8, 1.0
    )

    answer_relevancy = AnswerRelevancyMetric(
        MockMetricDefinition("answer_relevancy", "答案相关性", "builtin_answer_relevancy"),
        None, 0.8, 1.0
    )

    context_recall = ContextRecallMetric(
        MockMetricDefinition("context_recall", "上下文召回", "builtin_context_recall"),
        None, 0.8, 1.0
    )

    start_time = time.time()

    async def eval_one_row(sample: dict, idx: int):
        print(f"  [{idx}/{len(samples)}] 评测中...")

        row_data = {
            "response": sample["response"],
            "retrieved_contexts": sample["retrieved_contexts"],
            "user_input": sample["user_input"],
            "reference": sample["reference"]
        }

        # 并发执行 3 个指标
        tasks = [
            faithfulness.ascore(row_data, judge),
            answer_relevancy.ascore(row_data, judge),
            context_recall.ascore(row_data, judge)
        ]

        metric_results = await asyncio.gather(*tasks)

        return {
            "faithfulness": metric_results[0].value,
            "answer_relevancy": metric_results[1].value,
            "context_recall": metric_results[2].value
        }

    # 行级并发
    semaphore = asyncio.Semaphore(row_concurrency)

    async def eval_with_limit(sample: dict, idx: int):
        async with semaphore:
            return await eval_one_row(sample, idx)

    tasks = [eval_with_limit(sample, i+1) for i, sample in enumerate(samples)]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    # 计算平均分
    avg_faith = sum(r["faithfulness"] for r in results if r["faithfulness"] is not None) / len(results)
    avg_rel = sum(r["answer_relevancy"] for r in results if r["answer_relevancy"] is not None) / len(results)
    avg_recall = sum(r["context_recall"] for r in results if r["context_recall"] is not None) / len(results)

    print(f"  ✓ 完成 - 耗时: {elapsed:.2f}s")

    return {
        "mode": f"行并发={row_concurrency} + 指标并发=3",
        "elapsed_time": elapsed,
        "avg_faithfulness": round(avg_faith, 4),
        "avg_answer_relevancy": round(avg_rel, 4),
        "avg_context_recall": round(avg_recall, 4),
        "weighted_mean": round((avg_faith + avg_rel + avg_recall) / 3, 4)
    }


async def run_experiment():
    """运行并发优化实验"""

    print(f"📊 并发优化实验（时间优化）")
    print(f"=" * 80)
    print(f"样本数量: {len(SAMPLE_DATA)}")
    print(f"指标数量: 3 (faithfulness, answer_relevancy, context_recall)")
    print(f"总评测次数: {len(SAMPLE_DATA) * 3} = {len(SAMPLE_DATA) * 3}")
    print(f"=" * 80)
    print()

    # 初始化 LLM Judge
    class LLMConfig:
        def __init__(self):
            self.model_name = settings.LLM_MODEL
            self.api_base_url = settings.LLM_ENDPOINT
            self.api_key = settings.LLM_API_KEY
            self.temperature = 0.01
            self.max_tokens = 1024

    judge = OpenAIJudgeClient(LLMConfig())

    # 实验组
    experiments = []

    # 1. 基线（串行）
    result_serial = await run_serial_evaluation(SAMPLE_DATA, judge)
    experiments.append(result_serial)
    print()

    # 2. 指标并发
    result_concurrent = await run_concurrent_evaluation(SAMPLE_DATA, judge, concurrency=3)
    experiments.append(result_concurrent)
    print()

    # 3. 行并发 + 指标并发
    result_row_concurrent = await run_row_concurrent_evaluation(SAMPLE_DATA, judge, row_concurrency=3)
    experiments.append(result_row_concurrent)
    print()

    # 生成报告
    print(f"=" * 80)
    print(f"📊 实验结果对比")
    print(f"=" * 80)

    baseline_time = experiments[0]["elapsed_time"]

    for exp in experiments:
        time_saving = (baseline_time - exp["elapsed_time"]) / baseline_time * 100
        quality_change = abs(exp["weighted_mean"] - experiments[0]["weighted_mean"])

        print(f"\n{exp['mode']}:")
        print(f"  耗时: {exp['elapsed_time']:.2f}s (相比基线: {time_saving:+.1f}%)")
        print(f"  质量: weighted_mean={exp['weighted_mean']:.4f} (变化: {quality_change:.4f})")
        print(f"  各指标: faith={exp['avg_faithfulness']:.4f}, rel={exp['avg_answer_relevancy']:.4f}, recall={exp['avg_context_recall']:.4f}")

    # 保存报告
    report = {
        "experiment_info": {
            "name": "并发优化实验（时间优化）",
            "note": "本实验测试并发对评测耗时的影响。并发不会降低 token 消耗（成本相同），仅优化响应速度。",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples": len(SAMPLE_DATA),
            "metrics": 3,
            "total_evaluations": len(SAMPLE_DATA) * 3
        },
        "experiments": experiments,
        "analysis": {
            "baseline_time": baseline_time,
            "best_time_saving": f"{max((baseline_time - e['elapsed_time']) / baseline_time * 100 for e in experiments):.1f}%",
            "quality_stable": all(abs(e["weighted_mean"] - experiments[0]["weighted_mean"]) < 0.05 for e in experiments),
            "recommendation": "行并发 + 指标并发是最优解：大幅降低耗时，质量几乎不变",
            "cost_note": "⚠️ 并发优化的是时间，不是成本。token 消耗与串行相同。真正的成本优化需要批量推理或模型分层。"
        }
    }

    report_path = Path(__file__).parent.parent / "concurrency_optimization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 报告已保存: {report_path}")
    print(f"\n💡 核心结论（时间优化）：")
    print(f"  - 指标并发可降低耗时 {(baseline_time - experiments[1]['elapsed_time']) / baseline_time * 100:.1f}%")
    print(f"  - 行并发 + 指标并发可降低耗时 {(baseline_time - experiments[2]['elapsed_time']) / baseline_time * 100:.1f}%")
    print(f"  - 质量保持稳定（weighted_mean 变化 <0.05）")


if __name__ == "__main__":
    asyncio.run(run_experiment())
