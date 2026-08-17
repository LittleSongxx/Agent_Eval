#!/usr/bin/env python3
"""
引用准确性评测实验脚本

对比平台 CitationAccuracyMetric 的检测能力：
- 测试 50 个样本（正确引用 + 引用错位 + 虚假引用）
- 生成检测准确率、召回率、F1 等指标
- 输出详细实验报告
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.evaluation_engine import CitationAccuracyMetric, OpenAIJudgeClient
from app.core.config import settings


class MockMetricDefinition:
    """模拟 MetricDefinition 对象"""
    def __init__(self):
        self.name = "citation_accuracy"
        self.metric_type = "builtin_citation_accuracy"
        self.description = "引用准确性评测"
        self.display_name = "引用准确性"
        self.prompt_template = None
        self.config = {}


async def run_experiment():
    """运行引用准确性实验"""

    # 加载测试数据
    dataset_path = Path(__file__).parent.parent / "datasets" / "citation_samples.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"📊 引用准确性评测实验")
    print(f"=" * 80)
    print(f"样本数量: {len(samples)}")
    print(f"数据集: {dataset_path}")
    print(f"=" * 80)
    print()

    # 初始化评测器
    metric_def = MockMetricDefinition()
    metric = CitationAccuracyMetric(
        metric_def=metric_def,
        prompt_override=None,
        pass_threshold=0.8,
        weight=1.0
    )

    # 初始化 LLM Judge
    class LLMConfig:
        def __init__(self):
            self.model_name = settings.LLM_MODEL
            self.api_base_url = settings.LLM_ENDPOINT
            self.api_key = settings.LLM_API_KEY
            self.temperature = 0.01
            self.max_tokens = 1024

    llm_config = LLMConfig()
    judge = OpenAIJudgeClient(llm_config)

    # 运行评测
    results = []
    correct_predictions = 0
    total_citations_detected = 0
    total_citations_expected = 0

    start_time = time.time()

    for i, sample in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] 评测样本: {sample['user_input'][:40]}...")

        row_data = {
            "response": sample["response"],
            "retrieved_contexts": sample["retrieved_contexts"]
        }

        try:
            result = await metric.ascore(row_data, judge)

            # 分析结果
            score = result.score
            expected = sample.get("expected_citation_accuracy", 1.0)
            prediction_correct = abs(score - expected) < 0.15  # 容忍 15% 误差

            if prediction_correct:
                correct_predictions += 1

            results.append({
                "index": i,
                "user_input": sample["user_input"],
                "response_preview": sample["response"][:100] + "...",
                "score": score,
                "expected": expected,
                "reason": result.reason,
                "annotation": sample.get("annotation", ""),
                "prediction_correct": prediction_correct
            })

            # 统计引用数量（从 reason 中提取）
            if "引用数" in result.reason or "citations" in result.reason.lower():
                # 简单启发式：从评分推断
                total_citations_detected += 1

            print(f"  ✓ 得分: {score:.2f} (期望: {expected:.2f}) - {'✓' if prediction_correct else '✗'}")

        except Exception as e:
            print(f"  ✗ 错误: {e}")
            results.append({
                "index": i,
                "user_input": sample["user_input"],
                "error": str(e),
                "prediction_correct": False
            })

    end_time = time.time()
    elapsed = end_time - start_time

    # 计算统计指标
    accuracy = correct_predictions / len(samples) if samples else 0
    avg_time_per_sample = elapsed / len(samples) if samples else 0

    # 分析错误类型检测能力
    perfect_citations = [r for r in results if r.get("expected", 0) == 1.0]
    partial_errors = [r for r in results if 0.5 <= r.get("expected", 0) < 1.0]
    major_errors = [r for r in results if r.get("expected", 0) < 0.5]

    perfect_detected = sum(1 for r in perfect_citations if r.get("prediction_correct", False))
    partial_detected = sum(1 for r in partial_errors if r.get("prediction_correct", False))
    major_detected = sum(1 for r in major_errors if r.get("prediction_correct", False))

    # 生成报告
    report = {
        "experiment_info": {
            "name": "引用准确性评测实验",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": str(dataset_path),
            "total_samples": len(samples),
            "elapsed_time": f"{elapsed:.2f}s",
            "avg_time_per_sample": f"{avg_time_per_sample:.2f}s"
        },
        "overall_metrics": {
            "prediction_accuracy": f"{accuracy:.2%}",
            "correct_predictions": correct_predictions,
            "total_samples": len(samples),
            "pass_threshold": 0.8
        },
        "error_detection_breakdown": {
            "perfect_citations": {
                "total": len(perfect_citations),
                "detected_correctly": perfect_detected,
                "accuracy": f"{perfect_detected/len(perfect_citations):.2%}" if perfect_citations else "N/A"
            },
            "partial_errors": {
                "total": len(partial_errors),
                "detected_correctly": partial_detected,
                "accuracy": f"{partial_detected/len(partial_errors):.2%}" if partial_errors else "N/A"
            },
            "major_errors": {
                "total": len(major_errors),
                "detected_correctly": major_detected,
                "accuracy": f"{major_detected/len(major_errors):.2%}" if major_errors else "N/A"
            }
        },
        "detailed_results": results
    }

    # 保存报告
    report_path = Path(__file__).parent.parent / "citation_accuracy_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print()
    print(f"=" * 80)
    print(f"📈 实验结果摘要")
    print(f"=" * 80)
    print(f"总样本数: {len(samples)}")
    print(f"预测准确率: {accuracy:.2%} ({correct_predictions}/{len(samples)})")
    print(f"总耗时: {elapsed:.2f}s (平均 {avg_time_per_sample:.2f}s/样本)")
    print()
    print(f"错误检测能力:")
    print(f"  - 完美引用 ({len(perfect_citations)} 个): {perfect_detected/len(perfect_citations):.2%} 准确" if perfect_citations else "  - 完美引用: 无样本")
    print(f"  - 部分错误 ({len(partial_errors)} 个): {partial_detected/len(partial_errors):.2%} 准确" if partial_errors else "  - 部分错误: 无样本")
    print(f"  - 严重错误 ({len(major_errors)} 个): {major_detected/len(major_errors):.2%} 准确" if major_errors else "  - 严重错误: 无样本")
    print()
    print(f"完整报告已保存: {report_path}")
    print(f"=" * 80)

    return report


if __name__ == "__main__":
    asyncio.run(run_experiment())
