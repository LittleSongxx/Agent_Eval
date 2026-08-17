#!/usr/bin/env python3
"""Agent 轨迹级评测实验"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.evaluation_engine import (
    TrajectoryFaithfulnessMetric,
    ErrorRecoveryMetric,
    ToolSelectionRationalityMetric,
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


async def run_experiment():
    """运行 Agent 轨迹级评测实验"""

    # 加载样本
    dataset_path = Path(__file__).parent.parent / "datasets" / "agent_trajectory_samples.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"📊 Agent 轨迹级评测实验")
    print(f"=" * 80)
    print(f"样本数量: {len(samples)}")
    print(f"数据集: {dataset_path}")
    print(f"=" * 80)
    print()

    # 初始化指标
    trajectory_metric = TrajectoryFaithfulnessMetric(
        MockMetricDefinition("trajectory_faithfulness", "轨迹忠实度", "builtin_trajectory_faithfulness"),
        None, 0.8, 1.0
    )

    error_recovery_metric = ErrorRecoveryMetric(
        MockMetricDefinition("error_recovery", "错误恢复能力", "builtin_error_recovery"),
        None, 0.6, 1.0
    )

    tool_selection_metric = ToolSelectionRationalityMetric(
        MockMetricDefinition("tool_selection_rationality", "工具选择合理性", "builtin_tool_selection_rationality"),
        None, 0.9, 1.0
    )

    # 初始化 LLM Judge
    class LLMConfig:
        def __init__(self):
            self.model_name = settings.LLM_MODEL
            self.api_base_url = settings.LLM_ENDPOINT
            self.api_key = settings.LLM_API_KEY
            self.temperature = 0.01
            self.max_tokens = 1024

    judge = OpenAIJudgeClient(LLMConfig())

    # 结果收集
    results = []
    trajectory_scores = []
    error_recovery_scores = []
    tool_selection_scores = []

    start_time = time.time()

    # 评测每个样本
    for i, sample in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {sample['task']}")

        row_data = {
            "agent_trajectory": sample["agent_trajectory"],
            "available_tools": sample.get("available_tools", {})
        }

        # 1. 轨迹忠实度
        try:
            traj_result = await trajectory_metric.ascore(row_data, judge)
            traj_score = traj_result.value
            traj_reason = traj_result.reason
            if traj_score is not None:
                trajectory_scores.append(traj_score)
        except Exception as e:
            traj_score = None
            traj_reason = f"评测失败: {str(e)[:100]}"

        # 2. 错误恢复能力
        try:
            err_result = await error_recovery_metric.ascore(row_data, judge)
            err_score = err_result.value
            err_reason = err_result.reason
            if err_score is not None:
                error_recovery_scores.append(err_score)
        except Exception as e:
            err_score = None
            err_reason = f"评测失败: {str(e)[:100]}"

        # 3. 工具选择合理性
        try:
            tool_result = await tool_selection_metric.ascore(row_data, judge)
            tool_score = tool_result.value
            tool_reason = tool_result.reason
            if tool_score is not None:
                tool_selection_scores.append(tool_score)
        except Exception as e:
            tool_score = None
            tool_reason = f"评测失败: {str(e)[:100]}"

        # 输出结果
        print(f"  轨迹忠实度: {traj_score if traj_score is not None else 'N/A'} | 期望: {sample.get('expected_trajectory_faithfulness', 'N/A')}")
        print(f"  错误恢复: {err_score if err_score is not None else 'N/A'} | 期望: {sample.get('expected_error_recovery', 'N/A')}")
        print(f"  工具选择: {tool_score if tool_score is not None else 'N/A'} | 期望: {sample.get('expected_tool_selection_rationality', 'N/A')}")
        print(f"  注释: {sample['annotation']}")
        print()

        # 保存详细结果
        results.append({
            "task": sample["task"],
            "annotation": sample["annotation"],
            "trajectory_faithfulness": {
                "score": traj_score,
                "expected": sample.get("expected_trajectory_faithfulness"),
                "reason": traj_reason
            },
            "error_recovery": {
                "score": err_score,
                "expected": sample.get("expected_error_recovery"),
                "reason": err_reason
            },
            "tool_selection_rationality": {
                "score": tool_score,
                "expected": sample.get("expected_tool_selection_rationality"),
                "reason": tool_reason
            }
        })

    elapsed_time = time.time() - start_time

    # 计算统计指标
    avg_trajectory = sum(trajectory_scores) / len(trajectory_scores) if trajectory_scores else 0
    avg_error_recovery = sum(error_recovery_scores) / len(error_recovery_scores) if error_recovery_scores else 0
    avg_tool_selection = sum(tool_selection_scores) / len(tool_selection_scores) if tool_selection_scores else 0

    # 生成报告
    report = {
        "experiment_info": {
            "name": "Agent 轨迹级评测实验",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": str(dataset_path),
            "total_samples": len(samples),
            "elapsed_time": f"{elapsed_time:.2f}s",
            "avg_time_per_sample": f"{elapsed_time / len(samples):.2f}s"
        },
        "overall_metrics": {
            "trajectory_faithfulness_mean": round(avg_trajectory, 4),
            "trajectory_faithfulness_samples": len(trajectory_scores),
            "error_recovery_rate": round(avg_error_recovery, 4),
            "error_recovery_samples": len(error_recovery_scores),
            "tool_selection_rationality": round(avg_tool_selection, 4),
            "tool_selection_samples": len(tool_selection_scores)
        },
        "breakdown": {
            "perfect_faithfulness": sum(1 for s in trajectory_scores if s == 1.0),
            "partial_faithfulness": sum(1 for s in trajectory_scores if 0 < s < 1.0),
            "zero_faithfulness": sum(1 for s in trajectory_scores if s == 0.0),
            "successful_recovery": sum(1 for s in error_recovery_scores if s > 0),
            "failed_recovery": sum(1 for s in error_recovery_scores if s == 0),
            "optimal_tool_selection": sum(1 for s in tool_selection_scores if s == 1.0),
            "suboptimal_tool_selection": sum(1 for s in tool_selection_scores if s < 1.0)
        },
        "sample_results": results
    }

    # 保存报告
    report_path = Path(__file__).parent.parent / "agent_trajectory_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 输出总结
    print(f"=" * 80)
    print(f"📊 实验总结")
    print(f"=" * 80)
    print(f"轨迹忠实度均值: {avg_trajectory:.4f} ({len(trajectory_scores)} 个样本)")
    print(f"  - 完全忠实 (1.0): {report['breakdown']['perfect_faithfulness']} 个")
    print(f"  - 部分忠实 (0-1): {report['breakdown']['partial_faithfulness']} 个")
    print(f"  - 完全不忠实 (0): {report['breakdown']['zero_faithfulness']} 个")
    print()
    print(f"错误恢复率: {avg_error_recovery:.4f} ({len(error_recovery_scores)} 个样本)")
    print(f"  - 成功恢复: {report['breakdown']['successful_recovery']} 个")
    print(f"  - 恢复失败: {report['breakdown']['failed_recovery']} 个")
    print()
    print(f"工具选择合理性: {avg_tool_selection:.4f} ({len(tool_selection_scores)} 个样本)")
    print(f"  - 最优选择 (1.0): {report['breakdown']['optimal_tool_selection']} 个")
    print(f"  - 次优选择 (<1.0): {report['breakdown']['suboptimal_tool_selection']} 个")
    print()
    print(f"总耗时: {elapsed_time:.2f}s")
    print(f"平均耗时: {elapsed_time / len(samples):.2f}s/样本")
    print()
    print(f"✓ 报告已保存: {report_path}")


if __name__ == "__main__":
    asyncio.run(run_experiment())
