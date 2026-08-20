#!/usr/bin/env python3
"""Run the Agent trajectory meta-evaluation experiment.

This is an opt-in, remote-model experiment. ``--help`` and ``--dry-run`` are
side-effect free; a real run records enough provenance to make the numbers
defensible in an interview or a regression review.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from scripts.experiment_utils import (  # noqa: E402
    add_common_arguments,
    build_provenance,
    ensure_api_key,
    ensure_output_path,
    load_samples,
    write_json_report,
)


class MockMetricDefinition:
    def __init__(self, name: str, display_name: str, metric_type: str):
        self.name = name
        self.metric_type = metric_type
        self.display_name = display_name
        self.description = f"{display_name}评测"
        self.prompt_template = None
        self.config = {}


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _spearman(actual: list[float], expected: list[float]) -> float | None:
    if len(actual) < 2 or len(actual) != len(expected):
        return None
    actual_rank = _rank(actual)
    expected_rank = _rank(expected)
    actual_mean = sum(actual_rank) / len(actual_rank)
    expected_mean = sum(expected_rank) / len(expected_rank)
    numerator = sum((a - actual_mean) * (e - expected_mean) for a, e in zip(actual_rank, expected_rank))
    denominator = math.sqrt(
        sum((a - actual_mean) ** 2 for a in actual_rank)
        * sum((e - expected_mean) ** 2 for e in expected_rank)
    )
    return round(numerator / denominator, 4) if denominator else None


def _confusion(actual: list[float], expected: list[float]) -> dict[str, Any] | None:
    if not actual or not expected or not all(value in (0.0, 1.0) for value in expected):
        return None
    pairs = [(1 if value >= 0.5 else 0, int(target)) for value, target in zip(actual, expected)]
    tp = sum(pred == target == 1 for pred, target in pairs)
    tn = sum(pred == target == 0 for pred, target in pairs)
    fp = sum(pred == 1 and target == 0 for pred, target in pairs)
    fn = sum(pred == 0 and target == 1 for pred, target in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _metric_stats(results: list[dict[str, Any]], field: str, tolerance: float) -> dict[str, Any]:
    pairs = []
    for item in results:
        actual = item.get(field, {}).get("score")
        expected = item.get(field, {}).get("expected")
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            pairs.append((float(actual), float(expected)))
    if not pairs:
        return {"sample_count": 0, "note": "没有该指标的人工期望值或 Judge 未返回分数"}
    actual = [pair[0] for pair in pairs]
    expected = [pair[1] for pair in pairs]
    mae = sum(abs(a - e) for a, e in pairs) / len(pairs)
    rmse = math.sqrt(sum((a - e) ** 2 for a, e in pairs) / len(pairs))
    stats: dict[str, Any] = {
        "sample_count": len(pairs),
        "score_mean": round(sum(actual) / len(actual), 4),
        "expected_mean": round(sum(expected) / len(expected), 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "tolerance": tolerance,
        "tolerance_accuracy": round(sum(abs(a - e) <= tolerance for a, e in pairs) / len(pairs), 4),
        "spearman": _spearman(actual, expected),
    }
    confusion = _confusion(actual, expected)
    if confusion is not None:
        stats["binary_confusion"] = confusion
    return stats


async def run_experiment(samples: list[dict[str, Any]], dataset_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    # Imports happen only after argument parsing and dry-run validation.
    from app.core.config import settings
    from app.core.evaluation_engine import (
        ErrorRecoveryMetric,
        OpenAIJudgeClient,
        ToolSelectionRationalityMetric,
        TrajectoryFaithfulnessMetric,
    )

    ensure_api_key(settings)
    trajectory_metric = TrajectoryFaithfulnessMetric(
        MockMetricDefinition("trajectory_faithfulness", "轨迹忠实度", "builtin_trajectory_faithfulness"), None, 0.8, 1.0
    )
    error_recovery_metric = ErrorRecoveryMetric(
        MockMetricDefinition("error_recovery", "错误恢复能力", "builtin_error_recovery"), None, 0.6, 1.0
    )
    tool_selection_metric = ToolSelectionRationalityMetric(
        MockMetricDefinition("tool_selection_rationality", "工具选择合理性", "builtin_tool_selection_rationality"), None, 0.9, 1.0
    )

    class LLMConfig:
        model_name = settings.LLM_MODEL
        api_base_url = settings.LLM_ENDPOINT
        api_key = settings.LLM_API_KEY
        temperature = 0.01
        max_tokens = 1024

    judge = OpenAIJudgeClient(LLMConfig())
    results: list[dict[str, Any]] = []
    started = time.time()
    for index, sample in enumerate(samples, 1):
        print(f"[{index}/{len(samples)}] {sample.get('task', sample.get('user_input', ''))}")
        row_data = {"agent_trajectory": sample.get("agent_trajectory"), "available_tools": sample.get("available_tools", {})}
        result_item: dict[str, Any] = {
            "index": index,
            "sample_id": sample.get("sample_id", f"agent-{index:03d}"),
            "task": sample.get("task", ""),
            "annotation": sample.get("annotation", ""),
        }
        for key, metric, expected_key in (
            ("trajectory_faithfulness", trajectory_metric, "expected_trajectory_faithfulness"),
            ("error_recovery", error_recovery_metric, "expected_error_recovery"),
            ("tool_selection_rationality", tool_selection_metric, "expected_tool_selection_rationality"),
        ):
            try:
                metric_result = await metric.ascore(row_data, judge)
                score = metric_result.value
                reason = metric_result.reason
            except Exception as exc:
                score = None
                reason = f"评测失败: {str(exc)[:200]}"
            result_item[key] = {"score": score, "expected": sample.get(expected_key), "reason": reason}
            print(f"  {key}: {score if score is not None else 'N/A'} (期望: {sample.get(expected_key, 'N/A')})")
        results.append(result_item)

    elapsed = time.time() - started
    report = {
        "experiment_info": {
            "name": "Agent 轨迹级评测实验",
            "elapsed_seconds": round(elapsed, 3),
            "avg_seconds_per_sample": round(elapsed / len(samples), 3),
        },
        "provenance": build_provenance(
            script_path=Path(__file__),
            dataset_path=dataset_path,
            sample_count=len(samples),
            model=settings.LLM_MODEL,
            selected_samples=samples,
            extra={"endpoint": settings.LLM_ENDPOINT, "judge_temperature": 0.01},
        ),
        "metrics": {
            "trajectory_faithfulness": _metric_stats(results, "trajectory_faithfulness", 0.15),
            "error_recovery": _metric_stats(results, "error_recovery", 0.15),
            "tool_selection_rationality": _metric_stats(results, "tool_selection_rationality", 0.15),
        },
        "sample_results": results,
    }
    write_json_report(args.output, report, overwrite=args.overwrite)
    print(f"报告已保存: {args.output}")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent 轨迹级指标元评测（会调用远程 Judge）")
    dataset_path = BACKEND_DIR / "datasets" / "agent_trajectory_samples.json"
    add_common_arguments(parser, default_output=BACKEND_DIR / "agent_trajectory_report.json")
    parser.add_argument("--dataset", type=Path, default=dataset_path, help=f"数据集路径（默认: {dataset_path}）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        samples = load_samples(args.dataset, args.limit)
        print(f"数据集: {args.dataset}；样本数: {len(samples)}")
        if args.dry_run:
            print("dry-run: 校验通过，未创建 Judge，未发起网络请求，未写入报告。")
            return 0
        ensure_output_path(args.output, overwrite=args.overwrite)
        asyncio.run(run_experiment(samples, args.dataset, args))
        return 0
    except (FileNotFoundError, ValueError, FileExistsError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
