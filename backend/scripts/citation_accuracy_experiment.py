#!/usr/bin/env python3
"""Measure the citation metric against a labelled synthetic gold set.

The old report called a score-within-15%-of-expectation count "accuracy" and
claimed a major-error result although the fixture contained no major-error
labels.  This version reports regression-friendly MAE/tolerance statistics and
an explicit error-detection confusion matrix, including label coverage.
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
    name = "citation_accuracy"
    metric_type = "builtin_citation_accuracy"
    description = "引用准确性评测"
    display_name = "引用准确性"
    prompt_template = None
    config: dict[str, Any] = {}


def _binary_confusion(results: list[dict[str, Any]], decision_threshold: float) -> dict[str, Any]:
    pairs = []
    for item in results:
        score = item.get("score")
        expected = item.get("expected")
        if isinstance(score, (int, float)) and isinstance(expected, (int, float)):
            # positive means "contains a citation error".
            pairs.append((float(score) < decision_threshold, float(expected) < 1.0))
    tp = sum(pred and gold for pred, gold in pairs)
    tn = sum(not pred and not gold for pred, gold in pairs)
    fp = sum(pred and not gold for pred, gold in pairs)
    fn = sum(not pred and gold for pred, gold in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "decision_threshold": decision_threshold,
        "sample_count": len(pairs),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _stratum_stats(results: list[dict[str, Any]], name: str, predicate) -> dict[str, Any]:
    group = [item for item in results if predicate(item.get("expected"))]
    pairs = [
        (float(item["score"]), float(item["expected"]))
        for item in group
        if isinstance(item.get("score"), (int, float)) and isinstance(item.get("expected"), (int, float))
    ]
    if not pairs:
        return {"label": name, "total": len(group), "scored": 0, "note": "该类别无有效标注/模型分数"}
    errors = [abs(score - expected) for score, expected in pairs]
    return {
        "label": name,
        "total": len(group),
        "scored": len(pairs),
        "score_mean": round(sum(score for score, _ in pairs) / len(pairs), 4),
        "expected_mean": round(sum(expected for _, expected in pairs) / len(pairs), 4),
        "mae": round(sum(errors) / len(errors), 4),
        "tolerance_accuracy_0_15": round(sum(error <= 0.15 for error in errors) / len(errors), 4),
    }


async def run_experiment(samples: list[dict[str, Any]], dataset_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    from app.core.config import settings
    from app.core.evaluation_engine import CitationAccuracyMetric, OpenAIJudgeClient

    ensure_api_key(settings)
    metric = CitationAccuracyMetric(MockMetricDefinition(), None, 0.8, 1.0)

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
        print(f"[{index}/{len(samples)}] {str(sample.get('user_input', ''))[:60]}...")
        try:
            metric_result = await metric.ascore(
                {
                    "user_input": sample.get("user_input", ""),
                    "response": sample.get("response", ""),
                    "retrieved_contexts": sample.get("retrieved_contexts", []),
                },
                judge,
            )
            score = metric_result.value
            reason = metric_result.reason
        except Exception as exc:
            score = None
            reason = f"评测失败: {str(exc)[:200]}"
        item = {
            "index": index,
            "sample_id": sample.get("sample_id", f"citation-{index:03d}"),
            "score": score,
            "expected": sample.get("expected_citation_accuracy"),
            "annotation": sample.get("annotation", ""),
            "reason": reason,
        }
        results.append(item)
        print(f"  score={score if score is not None else 'N/A'} expected={item['expected']}")

    elapsed = time.time() - started
    pairs = [
        (float(item["score"]), float(item["expected"]))
        for item in results
        if isinstance(item.get("score"), (int, float)) and isinstance(item.get("expected"), (int, float))
    ]
    absolute_errors = [abs(score - expected) for score, expected in pairs]
    overall = {
        "scored_samples": len(pairs),
        "failed_samples": len(results) - len(pairs),
        "score_mean": round(sum(score for score, _ in pairs) / len(pairs), 4) if pairs else None,
        "expected_mean": round(sum(expected for _, expected in pairs) / len(pairs), 4) if pairs else None,
        "mae": round(sum(absolute_errors) / len(absolute_errors), 4) if pairs else None,
        "rmse": round(math.sqrt(sum(error * error for error in absolute_errors) / len(absolute_errors)), 4) if pairs else None,
        "tolerance": 0.15,
        "tolerance_accuracy": round(sum(error <= 0.15 for error in absolute_errors) / len(absolute_errors), 4) if pairs else None,
    }
    report = {
        "experiment_info": {
            "name": "引用准确性评测实验",
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
        "overall": overall,
        "error_detection": _binary_confusion(results, args.decision_threshold),
        "strata": [
            _stratum_stats(results, "perfect_citation", lambda value: value is not None and value >= 1.0),
            _stratum_stats(results, "partial_error", lambda value: value is not None and 0.5 <= value < 1.0),
            _stratum_stats(results, "major_error", lambda value: value is not None and value < 0.5),
        ],
        "sample_results": results,
    }
    write_json_report(args.output, report, overwrite=args.overwrite)
    print(f"报告已保存: {args.output}")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="引用准确性元评测（会调用远程 Judge）")
    dataset_path = BACKEND_DIR / "datasets" / "citation_samples.json"
    add_common_arguments(parser, default_output=BACKEND_DIR / "citation_accuracy_report.json")
    parser.add_argument("--dataset", type=Path, default=dataset_path, help=f"数据集路径（默认: {dataset_path}）")
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.85,
        help="把 citation score 低于该值视为检测到引用错误（默认: 0.85）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 <= args.decision_threshold <= 1:
        print("错误: --decision-threshold 必须在 0 和 1 之间", file=sys.stderr)
        return 2
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
