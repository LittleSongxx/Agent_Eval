#!/usr/bin/env python3
"""Benchmark Judge-call concurrency with an explicit execution contract.

The production evaluator currently guarantees row-level concurrency.  This
script is a small remote-call harness for comparing serial calls, independent
metric calls, and independent row/metric calls; its report must not be quoted
as a production double-concurrency guarantee without the contract field below.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
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


SAMPLE_DATASET = BACKEND_DIR / "datasets" / "concurrency_samples.json"


def _metric_definitions() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            id=index,
            name=name,
            display_name=display,
            metric_type=metric_type,
            description=display,
            prompt_template=None,
            config={},
            category="rag",
            is_builtin=True,
        )
        for index, (name, display, metric_type) in enumerate(
            (
                ("faithfulness", "忠实度", "builtin_faithfulness"),
                ("answer_relevancy", "答案相关性", "builtin_answer_relevancy"),
                ("context_recall", "上下文召回", "builtin_context_recall"),
            ),
            start=1,
        )
    ]


def _metric_instances():
    from app.core.evaluation_engine import build_metric

    scenario_metric = SimpleNamespace(pass_threshold=0.8, weight=1.0, prompt_override=None)
    return [
        (metric_def.name, build_metric(metric_def, scenario_metric=scenario_metric)[1])
        for metric_def in _metric_definitions()
    ]


def _row_data(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_input": sample.get("user_input", ""),
        "response": sample.get("response", ""),
        "retrieved_contexts": sample.get("retrieved_contexts", []),
        "reference": sample.get("reference", ""),
    }


def _new_judge(settings):
    from app.core.evaluation_engine import OpenAIJudgeClient

    config = SimpleNamespace(
        model_name=settings.LLM_MODEL,
        api_base_url=settings.LLM_ENDPOINT,
        api_key=settings.LLM_API_KEY,
        temperature=0.01,
        max_tokens=1024,
    )
    return OpenAIJudgeClient(config)


async def _score_one(sample: dict[str, Any], settings, *, concurrent_metrics: bool) -> dict[str, Any]:
    metrics = _metric_instances()

    async def score(metric_name: str, metric, judge) -> tuple[str, Any]:
        result = await metric.ascore(_row_data(sample), judge)
        return metric_name, result.value

    if concurrent_metrics:
        # Each concurrent call owns a Judge client.  Sharing one client would
        # corrupt its per-row token accounting and make the benchmark invalid.
        values = await asyncio.gather(
            *(score(name, metric, _new_judge(settings)) for name, metric in metrics)
        )
    else:
        values = []
        judge = _new_judge(settings)
        for name, metric in metrics:
            values.append(await score(name, metric, judge))
    return dict(values)


async def _run_mode(samples: list[dict[str, Any]], settings, mode: str, row_concurrency: int = 3) -> dict[str, Any]:
    started = time.time()
    if mode == "serial":
        values = [await _score_one(sample, settings, concurrent_metrics=False) for sample in samples]
    elif mode == "metric_concurrent":
        values = [await _score_one(sample, settings, concurrent_metrics=True) for sample in samples]
    elif mode == "row_metric_concurrent":
        semaphore = asyncio.Semaphore(max(1, row_concurrency))

        async def run_limited(sample):
            async with semaphore:
                return await _score_one(sample, settings, concurrent_metrics=True)

        values = await asyncio.gather(*(run_limited(sample) for sample in samples))
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"unknown mode: {mode}")

    elapsed = time.time() - started
    means: dict[str, float | None] = {}
    for name in ("faithfulness", "answer_relevancy", "context_recall"):
        scores = [float(item[name]) for item in values if isinstance(item.get(name), (int, float))]
        means[name] = round(sum(scores) / len(scores), 4) if scores else None
    numeric = [value for value in means.values() if value is not None]
    return {
        "mode": mode,
        "row_concurrency": row_concurrency if mode == "row_metric_concurrent" else 1,
        "elapsed_seconds": round(elapsed, 3),
        "scores": means,
        "weighted_mean": round(sum(numeric) / len(numeric), 4) if numeric else None,
        "sample_count": len(samples),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测 Judge 调用并发实验（会调用远程 Judge）")
    add_common_arguments(parser, default_output=BACKEND_DIR / "concurrency_optimization_report.json")
    parser.add_argument("--row-concurrency", type=int, default=3, help="行并发上限（默认: 3）")
    return parser.parse_args(argv)


async def run_experiment(samples: list[dict[str, Any]], dataset_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    from app.core.config import settings

    ensure_api_key(settings)
    experiments = []
    for mode in ("serial", "metric_concurrent", "row_metric_concurrent"):
        print(f"运行模式: {mode}")
        result = await _run_mode(samples, settings, mode, args.row_concurrency)
        experiments.append(result)
        print(f"  {result['elapsed_seconds']}s, weighted_mean={result['weighted_mean']}")

    baseline = experiments[0]["elapsed_seconds"]
    for item in experiments:
        item["time_change_vs_serial"] = round(
            (baseline - item["elapsed_seconds"]) / baseline, 4
        ) if baseline else None
        item["quality_delta_vs_serial"] = round(
            abs((item["weighted_mean"] or 0) - (experiments[0]["weighted_mean"] or 0)), 4
        )
    report = {
        "experiment_info": {
            "name": "Judge 并发优化实验（远程调用 harness）",
            "metrics": ["faithfulness", "answer_relevancy", "context_recall"],
        },
        "provenance": build_provenance(
            script_path=Path(__file__),
            dataset_path=dataset_path,
            sample_count=len(samples),
            model=settings.LLM_MODEL,
            selected_samples=samples,
            extra={"endpoint": settings.LLM_ENDPOINT, "row_concurrency": args.row_concurrency},
        ),
        "execution_contract": {
            "production_engine_row_concurrency": True,
            "metric_level_concurrency_in_this_script": True,
            "independent_judge_per_concurrent_call": True,
            "cost_claim": "并发只改变等待时间，不代表 token 成本下降",
        },
        "experiments": experiments,
    }
    write_json_report(args.output, report, overwrite=args.overwrite)
    print(f"报告已保存: {args.output}")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.row_concurrency <= 0:
        print("错误: --row-concurrency 必须是正整数", file=sys.stderr)
        return 2
    try:
        # Keep this fixture in the repository so its hash is part of evidence.
        samples = load_samples(SAMPLE_DATASET, args.limit)
        print(f"数据集: {SAMPLE_DATASET}；样本数: {len(samples)}")
        if args.dry_run:
            print("dry-run: 校验通过，未创建 Judge，未发起网络请求，未写入报告。")
            return 0
        ensure_output_path(args.output, overwrite=args.overwrite)
        asyncio.run(run_experiment(samples, SAMPLE_DATASET, args))
        return 0
    except (FileNotFoundError, ValueError, FileExistsError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
