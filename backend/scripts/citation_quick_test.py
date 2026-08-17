#!/usr/bin/env python3
"""快速测试 CitationAccuracyMetric - 5个样本"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.evaluation_engine import CitationAccuracyMetric, OpenAIJudgeClient
from app.core.config import settings


class MockMetricDefinition:
    def __init__(self):
        self.name = "citation_accuracy"
        self.metric_type = "builtin_citation_accuracy"
        self.description = "引用准确性评测"
        self.display_name = "引用准确性"
        self.prompt_template = None
        self.config = {}


async def quick_test():
    # 加载前5个样本
    dataset_path = Path(__file__).parent.parent / "datasets" / "citation_samples.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    samples = all_samples[:10]

    print(f"🧪 引用准确性快速测试")
    print(f"=" * 60)
    print(f"测试样本: {len(samples)}")
    print(f"=" * 60)
    print()

    # 初始化
    metric_def = MockMetricDefinition()
    metric = CitationAccuracyMetric(metric_def, None, 0.8, 1.0)

    class LLMConfig:
        def __init__(self):
            self.model_name = settings.LLM_MODEL
            self.api_base_url = settings.LLM_ENDPOINT
            self.api_key = settings.LLM_API_KEY
            self.temperature = 0.01
            self.max_tokens = 1024

    judge = OpenAIJudgeClient(LLMConfig())

    # 运行测试
    for i, sample in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {sample['user_input']}")

        row_data = {
            "response": sample["response"],
            "retrieved_contexts": sample["retrieved_contexts"]
        }

        try:
            result = await metric.ascore(row_data, judge)
            score = result.value
            expected = sample.get("expected_citation_accuracy", 1.0)

            print(f"  得分: {score:.2f} | 期望: {expected:.2f}")
            print(f"  理由: {result.reason[:80]}...")
            print(f"  {'✓ 准确' if abs(score - expected) < 0.15 else '✗ 偏差较大'}")
            print()

        except Exception as e:
            print(f"  ✗ 错误: {e}")
            print()

    print(f"=" * 60)
    print(f"✓ 快速测试完成")


if __name__ == "__main__":
    asyncio.run(quick_test())
