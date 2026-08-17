#!/usr/bin/env python3
"""最小化测试 - 单个样本"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.evaluation_engine import OpenAIJudgeClient, STRUCTURED_JSON_SYSTEM_PROMPT
from app.core.config import settings


async def minimal_test():
    print("🧪 测试 LLM Judge 调用")
    print("=" * 60)

    # 初始化 Judge
    class LLMConfig:
        def __init__(self):
            self.model_name = settings.LLM_MODEL
            self.api_base_url = settings.LLM_ENDPOINT
            self.api_key = settings.LLM_API_KEY
            self.temperature = 0.01
            self.max_tokens = 512

    judge = OpenAIJudgeClient(LLMConfig())

    # 简单测试
    print("测试 1: 基础调用")
    try:
        result = await judge.chat_json(
            STRUCTURED_JSON_SYSTEM_PROMPT,
            '判断"Python是编程语言"是否正确。返回 {"verdict": "true"或"false", "reason": "理由"}',
        )
        print(f"✓ 成功: {result}")
    except Exception as e:
        print(f"✗ 失败: {e}")
        return

    # 引用验证测试
    print("\n测试 2: 引用验证")
    try:
        verification_prompt = """判断以下文档是否支持所引用的内容。

被引用的内容：
根据文档[1]，Python 中列表是可变的。

被引用的文档：
Python 列表（list）是可变的数据结构，使用方括号 [] 定义。

如果文档明确支持该内容，返回 {"verdict": "supported", "reason": "具体理由"}；
否则返回 {"verdict": "not_supported", "reason": "具体理由"}。"""

        result = await judge.chat_json(STRUCTURED_JSON_SYSTEM_PROMPT, verification_prompt)
        print(f"✓ 成功: {result}")
    except Exception as e:
        print(f"✗ 失败: {e}")

    print("\n" + "=" * 60)
    print("✓ 测试完成")


if __name__ == "__main__":
    asyncio.run(minimal_test())
