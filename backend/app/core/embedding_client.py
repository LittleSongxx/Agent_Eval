"""OpenAI 兼容 embedding 客户端（平台原生实现）。

供生成式相关性指标（反向问题 + 语义相似度）与数据集污染检测脚本共用，
不依赖任何第三方向量库或评测框架。
"""

from __future__ import annotations

import typing as t


class OpenAICompatibleEmbeddingClient:
    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
    ):
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI(
            base_url=api_base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=2,
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量获取向量（每批 64 条，规避网关批量上限）。"""
        if not texts:
            return []
        results: list[list[float]] = []
        for start in range(0, len(texts), 64):
            batch = texts[start : start + 64]
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            results.extend(item.embedding for item in response.data)
        return results

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0] if vectors else []

    @staticmethod
    def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
        """余弦相似度（内积 / 模长积），数值 0-1。"""
        if not vector_a or not vector_b:
            return 0.0
        dot = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = sum(a * a for a in vector_a) ** 0.5
        norm_b = sum(b * b for b in vector_b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))
