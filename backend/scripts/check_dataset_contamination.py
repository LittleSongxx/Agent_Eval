"""数据集污染检测：n-gram 重叠 + embedding 相似度双信号。

评测集可能被污染（问题/答案已进入模型预训练数据或与语料高度重叠），
导致指标虚高。本工具对数据集样本与给定语料做两级检测：
1. 词面级：字符 5-gram 重叠率（挡近重复/逐字泄露）；
2. 语义级：OpenAI 兼容 embedding 余弦相似度（挡改写式泄露，需 --embed）。

用法：
    cd backend
    python -m scripts.check_dataset_contamination \
        --dataset "售后政策问答集（分章版）" \
        --corpus /tmp/policy_parts \
        [--embed] [--ngram-overlap 0.8] [--sim-threshold 0.92] [--limit 50]
"""

from __future__ import annotations

import argparse
import json
import re
import typing as t
from pathlib import Path


def _load_dataset_rows(dataset_name: str, limit: int) -> list[dict[str, t.Any]]:
    from app.core.database import SessionLocal
    from app.models.dataset import Dataset, DatasetRow

    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.name == dataset_name).first()
        if dataset is None:
            raise SystemExit(f"数据集不存在: {dataset_name}")
        rows = (
            db.query(DatasetRow)
            .filter(DatasetRow.dataset_id == dataset.id)
            .order_by(DatasetRow.row_index)
            .limit(limit)
            .all()
        )
        return [dict(row.data or {}) for row in rows]
    finally:
        db.close()


def _load_corpus_sentences(source: str) -> list[str]:
    source_path = Path(source)
    files = [source_path] if source_path.is_file() else sorted(source_path.glob("*.md")) + sorted(source_path.glob("*.txt"))
    sentences: list[str] = []
    for file_path in files:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        for block in re.split(r"[。！？\n]+", content):
            block = block.strip()
            if len(block) >= 10:
                sentences.append(block)
    if not sentences:
        raise SystemExit(f"语料为空: {source}")
    return sentences


def _char_ngrams(text: str, n: int = 5) -> set[str]:
    compact = re.sub(r"\s+", "", str(text).lower())
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def _max_ngram_overlap(text: str, sentences: list[str], n: int = 5) -> float:
    """文本与语料句子的最大 n-gram 重叠率（词面级泄露信号）。"""
    text_ngrams = _char_ngrams(text, n)
    if not text_ngrams:
        return 0.0
    best = 0.0
    for sentence in sentences:
        sentence_ngrams = _char_ngrams(sentence, n)
        if not sentence_ngrams:
            continue
        overlap = len(text_ngrams & sentence_ngrams) / len(text_ngrams)
        if overlap > best:
            best = overlap
            if best >= 1.0:
                break
    return best


def _max_embedding_similarity(text: str, sentences: list[str]) -> float:
    """文本与语料的最大 embedding 余弦相似度（语义级泄露信号）。"""
    import os

    from app.core.embedding_client import OpenAICompatibleEmbeddingClient
    from app.core.database import SessionLocal
    from app.models.llm_config import LLMConfig

    db = SessionLocal()
    try:
        llm_config = (
            db.query(LLMConfig).filter(LLMConfig.is_default == True).first() or db.query(LLMConfig).first()  # noqa: E712
        )
    finally:
        db.close()
    if llm_config is None:
        raise SystemExit("数据库中没有 LLM 配置，无法调用 embedding")

    client = OpenAICompatibleEmbeddingClient(
        llm_config.api_base_url,
        llm_config.api_key,
        os.getenv("LLM_EMBEDDING_MODEL", "text-embedding-v3"),
    )
    import asyncio

    async def _compute() -> float:
        vectors = await client.embed_texts([text] + sentences)
        text_vector = vectors[0]
        return max(
            OpenAICompatibleEmbeddingClient.cosine_similarity(text_vector, vector)
            for vector in vectors[1:]
        )

    return asyncio.run(_compute())


def main() -> None:
    parser = argparse.ArgumentParser(description="数据集污染检测")
    parser.add_argument("--dataset", required=True, help="数据集名称")
    parser.add_argument("--corpus", required=True, help="比对语料（文件或目录）")
    parser.add_argument("--embed", action="store_true", help="启用 embedding 语义级检测")
    parser.add_argument("--ngram-overlap", type=float, default=0.8, help="n-gram 重叠率阈值")
    parser.add_argument("--sim-threshold", type=float, default=0.92, help="embedding 相似度阈值")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    rows = _load_dataset_rows(args.dataset, args.limit)
    sentences = _load_corpus_sentences(args.corpus)
    print(f"数据集 {args.dataset}：{len(rows)} 条样本；语料：{len(sentences)} 个句子")

    flagged: list[dict[str, t.Any]] = []
    for idx, row in enumerate(rows, start=1):
        user_input = str(row.get("user_input") or "")
        reference = str(row.get("reference") or "")
        overlap = _max_ngram_overlap(user_input, sentences)
        embedding_sim = _max_embedding_similarity(user_input, sentences) if args.embed else None
        overlap_flag = overlap >= args.ngram_overlap
        embed_flag = embedding_sim is not None and embedding_sim >= args.sim_threshold
        if overlap_flag or embed_flag:
            flagged.append(
                {
                    "row_index": idx,
                    "user_input": user_input[:80],
                    "ngram_overlap": round(overlap, 4),
                    "embedding_sim": round(embedding_sim, 4) if embedding_sim is not None else None,
                    "reason": "词面级疑似污染" if overlap_flag else "语义级疑似污染",
                }
            )
        print(f"  行 {idx}: ngram={overlap:.2f}{'  ⚠' if overlap_flag else ''}")

    print("=" * 70)
    print(f"疑似污染样本: {len(flagged)}/{len(rows)}")
    for item in flagged:
        print(
            f"  ⚠ 行 {item['row_index']} [{item['reason']}] "
            f"ngram={item['ngram_overlap']} sim={item['embedding_sim']}"
        )
        print(f"    {item['user_input']}")

    output = "contamination_report.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "flagged": flagged}, f, ensure_ascii=False, indent=2)
    print(f"报告已写入: {output}")
    print("建议: 词面级重叠高的样本建议剔除或改写；语义级重叠高的样本建议用私有/未见数据替换。")


if __name__ == "__main__":
    main()
