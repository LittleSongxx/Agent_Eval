"""检索噪声注入实验：验证检索侧指标对噪声的敏感性与区分度。

背景：合成数据集的 retrieved_contexts 直接取自源 chunk（= 完美检索），
检索侧指标天然满分、无方差（天花板效应）。本实验向检索结果头部注入
无关 chunk（模拟"无关文档排前面"的检索退化），验证：
1. 确定性指标 HitRate@K / MRR 随噪声数量单调下降（区分度证据）；
2. 自研"规则版"上下文相关性（查询覆盖度阈值 0.4，与 RAGAS NonLLM
   规则版同思路）随噪声数量下降，且与确定性指标方向一致。

用法：
    cd backend
    python -m scripts.retrieval_noise_experiment \
        --dataset "售后政策对照集（含坏样本）" \
        --noise-corpus /tmp/unrelated.md \
        --noise-counts 0,2,5,10
"""

from __future__ import annotations

import argparse
import json
import random
import re
import typing as t
from pathlib import Path

from scripts.bm25_mock_retriever import bigram_tokens

_K = 5


def _load_rows(dataset_name: str) -> list[dict[str, t.Any]]:
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
            .all()
        )
        if not rows:
            raise SystemExit("数据集为空")
        return [dict(row.data or {}) for row in rows]
    finally:
        db.close()


def _load_noise_chunks(noise_source: str) -> list[str]:
    """从文件/目录加载噪声 chunk（与知识库主题不同的内容）。"""
    source = Path(noise_source)
    chunks: list[str] = []
    files = [source] if source.is_file() else sorted(source.glob("*.md")) + sorted(source.glob("*.txt"))
    for file_path in files:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        for block in re.split(r"\n{2,}", content):
            block = block.strip()
            if len(block) >= 20:
                chunks.append(block)
    if not chunks:
        raise SystemExit(f"噪声语料为空: {noise_source}")
    return chunks


def _sentence_overlap(text: str, chunk: str) -> float:
    """文本在 chunk 中的查询覆盖度：文本词项被 chunk 任一句子覆盖的比例。

    短查询 vs 长 chunk 的 Jaccard 会被稀释，覆盖度更符合"检索相关"直觉。
    """
    text_terms = set(bigram_tokens(text))
    if not text_terms:
        return 0.0
    sentences = [s.strip() for s in re.split(r"[。；\n]", chunk) if s.strip()]
    if not sentences:
        sentences = [chunk]
    return max(
        len(text_terms & set(bigram_tokens(sentence))) / len(text_terms)
        for sentence in sentences
    )


RULE_RELEVANCE_THRESHOLD = 0.4


def _rule_context_precision(user_input: str, contexts: list[str]) -> float:
    """规则版上下文相关性（NonLLM 风格）：查询覆盖度 ≥ 阈值的 chunk 占比。"""
    if not contexts:
        return 0.0
    relevant = sum(
        1 for ctx in contexts if _sentence_overlap(user_input, ctx) >= RULE_RELEVANCE_THRESHOLD
    )
    return relevant / len(contexts)


def _rule_context_recall(reference: str, contexts: list[str]) -> float:
    """规则版上下文覆盖度：参考要点被任一 chunk 覆盖即算命中。"""
    if not contexts or not reference:
        return 0.0
    ref_sentences = [s.strip() for s in re.split(r"[。；\n]", reference) if s.strip()]
    if not ref_sentences:
        ref_sentences = [reference]
    hit = sum(
        1
        for sentence in ref_sentences
        if any(_sentence_overlap(sentence, ctx) >= RULE_RELEVANCE_THRESHOLD for ctx in contexts)
    )
    return hit / len(ref_sentences)


def _hit_rate_at_k(retrieved_ids: list[str], reference_ids: list[str], k: int = _K) -> float:
    if not reference_ids:
        return None
    return 1.0 if set(retrieved_ids[:k]) & set(reference_ids) else 0.0


def _mrr(retrieved_ids: list[str], reference_ids: list[str]) -> float:
    if not reference_ids:
        return None
    expected = set(reference_ids)
    for idx, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected:
            return 1.0 / idx
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="检索噪声注入实验")
    parser.add_argument("--dataset", required=True, help="数据集名称")
    parser.add_argument("--noise-corpus", required=True, help="无关噪声 chunk 来源（文件或目录）")
    parser.add_argument("--noise-counts", default="0,2,5,10", help="注入的噪声 chunk 数量（逗号分隔）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    noise_counts = [int(v) for v in args.noise_counts.split(",")]

    rows = _load_rows(args.dataset)
    noise_chunks = _load_noise_chunks(args.noise_corpus)
    rng = random.Random(args.seed)
    print(f"样本 {len(rows)} 条，噪声 chunk {len(noise_chunks)} 个，注入数量 {noise_counts}")

    # 基线：每行正确检索上下文 = 原始 retrieved_contexts，正确 ID 优先取
    # reference_context_ids[0]（生成数据集格式），缺失时按内容匹配
    base_rows = []
    for row in rows:
        base_contexts = list(row.get("retrieved_contexts") or [])
        if not base_contexts:
            continue
        base_reference_ids = list(row.get("reference_context_ids") or [])
        if not base_reference_ids:
            print("  ⚠ 跳过无 reference_context_ids 的行（无法计算 HitRate@K/MRR）")
            continue
        base_rows.append((row, base_contexts, base_reference_ids))
    if not base_rows:
        raise SystemExit("没有可参与检索指标计算的样本（需要 retrieved_contexts + reference_context_ids）")

    print("\n" + "=" * 70)
    header = f"{'噪声数':<8}{'HitRate@5':<12}{'MRR':<10}{'规则版Precision':<16}{'规则版Recall':<12}"
    print(header)
    print("-" * 70)
    report: dict[str, t.Any] = {"dataset": args.dataset, "rows": len(base_rows), "results": {}}

    for noise_count in noise_counts:
        hit_values: list[float] = []
        mrr_values: list[float] = []
        precision_values: list[float] = []
        recall_values: list[float] = []
        for row, base_contexts, base_reference_ids in base_rows:
            noise_count_clamped = max(0, min(noise_count, len(noise_chunks)))
            noise_ids = [f"noise-{rng.randint(0, 10 ** 6)}" for _ in range(noise_count_clamped)]
            noise_texts = [
                noise_chunks[rng.randrange(len(noise_chunks))] for _ in range(noise_count_clamped)
            ]

            # 噪声插入到检索结果头部（模拟"无关文档排前面"的检索退化）
            retrieved_ids = noise_ids + list(base_reference_ids)
            retrieved_contexts = noise_texts + base_contexts

            hit = _hit_rate_at_k(retrieved_ids, base_reference_ids)
            mrr = _mrr(retrieved_ids, base_reference_ids)
            precision = _rule_context_precision(str(row.get("user_input") or ""), retrieved_contexts)
            recall = _rule_context_recall(str(row.get("reference") or ""), retrieved_contexts)
            if hit is not None:
                hit_values.append(hit)
            if mrr is not None:
                mrr_values.append(mrr)
            precision_values.append(precision)
            recall_values.append(recall)

        hit_mean = sum(hit_values) / len(hit_values) if hit_values else 0.0
        mrr_mean = sum(mrr_values) / len(mrr_values) if mrr_values else 0.0
        precision_mean = sum(precision_values) / len(precision_values)
        recall_mean = sum(recall_values) / len(recall_values)
        print(
            f"{noise_count:<8d}{hit_mean:<12.4f}{mrr_mean:<10.4f}"
            f"{precision_mean:<16.4f}{recall_mean:<12.4f}"
        )
        report["results"][str(noise_count)] = {
            "hit_rate_at_5": round(hit_mean, 4),
            "mrr": round(mrr_mean, 4),
            "rule_precision": round(precision_mean, 4),
            "rule_recall": round(recall_mean, 4),
        }

    output = "retrieval_noise_report.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("-" * 70)
    print(f"报告已写入: {output}")
    print("解读: 确定性检索指标与规则版指标应随噪声比例单调下降；")
    print("      若某个指标对噪声不敏感，说明它无法区分检索质量的好坏。")


if __name__ == "__main__":
    main()
