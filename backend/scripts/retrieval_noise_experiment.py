"""检索侧实验：用真实 BM25 检索产出 retrieved_ids，度量检索指标对干扰项的敏感性。

背景与本脚本的一次自我纠错
--------------------------
合成数据集的 `retrieved_contexts` 直接取自源 chunk（= 完美检索），检索侧指标
天然满分、无方差（天花板效应）。本脚本的**上一版**试图这样制造方差：

    retrieved_ids = noise_ids + list(base_reference_ids)   # 噪声恒定插在金标前面

这个构造是**同义反复**，不是测量：金标恒定排在第 noise_count+1 位，于是
MRR ≡ 1/(noise_count+1)、HitRate@5 ≡ (noise_count < 5)，与数据集内容、
与检索器质量、与 chunk 文本全都无关——换一份完全不相干的数据也是同一串数字。
把它当作"检索指标随噪声单调下降 ⇒ 有区分度"的证据是循环论证。

本版改为**真的检索**：把金标 chunk 和干扰 chunk 一起灌进 BM25 索引，让排序
由 BM25 打分函数决定。`retrieved_ids` 只能来自 `index.search_with_ids()`，
脚本任何位置都不得把 `reference_context_ids` 拼进检索结果——这一点由
`tests/test_retrieval_experiment.py::test_experiment_ranking_is_data_dependent`
把守（打乱金标文本后 HitRate 必须掉下来；同义反复的实现过不了这条）。

两类干扰项，难度不同：
1. **无关语料**（`_fixtures/distractor_corpus.md`）：手机/咖啡/编程/旅行等，
   与售后政策零词汇重叠 —— 检索器应当几乎不受影响（易）；
2. **同域难负例**（`_fixtures/hard_negatives.md`）：虚构"优选商城"的售后政策，
   与金标同主题、同术语、不同数字 —— 压力测试 BM25 的区分能力（难）。

知识库口径：索引 = 数据集 `reference_context_ids` 引用到的那些 chunk（本数据集
为 job 2 的 6 个 chunk）+ 干扰项。刻意不含 job 1 的 `doc-1-chunk-*`：那两个 chunk
是同一份政策文档的粗粒度切分，内容**包含** `doc-2-chunk-0` 的全文，一旦入索引
就会出现"检索到的 chunk 确实含答案、但 ID 不等于金标 ID"的假阴性，污染 ID 级
命中率的口径。需要复现该重复内容场景时用 `--include-chunk-keys` 显式加入。

用法：
    cd backend
    python -m scripts.retrieval_noise_experiment --dataset "售后政策对照集（含坏样本）"
"""

from __future__ import annotations

import argparse
import json
import re
import typing as t
from pathlib import Path

from scripts.bm25_mock_retriever import BM25Index, bigram_tokens

_K = 5
_FIXTURES = Path(__file__).resolve().parent / "_fixtures"


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


def _load_kb_chunks(chunk_keys: t.Iterable[str]) -> dict[str, str]:
    """按 chunk_key 从 rag_dataset_chunks 取知识库正文，返回 {chunk_key: content}。"""
    from app.core.database import SessionLocal
    from app.models.rag_dataset_job import RagDatasetChunk

    wanted = list(dict.fromkeys(chunk_keys))
    db = SessionLocal()
    try:
        found = (
            db.query(RagDatasetChunk)
            .filter(RagDatasetChunk.chunk_key.in_(wanted))
            .all()
        )
        mapping = {chunk.chunk_key: (chunk.content or "") for chunk in found}
    finally:
        db.close()

    missing = [key for key in wanted if not mapping.get(key)]
    if missing:
        raise SystemExit(f"知识库缺少 chunk（无法建索引）: {missing}")
    return {key: mapping[key] for key in wanted}


def _load_corpus_chunks(source: str | Path) -> list[str]:
    """从文件/目录加载干扰 chunk（按空行切分，最短 20 字符）。"""
    source_path = Path(source)
    chunks: list[str] = []
    if source_path.is_file():
        files = [source_path]
    else:
        files = sorted(source_path.glob("*.md")) + sorted(source_path.glob("*.txt"))
    for file_path in files:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        for block in re.split(r"\n{2,}", content):
            block = block.strip()
            if len(block) >= 20:
                chunks.append(block)
    if not chunks:
        raise SystemExit(f"干扰语料为空: {source}")
    return chunks


def _strip_fixture_preamble(chunks: list[str]) -> list[str]:
    """丢掉 fixture 顶部的说明段落，只留真正的干扰内容。

    说明段带 markdown 标题或"本文件/用法/选材原则"等字样，不该进索引。
    """
    dropped_markers = ("#", "---")
    prose_markers = ("本文件", "用法", "选材原则", "每段", "每个段落")
    kept: list[str] = []
    for chunk in chunks:
        stripped = chunk.strip()
        if stripped.startswith(dropped_markers):
            continue
        if any(marker in stripped for marker in prose_markers):
            continue
        kept.append(chunk)
    return kept


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

# 注意：规则版 precision（查询覆盖度阈值法）已从本实验移除。实测对问句-段落匹配
# 不可校准：纯知识库下按句覆盖度仅 0.13、整 chunk 覆盖度 0.21，且随同域难负例注入
# 反向微升（硬负例与金标共享词汇，覆盖度无法区分）——该启发式对短中文问句
# 无判别力，不构成有效信号。规则版 recall 保留：参考要点按句覆盖，作为
# "top-K 是否真的覆盖了答案"的内容级召回 sanity（实测各场景恒为 1.0）。
# 规则版对标路径保留给 RAGAS NonLLMContextPrecision（见路线图第四章）。


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


def _hit_rate_at_k(
    retrieved_ids: list[str], reference_ids: list[str], k: int = _K
) -> float | None:
    """Top-K 里命中任一金标即 1.0。无金标时返回 None（不参与均值）。"""
    if not reference_ids:
        return None
    return 1.0 if set(retrieved_ids[:k]) & set(reference_ids) else 0.0


def _mrr(retrieved_ids: list[str], reference_ids: list[str]) -> float | None:
    """首个金标命中的倒数排名。无金标时返回 None（不参与均值）。"""
    if not reference_ids:
        return None
    expected = set(reference_ids)
    for idx, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected:
            return 1.0 / idx
    return 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class RetrievalScenario(t.NamedTuple):
    label: str
    distractors: int
    hard_negatives: int


def run_scenario(
    rows: list[dict[str, t.Any]],
    kb_chunks: dict[str, str],
    distractors: list[str],
    hard_negatives: list[str],
    scenario: RetrievalScenario,
    k: int = _K,
) -> dict[str, t.Any]:
    """在 (金标 chunk + 干扰项) 索引上真实检索 25 条查询，返回聚合指标。

    干扰项按固定顺序取前 N 个（不随机抽样），保证同一 scenario 可复现。
    """
    corpus_ids = list(kb_chunks.keys())
    corpus_texts = [kb_chunks[key] for key in corpus_ids]
    for idx, text in enumerate(distractors[: scenario.distractors]):
        corpus_ids.append(f"distractor-{idx}")
        corpus_texts.append(text)
    for idx, text in enumerate(hard_negatives[: scenario.hard_negatives]):
        corpus_ids.append(f"hardneg-{idx}")
        corpus_texts.append(text)

    index = BM25Index(corpus_texts)
    gold_id_set = set(kb_chunks.keys())
    text_by_id = dict(zip(corpus_ids, corpus_texts))

    hits_at_1: list[float] = []
    hits_at_3: list[float] = []
    hits_at_k: list[float] = []
    mrr_values: list[float] = []
    recall_values: list[float] = []
    per_kind: dict[str, list[float]] = {}
    detail: list[dict[str, t.Any]] = []

    for row in rows:
        user_input = str(row.get("user_input") or "")
        reference_ids = [
            rid for rid in (row.get("reference_context_ids") or []) if rid in gold_id_set
        ]
        if not reference_ids:
            continue

        # 唯一的检索来源：BM25 打分排序。绝不把 reference_ids 拼进来。
        ranked = index.search_with_ids(user_input, corpus_ids, top_k=k)
        retrieved_ids = [doc_id for doc_id, _score in ranked]
        retrieved_texts = [text_by_id[doc_id] for doc_id in retrieved_ids]

        hit_1 = _hit_rate_at_k(retrieved_ids, reference_ids, k=1)
        hit_3 = _hit_rate_at_k(retrieved_ids, reference_ids, k=3)
        hit_k = _hit_rate_at_k(retrieved_ids, reference_ids, k=k)
        rr = _mrr(retrieved_ids, reference_ids)
        recall = _rule_context_recall(str(row.get("reference") or ""), retrieved_texts)

        for bucket, value in ((hits_at_1, hit_1), (hits_at_3, hit_3), (hits_at_k, hit_k), (mrr_values, rr)):
            if value is not None:
                bucket.append(value)
        recall_values.append(recall)

        kind = str((row.get("generation_meta") or {}).get("kind") or "unknown")
        if hit_k is not None:
            per_kind.setdefault(kind, []).append(hit_k)

        reference_id_set = set(reference_ids)
        detail.append(
            {
                "user_input": user_input,
                "kind": kind,
                "gold": reference_ids,
                "retrieved": retrieved_ids,
                "rank_of_gold": next(
                    (i + 1 for i, rid in enumerate(retrieved_ids) if rid in reference_id_set),
                    None,
                ),
            }
        )

    return {
        "label": scenario.label,
        "corpus_size": len(corpus_ids),
        "distractors": scenario.distractors,
        "hard_negatives": scenario.hard_negatives,
        "scored_rows": len(recall_values),
        "hit_rate_at_1": round(_mean(hits_at_1), 4),
        "hit_rate_at_3": round(_mean(hits_at_3), 4),
        f"hit_rate_at_{k}": round(_mean(hits_at_k), 4),
        "mrr": round(_mean(mrr_values), 4),
        "rule_recall": round(_mean(recall_values), 4),
        "hit_rate_by_kind": {
            kind: round(_mean(values), 4) for kind, values in sorted(per_kind.items())
        },
        "detail": detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 端到端检索实验")
    parser.add_argument("--dataset", required=True, help="数据集名称")
    parser.add_argument(
        "--distractor-corpus",
        default=str(_FIXTURES / "distractor_corpus.md"),
        help="无关干扰语料（易）",
    )
    parser.add_argument(
        "--hard-negative-corpus",
        default=str(_FIXTURES / "hard_negatives.md"),
        help="同域难负例语料（难）",
    )
    parser.add_argument("--k", type=int, default=_K, help="Top-K")
    parser.add_argument(
        "--include-chunk-keys",
        default="",
        help="额外加入索引的 chunk_key（逗号分隔），用于复现重复内容场景",
    )
    parser.add_argument("--output", default="retrieval_bm25_report.json")
    args = parser.parse_args()

    rows = _load_rows(args.dataset)
    gold_keys = [
        rid
        for row in rows
        for rid in (row.get("reference_context_ids") or [])
    ]
    extra_keys = [key.strip() for key in args.include_chunk_keys.split(",") if key.strip()]
    kb_chunks = _load_kb_chunks(gold_keys + extra_keys)

    distractors = _strip_fixture_preamble(_load_corpus_chunks(args.distractor_corpus))
    hard_negatives = _strip_fixture_preamble(_load_corpus_chunks(args.hard_negative_corpus))

    print(f"数据集 {args.dataset}: {len(rows)} 行")
    print(f"知识库 chunk: {len(kb_chunks)} 个 -> {list(kb_chunks)}")
    print(f"无关干扰 chunk: {len(distractors)} 个；同域难负例: {len(hard_negatives)} 个")

    scenarios = [
        RetrievalScenario("纯知识库（无干扰）", 0, 0),
        RetrievalScenario("无关干扰 ×5", 5, 0),
        RetrievalScenario("无关干扰 ×10", 10, 0),
        RetrievalScenario("无关干扰 ×20", 20, 0),
        RetrievalScenario("无关干扰 ×全部", len(distractors), 0),
        RetrievalScenario("同域难负例 ×3", 0, 3),
        RetrievalScenario("同域难负例 ×5", 0, 5),
        RetrievalScenario("同域难负例 ×全部", 0, len(hard_negatives)),
        RetrievalScenario("无关全部 + 难负例全部", len(distractors), len(hard_negatives)),
    ]

    results = [
        run_scenario(rows, kb_chunks, distractors, hard_negatives, scenario, k=args.k)
        for scenario in scenarios
    ]

    print("\n" + "=" * 92)
    print(
        f"{'场景':<24}{'库':<6}{'H@1':<8}{'H@3':<8}{f'H@{args.k}':<8}"
        f"{'MRR':<8}{'规则R':<8}"
    )
    print("-" * 76)
    for result in results:
        print(
            f"{result['label']:<24}{result['corpus_size']:<6}"
            f"{result['hit_rate_at_1']:<8.4f}{result['hit_rate_at_3']:<8.4f}"
            f"{result[f'hit_rate_at_{args.k}']:<8.4f}{result['mrr']:<8.4f}"
            f"{result['rule_recall']:<8.4f}"
        )
    print("-" * 76)

    report = {
        "dataset": args.dataset,
        "rows": len(rows),
        "k": args.k,
        "kb_chunk_keys": list(kb_chunks),
        "distractor_pool": len(distractors),
        "hard_negative_pool": len(hard_negatives),
        "retrieval": "BM25 (bigram, k1=1.5, b=0.75)",
        "note": (
            "retrieved_ids 全部来自 BM25 排序，未混入 reference_context_ids；"
            "上一版脚本用 noise_ids + reference_ids 构造检索结果，指标是闭式解而非测量，已废弃。"
            "规则版 precision（查询覆盖度阈值法）因对问句-段落匹配不可校准而移除，见脚本内注释。"
        ),
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已写入: {args.output}")
    print("解读: 无关干扰下指标应基本不动（检索器能区分主题）；")
    print("      同域难负例下若指标明显下降，说明 BM25 词面匹配在近义政策文本上力不从心——")
    print("      这是真实的检索质量信号，不是构造出来的。")


if __name__ == "__main__":
    main()
