"""检索侧结构效度：平台确定性指标 vs RAGAS NonLLM 规则版（确定性 ↔ 确定性）。

背景：检索侧平台已有 ID 级确定性指标（HitRate@K / MRR，BM25 实测，见
retrieval_bm25_report.json）。结构效度还差一块：与业界规则版指标对照。
自研"查询覆盖度阈值法"已被证明不可校准（retrieval_noise_experiment.py 注释），
不重复造轮子——直接用 RAGAS 的 NonLLMContextRecall / NonLLMContextPrecision
（纯字符级 Levenshtein 相似度 + 阈值，零 LLM、零 embedding，完全确定性）。

口径：同一批 BM25 检索输出（含四场景：纯知识库 / 无关干扰 / 同域难负例 /
全压力），把检索到的 chunk 文本喂给 RAGAS NonLLM 指标，金标 chunk 文本作
reference_contexts。RAGAS NonLLM 分数与平台 ID 级命中率在相同检索结果上
对齐——两者任何差异都来自"文本相似度判定" vs "ID 精确命中"的机制差异。

注意：本脚本需要 RAGAS 环境（conda env langchain，python 路径见下方
RAGAS_PYTHON），平台依赖用当前环境。RAGAS 0.4.x 导入需要 langchain-community
的 vertexai shim（scripts/_shims/），见 compare_with_ragas.py。

用法（分两步，避免环境混用）：
    # 1. 用平台环境产出检索结果
    python -m scripts.compare_ragas_nonllm --phase dump --dataset "售后政策对照集（含坏样本）"
    # 2. 用 langchain env 跑 RAGAS NonLLM
    /home/song/anaconda3/envs/langchain/bin/python -m scripts.compare_ragas_nonllm --phase score
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
import typing as t

BACKEND = pathlib.Path(__file__).resolve().parent.parent
DUMP_FILE = BACKEND / "ragas_nonllm_dump.json"
OUT_FILE = BACKEND / "ragas_nonllm_report.json"

_FIXTURES = pathlib.Path(__file__).resolve().parent / "_fixtures"


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
        return [dict(row.data or {}) for row in rows]
    finally:
        db.close()


def _load_kb_chunks(chunk_keys: t.Iterable[str]) -> dict[str, str]:
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


def _load_corpus_chunks(source: str | pathlib.Path) -> list[str]:
    source_path = pathlib.Path(source)
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
    return chunks


def _strip_fixture_preamble(chunks: list[str]) -> list[str]:
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


def _scenarios(rows: list[dict], kb_chunks: dict[str, str]) -> list[dict]:
    """构造四场景：与 retrieval_noise_experiment 的 K=5、干扰口径一致。"""
    from scripts.bm25_mock_retriever import BM25Index, bigram_tokens

    distractors = _strip_fixture_preamble(
        _load_corpus_chunks(_FIXTURES / "distractor_corpus.md")
    )
    hard_negatives = _strip_fixture_preamble(
        _load_corpus_chunks(_FIXTURES / "hard_negatives.md")
    )
    base_keys = list(kb_chunks.keys())

    scenarios = [
        ("纯知识库（无干扰）", base_keys, [], []),
        ("无关干扰 ×全部", base_keys + [f"dist-{i}" for i in range(len(distractors))],
         distractors, []),
        ("同域难负例 ×全部", base_keys + [f"hard-{i}" for i in range(len(hard_negatives))],
         [], hard_negatives),
        ("无关全部 + 难负例全部",
         base_keys + [f"dist-{i}" for i in range(len(distractors))]
         + [f"hard-{i}" for i in range(len(hard_negatives))],
         distractors, hard_negatives),
    ]

    out = []
    for label, keys, dist_chunks, hard_chunks in scenarios:
        kb = dict(kb_chunks)
        for i, c in enumerate(dist_chunks):
            kb[f"dist-{i}"] = c
        for i, c in enumerate(hard_chunks):
            kb[f"hard-{i}"] = c
        corpus_ids = list(kb.keys())
        index = BM25Index([kb[key] for key in corpus_ids])
        scored = []
        for row in rows:
            query = row.get("user_input", "")
            if not query:
                continue
            reference_ids = [rid for rid in (row.get("reference_context_ids") or []) if rid in kb]
            if not reference_ids:
                continue
            results = index.search_with_ids(query, corpus_ids, top_k=5)
            retrieved_ids = [key for key, _score in results]
            scored.append(
                {
                    "row_index": row.get("row_index", -1),
                    "kind": (row.get("generation_meta") or {}).get("kind", "?"),
                    "user_input": query,
                    "reference_ids": reference_ids,
                    "reference_contexts": [kb.get(kid, "") for kid in reference_ids],
                    "retrieved_ids": retrieved_ids,
                    "retrieved_contexts": [kb.get(kid, "") for kid in retrieved_ids],
                }
            )
        out.append({"label": label, "corpus_size": len(corpus_ids), "scored": scored})
    return out


def phase_dump(dataset_name: str) -> int:
    rows = _load_rows(dataset_name)
    reference_ids = set()
    for row in rows:
        reference_ids.update(row.get("reference_context_ids") or [])
    kb_chunks = _load_kb_chunks(sorted(reference_ids))
    scenarios = _scenarios(rows, kb_chunks)
    DUMP_FILE.write_text(
        json.dumps(
            {"dataset": dataset_name, "rows": len(rows), "scenarios": scenarios},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    total = sum(len(s["scored"]) for s in scenarios)
    print(f"dump 完成: {len(scenarios)} 场景 × {total} 行 → {DUMP_FILE.name}")
    return 0


def phase_score() -> int:
    """用 RAGAS NonLLM 指标给 dump 的 retrieved_contexts 打分。"""
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics._context_precision import NonLLMContextPrecisionWithReference
    from ragas.metrics._context_recall import NonLLMContextRecall

    dump = json.loads(DUMP_FILE.read_text(encoding="utf-8"))
    recall = NonLLMContextRecall()
    precision = NonLLMContextPrecisionWithReference()

    async def score_all() -> None:
        for scenario in dump["scenarios"]:
            for item in scenario["scored"]:
                if not item["retrieved_contexts"] or not item["reference_contexts"]:
                    item["nonllm_recall"] = None
                    item["nonllm_precision"] = None
                    continue
                sample = SingleTurnSample(
                    retrieved_contexts=item["retrieved_contexts"],
                    reference_contexts=item["reference_contexts"],
                )
                item["nonllm_recall"] = await recall._single_turn_ascore(sample, None)
                item["nonllm_precision"] = await precision._single_turn_ascore(sample, None)

    asyncio.run(score_all())

    # 汇总每场景：平台 ID 级指标（H@1 / MRR）vs RAGAS NonLLM 分数
    import statistics

    for scenario in dump["scenarios"]:
        scored = scenario["scored"]
        ranks = []
        for item in scored:
            rid = item["reference_ids"][0]
            rank = item["retrieved_ids"].index(rid) + 1 if rid in item["retrieved_ids"] else None
            item["rank_of_gold"] = rank
            ranks.append(rank)
        ranks = [r for r in ranks if r is not None]
        h1 = sum(1 for r in ranks if r == 1) / len(ranks)
        mrr = sum(1.0 / r for r in ranks) / len(ranks)

        recalls = [x["nonllm_recall"] for x in scored if x["nonllm_recall"] is not None]
        precs = [x["nonllm_precision"] for x in scored if x["nonllm_precision"] is not None]

        def spearman(xs, ys):
            """Spearman 秩相关（平均秩处理 tie）。"""

            def avg_rank(v):
                n = len(v)
                idx = sorted(range(n), key=lambda i: (v[i], i))
                rk = [0.0] * n
                i = 0
                while i < n:
                    j = i
                    while j + 1 < n and v[idx[j + 1]] == v[idx[i]]:
                        j += 1
                    avg = (i + j) / 2 + 1
                    for k in range(i, j + 1):
                        rk[idx[k]] = avg
                    i = j + 1
                return rk

            def pearson(a, b):
                n = len(a)
                ma, mb = sum(a) / n, sum(b) / n
                cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
                va = sum((x - ma) ** 2 for x in a) ** 0.5
                vb = sum((y - mb) ** 2 for y in b) ** 0.5
                return cov / (va * vb) if va and vb else None

            return pearson(avg_rank(xs), avg_rank(ys))

        rank_sp = spearman(
            [x["rank_of_gold"] or 6 for x in scored],
            [x["nonllm_precision"] or 0 for x in scored],
        )
        scenario["stats"] = {
            "rows": len(scored),
            "platform_hit_rate_at_1": round(h1, 4) if ranks else None,
            "platform_mrr": round(mrr, 4) if ranks else None,
            "ragas_nonllm_recall_mean": round(statistics.mean(recalls), 4) if recalls else None,
            "ragas_nonllm_precision_mean": round(statistics.mean(precs), 4) if precs else None,
            "spearman_precision_vs_rank": round(rank_sp, 4) if rank_sp is not None else None,
        }

    OUT_FILE.write_text(
        json.dumps(
            {
                "provenance": {
                    "script": "scripts/compare_ragas_nonllm.py",
                    "ragas_version": __import__("ragas").__version__,
                    "distance": "Levenshtein normalized similarity, threshold=0.5",
                    "note": "确定性 ↔ 确定性：平台 ID 级命中 vs RAGAS NonLLM 文本级判定",
                },
                "dataset": dump["dataset"],
                "scenarios": dump["scenarios"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for scenario in dump["scenarios"]:
        s = scenario["stats"]
        print(
            f"{scenario['label']}: rows={s['rows']} | 平台 H@1 {s['platform_hit_rate_at_1']} / "
            f"MRR {s['platform_mrr']} | RAGAS NonLLM recall {s['ragas_nonllm_recall_mean']} / "
            f"precision {s['ragas_nonllm_precision_mean']} | spearman {s['spearman_precision_vs_rank']}"
        )
    print(f"报告已写入: {OUT_FILE.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["dump", "score"], required=True)
    parser.add_argument("--dataset", default="售后政策对照集（含坏样本）")
    args = parser.parse_args()
    if args.phase == "dump":
        return phase_dump(args.dataset)
    return phase_score()


if __name__ == "__main__":
    raise SystemExit(main())
