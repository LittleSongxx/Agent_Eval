"""生成检索侧黄金集标注表（docs/检索侧黄金集标注表.md）。

输入：retrieval_bm25_report.json 选定场景的逐条检索 detail +
      retrieval_goldset_annotations.json（独立判定结果）。支持任意行数；
      有稳定 sample_id 时优先按 sample_id 对齐，否则兼容历史 1-based 行号。
输出：markdown 标注表——每条查询 + top-5 检索结果（含 chunk 文本摘要）+ 判定。

说明：判定列来自 scripts/retrieval_goldset_annotations.json——人工黄金集
（2026-08-06：Claude 独立判定初标 + 用户逐条人工严格复核，25/25 一致，
人工审核确认为最终真值）。若后续人工修正，覆盖该 JSON 的 annotations
字段后重新运行本脚本即可。
"""

from __future__ import annotations

import argparse
import json
import pathlib

BACKEND = pathlib.Path(__file__).resolve().parent.parent
REPORT = BACKEND / "retrieval_bm25_report.json"
ANNOTATIONS = pathlib.Path(__file__).resolve().parent / "retrieval_goldset_annotations.json"
OUT = BACKEND.parent / "docs" / "检索侧黄金集标注表.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=pathlib.Path, default=REPORT,
                        help="BM25 检索报告 JSON（默认 backend/retrieval_bm25_report.json）")
    parser.add_argument("--annotations", type=pathlib.Path, default=ANNOTATIONS,
                        help="人工黄金集标注 JSON（默认 scripts/retrieval_goldset_annotations.json）")
    parser.add_argument("--out", type=pathlib.Path, default=OUT,
                        help="输出标注表（默认 docs/检索侧黄金集标注表.md）")
    parser.add_argument("--scenario-index", type=int, default=-1,
                        help="从 report.results 选择场景的下标（默认 -1，即最后一个）")
    parser.add_argument("--overwrite", action="store_true",
                        help="允许覆盖已有标注表；默认保护已有人工记录")
    args = parser.parse_args()

    rep = json.loads(args.report.read_text(encoding="utf-8"))
    # 最重压力场景 = 无关全部 + 难负例全部（库 42）
    try:
        scenario = rep["results"][args.scenario_index]
    except (KeyError, IndexError) as exc:
        raise SystemExit(f"scenario-index 越界或报告缺少 results: {args.scenario_index}") from exc

    from app.core.database import SessionLocal
    from app.models.rag_dataset_job import RagDatasetChunk

    db = SessionLocal()
    try:
        keys = list(rep["kb_chunk_keys"])
        chunks = {
            c.chunk_key: c.content
            for c in db.query(RagDatasetChunk)
            .filter(RagDatasetChunk.chunk_key.in_(keys))
            .all()
        }
    finally:
        db.close()

    # 干扰语料文本（与 compare_ragas_nonllm 相同的 preamble 清洗），供标注者判断
    from scripts.compare_ragas_nonllm import _load_corpus_chunks, _strip_fixture_preamble

    _FIXTURES = pathlib.Path(__file__).resolve().parent / "_fixtures"
    for prefix, filename in (
        ("hardneg", "hard_negatives.md"),
        ("dist", "distractor_corpus.md"),
    ):
        for i, text in enumerate(
            _strip_fixture_preamble(_load_corpus_chunks(_FIXTURES / filename))
        ):
            chunks[f"{prefix}-{i}"] = text

    def short(text: str, n: int = 60) -> str:
        return (text or "").replace("\n", " ")[:n]

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))["annotations"]

    def annotation_for(row: dict, row_number: int) -> dict:
        sample_id = row.get("sample_id")
        value = annotations.get(sample_id) if sample_id else None
        if value is None:
            value = annotations.get(str(row_number), {})
        return value if isinstance(value, dict) else {}

    lines = [
        "# 检索侧黄金集标注表",
        "",
        "> 场景：最重压力（无关 26 + 同域难负例 10，共 42 chunk 库），25 条互异查询，"
        "BM25 top-5 结果",
        "> 判定来源：`scripts/retrieval_goldset_annotations.json`——**人工黄金集**"
        "（2026-08-06：Claude 独立判定初标 + 用户逐条人工严格复核，25/25 一致，"
        "人工审核确认为最终真值）。",
        "> 判定标准：金标是否包含查询答案的唯一合理来源 → Y；多个合理来源 → 填多个 "
        "ID（MULTI）；金标不是答案来源 → N。同主题但商城/数值不同的难负例不视为"
        "答案来源（会误导回答）。",
        "> 若后续人工复核，覆盖该 JSON 的 `annotations` 字段后重跑本脚本即可。",
        "",
        "| # | sample_id | 查询 | 类别 | 金标 | 检索 top-5（ID + 摘要） | 判定 |",
        "|---|-----------|------|------|------|----------------------|------|",
    ]
    for i, row in enumerate(scenario["detail"], 1):
        retrieved = "; ".join(
            f"{rid}:{short(chunks.get(rid, '?'))}" for rid in row["retrieved"]
        )
        a = annotation_for(row, i)
        verdict = a.get("verdict", "")
        if verdict == "Y":
            verdict_display = "✅ Y"
        elif verdict == "MULTI":
            verdict_display = "🔶 MULTI " + "+".join(a.get("expected_hits", []))
        else:
            verdict_display = f"❌ {verdict}"
        lines.append(
            f"| {i} | {row.get('sample_id', '')} | {row['user_input']} | {row['kind']} | {row['gold'][0]} "
            f"| {retrieved} | {verdict_display} |"
        )

    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"标注表已存在: {args.out}；请指定 --overwrite 以明确覆盖")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"标注表已写入: {args.out}（{len(scenario['detail'])} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
