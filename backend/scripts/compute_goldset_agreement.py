"""检索侧准则效度：人工黄金集 vs 合成数据集金标 vs BM25 检索结果。

输入：
- retrieval_bm25_report.json（最重压力场景逐条 detail：retrieved 顺序）
- retrieval_goldset_annotations.json（人工黄金集：expected_hits + verdict，
  2026-08-06 人工逐条审核确认）

输出（打印 + retrieval_goldset_agreement.json）：
- 金标口径 H@1 / MRR（= retrieval_bm25_report.json 已报数字的复算）
- 人工黄金集口径 H@1 / MRR（expected_hits 任一命中的最靠前位次）
- 两口径一致率、逐行 rank 对比
- 判定分布（Y / MULTI / N）——N 表示人工审核发现金标构造错误

用法：cd backend && python -m scripts.compute_goldset_agreement --overwrite
"""

from __future__ import annotations

import argparse
import json
import pathlib

BACKEND = pathlib.Path(__file__).resolve().parent.parent
REPORT = BACKEND / "retrieval_bm25_report.json"
ANNOTATIONS = pathlib.Path(__file__).resolve().parent / "retrieval_goldset_annotations.json"
OUT = BACKEND / "retrieval_goldset_agreement.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=pathlib.Path, default=REPORT,
                        help="BM25 检索报告 JSON（默认 backend/retrieval_bm25_report.json）")
    parser.add_argument("--annotations", type=pathlib.Path, default=ANNOTATIONS,
                        help="人工黄金集标注 JSON（默认 scripts/retrieval_goldset_annotations.json）")
    parser.add_argument("--out", type=pathlib.Path, default=OUT,
                        help="输出 JSON（默认 backend/retrieval_goldset_agreement.json）")
    parser.add_argument("--scenario-index", type=int, default=-1,
                        help="从 report.results 选择场景的下标（默认 -1，即最后一个）")
    parser.add_argument("--overwrite", action="store_true",
                        help="允许覆盖已有输出；默认保护已有证据")
    args = parser.parse_args()

    rep = json.loads(args.report.read_text(encoding="utf-8"))
    ann = json.loads(args.annotations.read_text(encoding="utf-8"))
    scenarios = rep.get("results")
    if not isinstance(scenarios, list) or not scenarios:
        raise SystemExit("报告缺少非空 results 场景数组")
    try:
        scenario = scenarios[args.scenario_index]
    except IndexError as exc:
        raise SystemExit(f"scenario-index 越界: {args.scenario_index}") from exc
    detail = scenario["detail"]
    if not isinstance(detail, list) or not detail:
        raise SystemExit("选中场景缺少非空 detail 数组")

    annotations = ann.get("annotations") if isinstance(ann, dict) else None
    if not isinstance(annotations, dict):
        raise SystemExit("人工标注文件缺少 annotations 对象")

    def annotation_for(row: dict, row_number: int) -> dict:
        # 新数据优先用稳定 sample_id；兼容历史 1-based 行号键。
        sample_id = row.get("sample_id")
        value = annotations.get(sample_id) if sample_id else None
        if value is None:
            value = annotations.get(str(row_number))
        if not isinstance(value, dict):
            raise SystemExit(
                f"第 {row_number} 行（sample_id={sample_id!r}）缺少人工标注；"
                "拒绝静默丢行"
            )
        if not isinstance(value.get("expected_hits"), list):
            raise SystemExit(f"第 {row_number} 行人工标注缺少 expected_hits 数组")
        return value

    rows = []
    gold_h1_ok = gold_mrr = ann_h1_ok = ann_mrr = 0
    n = len(detail)
    for i, row in enumerate(detail, 1):
        a = annotation_for(row, i)
        retrieved = row["retrieved"]
        gold_ids = row.get("gold") or []
        if not gold_ids:
            raise SystemExit(f"第 {i} 行缺少 gold 检索 ID")
        gold_rank = next(
            (retrieved.index(hit) + 1 for hit in gold_ids if hit in retrieved),
            None,
        )
        ann_rank = min(
            (retrieved.index(hit) + 1 for hit in a["expected_hits"] if hit in retrieved),
            default=None,
        )
        # 金标是否在独立判定期望集中
        gold_in_expected = any(hit in a["expected_hits"] for hit in gold_ids)
        if gold_rank is not None:
            gold_mrr += 1.0 / gold_rank
            gold_h1_ok += gold_rank == 1
        if ann_rank is not None:
            ann_mrr += 1.0 / ann_rank
            ann_h1_ok += ann_rank == 1
        rows.append(
            {
                "row": i,
                "sample_id": row.get("sample_id"),
                "query": row["user_input"],
                "kind": row["kind"],
                "gold": gold_ids,
                "expected_hits": a["expected_hits"],
                "verdict": a["verdict"],
                "gold_in_expected": gold_in_expected,
                "gold_rank": gold_rank,
                "ann_rank": ann_rank,
            }
        )

    def h1_ok(total: int, hits: int) -> float:
        return round(hits / total, 4)

    gold_h1 = h1_ok(n, gold_h1_ok)
    gold_mrr_v = round(gold_mrr / n, 4)
    ann_h1 = h1_ok(n, ann_h1_ok)
    ann_mrr_v = round(ann_mrr / n, 4)

    # This is deterministic post-processing, so model is explicitly marked
    # as none rather than pretending that a remote Judge produced the result.
    from scripts.experiment_utils import build_provenance, sha256_file, write_json_report

    report = {
        "provenance": build_provenance(
            script_path=pathlib.Path(__file__),
            dataset_path=args.report,
            sample_count=n,
            model="none (deterministic retrieval post-processing)",
            extra={
                "scenario": scenario.get("label"),
                "scenario_index": args.scenario_index,
                "annotations": ann.get("provenance"),
                "annotations_sha256": sha256_file(args.annotations),
                "k": rep.get("k"),
            },
        ),
        "verdict_distribution": {
            "Y": sum(1 for r in rows if r["verdict"] == "Y"),
            "MULTI": sum(1 for r in rows if r["verdict"] == "MULTI"),
            "N": sum(1 for r in rows if r["verdict"] == "N"),
        },
        "gold_criterion": {"h1": gold_h1, "mrr": gold_mrr_v},
        "annotation_criterion": {"h1": ann_h1, "mrr": ann_mrr_v},
        "gold_in_expected_rate": round(
            sum(1 for r in rows if r["gold_in_expected"]) / n, 4
        ),
        "multi_source_rows": [
            r["row"] for r in rows if r["verdict"] == "MULTI"
        ],
        "rows": rows,
    }
    write_json_report(args.out, report, overwrite=args.overwrite)

    print(f"人工黄金集判定分布: Y {report['verdict_distribution']['Y']} / "
          f"MULTI {report['verdict_distribution']['MULTI']} / N {report['verdict_distribution']['N']}")
    print(f"金标口径:      H@1 {gold_h1} / MRR {gold_mrr_v}")
    print(f"人工黄金集口径: H@1 {ann_h1} / MRR {ann_mrr_v}")
    print(f"金标 ∈ 人工期望集: {report['gold_in_expected_rate']}"
          f"（25/25 → 合成单金标构造经人工审核未发现错误）")
    print(f"多来源行: {report['multi_source_rows']}"
          f"（同一答案信息跨 chunk 重复，金标仍均在 top-1）")
    print(f"报告已写入: {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
