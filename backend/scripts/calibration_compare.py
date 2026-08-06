"""校准前后对比：人工标注 vs 自动判定（kappa / 一致率 / 分歧行 / 阈值扫描）。

背景：准则效度首轮实测（任务 10，阈值 0.5）人工-自动 kappa = 0.4582，分歧 7 处
其中 6 处为"自动过松"（2 幻觉 + 4 截断）。经业界调研定位三层根因后执行校准：
  1. 场景 4 阈值 0.5 → 0.7（与 seed 模板一致）；
  2. `builtin_answer_relevancy` criteria 修订（prompt_manager.py）：
     截断可观察定义 + 防格式误杀 + 禁止常识外推。
重跑任务（任务 11，同 25 行 × 5 采样，口径与任务 10 一致）后，本脚本输出
校准前后与人工标注的一致性对比，并为每个数字标注来源任务与阈值，避免人工转抄。

用法：
    python -m scripts.calibration_compare \
        --before 10 --after 11 --db backend/eval_platform.db

输出：
    backend/calibration_compare_report.json（带 provenance）
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sqlite3

BACKEND = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DB = BACKEND / "eval_platform.db"
OUT = BACKEND / "calibration_compare_report.json"

METRICS = [
    "faithfulness",
    "faithfulness_claim",
    "answer_relevancy",
    "answer_relevancy_generative",
    "context_recall",
    "context_precision",
]


def _load_rows(db_path: pathlib.Path, task_id: int) -> dict[int, dict]:
    """按 row_index 取任务的 metric_scores + is_pass + 样本 generation_meta.kind。"""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT row_index, is_pass, metric_scores, manual_status FROM eval_row_results "
        "WHERE eval_task_id = ? ORDER BY row_index",
        (task_id,),
    ).fetchall()
    conn.close()
    out = {}
    for row_index, is_pass, metric_scores, manual_status in rows:
        out[row_index] = {
            "is_pass": bool(is_pass),
            "manual_status": manual_status,
            "metric_scores": json.loads(metric_scores or "{}"),
        }
    return out


def _load_kinds(db_path: pathlib.Path, dataset_id: int) -> dict[int, str]:
    """取数据集每行的 generation_meta.kind（correct/hallucinated/truncated）。"""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT row_index, data FROM dataset_rows WHERE dataset_id = ?", (dataset_id,)
    ).fetchall()
    conn.close()
    out = {}
    for row_index, data in rows:
        try:
            parsed = json.loads(data) if isinstance(data, str) else data
            kind = (parsed or {}).get("generation_meta", {}).get("kind", "?")
        except (json.JSONDecodeError, AttributeError):
            kind = "?"
        out[row_index] = kind
    return out


def _get_task_meta(db_path: pathlib.Path, task_id: int) -> dict:
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT id, name, dataset_id, scenario_id, created_at, status FROM eval_tasks "
        "WHERE id = ?",
        (task_id,),
    ).fetchone()
    conn.close()
    if row is None:
        raise SystemExit(f"任务 {task_id} 不存在")
    return {
        "id": row[0],
        "name": row[1],
        "dataset_id": row[2],
        "scenario_id": row[3],
        "created_at": row[4],
        "status": row[5],
    }


def _get_thresholds(db_path: pathlib.Path, scenario_id: int) -> dict[str, float]:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT m.name, sm.pass_threshold FROM scenario_metrics sm "
        "JOIN metric_definitions m ON m.id = sm.metric_definition_id "
        "WHERE sm.scenario_id = ?",
        (scenario_id,),
    ).fetchall()
    conn.close()
    return {name: thr for name, thr in rows if thr is not None}


def kappa_matrix(manual: dict[int, str], auto: dict[int, bool]) -> tuple:
    """人工 pass/fail vs 自动 pass/fail 的 2×2 Cohen's kappa（口径与 report.py 一致）。"""
    both_pass = sum(1 for i in manual if manual[i] == "pass" and auto[i])
    manual_pass_auto_fail = sum(1 for i in manual if manual[i] == "pass" and not auto[i])
    manual_fail_auto_pass = sum(1 for i in manual if manual[i] == "fail" and auto[i])
    both_fail = sum(1 for i in manual if manual[i] == "fail" and not auto[i])
    total = len(manual)
    observed = (both_pass + both_fail) / total
    expected = (
        (both_pass + manual_pass_auto_fail) * (both_pass + manual_fail_auto_pass)
        + (manual_fail_auto_pass + both_fail) * (manual_pass_auto_fail + both_fail)
    ) / (total * total)
    kappa = None if expected >= 1.0 else (observed - expected) / (1.0 - expected)
    return {
        "total": total,
        "both_pass": both_pass,
        "manual_pass_auto_fail": manual_pass_auto_fail,
        "manual_fail_auto_pass": manual_fail_auto_pass,
        "both_fail": both_fail,
        "agreement_rate": round(observed, 4),
        "kappa": round(kappa, 4) if kappa is not None else None,
    }


def threshold_sweep(rows: dict[int, dict], manual: dict[int, str]) -> list[dict]:
    """用给定任务的分数做阈值扫描（0.5~0.9 统一阈值），零成本模拟 kappa。"""
    results = []
    for t in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        auto = {}
        for i, r in rows.items():
            scores = r["metric_scores"]
            ok = all(
                (scores.get(m, {}).get("score") or 0) >= t for m in METRICS
            )
            auto[i] = ok
        matrix = kappa_matrix(manual, auto)
        results.append({"threshold": t, **matrix})
    return results


def _score_table(row: dict) -> dict:
    """每指标 score 的紧凑摘要，供分歧行明细。"""
    return {
        m: row["metric_scores"].get(m, {}).get("score")
        for m in METRICS
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=int, default=10, help="校准前任务 ID")
    parser.add_argument("--after", type=int, default=11, help="校准后任务 ID")
    parser.add_argument("--db", type=pathlib.Path, default=DEFAULT_DB)
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"DB 不存在: {args.db}")

    before_rows = _load_rows(args.db, args.before)
    after_rows = _load_rows(args.db, args.after)
    before_meta = _get_task_meta(args.db, args.before)
    after_meta = _get_task_meta(args.db, args.after)
    if before_meta["dataset_id"] != after_meta["dataset_id"]:
        raise SystemExit("校准前后任务的数据集不一致，无法对比")
    kinds = _load_kinds(args.db, before_meta["dataset_id"])
    thresholds = _get_thresholds(args.db, after_meta["scenario_id"])

    # 人工标注：以校准前任务的行级 manual_status 为准（25 条均已有明确 pass/fail）
    manual = {
        i: r["manual_status"]
        for i, r in before_rows.items()
        if r["manual_status"] in ("pass", "fail")
    }
    if len(manual) < 10:
        raise SystemExit(f"人工标注不足（{len(manual)} 条），kappa 无意义")

    before_auto = {i: r["is_pass"] for i, r in before_rows.items()}
    after_auto = {i: r["is_pass"] for i, r in after_rows.items()}

    # 容错：取行交集，任务未跑完时降级为"已完成行"对比（完成后自动是完整 25 行）
    common = sorted(set(manual) & set(before_auto) & set(after_auto))
    if len(common) != len(manual):
        print(f"注意：任务 {args.after} 尚未完成，仅对比已完成行 {len(common)}/{len(manual)}")
    manual = {i: manual[i] for i in common}

    before_k = kappa_matrix(manual, before_auto)
    after_k = kappa_matrix(manual, after_auto)

    # 分层一致率（按 generation kind）
    def stratified(rows_auto: dict[int, bool]) -> dict:
        by_kind: dict[str, dict] = {}
        for i in sorted(manual):
            kind = kinds.get(i, "?")
            bucket = by_kind.setdefault(kind, {"agree": 0, "disagree": 0, "rows": []})
            if (manual[i] == "pass") == (rows_auto[i] is True):
                bucket["agree"] += 1
            else:
                bucket["disagree"] += 1
                bucket["rows"].append(i)
        return {
            kind: {
                "agree": v["agree"],
                "disagree": v["disagree"],
                "rate": round(v["agree"] / (v["agree"] + v["disagree"]), 4),
                "disagree_rows": v["rows"],
            }
            for kind, v in by_kind.items()
        }

    # 翻转明细：校准后自动判定相比校准前的变化 + 与人工的一致性
    flips = []
    for i in sorted(manual):
        before_pass, after_pass = before_auto[i], after_auto[i]
        if before_pass != after_pass:
            flips.append(
                {
                    "row": i,
                    "kind": kinds.get(i, "?"),
                    "before": "pass" if before_pass else "fail",
                    "after": "pass" if after_pass else "fail",
                    "manual": manual[i],
                    "after_consistent_with_human": (manual[i] == "pass") == after_pass,
                    "scores": _score_table(after_rows[i]),
                }
            )

    # 校准后仍与人工不一致的行（两种方向）
    after_disagreements = [
        {
            "row": i,
            "kind": kinds.get(i, "?"),
            "manual": manual[i],
            "auto": "pass" if after_auto[i] else "fail",
            "scores": _score_table(after_rows[i]),
        }
        for i in sorted(manual)
        if (manual[i] == "pass") != (after_auto[i] is True)
    ]

    report = {
        "provenance": {
            "script": "scripts/calibration_compare.py",
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "db": str(args.db),
            "before_task": before_meta,
            "after_task": after_meta,
            "thresholds_after": thresholds,
            "annotation_source": f"task {args.before} manual_status（25 条 pass/fail）",
        },
        "before_calibration": before_k,
        "after_calibration": after_k,
        "stratified_agreement_before": stratified(before_auto),
        "stratified_agreement_after": stratified(after_auto),
        "verdict_flips": flips,
        "after_disagreements": after_disagreements,
        "threshold_sweep_after_scores": threshold_sweep(after_rows, manual),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"=== 校准对比（人工 {len(manual)} 条，kind 分层: {dict(sorted(kinds.items()))}）===")
    print(f"阈值（校准后）: {thresholds}")
    print(f"校准前(任务{before_meta['id']}): 一致率 {before_k['agreement_rate']}  kappa {before_k['kappa']}"
          f"  (人工过松 {before_k['manual_pass_auto_fail']} / 自动过松 {before_k['manual_fail_auto_pass']})")
    print(f"校准后(任务{after_meta['id']}): 一致率 {after_k['agreement_rate']}  kappa {after_k['kappa']}"
          f"  (人工过松 {after_k['manual_pass_auto_fail']} / 自动过松 {after_k['manual_fail_auto_pass']})")
    print("分层一致率 校准前→后:")
    sb, sa = stratified(before_auto), stratified(after_auto)
    for kind in sb:
        print(f"  {kind}: {sb[kind]['rate']} → {sa.get(kind, {}).get('rate')}"
              f"  (分歧行: {sb[kind]['disagree_rows']} → {sa.get(kind, {}).get('disagree_rows')})")
    print(f"判定翻转 {len(flips)} 行:")
    for f in flips:
        consistent = "✓ 与人工一致" if f["after_consistent_with_human"] else "✗ 与人工相悖"
        print(f"  row {f['row']} ({f['kind']}): {f['before']} → {f['after']}  [人工={f['manual']}] {consistent}")
    print(f"校准后仍分歧 {len(after_disagreements)} 行:")
    for d in after_disagreements:
        print(f"  row {d['row']} ({d['kind']}): 人工={d['manual']} 自动={d['auto']} {d['scores']}")
    print(f"报告已写入: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
