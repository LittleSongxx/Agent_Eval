"""双通道仲裁分析：整体 answer_relevancy vs 生成式 answer_relevancy_generative。

背景：整体通道（判定版）与生成式通道（反推问题 + 相似度）机制不同——
整体通道对截断/答非所问敏感，但会受裁判"外部知识污染"误伤（row 7 案例）；
生成式通道温和（截断也给高分），但不会被污染干扰。仲裁目标：分歧行
（两通道差大）上，用人工标签为真值量化哪种仲裁规则一致性最高。

输入：eval_row_results（task_id 参数默认 10，校准前）+ dataset_rows 的 kind +
      manual_status（人工 pass/fail 标注）。
输出：打印 + dual_channel_arbitration_report.json。

用法：cd backend && python -m scripts.dual_channel_arbitration --task 10
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3

BACKEND = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DB = BACKEND / "eval_platform.db"
OUT = BACKEND / "dual_channel_arbitration_report.json"

THRESHOLDS = [0.5, 0.7]  # 校准前政策 0.5；校准后场景 4 为 0.7
DIFF_CUT = 0.2  # 分歧行定义：两通道分数差


def _load(db: pathlib.Path, task_id: int) -> dict:
    conn = sqlite3.connect(str(db))
    rows = conn.execute(
        "SELECT row_index, metric_scores, manual_status FROM eval_row_results "
        "WHERE eval_task_id = ? ORDER BY row_index",
        (task_id,),
    ).fetchall()
    dataset_id = conn.execute(
        "SELECT dataset_id FROM eval_tasks WHERE id = ?", (task_id,)
    ).fetchone()[0]
    kinds = dict(
        conn.execute(
            "SELECT row_index, json_extract(data, '$.generation_meta.kind') "
            "FROM dataset_rows WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchall()
    )
    conn.close()
    out = {}
    for ri, ms, manual in rows:
        ms = json.loads(ms or "{}")
        out[ri] = {
            "kind": kinds.get(ri, "?"),
            "manual": manual,
            "ar": ms.get("answer_relevancy", {}).get("score"),
            "ar_gen": ms.get("answer_relevancy_generative", {}).get("score"),
        }
    return out


def kappa(manual: list[str], auto: list[bool]) -> dict:
    n = len(manual)
    bp = sum(1 for m, a in zip(manual, auto) if m == "pass" and a)
    mpaf = sum(1 for m, a in zip(manual, auto) if m == "pass" and not a)
    mfap = sum(1 for m, a in zip(manual, auto) if m == "fail" and not a) if False else \
           sum(1 for m, a in zip(manual, auto) if m == "fail" and a)
    bf = sum(1 for m, a in zip(manual, auto) if m == "fail" and not a)
    obs = (bp + bf) / n
    exp = ((bp + mpaf) * (bp + mfap) + (mfap + bf) * (mpaf + bf)) / (n * n)
    k = None if exp >= 1.0 else (obs - exp) / (1.0 - exp)
    return {
        "agree": bp + bf,
        "total": n,
        "rate": round(obs, 4),
        "kappa": round(k, 4) if k is not None else None,
        "human_pass_auto_fail": mpaf,
        "human_fail_auto_pass": mfap,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=10)
    parser.add_argument("--manual-task", type=int, default=None,
                        help="人工标注来源任务（默认=--task 自身；校准后重跑用旧任务的标注）")
    parser.add_argument("--db", type=pathlib.Path, default=DEFAULT_DB)
    args = parser.parse_args()

    data = _load(args.db, args.task)
    if args.manual_task and args.manual_task != args.task:
        conn = sqlite3.connect(str(args.db))
        manual_rows = conn.execute(
            "SELECT row_index, manual_status FROM eval_row_results "
            "WHERE eval_task_id = ?",
            (args.manual_task,),
        ).fetchall()
        conn.close()
        manual_by_row = {ri: m for ri, m in manual_rows}
        for ri in data:
            if data[ri]["manual"] is None:
                data[ri]["manual"] = manual_by_row.get(ri)
    rows = sorted(data)
    if len(rows) != 25:
        print(f"警告: {len(rows)} 行（预期 25）")
    manual = [data[i]["manual"] for i in rows]
    if not all(m in ("pass", "fail") for m in manual):
        raise SystemExit("存在未标注行，无法做人工真值仲裁")

    report = {
        "provenance": {
            "script": "scripts/dual_channel_arbitration.py",
            "task_id": args.task,
            "thresholds_evaluated": THRESHOLDS,
            "dispute_definition": f"|ar - ar_gen| > {DIFF_CUT}",
            "note": "零 LLM 成本：用已有双通道分数 + 人工标签做仲裁分析",
        }
    }

    for t in THRESHOLDS:
        ar_pass = [data[i]["ar"] is not None and data[i]["ar"] >= t for i in rows]
        gen_pass = [data[i]["ar_gen"] is not None and data[i]["ar_gen"] >= t for i in rows]
        report[f"threshold_{t}"] = {
            "ar_channel_vs_human": kappa(manual, ar_pass),
            "generative_channel_vs_human": kappa(manual, gen_pass),
            "min_channel_vs_human": kappa(manual, [a and g for a, g in zip(ar_pass, gen_pass)]),
            "max_channel_vs_human": kappa(manual, [a or g for a, g in zip(ar_pass, gen_pass)]),
            "mean_channel_vs_human": kappa(
                manual,
                [
                    (data[i]["ar"] is not None and data[i]["ar_gen"] is not None
                     and (data[i]["ar"] + data[i]["ar_gen"]) / 2 >= t)
                    for i in rows
                ],
            ),
        }

    # 分歧行明细（用 0.7 阈值语境）——diff 与阈值无关，只需一列
    disputes = []
    for i in rows:
        d = data[i]
        if d["ar"] is None or d["ar_gen"] is None:
            continue
        diff = abs(d["ar"] - d["ar_gen"])
        if diff > DIFF_CUT:
            disputes.append(
                {
                    "row": i,
                    "kind": d["kind"],
                    "manual": d["manual"],
                    "ar": d["ar"],
                    "ar_gen": d["ar_gen"],
                    "diff": round(diff, 3),
                    "low_channel": "ar" if d["ar"] < d["ar_gen"] else "ar_gen",
                }
            )
    report["dispute_rows"] = disputes

    # 分歧行分层：低通道 vs 高通道 与人工的一致性（0.7 阈值）
    t = 0.7
    dispute_idx = [d["row"] for d in disputes]
    if dispute_idx:
        low_pass = [data[i]["ar"] >= t if data[i]["ar"] <= data[i]["ar_gen"]
                    else data[i]["ar_gen"] >= t for i in dispute_idx]
        high_pass = [data[i]["ar"] >= t if data[i]["ar"] >= data[i]["ar_gen"]
                     else data[i]["ar_gen"] >= t for i in dispute_idx]
        dm = [data[i]["manual"] for i in dispute_idx]
        report["dispute_resolution_@0.7"] = {
            "n_disputes": len(dispute_idx),
            "use_low_channel": kappa(dm, low_pass),
            "use_high_channel": kappa(dm, high_pass),
            "human_labels": {str(i): data[i]["manual"] for i in dispute_idx},
        }

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印
    print(f"=== 双通道仲裁分析（任务 {args.task}，{len(rows)} 行，人工真值）===")
    for t in THRESHOLDS:
        blk = report[f"threshold_{t}"]
        print(f"--- 阈值 {t} ---")
        for name in ("ar", "generative", "min", "max", "mean"):
            m = blk[f"{name}_channel_vs_human"]
            print(f"  {name:>10}: 一致率 {m['rate']}  kappa {m['kappa']}  "
                  f"(人工过松 {m['human_pass_auto_fail']} / 自动过松 {m['human_fail_auto_pass']})")
    print(f"--- 分歧行（|diff| > {DIFF_CUT}，{len(disputes)} 条）---")
    for d in disputes:
        print(f"  row {d['row']:>2} ({d['kind']:>11}): 人工={d['manual']} "
              f"AR={d['ar']} AR_gen={d['ar_gen']} 低通道={d['low_channel']}")
    if "dispute_resolution_@0.7" in report:
        dr = report["dispute_resolution_@0.7"]
        lo, hi = dr["use_low_channel"], dr["use_high_channel"]
        print(f"--- 分歧行仲裁（阈值 0.7）---")
        print(f"  用低通道: 一致率 {lo['rate']} kappa {lo['kappa']}")
        print(f"  用高通道: 一致率 {hi['rate']} kappa {hi['kappa']}")
    print(f"报告已写入: {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
