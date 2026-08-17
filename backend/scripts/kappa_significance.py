"""校准提升的统计显著性：配对 bootstrap + 基线敏感性 + 裁判噪声底 + 样本量外推。

为什么要有这个脚本（它是用来推翻本项目自己的标题数字的）：
  `calibration_compare.py` 报出"kappa 0.4582 → 0.8175"，文档一度把它写成
  "校准把一致性从中等提到显著"。给 `core/agreement.py` 补上 bootstrap 之后
  回头量这句话，发现三处站不住：

  1. **两个 kappa 的区间都横跨多个 Landis-Koch 档位**（[0.1325, 0.7634] 与
     [0.5192, 1.0]），点估计所在的那一档不是能下的结论；
  2. **配对差的双侧下界压在 0 上**。两个 kappa 对着同一批 25 条标注算，是配对
     数据，必须一次抽定行下标、在同一份 resample 上重算两侧再相减（理由见
     `paired_kappa_delta_ci` 的 docstring）；
  3. **换一个同配置的基线，显著性判定就翻**。任务 9 与任务 10 是**同一个校准前
     配置**的两次完整跑（当初为重测信度多跑的一轮），pass/fail 只差 1/25 行，
     却把基线 kappa 从 0.4582 挪到 0.3939。这 0.0643 是**裁判噪声底**——同一
     配置重跑就能产生的抖动，占声称效应量的 17.9%。以任务 10 为基线区间含 0，
     以任务 9 为基线区间排除 0：文档里用的恰好是更保守的那个，属于运气。

  这三条的结论是：n=25 上的这个提升，正确表述是"方向明确、量级待定"，不是
  "已验证"。第 4 项外推给出让下界离开 0 所需的样本量，把"该扩样"从泛泛之谈
  变成定量目标。

口径说明（与 `calibration_compare.py` 完全一致，避免两处漂移）：
  - 人工标注只存在于任务 10 的 `manual_status`（9 pass / 16 fail），任务 9 与
    任务 13 的该字段为空，故三个任务共用这一批人工真值；
  - 自动判定取各任务入库的 `is_pass`（任务 9/10 为阈值 0.5 口径，任务 13 为
    校准后 0.7 口径）——即"同一批人工标注，只换自动判定那一列"。

用法：
    cd backend && python -m scripts.kappa_significance \
        --baseline 10 --calibrated 13 --alt-baseline 9

输出：
    backend/kappa_significance_report.json（带 provenance）
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sqlite3

BACKEND = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DB = BACKEND / "eval_platform.db"
OUT = BACKEND / "kappa_significance_report.json"


def _load_auto(db_path: pathlib.Path, task_id: int) -> dict[int, bool]:
    """按 row_index 取任务的自动判定（入库 is_pass，即该任务运行时的阈值口径）。"""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT row_index, is_pass FROM eval_row_results "
            "WHERE eval_task_id = ? ORDER BY row_index",
            (task_id,),
        ).fetchall()
    finally:
        conn.close()
    return {row_index: bool(is_pass) for row_index, is_pass in rows}


def _load_manual(db_path: pathlib.Path, task_id: int) -> dict[int, str]:
    """按 row_index 取人工标注（只保留明确 pass/fail 的行）。"""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT row_index, manual_status FROM eval_row_results "
            "WHERE eval_task_id = ? AND manual_status IN ('pass', 'fail') "
            "ORDER BY row_index",
            (task_id,),
        ).fetchall()
    finally:
        conn.close()
    return {row_index: status for row_index, status in rows}


def _task_meta(db_path: pathlib.Path, task_id: int) -> dict:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT id, name, status, scenario_id, dataset_id, finished_at "
            "FROM eval_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise SystemExit(f"任务 {task_id} 不存在于 {db_path}")
    return {
        "task_id": row[0],
        "name": row[1],
        "status": row[2],
        "scenario_id": row[3],
        "dataset_id": row[4],
        "finished_at": row[5],
    }


def _pairs(manual: dict[int, str], auto: dict[int, bool]) -> list[tuple[bool, bool]]:
    """对齐成 (人工通过, 自动通过) 布尔对，行序固定，与 calibration_compare 同口径。

    缺行直接报错而不是跳过：静默少几行会让 n 和 kappa 都对不上账，
    而这类不一致最难发现——两边都"看起来算对了"。
    """
    missing = [i for i in sorted(manual) if i not in auto]
    if missing:
        raise SystemExit(f"自动判定缺少人工已标注的行：{missing}")
    return [(manual[i] == "pass", auto[i] is True) for i in sorted(manual)]


def _replicate(pairs: list[tuple[bool, bool]], factor: int) -> list[tuple[bool, bool]]:
    """把观测到的联合模式复制 factor 份，用于样本量外推。"""
    return list(pairs) * factor


def extrapolate_sample_size(
    baseline_pairs: list[tuple[bool, bool]],
    calibrated_pairs: list[tuple[bool, bool]],
    factors: tuple[int, ...] = (1, 2, 3, 4),
) -> list[dict]:
    """样本量外推：把当前联合模式复制 N 份后重算配对区间。

    **这是功效（power）计算，不是效应量的证据。** 复制观测模式等于假设"这 25 行
    的联合分布就是总体分布"——真扩样时新样本几乎必然带来新的分歧模式，所以这里
    回答的是"若效应确实是这个量级，多少样本才能让下界离开 0"，而不是"效应是真的"。
    把这一点写在输出里，避免这组数字被当成"扩样后就能得到 0.8175"来引用。
    """
    from app.core.agreement import paired_kappa_delta_ci

    out = []
    for factor in factors:
        base = _replicate(baseline_pairs, factor)
        cal = _replicate(calibrated_pairs, factor)
        delta_ci = paired_kappa_delta_ci(base, cal)
        if delta_ci is None:
            continue
        out.append(
            {
                "n": len(base),
                "replication_factor": factor,
                "delta": delta_ci["delta"],
                "ci_low": delta_ci["ci_low"],
                "ci_high": delta_ci["ci_high"],
                "excludes_zero": delta_ci["excludes_zero"],
            }
        )
    return out


def analyse(
    db_path: pathlib.Path,
    manual_task: int,
    baseline_task: int,
    calibrated_task: int,
    alt_baseline_task: int | None,
) -> dict:
    from app.core.agreement import (
        bootstrap_kappa_ci,
        cohens_kappa,
        interpret_kappa,
        paired_kappa_delta_ci,
    )

    manual = _load_manual(db_path, manual_task)
    if not manual:
        raise SystemExit(f"任务 {manual_task} 没有 pass/fail 人工标注")

    def side(task_id: int) -> dict:
        pairs = _pairs(manual, _load_auto(db_path, task_id))
        point = cohens_kappa(pairs)
        return {
            "task": _task_meta(db_path, task_id),
            "pairs": pairs,
            "matrix": point,
            "kappa_band": interpret_kappa(point["kappa"]),
            "kappa_ci": bootstrap_kappa_ci(pairs),
        }

    baseline = side(baseline_task)
    calibrated = side(calibrated_task)

    primary = paired_kappa_delta_ci(baseline["pairs"], calibrated["pairs"])

    result: dict = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "provenance": {
            "db": str(db_path),
            "manual_annotation_source": (
                f"task {manual_task} manual_status（{len(manual)} 条 pass/fail）"
            ),
            "baseline_task": baseline["task"],
            "calibrated_task": calibrated["task"],
            "auto_verdict_field": "eval_row_results.is_pass（各任务运行时阈值口径）",
            "note": (
                "同一批人工标注，只换自动判定那一列——这是配对 bootstrap 成立的前提"
            ),
        },
        "baseline": {
            "task_id": baseline_task,
            "matrix": baseline["matrix"],
            "kappa_band": baseline["kappa_band"],
            "kappa_ci": baseline["kappa_ci"],
        },
        "calibrated": {
            "task_id": calibrated_task,
            "matrix": calibrated["matrix"],
            "kappa_band": calibrated["kappa_band"],
            "kappa_ci": calibrated["kappa_ci"],
        },
        "paired_delta": primary,
    }

    # 逐行联合模式：校准修对了几行、弄坏了几行（净收益不等于只赚不赔）
    fixed, broken = [], []
    for idx, (m, base_auto), (_, cal_auto) in zip(
        sorted(manual), baseline["pairs"], calibrated["pairs"]
    ):
        if base_auto != m and cal_auto == m:
            fixed.append(idx)
        elif base_auto == m and cal_auto != m:
            broken.append(idx)
    result["per_row_effect"] = {
        "fixed_rows": fixed,
        "broken_rows": broken,
        "fixed_count": len(fixed),
        "broken_count": len(broken),
        "note": "broken_count > 0 说明校准不是只赚不赔，净收益才是真实收益",
    }

    # 基线敏感性：换一个同配置的重跑当基线，显著性判定是否翻转
    if alt_baseline_task is not None:
        alt = side(alt_baseline_task)
        alt_delta = paired_kappa_delta_ci(alt["pairs"], calibrated["pairs"])
        base_kappa = baseline["matrix"]["kappa"]
        alt_kappa = alt["matrix"]["kappa"]
        noise_floor = (
            None
            if base_kappa is None or alt_kappa is None
            else round(abs(base_kappa - alt_kappa), 4)
        )
        row_diffs = [
            idx
            for idx, (_, a1), (_, a2) in zip(
                sorted(manual), baseline["pairs"], alt["pairs"]
            )
            if a1 != a2
        ]
        share = None
        if noise_floor is not None and primary and primary["delta"]:
            share = round(noise_floor / abs(primary["delta"]), 4)
        verdict_flips = bool(
            primary
            and alt_delta
            and primary["excludes_zero"] != alt_delta["excludes_zero"]
        )
        result["baseline_sensitivity"] = {
            "alt_baseline_task": alt["task"],
            "alt_baseline_kappa": alt_kappa,
            "primary_baseline_kappa": base_kappa,
            "judge_noise_floor": noise_floor,
            "noise_floor_share_of_effect": share,
            "rows_differing_between_same_config_runs": row_diffs,
            "alt_paired_delta": alt_delta,
            "verdict_flips": verdict_flips,
            "note": (
                "两个基线任务是同一校准前配置的两次完整跑；verdict_flips=true 表示"
                "选哪个基线决定了'提升是否显著'的结论——这本身就是 n=25 不足的证据"
            ),
        }

    result["sample_size_extrapolation"] = {
        "caveat": (
            "功效计算，非效应量证据：复制观测模式=假设这 25 行的联合分布即总体分布。"
            "回答的是'若效应确为此量级，多少样本能让下界离开 0'，"
            "不是'扩样后就能得到该 kappa'"
        ),
        "curve": extrapolate_sample_size(baseline["pairs"], calibrated["pairs"]),
    }

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="校准提升的统计显著性检验（配对 bootstrap + 基线敏感性）"
    )
    parser.add_argument("--baseline", type=int, default=10, help="校准前任务 ID")
    parser.add_argument("--calibrated", type=int, default=13, help="校准后任务 ID")
    parser.add_argument(
        "--alt-baseline",
        type=int,
        default=9,
        help="同配置的另一次校准前重跑，用于量化裁判噪声底（传 -1 关闭）",
    )
    parser.add_argument(
        "--manual-task",
        type=int,
        default=10,
        help="人工标注所在任务（manual_status 只存在于该任务）",
    )
    parser.add_argument("--db", type=pathlib.Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"数据库不存在：{args.db}")

    alt = None if args.alt_baseline is not None and args.alt_baseline < 0 else args.alt_baseline

    report = analyse(
        db_path=args.db,
        manual_task=args.manual_task,
        baseline_task=args.baseline,
        calibrated_task=args.calibrated,
        alt_baseline_task=alt,
    )

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    base = report["baseline"]
    cal = report["calibrated"]
    delta = report["paired_delta"]

    print(f"人工真值: {report['provenance']['manual_annotation_source']}")
    print(
        f"校准前（任务 {base['task_id']}）: kappa={base['matrix']['kappa']} "
        f"CI=[{base['kappa_ci']['ci_low']}, {base['kappa_ci']['ci_high']}] "
        f"跨档={base['kappa_ci']['spans_bands']}"
    )
    print(
        f"校准后（任务 {cal['task_id']}）: kappa={cal['matrix']['kappa']} "
        f"CI=[{cal['kappa_ci']['ci_low']}, {cal['kappa_ci']['ci_high']}] "
        f"跨档={cal['kappa_ci']['spans_bands']}"
    )
    if delta:
        print(
            f"配对 Δkappa={delta['delta']} "
            f"95%双侧 CI=[{delta['ci_low']}, {delta['ci_high']}] "
            f"排除0={delta['excludes_zero']} | "
            f"单侧下界={delta['one_sided_ci_low']} "
            f"P(Δ>0)={delta['p_delta_gt_zero']}"
        )
    eff = report["per_row_effect"]
    print(f"逐行效果: 修对 {eff['fixed_count']} 行 {eff['fixed_rows']}，"
          f"弄坏 {eff['broken_count']} 行 {eff['broken_rows']}")

    sens = report.get("baseline_sensitivity")
    if sens:
        print(
            f"基线敏感性: 任务 {sens['alt_baseline_task']['task_id']} 基线 "
            f"kappa={sens['alt_baseline_kappa']}（vs {sens['primary_baseline_kappa']}）"
            f"，噪声底={sens['judge_noise_floor']}"
            f"（占效应 {sens['noise_floor_share_of_effect']}）"
            f"，同配置两轮判定差异行={sens['rows_differing_between_same_config_runs']}"
        )
        alt_delta = sens["alt_paired_delta"]
        if alt_delta:
            print(
                f"  换基线后: Δkappa={alt_delta['delta']} "
                f"CI=[{alt_delta['ci_low']}, {alt_delta['ci_high']}] "
                f"排除0={alt_delta['excludes_zero']}"
            )
        print(f"  显著性判定翻转: {sens['verdict_flips']}")

    print("样本量外推（功效计算，非效应量证据）:")
    for point in report["sample_size_extrapolation"]["curve"]:
        print(
            f"  n={point['n']}: Δ={point['delta']} "
            f"CI=[{point['ci_low']}, {point['ci_high']}] "
            f"排除0={point['excludes_zero']}"
        )

    print(f"\n报告已写入: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
