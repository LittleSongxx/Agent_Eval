"""合并 n=25 对照报告，产出可追溯的权威版本。

背景：修复两个平台自身缺陷（见 docs/meta-evaluation.md §5.3）之后重跑了对照，
但为省调用量只重跑了 4 个走 ``_sample_payload`` 的基础指标。

两个双通道指标（faithfulness_claim / answer_relevancy_generative）在代码上根本
不经过 ``_sample_payload`` —— 它们只读自己声明的字段，既没见过 generation_meta
（缺陷 A），也不受 judge_fields 白名单影响（缺陷 B）。两个缺陷的修复 diff 都不
落在这两个类里，故其修复前数值依然有效，直接沿用。

本脚本把两份来源按指标拼成一份报告，并为每个指标标注数据来源文件，
避免任何数值靠人工转抄。
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

BACKEND = pathlib.Path(__file__).resolve().parent.parent

SCOPED = BACKEND / "ragas_comparison_report_n25_scoped.json"  # 缺陷 A+B 均已修复
PREFIX_RUN = BACKEND / "ragas_comparison_report_n25.json"      # 两个缺陷都还在
OUT = BACKEND / "ragas_comparison_report.json"

# 不走 _sample_payload 的指标：只读自己声明的字段，两个缺陷都影响不到它们
CARRY_FORWARD = ["faithfulness_claim", "answer_relevancy_generative"]


def _mtime(path: pathlib.Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    for path in (SCOPED, PREFIX_RUN):
        if not path.exists():
            print(f"缺少来源文件: {path}")
            return 1

    scoped = json.loads(SCOPED.read_text(encoding="utf-8"))
    prefix = json.loads(PREFIX_RUN.read_text(encoding="utf-8"))

    # 合并的前提：两次跑的是同一个数据集、同一个顺序。否则逐指标拼接没有意义。
    if scoped["kinds"] != prefix["kinds"]:
        print("两次运行的样本类型序列不一致，禁止合并。")
        return 1
    if scoped["rows"] != prefix["rows"]:
        print(f"行数不一致: scoped={scoped['rows']} prefix={prefix['rows']}")
        return 1

    merged_metrics: dict[str, dict] = {}

    for name, entry in scoped["metrics"].items():
        merged_metrics[name] = {
            **entry,
            "_provenance": {
                "source": SCOPED.name,
                "run_at": _mtime(SCOPED),
                "defect_a_fixed": True,
                "defect_b_fixed": True,
                "note": "标签泄露 + 口径越界两个缺陷均已修复后重跑",
            },
        }

    for name in CARRY_FORWARD:
        if name not in prefix["metrics"]:
            print(f"来源缺少待沿用指标: {name}")
            return 1
        merged_metrics[name] = {
            **prefix["metrics"][name],
            "_provenance": {
                "source": PREFIX_RUN.name,
                "run_at": _mtime(PREFIX_RUN),
                "defect_a_fixed": False,
                "defect_b_fixed": False,
                "note": (
                    "该指标不经过 _sample_payload，只读取自身声明的字段："
                    "既未收到 generation_meta（缺陷 A），也不受 judge_fields "
                    "白名单约束（缺陷 B）。两个修复的 diff 均不落在该类，"
                    "故沿用修复前数值。"
                ),
            },
        }

    # 按对照可读性排序：整体 / 断言级成对出现
    order = [
        "faithfulness",
        "faithfulness_claim",
        "answer_relevancy",
        "answer_relevancy_generative",
        "context_precision",
        "context_recall",
    ]
    ordered = {k: merged_metrics[k] for k in order if k in merged_metrics}
    ordered.update({k: v for k, v in merged_metrics.items() if k not in ordered})

    out = {
        "_note": (
            "n=25 平台 vs RAGAS 对照（9 正确 / 8 幻觉注入 / 8 截断）。"
            "各指标数据来源见 _provenance：基础指标为两个缺陷修复后重跑；"
            "双通道指标因机制上不受两个缺陷影响而沿用。"
        ),
        "kinds": scoped["kinds"],
        "rows": scoped["rows"],
        "metrics": ordered,
    }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已写出 {OUT.name}（{len(ordered)} 个指标, n={scoped['rows']}）\n")
    header = f"{'指标':<32}{'平台':>8}{'RAGAS':>9}{'Pearson':>10}{'Spearman':>10}   来源"
    print(header)
    print("-" * len(header))
    for name, entry in ordered.items():
        carried = "" if entry["_provenance"]["defect_b_fixed"] else "  (沿用)"
        print(
            f"{name:<32}"
            f"{entry['platform_mean']:>8.4f}"
            f"{entry['ragas_mean']:>9.4f}"
            f"{str(entry['pearson']):>10}"
            f"{str(entry['spearman']):>10}"
            f"   {entry['_provenance']['source']}{carried}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
