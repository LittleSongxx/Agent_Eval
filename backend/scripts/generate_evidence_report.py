#!/usr/bin/env python3
"""Generate the canonical, provenance-aware秋招 evidence report.

This command is local and deterministic.  It never calls a model; remote
experiments are reported as ``待生成`` until their new JSON report exists.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from scripts.experiment_utils import ensure_output_path, load_samples, sha256_file  # noqa: E402
from scripts.meta_evaluation_audit import audit as audit_meta_evaluation  # noqa: E402


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _report_status(path: Path) -> str:
    if not path.is_file():
        return "待生成（旧结果已清理；需要显式运行远程实验）"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        provenance = value.get("provenance") or {}
        if not provenance.get("dataset_sha256") or not provenance.get("git_commit"):
            return "存在但 provenance 不完整，不作为面试证据"
        return f"已生成（{provenance.get('generated_at_utc', '时间未知')}，commit {str(provenance.get('git_commit'))[:12]}）"
    except (OSError, ValueError, TypeError):
        return "存在但 JSON 无法解析，不作为面试证据"


def _run_verification() -> str:
    # The canonical report should record the full regression gate, not only
    # the low-cost smoke subset used by the script's own unit tests.
    command = [sys.executable, "-m", "pytest", "-q"]
    result = subprocess.run(command, cwd=BACKEND_DIR, capture_output=True, text=True)
    lines = (result.stdout + "\n" + result.stderr).strip().splitlines()
    matching = [line.strip() for line in lines if " passed" in line or " failed" in line]
    summary = matching[-1] if matching else (lines[-1] if lines else f"exit={result.returncode}")
    return f"`{' '.join(command)}` -> {summary}"


def _render(*, verification: str | None = None) -> str:
    agent_path = BACKEND_DIR / "datasets" / "agent_trajectory_samples.json"
    citation_path = BACKEND_DIR / "datasets" / "citation_samples.json"
    concurrency_path = BACKEND_DIR / "datasets" / "concurrency_samples.json"
    agent = load_samples(agent_path)
    citation = load_samples(citation_path)
    concurrency = load_samples(concurrency_path)
    agent_error_labels = sum(item.get("expected_error_recovery") is not None for item in agent)
    agent_tool_labels = sum(item.get("expected_tool_selection_rationality") is not None for item in agent)
    citation_strata = {
        "完美引用": sum(item.get("expected_citation_accuracy") == 1.0 for item in citation),
        "部分错误": sum(
            item.get("expected_citation_accuracy") is not None
            and 0.5 <= item.get("expected_citation_accuracy") < 1.0
            for item in citation
        ),
        "严重错误": sum(
            item.get("expected_citation_accuracy") is not None
            and item.get("expected_citation_accuracy") < 0.5
            for item in citation
        ),
    }
    report_status = {
        "Agent 轨迹": _report_status(BACKEND_DIR / "agent_trajectory_report.json"),
        "引用准确性": _report_status(BACKEND_DIR / "citation_accuracy_report.json"),
        "并发 harness": _report_status(BACKEND_DIR / "concurrency_optimization_report.json"),
    }
    meta_audit = audit_meta_evaluation(
        BACKEND_DIR / "datasets" / "meta_evaluation_manifest.json",
        BACKEND_DIR / "eval_platform.db",
    )
    generated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines = [
        "# Agent_Eval 秋招项目证据报告",
        "",
        "> 本文件由 `backend/scripts/generate_evidence_report.py` 生成。远程模型实验没有报告文件时明确标记为“待生成”，不沿用旧文档数字。",
        "",
        "## 当前结论",
        "",
        "项目真实定位是 Python/FastAPI + React/TypeScript 的 AI 评测工作台，不是 Java/Spring AI 后端，也不是完整 Agent Runtime。可描述的闭环是：数据集/Agent Trace 导入 -> 场景、指标、Judge、Endpoint 配置 -> 离线或实时评测 -> 行级分数、成本、延迟和报告 -> 人工复核/Bad Case -> 回归集和质量门禁。",
        "",
        "本轮秋招高性价比改造的重点是评测证据可信度：任务创建时冻结行输入和 Judge 运行参数，执行时使用冻结值；实验脚本默认不误调用、不覆盖旧结果，并记录 provenance。",
        "",
        "## 可复核的数据集",
        "",
        "| 数据集 | 样本数 | 标注覆盖 | SHA256 |",
        "|---|---:|---|---|",
        f"| Agent 轨迹 | {len(agent)} | trajectory={len(agent)}/{len(agent)}；error_recovery={agent_error_labels}/{len(agent)}；tool_selection={agent_tool_labels}/{len(agent)} | `{sha256_file(agent_path)}` |",
        f"| 引用准确性 | {len(citation)} | 完美={citation_strata['完美引用']}；部分错误={citation_strata['部分错误']}；严重错误={citation_strata['严重错误']} | `{sha256_file(citation_path)}` |",
        f"| 并发 harness | {len(concurrency)} | 无人工质量金标，仅用于耗时对照 | `{sha256_file(concurrency_path)}` |",
        "",
        "注意：引用数据集当前没有严重错误样本，不能据此声称“严重错误召回率”。Agent 的错误恢复只有标注样本子集，不能把其均值当成全体 30 条的恢复率。",
        "",
        "## 元评价证据门禁",
        "",
        f"当前审计状态：**{meta_audit['status']}**；阻塞门禁 {len(meta_audit['blocking_gates'])} 项。该状态来自 `backend/scripts/meta_evaluation_audit.py`，不是人工转抄。",
        "",
        "| 门禁 | 当前值 | 最低要求 | 结果 |",
        "|---|---:|---:|---|",
    ]
    for name, gate in meta_audit["gates"].items():
        lines.append(
            f"| {name} | {gate.get('count', '—')} | {gate.get('minimum', '—')} | "
            f"{'通过' if gate.get('pass') else '阻塞'} |"
        )
    lines.extend([
        "",
        "当前元评价只能作为 pilot：需要至少两位独立标注者、显式 calibration/held-out 划分、50 条以上 Agent 轨迹（含至少 10 条恢复样本）、60 条以上 Citation（含至少 10 条严重错误），并为每份结果补齐数据 hash、代码 commit、模型和样本数。",
        "",
        "## 远程实验状态",
        "",
        "| 实验 | 当前状态 | 正确统计口径 |",
        "|---|---|---|",
        "\n".join(
            f"| {name} | {status} | {desc} |"
            for (name, status), desc in zip(
                report_status.items(),
                (
                    "MAE、Spearman、容差准确率；二分类时再看 confusion matrix/F1",
                    "MAE/RMSE、容差准确率、完美/部分/严重分层和引用错误 confusion matrix",
                    "serial vs metric-concurrent vs row/metric harness；必须注明不是生产双层并发",
                ),
            )
        ),
        "",
        "运行命令（会调用远程模型并产生费用，默认防止覆盖）：",
        "",
        "```bash",
        "cd backend",
        "python -m scripts.agent_trajectory_experiment --output agent_trajectory_report.json",
        "python -m scripts.citation_accuracy_experiment --output citation_accuracy_report.json",
        "python -m scripts.concurrency_optimization_experiment --output concurrency_optimization_report.json",
        "python -m scripts.generate_evidence_report --overwrite",
        "python -m scripts.meta_evaluation_audit --output meta_evaluation_audit.json --overwrite --allow-blocked",
        "```",
        "",
        "## 已实现的正确性保护",
        "",
        "- `EvalTask` 保存 `dataset_snapshot`、`dataset_snapshot_digest` 和完整 Judge runtime snapshot；公开响应只返回去密钥的 `judge_snapshot`。",
        "- `EvalTask` 同时冻结 `judge_samples`，质量门禁能识别重采样次数变化，不把稳定性/成本变化误归因于被测系统。",
        "- 执行器用任务快照构造行输入和 Judge 客户端；旧任务仅走兼容回退，因此旧任务仍标记为不可严格复现。",
        "- Endpoint `write_back` 真正改变行或 schema 时递增 `Dataset.version`，避免后续指纹把不同数据当成同一版本。",
        "- Agent Trace 同时提供 `events` 与 `agent_trajectory` 时校验冲突；并行 tool call 缺少 `id` 时拒绝导入，避免静默错配。",
        "- 三个远程实验脚本具备 `--help`、`--dry-run`、`--limit`、`--output`、`--overwrite`、API Key 前置检查和 provenance。",
        "",
        "## 本次生成信息",
        "",
        f"- 生成时间（UTC）：`{generated}`",
        f"- 源代码 HEAD：`{_git_revision()}`",
        f"- 工作树：`{'clean' if not subprocess.run(['git', '-C', str(REPO_ROOT), 'status', '--porcelain'], capture_output=True, text=True).stdout.strip() else '有未提交改动，提交后应重新生成'}`",
    ])
    if verification:
        lines.extend(["", "## 本地验证", "", verification])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成不含虚假实验数字的秋招证据报告")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs" / "秋招项目证据报告.md")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有报告")
    parser.add_argument("--run-verification", action="store_true", help="额外运行低成本回归测试并记录最后一行摘要")
    args = parser.parse_args(argv)
    try:
        ensure_output_path(args.output, overwrite=args.overwrite)
        verification = _run_verification() if args.run_verification else None
        temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(_render(verification=verification), encoding="utf-8")
        os.replace(temporary, args.output)
        print(f"证据报告已生成: {args.output}")
        return 0
    except (OSError, ValueError, FileExistsError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
