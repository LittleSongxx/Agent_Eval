#!/usr/bin/env python3
"""Audit whether the project's meta-evaluation evidence is interview-ready.

The audit is intentionally local and deterministic apart from the report
timestamp.  It does not call a model and it never upgrades a small synthetic
fixture into a claim about general accuracy.  The manifest defines the target
sample sizes and annotation protocol; this command checks the current files
and (when available) the SQLite annotation table against those targets.

Usage::

    cd backend
    python -m scripts.meta_evaluation_audit --output meta_evaluation_audit.json

Exit status is 0 for a complete gate and 2 when the evidence is still
blocked.  ``--allow-blocked`` is useful when generating a report during
development; it does not change the report's status.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
DEFAULT_MANIFEST = BACKEND_DIR / "datasets" / "meta_evaluation_manifest.json"
DEFAULT_OUTPUT = BACKEND_DIR / "meta_evaluation_audit.json"

REQUIRED_PROVENANCE = {
    "generated_at_utc",
    "git_commit",
    "dataset_sha256",
    "sample_count",
    "model",
}


def _git_revision() -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return value or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_backend_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else BACKEND_DIR / path


def _display_path(path: Path) -> str:
    """Keep generated reports portable instead of embedding this checkout path."""

    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _field_coverage(rows: list[dict[str, Any]], field: str) -> int:
    return sum(row.get(field) is not None for row in rows)


def _dataset_audit(entry: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_backend_path(str(entry["path"]))
    result: dict[str, Any] = {
        "id": entry.get("id"),
        "path": _display_path(path),
        "domain": entry.get("domain"),
        "source_status": entry.get("source_status", "unknown"),
        "target_rows": entry.get("target_rows"),
        "exists": path.is_file(),
    }
    if not path.is_file():
        result.update({"status": "missing", "sample_count": 0, "findings": ["数据集文件不存在"]})
        return result

    try:
        value = _load_json(path)
    except (OSError, ValueError) as exc:
        result.update({"status": "invalid", "sample_count": 0, "findings": [f"JSON 无法解析: {exc}"]})
        return result

    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        result.update({"status": "invalid", "sample_count": 0, "findings": ["数据集必须是 JSON 对象数组"]})
        return result

    rows = value
    result["sample_count"] = len(rows)
    result["sha256"] = _sha256(path)
    result["coverage"] = {
        field: {"count": _field_coverage(rows, field), "total": len(rows)}
        for field in (
            entry.get("trajectory_gold_field"),
            entry.get("recovery_gold_field"),
            entry.get("tool_selection_gold_field"),
            entry.get("score_gold_field"),
        )
        if field
    }
    split_counts: dict[str, int] = {}
    for row in rows:
        split = row.get("split")
        if split not in (None, ""):
            split_counts[str(split)] = split_counts.get(str(split), 0) + 1
    result["split_counts"] = split_counts
    # A fixture without a stable sample id is still usable for a pilot, but
    # cannot support reliable cross-run row alignment for a held-out study.
    ids = [row.get("sample_id") for row in rows]
    valid_ids = bool(ids) and all(isinstance(item, str) and bool(item.strip()) for item in ids)
    result["duplicate_sample_ids"] = len(ids) - len(set(ids)) if valid_ids else None
    result["stable_sample_id"] = bool(valid_ids and result["duplicate_sample_ids"] == 0)
    findings: list[str] = []
    target = entry.get("target_rows")
    if isinstance(target, int) and len(rows) < target:
        findings.append(f"样本数 {len(rows)} < 目标 {target}")
    if entry.get("source_status") == "synthetic_fixture":
        findings.append("合成 fixture，只能作为方法冒烟/初步对照")
    if not result["stable_sample_id"]:
        if not valid_ids:
            findings.append("缺少有效字符串 sample_id，无法严格做跨标注/held-out 行对齐")
        else:
            findings.append(f"sample_id 有 {result['duplicate_sample_ids']} 个重复，无法严格做行对齐")
    for field, coverage in result["coverage"].items():
        if coverage["count"] == 0:
            findings.append(f"字段 {field} 没有 gold 覆盖")
    result["findings"] = findings
    result["status"] = "pilot" if findings else "ready"
    return result


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _annotation_audit(db_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": _display_path(db_path),
        "exists": db_path.is_file(),
        "status": "missing",
        "annotator_count": 0,
        "independent_annotator_count": 0,
        "adjudication_count": 0,
        "shared_row_count": 0,
        "findings": [],
    }
    if not db_path.is_file():
        result["findings"] = ["SQLite 数据库不存在，无法核验独立标注覆盖"]
        return result

    try:
        conn = sqlite3.connect(str(db_path))
        if not _table_exists(conn, "row_annotations"):
            result["findings"] = ["缺少 row_annotations 表，无法核验标注者一致性"]
            conn.close()
            return result
        rows = conn.execute(
            "SELECT annotator, is_adjudication, row_result_id FROM row_annotations"
        ).fetchall()
        conn.close()
    except sqlite3.Error as exc:
        result["findings"] = [f"读取标注表失败: {exc}"]
        return result

    independent: dict[str, set[int]] = {}
    for annotator, is_adjudication, row_id in rows:
        if bool(is_adjudication):
            result["adjudication_count"] += 1
            continue
        name = str(annotator or "").strip() or "<empty>"
        independent.setdefault(name, set()).add(int(row_id))
    result["annotators"] = sorted(independent)
    result["annotator_count"] = len(independent)
    result["independent_annotator_count"] = len(independent)
    if len(independent) >= 2:
        names = sorted(independent)
        shared = set.intersection(*(independent[name] for name in names))
        result["shared_row_count"] = len(shared)
    findings: list[str] = []
    if len(independent) < 2:
        findings.append("当前只有 1 位独立标注者，人-人 kappa 上界不可测")
    if not rows:
        findings.append("row_annotations 为空")
    result["findings"] = findings
    result["status"] = "ready" if not findings else "pilot"
    return result


def _report_audit(entry: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_backend_path(str(entry["path"]))
    result: dict[str, Any] = {
        "id": entry.get("id"),
        "path": _display_path(path),
        "kind": entry.get("kind"),
        "required_for_release": bool(entry.get("required_for_release", False)),
        "exists": path.is_file(),
    }
    if not path.is_file():
        result.update({"status": "missing", "findings": ["报告文件不存在"]})
        return result
    try:
        value = _load_json(path)
    except (OSError, ValueError) as exc:
        result.update({"status": "invalid", "findings": [f"JSON 无法解析: {exc}"]})
        return result
    provenance = value.get("provenance") if isinstance(value, dict) else None
    result["provenance_keys"] = sorted(provenance) if isinstance(provenance, dict) else []
    missing = sorted(REQUIRED_PROVENANCE - set(provenance or {}))
    result["missing_provenance"] = missing
    findings: list[str] = []
    if not isinstance(provenance, dict):
        findings.append("缺少 provenance 对象")
    elif missing:
        findings.append("provenance 缺字段: " + ", ".join(missing))
    result["findings"] = findings
    result["status"] = "ready" if not findings else "pilot"
    return result


def audit(manifest_path: Path, db_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest 必须是 JSON 对象")
    datasets = [_dataset_audit(entry) for entry in manifest.get("datasets", [])]
    reports = [_report_audit(entry) for entry in manifest.get("reports", [])]
    annotations = _annotation_audit(db_path)
    targets = manifest.get("targets") or {}

    def coverage(domain: str, field: str, minimum: int) -> dict[str, Any]:
        matching = [item for item in datasets if item.get("domain") == domain]
        count = sum(
            item.get("coverage", {}).get(field, {}).get("count", 0)
            for item in matching
        )
        return {"count": count, "minimum": minimum, "pass": count >= minimum}

    agent = next((item for item in datasets if item.get("domain") == "agent"), {})
    citation = next((item for item in datasets if item.get("domain") == "citation"), {})
    # For citation, values below 0.5 are the explicitly severe-error stratum.
    citation_major = 0
    if citation.get("exists") and citation.get("status") != "invalid":
        try:
            citation_rows = _load_json(_resolve_backend_path(str(citation.get("path"))))
        except (OSError, ValueError):
            citation_rows = []
        citation_major = sum(
            1
            for row in citation_rows
            if isinstance(row, dict)
            and isinstance(row.get("expected_citation_accuracy"), (int, float))
            and row["expected_citation_accuracy"] < 0.5
        )
    held_out_count = sum(
        item.get("split_counts", {}).get("held_out", 0) for item in datasets
    )
    stable_id_count = sum(
        item.get("sample_count", 0) for item in datasets if item.get("stable_sample_id")
    )
    gates = {
        "agent_dataset_scale": {
            "count": agent.get("sample_count", 0),
            "minimum": targets.get("agent_rows_min", 0),
            "pass": agent.get("sample_count", 0) >= targets.get("agent_rows_min", 0),
        },
        "agent_recovery_gold": coverage("agent", "expected_error_recovery", targets.get("agent_recovery_gold_min", 0)),
        "citation_dataset_scale": {
            "count": citation.get("sample_count", 0),
            "minimum": targets.get("citation_rows_min", 0),
            "pass": citation.get("sample_count", 0) >= targets.get("citation_rows_min", 0),
        },
        "citation_major_error_coverage": {
            "count": citation_major,
            "minimum": targets.get("citation_major_error_min", 0),
            "pass": citation_major >= targets.get("citation_major_error_min", 0),
        },
        "independent_annotators": {
            "count": annotations.get("independent_annotator_count", 0),
            "minimum": targets.get("independent_annotators_min", 0),
            "pass": annotations.get("independent_annotator_count", 0) >= targets.get("independent_annotators_min", 0),
        },
        "held_out_split": {
            "count": held_out_count,
            "minimum": targets.get("held_out_rows_min", 0),
            "pass": held_out_count >= targets.get("held_out_rows_min", 0),
            "note": "当前 fixture 没有 split/held_out 标记；必须在新 gold set 中显式提供",
        },
        "stable_sample_ids": {
            "count": stable_id_count,
            "minimum": targets.get("human_gold_rows_min", 0),
            "pass": stable_id_count >= targets.get("human_gold_rows_min", 0),
            "note": "跨标注和 held-out 评测需要稳定 sample_id",
        },
    }
    gates["report_provenance"] = {
        "count": sum(report.get("status") == "ready" for report in reports),
        "minimum": len(reports),
        "pass": all(report.get("status") == "ready" for report in reports),
        "note": "历史报告即使不作为核心简历数字，也必须标为 pilot 或补齐 provenance",
    }
    blocking = [name for name, gate in gates.items() if not gate.get("pass")]
    return {
        "schema_version": 1,
        "benchmark_id": manifest.get("benchmark_id"),
        "status": "ready" if not blocking else "blocked",
        "blocking_gates": blocking,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_revision(),
        "git_dirty": _git_dirty(),
        "manifest": _display_path(manifest_path),
        "datasets": datasets,
        "annotations": annotations,
        "reports": reports,
        "gates": gates,
        "interpretation": {
            "current_claim": "pilot_only",
            "blocked_reason": "缺少双人独立标注、held-out 划分以及足够 Agent/Citation 严重错误覆盖" if blocking else None,
            "do_not_claim": [
                "通用 Agent 能力准确率",
                "严重引用错误召回率（无 major error gold 时）",
                "跨模型/跨数据集泛化稳定性",
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="审计元评价数据、标注覆盖和报告 provenance")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--db", type=Path, default=BACKEND_DIR / "eval_platform.db")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-blocked", action="store_true", help="即使门禁 blocked 也返回 0")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"报告已存在: {args.output}；请指定 --overwrite")
        result = audit(args.manifest, args.db)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        print(f"元评价审计: {result['status']}；阻塞门禁 {len(result['blocking_gates'])}；报告 {args.output}")
        return 0 if result["status"] == "ready" or args.allow_blocked else 2
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
