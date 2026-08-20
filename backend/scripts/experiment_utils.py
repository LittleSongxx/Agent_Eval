"""Shared safety and provenance helpers for reproducible experiment scripts.

The experiment scripts are allowed to call a paid/remote Judge, so their
command-line boundary is part of the evidence chain.  This module keeps the
boring safeguards identical across experiments: ``--dry-run`` never creates a
client, output files are not overwritten accidentally, and every report records
the data hash and source revision used to produce it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


def add_common_arguments(parser: argparse.ArgumentParser, *, default_output: Path) -> None:
    """Add the common, deliberately explicit experiment flags."""

    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"报告输出路径（默认: {default_output}）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有报告；默认发现同名文件就中止",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验数据集和执行计划，不创建 Judge、不发起网络请求",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只运行前 N 条样本（用于小规模验证；不传则运行全部）",
    )


def load_samples(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"实验数据集不存在: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"实验数据集必须是 JSON 对象数组: {path}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit 必须是正整数")
        value = value[:limit]
    if not value:
        raise ValueError(f"实验数据集为空: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a canonical JSON value for the exact selected sample subset."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_revision(repo_root: Path) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            or None
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_status(repo_root: Path) -> tuple[bool, str | None]:
    """Return dirty state and a hash of status text for uncommitted runs."""

    try:
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False, None
    status = status.strip()
    return bool(status), hashlib.sha256(status.encode("utf-8")).hexdigest() if status else None


def _portable_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def build_provenance(
    *,
    script_path: Path,
    dataset_path: Path,
    sample_count: int,
    model: str | None,
    selected_samples: list[dict[str, Any]] | None = None,
    argv: Iterable[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = script_path.resolve().parents[1]
    git_dirty, worktree_status_sha256 = git_worktree_status(repo_root)
    provenance: dict[str, Any] = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": _portable_path(script_path, repo_root),
        "git_commit": git_revision(repo_root),
        "git_dirty": git_dirty,
        "worktree_status_sha256": worktree_status_sha256,
        "dataset": _portable_path(dataset_path, repo_root),
        "dataset_sha256": sha256_file(dataset_path),
        "sample_count": sample_count,
        "model": model,
        "python": sys.version.split()[0],
        "command": list(argv) if argv is not None else sys.argv,
    }
    if extra:
        provenance.update(extra)
    if selected_samples is not None:
        # ``dataset_sha256`` identifies the complete source file.  When
        # ``--limit`` is used, this second digest identifies the rows that
        # were actually scored; otherwise a report could not be reproduced
        # from its own sample_count.
        provenance["selected_sample_count"] = len(selected_samples)
        provenance["selected_samples_sha256"] = sha256_json(selected_samples)
    return provenance


def ensure_api_key(settings: Any) -> None:
    key = str(getattr(settings, "LLM_API_KEY", "") or "").strip()
    if not key:
        raise RuntimeError(
            "未配置 LLM_API_KEY；为避免实验半途失败，请先设置 API Key，或使用 --dry-run。"
        )


def ensure_output_path(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"报告已存在: {path}。为避免覆盖证据，请改用 --output 或显式传 --overwrite。"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json_report(path: Path, report: dict[str, Any], *, overwrite: bool) -> None:
    ensure_output_path(path, overwrite=overwrite)
    # 同目录临时文件 + replace，避免中途网络异常留下半份 JSON 报告。
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
