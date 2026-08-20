"""评测脚本冒烟测试（CI 门禁）：所有 scripts/*.py 必须可正常解析参数并退出。

背景：校准/仲裁/对标脚本是评测闭环的复现工具（calibration_compare、
dual_channel_arbitration 等）。脚本一旦损坏（import 错误、参数解析回归），
"检测 → 定位 → 修复 → 复评"链路就断了，CI 必须抓住。本测试只做零成本冒烟
（--help 退出码 0），不跑真实评测。
"""

import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "calibration_compare",
    "dual_channel_arbitration",
    "compare_ragas_nonllm",
    "compute_goldset_agreement",
    "make_goldset_worksheet",
    "agent_trajectory_experiment",
    "citation_accuracy_experiment",
    "concurrency_optimization_experiment",
    "generate_evidence_report",
    "meta_evaluation_audit",
]


def test_scripts_parse_help():
    for name in SCRIPTS:
        result = subprocess.run(
            [sys.executable, "-m", f"scripts.{name}", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0, (
            f"{name} --help 退出码 {result.returncode}: {result.stderr[:500]}"
        )
        assert "usage" in result.stdout.lower(), f"{name} --help 未输出 usage"


def test_remote_experiment_scripts_dry_run_without_api_call():
    """The three remote experiments must have a zero-cost CI path."""
    for name in ("agent_trajectory_experiment", "citation_accuracy_experiment", "concurrency_optimization_experiment"):
        result = subprocess.run(
            [sys.executable, "-m", f"scripts.{name}", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0, f"{name} --dry-run 失败: {result.stderr[:500]}"
        assert "未发起网络请求" in result.stdout


def test_remote_experiment_refuses_existing_output_before_running(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.agent_trajectory_experiment",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 2
    assert "报告已存在" in result.stderr


def test_meta_evaluation_audit_is_explicitly_blocked_until_goldset_is_complete(tmp_path):
    output = tmp_path / "meta-audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.meta_evaluation_audit",
            "--output",
            str(output),
            "--overwrite",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 2
    payload = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert "independent_annotators" in payload["blocking_gates"]
    assert "held_out_split" in payload["blocking_gates"]


def test_meta_evaluation_audit_allow_blocked_is_nonzero_data_but_zero_exit(tmp_path):
    output = tmp_path / "meta-audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.meta_evaluation_audit",
            "--output",
            str(output),
            "--overwrite",
            "--allow-blocked",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0
    assert "blocked" in result.stdout


def test_goldset_agreement_honors_custom_annotations_path(tmp_path):
    """A custom annotation file must not silently fall back to the default."""
    backend = Path(__file__).resolve().parent.parent
    source_report = backend / "retrieval_bm25_report.json"
    source_annotations = backend / "scripts" / "retrieval_goldset_annotations.json"
    if not source_report.is_file() or not source_annotations.is_file():
        # The repository may intentionally omit historical retrieval artifacts.
        return
    annotations = json.loads(source_annotations.read_text(encoding="utf-8"))
    annotations["annotations"]["1"] = {"verdict": "N", "expected_hits": []}
    custom = tmp_path / "annotations.json"
    custom.write_text(json.dumps(annotations, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "agreement.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.compute_goldset_agreement",
            "--report",
            str(source_report),
            "--annotations",
            str(custom),
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=backend,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["verdict_distribution"]["N"] == 1
    assert payload["gold_in_expected_rate"] < 1.0


def test_provenance_records_hash_for_selected_subset(tmp_path):
    from scripts.experiment_utils import build_provenance

    dataset = tmp_path / "dataset.json"
    dataset.write_text('[{"sample_id":"a"},{"sample_id":"b"}]', encoding="utf-8")
    selected = [{"sample_id": "a"}]
    provenance = build_provenance(
        script_path=Path(__file__).resolve().parent.parent / "scripts" / "agent_trajectory_experiment.py",
        dataset_path=dataset,
        sample_count=1,
        model="test-model",
        selected_samples=selected,
        argv=["experiment", "--limit", "1"],
    )
    assert provenance["sample_count"] == 1
    assert provenance["selected_sample_count"] == 1
    assert provenance["selected_samples_sha256"]
    assert provenance["selected_samples_sha256"] != provenance["dataset_sha256"]


def test_meta_audit_rejects_duplicate_sample_ids(tmp_path):
    from scripts.meta_evaluation_audit import _dataset_audit

    dataset = tmp_path / "duplicate.json"
    dataset.write_text(
        json.dumps([{"sample_id": "same"}, {"sample_id": "same"}], ensure_ascii=False),
        encoding="utf-8",
    )
    result = _dataset_audit(
        {
            "id": "duplicate",
            "path": str(dataset),
            "domain": "agent",
            "source_status": "test",
            "target_rows": 1,
        }
    )
    assert result["stable_sample_id"] is False
    assert result["duplicate_sample_ids"] == 1
