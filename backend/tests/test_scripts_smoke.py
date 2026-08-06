"""评测脚本冒烟测试（CI 门禁）：所有 scripts/*.py 必须可正常解析参数并退出。

背景：校准/仲裁/对标脚本是评测闭环的复现工具（calibration_compare、
dual_channel_arbitration 等）。脚本一旦损坏（import 错误、参数解析回归），
"检测 → 定位 → 修复 → 复评"链路就断了，CI 必须抓住。本测试只做零成本冒烟
（--help 退出码 0），不跑真实评测。
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "calibration_compare",
    "dual_channel_arbitration",
    "compare_ragas_nonllm",
    "compute_goldset_agreement",
    "make_goldset_worksheet",
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
