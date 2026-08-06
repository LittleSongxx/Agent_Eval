"""A/B 实验：response 字段是否真的改变了上下文侧指标的打分。

context_recall 的口径声明只需要 reference + retrieved_contexts，
但平台实际把整条样本（含 response）下发给裁判。本实验在同一批幻觉样本上
分别以「带 response」和「不带 response」两种 payload 打分，
若分数出现系统性差异，即证明生成质量串进了检索侧指标。

运行（须在 backend/ 目录下，数据库路径为相对路径）：
    python -m scripts.ab_response_leak

前置条件：标签泄漏（缺陷 A）必须已修复，否则两个条件的 payload 里都带着
generation_meta 的真值标记，裁判锚定标签后本实验测不出 response 的净效应。
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

# 指向 backend/，使 app 包在直接运行和 -m 两种方式下都可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.WARNING)
logging.getLogger("app.core.evaluation_engine").setLevel(logging.WARNING)


def build(name: str):
    from app.core.evaluation_engine import build_metric

    _kind, metric = build_metric(
        SimpleNamespace(
            id=0,
            name=name,
            display_name=name,
            metric_type=f"builtin_{name}",
            config={},
            required_fields=[],
            category="rag",
            is_builtin=True,
        ),
        llm=None,
    )
    return metric


async def main() -> None:
    from app.core.database import SessionLocal
    from app.core.evaluation_engine import OpenAIJudgeClient
    from app.models.dataset import Dataset, DatasetRow
    from app.models.llm_config import LLMConfig

    db = SessionLocal()
    try:
        ds = db.query(Dataset).filter(Dataset.name == "售后政策对照集（含坏样本）").first()
        rows = [
            dict(r.data or {})
            for r in db.query(DatasetRow)
            .filter(DatasetRow.dataset_id == ds.id)
            .order_by(DatasetRow.row_index)
            .all()
        ]
        cfg = db.query(LLMConfig).filter(LLMConfig.is_default == True).first()  # noqa: E712
    finally:
        db.close()

    bad = [r for r in rows if (r.get("generation_meta") or {}).get("kind") == "hallucinated"]
    print(f"幻觉样本数: {len(bad)}")

    judge = OpenAIJudgeClient(cfg)

    for metric_name in ("context_recall", "context_precision"):
        print(f"\n{'=' * 70}\n{metric_name}\n{'=' * 70}")
        with_resp: list[float] = []
        without_resp: list[float] = []
        for i, row in enumerate(bad, start=1):
            a = await build(metric_name).ascore(dict(row), judge)
            stripped = {k: v for k, v in row.items() if k != "response"}
            b = await build(metric_name).ascore(stripped, judge)
            va = a.value if isinstance(a.value, (int, float)) else None
            vb = b.value if isinstance(b.value, (int, float)) else None
            if va is not None:
                with_resp.append(va)
            if vb is not None:
                without_resp.append(vb)
            flag = "  <-- 分数改变" if va != vb else ""
            print(f"  行{i}: 带response={va}  去掉response={vb}{flag}")

        ma = sum(with_resp) / len(with_resp) if with_resp else None
        mb = sum(without_resp) / len(without_resp) if without_resp else None
        print(f"\n  均值  带response={ma}   去掉response={mb}   RAGAS=1.0")
        if ma is not None and mb is not None:
            print(f"  差异  {round(mb - ma, 4)}（去掉 response 后的变化）")
            changed = sum(1 for x, y in zip(with_resp, without_resp) if x != y)
            print(f"  {changed}/{len(with_resp)} 行分数因 response 是否可见而改变")


if __name__ == "__main__":
    asyncio.run(main())
