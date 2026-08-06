"""与 RAGAS 的对照实验：验证平台原生 Judge 指标的打分可信度。

在同一批数据集样本上，分别用本平台的原生指标和 RAGAS 计算
faithfulness / answer_relevancy / context_precision / context_recall，
再计算两组分数的 Pearson / Spearman 相关性。

相关性高说明平台原生指标与业界主流框架在"好坏判断"上方向一致，
为自研指标提供实证背书（面试/技术方案评审中的量化产出）。

用法：
    cd backend
    python -m scripts.compare_with_ragas --dataset "RAG 示例数据集" --limit 20

可选参数：
    --dry-run        不调用真实 LLM，用固定分数验证脚本链路
    --limit N        最多评测 N 条样本
    --metric NAME    只对比单个指标（faithfulness / answer_relevancy /
                     context_precision / context_recall）

依赖（可选用）：pip install ragas==0.3.7 langchain-openai
RAGAS 侧 answer_relevancy 需要 embedding 模型（OpenAI 兼容 /embeddings
接口），默认 text-embedding-v3，可用环境变量 RAGAS_EMBEDDING_MODEL 覆盖。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import typing as t
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger("compare_with_ragas")

COMPARABLE_METRICS = [
    "faithfulness",
    "faithfulness_claim",
    "answer_relevancy",
    "answer_relevancy_generative",
    "context_precision",
    "context_recall",
]
# answer_relevancy（RAGAS 侧）需要 embedding 模型；平台生成式指标同样依赖 embedding
METRICS_REQUIRING_EMBEDDINGS = {"answer_relevancy", "answer_relevancy_generative"}


def _install_ragas_shim() -> None:
    """ragas 0.3.x 强依赖 langchain-community 已移除的 VertexAI 模块。

    把项目内的最小 shim（scripts/_shims）前置到 sys.path，整体遮蔽
    langchain_community：ragas 只用其中的 VertexAI 占位类（LLM 工厂注册表
    引用，注入自定义 wrapper 后不会被实例化），其余全走 langchain-core /
    langchain-openai，因此遮蔽是安全的。本脚本（以及平台本身）不依赖
    langchain_community 的任何真实功能。
    详见 scripts/_shims/langchain_community/__init__.py。
    """
    shim_dir = Path(__file__).resolve().parent / "_shims"
    if shim_dir not in sys.path:
        sys.path.insert(0, str(shim_dir))


class _FakeJudge:
    """Dry-run 用：返回固定分数，验证脚本链路本身。"""

    async def judge_json(self, payload):
        return {"score": 0.75, "reason": "dry-run 固定分数"}


def _load_rows(dataset_name: str | None, limit: int) -> list[dict[str, t.Any]]:
    from app.core.database import SessionLocal
    from app.models.dataset import Dataset, DatasetRow

    db = SessionLocal()
    try:
        query = db.query(DatasetRow)
        if dataset_name:
            dataset = db.query(Dataset).filter(Dataset.name == dataset_name).first()
            if dataset is None:
                raise SystemExit(f"数据集不存在: {dataset_name}")
            query = query.filter(DatasetRow.dataset_id == dataset.id)
        rows = query.order_by(DatasetRow.row_index).limit(limit).all()
        if not rows:
            raise SystemExit("数据集为空，请先导入或生成数据")
        return [dict(row.data or {}) for row in rows]
    finally:
        db.close()


def _build_platform_metric(name: str):
    """按平台内置指标规格构造原生指标实例。"""
    from app.core.evaluation_engine import build_metric

    metric_def = SimpleNamespace(
        id=0,
        name=name,
        display_name=name,
        metric_type=f"builtin_{name}",
        config={},
        required_fields=[],
        category="rag",
        is_builtin=True,
    )
    _kind, metric = build_metric(metric_def, llm=None)
    return metric


async def _run_platform_scores(
    rows: list[dict[str, t.Any]],
    metrics: list[str],
    llm_config,
    dry_run: bool,
) -> dict[str, list[float]]:
    from app.core.evaluation_engine import OpenAIJudgeClient

    judge = _FakeJudge() if dry_run else OpenAIJudgeClient(llm_config)
    platform_scores: dict[str, list[float]] = {name: [] for name in metrics}
    for idx, row in enumerate(rows, start=1):
        for name in metrics:
            metric = _build_platform_metric(name)
            try:
                result = await metric.ascore(row, judge)
            except Exception as exc:  # 单条失败不中断整体对照（与 RAGAS 侧对齐）
                print(f"  ⚠ 平台 [{name}] 行 {idx} 失败: {str(exc)[:200]}")
                platform_scores[name].append(None)
                continue
            if isinstance(result.value, (int, float)) and result.value is not None:
                platform_scores[name].append(float(result.value))
            else:
                platform_scores[name].append(None)
        _write_checkpoint("platform", platform_scores)
        print(f"  平台指标 进度 {idx}/{len(rows)}")
    return platform_scores


async def _run_ragas_scores(
    rows: list[dict[str, t.Any]],
    metrics: list[str],
    llm_config,
    dry_run: bool,
) -> dict[str, list[float]]:
    if dry_run:
        return {name: [0.75] * len(rows) for name in metrics}

    _install_ragas_shim()
    try:
        from ragas import SingleTurnSample
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError:
        print("未安装 ragas，跳过 RAGAS 对照（pip install ragas==0.3.7 langchain-openai）")
        return {}

    # 平台指标 → RAGAS 指标的映射：断言级忠实度对标 RAGAS faithfulness（同为
    # claim 级判定），生成式相关性对标 RAGAS answer_relevancy（同为生成式判定）
    ragas_metric_by_name = {
        "faithfulness": faithfulness,
        "faithfulness_claim": faithfulness,
        "answer_relevancy": answer_relevancy,
        "answer_relevancy_generative": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }

    try:
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(
            base_url=llm_config.api_base_url,
            api_key=llm_config.api_key,
            model=llm_config.model_name,
            temperature=0.01,
            # Qwen3 系模型默认开启 thinking，而 RAGAS 的 answer_relevancy 会
            # 一次请求 n=strictness 个补全，thinking 模式会被网关拒绝；
            # 显式关闭以对齐标准 RAGAS 评测配置
            extra_body={"enable_thinking": False},
        )
        wrapper = LangchainLLMWrapper(chat)
        for ragas_metric in ragas_metric_by_name.values():
            ragas_metric.llm = wrapper
    except ImportError:
        print("未安装 langchain-openai，跳过 RAGAS 对照（pip install ragas==0.3.7 langchain-openai）")
        return {}

    # answer_relevancy 需要 embedding 模型（OpenAI 兼容 /embeddings）
    needed_embeddings = METRICS_REQUIRING_EMBEDDINGS & set(metrics)
    if needed_embeddings:
        try:
            from langchain_openai import OpenAIEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper

            embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(
                    base_url=llm_config.api_base_url,
                    api_key=llm_config.api_key,
                    model=os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-v3"),
                    # 默认的 tiktoken 长度安全路径会因模型名不在白名单而破坏 input，
                    # 对短文本评测样本直接走直传路径
                    check_embedding_ctx_length=False,
                )
            )
            answer_relevancy.embeddings = embeddings
        except Exception as exc:
            print(f"⚠ embedding 配置失败，answer_relevancy 将无法打分: {str(exc)[:200]}")
            ragas_metric_by_name.pop("answer_relevancy", None)

    ragas_scores: dict[str, list[float]] = {name: [] for name in metrics}
    for idx, row in enumerate(rows, start=1):
        sample = SingleTurnSample(
            user_input=row.get("user_input"),
            response=row.get("response"),
            retrieved_contexts=row.get("retrieved_contexts") or [],
            reference=row.get("reference"),
        )
        for name in metrics:
            ragas_metric = ragas_metric_by_name.get(name)
            if ragas_metric is None:
                ragas_scores[name].append(None)
                continue
            try:
                result = ragas_metric.single_turn_score(sample)
                if asyncio.iscoroutine(result):
                    result = await result
                ragas_scores[name].append(float(result))
            except Exception as exc:  # 单条失败不中断整体对照
                print(f"  ⚠ RAGAS [{name}] 行 {idx} 失败: {str(exc)[:200]}")
                ragas_scores[name].append(None)
        _write_checkpoint("ragas", ragas_scores)
        print(f"  RAGAS 进度 {idx}/{len(rows)}")
    return ragas_scores


CHECKPOINT_PATH = "ragas_comparison_checkpoint.json"


def _write_checkpoint(side: str, scores: dict[str, list[t.Optional[float]]]) -> None:
    """每完成一行就落盘一次。

    长跑（25 行 × 6 指标，两侧合计数百次 LLM 调用）中途若因网关限流或配额
    耗尽中断，已完成的分数不会丢失，可直接从检查点算相关性。
    """
    try:
        existing: dict[str, t.Any] = {}
        if Path(CHECKPOINT_PATH).exists():
            with open(CHECKPOINT_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        existing[side] = scores
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # 检查点本身失败不能影响主流程
        logger.warning("写检查点失败: %s", exc)


def _load_reused_ragas_scores(
    report_path: str, metrics: list[str], expected_rows: int
) -> dict[str, list[t.Optional[float]]]:
    """从既有报告读取 RAGAS 逐行分数，避免重复消耗配额。

    使用前提：RAGAS 侧的输入未发生变化。RAGAS 只从 user_input / response /
    retrieved_contexts / reference 构造 SingleTurnSample，因此平台侧 payload
    的改动不影响 RAGAS 分数，可以安全复用；但若数据集本身变了就必须重跑，
    因此这里强校验行数一致。
    """
    path = Path(report_path)
    if not path.exists():
        raise SystemExit(f"复用的报告不存在: {report_path}")
    with open(path, encoding="utf-8") as f:
        prior = json.load(f)

    prior_metrics = prior.get("metrics") or {}
    reused: dict[str, list[t.Optional[float]]] = {}
    for name in metrics:
        scores = (prior_metrics.get(name) or {}).get("ragas_scores")
        if scores is None:
            raise SystemExit(f"报告 {report_path} 中缺少指标 {name} 的 ragas_scores，无法复用")
        if len(scores) != expected_rows:
            raise SystemExit(
                f"报告 {report_path} 中 {name} 的行数为 {len(scores)}，"
                f"与当前样本数 {expected_rows} 不一致，必须重跑 RAGAS"
            )
        reused[name] = scores
    return reused


def _correlation(pairs: list[tuple[float, float]]) -> dict[str, float]:
    """计算 Pearson / Spearman 相关性（pandas 自带，兼容缺失值）。"""
    if len(pairs) < 3:
        return {"pearson": None, "spearman": None, "n": len(pairs)}
    import pandas as pd

    frame = pd.DataFrame(pairs, columns=["platform", "ragas"]).dropna()
    if len(frame) < 3:
        return {"pearson": None, "spearman": None, "n": len(frame)}
    return {
        "pearson": round(float(frame["platform"].corr(frame["ragas"], method="pearson")), 4),
        "spearman": round(float(frame["platform"].corr(frame["ragas"], method="spearman")), 4),
        "n": len(frame),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="平台原生指标 vs RAGAS 对照实验")
    parser.add_argument("--dataset", default=None, help="数据集名称（默认全部）")
    parser.add_argument("--limit", type=int, default=10, help="最多评测样本数")
    parser.add_argument(
        "--metric",
        choices=COMPARABLE_METRICS,
        action="append",
        default=None,
        help="只对比指定指标，可重复传入（默认全部）",
    )
    parser.add_argument("--dry-run", action="store_true", help="用固定分数验证链路，不调用 LLM")
    parser.add_argument(
        "--reuse-ragas",
        default=None,
        metavar="REPORT_JSON",
        help=(
            "复用既有报告里的 RAGAS 逐行分数，跳过 RAGAS 侧重跑。"
            "仅在平台侧改动、RAGAS 侧输入未变时可用："
            "RAGAS 只读 user_input/response/retrieved_contexts/reference，"
            "不受平台 payload 改动影响。"
        ),
    )
    parser.add_argument(
        "--out", default="ragas_comparison_report.json", metavar="PATH", help="报告输出路径"
    )
    args = parser.parse_args()

    metrics = args.metric if args.metric else COMPARABLE_METRICS

    print(f"加载样本: dataset={args.dataset or '全部'} limit={args.limit}")
    rows = _load_rows(args.dataset, args.limit)
    print(f"样本数: {len(rows)}，指标: {', '.join(metrics)}")

    # 平台 Judge 配置：复用数据库里默认的 LLM 配置
    from app.core.database import SessionLocal
    from app.models.llm_config import LLMConfig

    db = SessionLocal()
    try:
        llm_config = (
            db.query(LLMConfig).filter(LLMConfig.is_default == True).first() or db.query(LLMConfig).first()  # noqa: E712
        )
    finally:
        db.close()
    if llm_config is None:
        raise SystemExit("数据库中没有 LLM 配置，请先在前端配置 Judge LLM")

    # 复用 RAGAS 分数时先校验再跑平台侧：行数/指标不匹配必须立即失败，
    # 而不是白跑一轮平台指标（25 行 × 4 指标约 100 次 LLM 调用）后才发现。
    reused_ragas: dict[str, list[t.Optional[float]]] | None = None
    if args.reuse_ragas:
        reused_ragas = _load_reused_ragas_scores(args.reuse_ragas, metrics, len(rows))
        print(f"复用 RAGAS 逐行分数: {args.reuse_ragas}（跳过 RAGAS 重跑）")

    print("=" * 60)
    print("运行平台原生指标...")
    platform_scores = await _run_platform_scores(rows, metrics, llm_config, args.dry_run)

    if reused_ragas is not None:
        ragas_scores = reused_ragas
    else:
        print("运行 RAGAS 指标...")
        ragas_scores = await _run_ragas_scores(rows, metrics, llm_config, args.dry_run)

    print("=" * 60)
    print("对照结果（分值范围 0~1）")
    # 样本类型标签（correct / hallucinated / truncated 等），用于分层分析：
    # 整体相关性偏低时，可定位是哪一类样本造成的分歧
    kinds = [(row.get("generation_meta") or {}).get("kind") for row in rows]
    report: dict[str, t.Any] = {"rows": len(rows), "kinds": kinds, "metrics": {}}
    for name in metrics:
        platform_row_scores = platform_scores.get(name, [])
        ragas_row_scores = ragas_scores.get(name, [])
        platform_values = [v for v in platform_row_scores if v is not None]
        ragas_values = [v for v in ragas_row_scores if v is not None]
        platform_mean = sum(platform_values) / len(platform_values) if platform_values else None
        ragas_mean = sum(ragas_values) / len(ragas_values) if ragas_values else None
        pairs = list(zip(platform_row_scores, ragas_row_scores))
        corr = _correlation(pairs)
        print(f"\n[{name}]")
        print(f"  平台原生指标均值: {platform_mean}")
        print(f"  RAGAS 均值:       {ragas_mean}")
        print(f"  相关性:           Pearson={corr['pearson']} Spearman={corr['spearman']} (n={corr['n']})")

        # 按样本类型分组的均值对照
        by_kind: dict[str, t.Any] = {}
        for kind in sorted({k for k in kinds if k}):
            idxs = [i for i, k in enumerate(kinds) if k == kind]
            p_vals = [platform_row_scores[i] for i in idxs if i < len(platform_row_scores) and platform_row_scores[i] is not None]
            r_vals = [ragas_row_scores[i] for i in idxs if i < len(ragas_row_scores) and ragas_row_scores[i] is not None]
            by_kind[kind] = {
                "n": len(idxs),
                "platform_mean": round(sum(p_vals) / len(p_vals), 4) if p_vals else None,
                "ragas_mean": round(sum(r_vals) / len(r_vals), 4) if r_vals else None,
            }
            print(
                f"    ├ {kind:<14} n={len(idxs):<3} 平台={by_kind[kind]['platform_mean']} "
                f"RAGAS={by_kind[kind]['ragas_mean']}"
            )

        report["metrics"][name] = {
            "platform_mean": platform_mean,
            "ragas_mean": ragas_mean,
            **corr,
            "by_kind": by_kind,
            # 保留逐行原始分，便于复核与二次分析（无需重跑 LLM）
            "platform_scores": platform_row_scores,
            "ragas_scores": ragas_row_scores,
        }

    output = args.out
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {output}")
    print("解释: 相关系数 > 0.8 说明与 RAGAS 的排序判断高度一致；")
    print("      > 0.6 说明方向一致，可以结合人工抽检校准绝对分。")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
