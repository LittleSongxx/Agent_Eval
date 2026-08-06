"""检索实验脚本的回归测试，重点是防"闭式循环"回归。

上一版实验用 ``noise_ids + reference_ids`` 构造检索结果，MRR/HitRate 是
列表拼接方式的闭式解，与数据集内容、检索器质量无关。本文件的核心守门测试
``test_ranking_is_data_dependent`` 专门防这类回归：把金标正文换成无关文本后，
HitRate 必须掉下来——同义反复的实现过不了这条。
"""

from __future__ import annotations

import pytest

from scripts.retrieval_noise_experiment import (
    RetrievalScenario,
    _hit_rate_at_k,
    _mrr,
    _strip_fixture_preamble,
    run_scenario,
)

KB = {
    "k1": "云购商城支持7天无理由退货、15天换货、30天维修的三级售后保障，自签收次日零时起算。",
    "k2": "生鲜食品质量问题需在签收24小时内提供照片和视频凭证，超时无法受理。",
    "k3": "大件家具类商品支持预约上门取件，海外购商品无理由退货需用户承担国际运费。",
}

ROWS = [
    {
        "user_input": "云购商城的三级售后保障分别是什么？",
        "reference": "7天无理由退货、15天换货、30天维修。",
        "reference_context_ids": ["k1"],
        "generation_meta": {"kind": "correct"},
    },
    {
        "user_input": "生鲜食品质量问题需要提供什么凭证？",
        "reference": "照片和视频凭证。",
        "reference_context_ids": ["k2"],
        "generation_meta": {"kind": "correct"},
    },
    {
        "user_input": "大件家具退货如何取件？",
        "reference": "预约上门取件。",
        "reference_context_ids": ["k3"],
        "generation_meta": {"kind": "correct"},
    },
]

SCENARIO = RetrievalScenario("test", 0, 0)


def _run(kb: dict[str, str]) -> dict:
    return run_scenario(ROWS, kb, [], [], SCENARIO, k=5)


class TestRunScenario:
    def test_retrieves_gold_at_rank_one(self):
        result = _run(KB)
        assert result["hit_rate_at_1"] == 1.0
        assert result["mrr"] == 1.0
        assert result["hit_rate_at_5"] == 1.0

    def test_ranking_is_data_dependent(self):
        """守门测试：金标正文换成无关文本，HitRate 必须归零。

        旧版"noise_ids + reference_ids 拼列表"的构造在正/乱两种正文下
        给出完全相同的结果，过不了这条。
        """
        scrambled = {key: "手机采用六轴防抖主摄模组。手冲咖啡水温 92 度。Python 列表推导式更快。" for key in KB}
        result = _run(scrambled)
        assert result["hit_rate_at_1"] == 0.0
        assert result["mrr"] == 0.0

    def test_unrelated_distractors_do_not_hurt(self):
        distractors = [
            "手机采用六轴防抖的主摄模组，等效焦距 24mm。",
            "手冲咖啡的水温建议控制在 92 到 94 摄氏度之间。",
            "云南夏季适合去香格里拉和泸沽湖，紫外线强烈。",
        ]
        scenario = RetrievalScenario("with-distractors", 3, 0)
        result = run_scenario(ROWS, KB, distractors, [], scenario, k=5)
        assert result["hit_rate_at_1"] == 1.0, "无关干扰不应挤掉金标（主题可分）"

    def test_retrieved_ids_only_from_corpus(self):
        """防作弊完整性：retrieved 必须 ⊆ 语料，不得混入 reference_context_ids。"""
        result = _run(KB)
        corpus_ids = set(KB) | {f"distractor-{i}" for i in range(0)} | {f"hardneg-{i}" for i in range(0)}
        for detail in result["detail"]:
            assert set(detail["retrieved"]) <= corpus_ids

    def test_gold_id_not_appendable_in_disguise(self):
        """即使检索结果为空，也不得回填金标 ID（否则同样是闭式作弊）。"""
        scrambled = {key: "与查询完全无关的内容。" for key in KB}
        result = _run(scrambled)
        for detail in result["detail"]:
            assert detail["retrieved"] == []
            assert detail["rank_of_gold"] is None


class TestMetricSemantics:
    def test_hit_rate_none_on_empty_gold(self):
        assert _hit_rate_at_k(["a", "b"], []) is None
        assert _mrr(["a", "b"], []) is None

    def test_hit_rate_k_cutoff(self):
        assert _hit_rate_at_k(["a", "b"], ["z"], k=1) == 0.0
        assert _hit_rate_at_k(["a", "b"], ["z"], k=2) == 0.0
        assert _hit_rate_at_k(["a", "b"], ["b"], k=1) == 0.0, "金标在位置 2，top-1 不含它"
        assert _hit_rate_at_k(["a", "b"], ["b"], k=2) == 1.0

    def test_mrr_rank_reciprocal(self):
        assert _mrr(["x", "y", "z"], ["z"]) == pytest.approx(1 / 3)
        assert _mrr(["x", "y"], ["z"]) == 0.0
        assert _mrr(["z"], ["z"]) == 1.0


class TestStripFixturePreamble:
    def test_drops_markdown_and_prose_blocks(self):
        chunks = [
            "# 标题说明",
            "---",
            "本文件是干扰语料，用法见文档。",
            "选材原则：主题完全无关。",
            "手机采用六轴防抖的主摄模组。",
        ]
        kept = _strip_fixture_preamble(chunks)
        assert kept == ["手机采用六轴防抖的主摄模组。"]

    def test_keeps_all_content_blocks(self):
        chunks = ["手机防抖模组。", "咖啡水温建议 92 度。"]
        assert _strip_fixture_preamble(chunks) == chunks
