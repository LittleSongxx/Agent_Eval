"""一致性统计的防回归测试：kappa 点估计口径 + bootstrap 置信区间。

这个文件锁两件事：

1. **重构不能改数字。** kappa 公式从 `api/report.py` 和
   `scripts/calibration_compare.py` 两份复制实现收敛到 `core/agreement.py`。
   项目文档里已经写死了两个数字（校准前 0.4582、校准后 0.8175），重构后
   必须还是这两个数字，否则所有对外结论都得重算。所以测试直接用真实的
   2x2 混淆矩阵做夹具，而不是随手编几个 pass/fail。

2. **点估计不等于结论。** n=25 时 kappa=0.8175 落在 Landis-Koch 的
   "几乎完全一致"档，但 95% 区间下界远低于此。原先门禁只看点估计
   （`kappa < 0.7` 才提示），意味着"样本太少所以看起来很好"和
   "口径真的对齐了"在报告里长得一模一样。这正是本平台要检测的
   那类缺陷，出现在平台自己的统计口径上。
"""

import pytest

from app.core.agreement import (
    MIN_N_FOR_CI,
    bootstrap_kappa_ci,
    build_calibration_suggestion,
    cohens_kappa,
    interpret_kappa,
    paired_kappa_delta_ci,
)


def _pairs(both_pass: int, manual_pass_auto_fail: int, manual_fail_auto_pass: int, both_fail: int):
    """由 2x2 混淆矩阵展开成 (人工通过, 自动通过) 布尔对列表。"""
    return (
        [(True, True)] * both_pass
        + [(True, False)] * manual_pass_auto_fail
        + [(False, True)] * manual_fail_auto_pass
        + [(False, False)] * both_fail
    )


# 校准前（任务 10，阈值 0.5）：25 行 7 处分歧，其中 6 处自动过松。
# 该矩阵复算出的 kappa 必须等于文档中已发布的 0.4582。
BEFORE_CALIBRATION = _pairs(both_pass=10, manual_pass_auto_fail=1, manual_fail_auto_pass=6, both_fail=8)
# 校准后（阈值 0.7 + answer_relevancy criteria 修订）：25 行 2 处分歧，
# 复算 kappa 必须等于已发布的 0.8175。
AFTER_CALIBRATION = _pairs(both_pass=7, manual_pass_auto_fail=0, manual_fail_auto_pass=2, both_fail=16)


class TestCohensKappa:
    def test_reproduces_published_before_calibration_kappa(self):
        """校准前 kappa 必须仍是 0.4582——文档和简历都引用了这个数字。"""
        result = cohens_kappa(BEFORE_CALIBRATION)
        assert result["kappa"] == 0.4582
        assert result["total"] == 25
        assert result["agreement_rate"] == 0.72
        assert result["manual_fail_auto_pass"] == 6  # 自动过松 6 处

    def test_reproduces_published_after_calibration_kappa(self):
        """校准后 kappa 必须仍是 0.8175。"""
        result = cohens_kappa(AFTER_CALIBRATION)
        assert result["kappa"] == 0.8175
        assert result["total"] == 25
        assert result["agreement_rate"] == 0.92

    def test_matches_legacy_inline_formula(self):
        """与重构前 report.py 里的内联公式逐位一致。

        这里刻意把旧实现重写一遍作为参照，而不是相信"我抄对了"。
        重构一个已发布数字的计算路径，唯一可信的验收是双实现比对。
        """
        for pairs in (BEFORE_CALIBRATION, AFTER_CALIBRATION):
            both_pass = sum(1 for m, a in pairs if m and a)
            manual_pass_auto_fail = sum(1 for m, a in pairs if m and not a)
            manual_fail_auto_pass = sum(1 for m, a in pairs if not m and a)
            both_fail = sum(1 for m, a in pairs if not m and not a)
            total = len(pairs)
            observed = (both_pass + both_fail) / total
            expected = (
                (both_pass + manual_pass_auto_fail) * (both_pass + manual_fail_auto_pass)
                + (manual_fail_auto_pass + both_fail) * (manual_pass_auto_fail + both_fail)
            ) / (total * total)
            legacy = round((observed - expected) / (1.0 - expected), 4)
            assert cohens_kappa(pairs)["kappa"] == legacy

    def test_perfect_agreement_is_one(self):
        pairs = _pairs(both_pass=12, manual_pass_auto_fail=0, manual_fail_auto_pass=0, both_fail=13)
        assert cohens_kappa(pairs)["kappa"] == 1.0

    def test_undefined_when_no_variance(self):
        """两侧全判通过时 kappa 无定义，必须返回 None 而不是凑一个 1.0。

        期望一致率为 1.0，公式分母为 0。返回 1.0 会把"没有信息"
        伪装成"完全一致"——一个只会输出 pass 的裁判会拿到满分。
        """
        pairs = _pairs(both_pass=25, manual_pass_auto_fail=0, manual_fail_auto_pass=0, both_fail=0)
        result = cohens_kappa(pairs)
        assert result["kappa"] is None
        assert result["agreement_rate"] == 1.0  # 裸一致率仍然是 100%，对比之下更能说明问题

    def test_empty_input_does_not_crash(self):
        result = cohens_kappa([])
        assert result["kappa"] is None
        assert result["total"] == 0


class TestInterpretKappa:
    @pytest.mark.parametrize(
        "kappa,band",
        [
            (-0.2, "差"),
            (0.10, "轻微"),
            (0.30, "一般"),
            (0.4582, "中等"),   # 校准前
            (0.70, "显著"),
            (0.8175, "几乎完全一致"),  # 校准后
            (1.0, "几乎完全一致"),
        ],
    )
    def test_landis_koch_bands(self, kappa, band):
        assert interpret_kappa(kappa) == band

    def test_none_passes_through(self):
        assert interpret_kappa(None) is None


class TestBootstrapCI:
    def test_returns_none_below_min_sample_size(self):
        """n < 10 不报区间：bootstrap 不能凭空造出信息。"""
        pairs = _pairs(both_pass=3, manual_pass_auto_fail=1, manual_fail_auto_pass=1, both_fail=3)
        assert len(pairs) < MIN_N_FOR_CI
        assert bootstrap_kappa_ci(pairs) is None

    def test_deterministic_across_calls(self):
        """同一份数据两次调用必须得到完全相同的区间。

        区间随刷新而变的报告没法用来做决策，也没法写进文档。
        """
        first = bootstrap_kappa_ci(AFTER_CALIBRATION)
        second = bootstrap_kappa_ci(AFTER_CALIBRATION)
        assert first == second

    def test_interval_brackets_point_estimate(self):
        for pairs in (BEFORE_CALIBRATION, AFTER_CALIBRATION):
            ci = bootstrap_kappa_ci(pairs)
            assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]

    def test_n25_interval_is_too_wide_to_support_the_headline_band(self):
        """核心发现：0.8175 的"几乎完全一致"这个结论，n=25 撑不住。

        点估计落在最高档，但区间下界掉到更低的档位——能站得住的结论
        只有下界那一档。断言 spans_bands 就是断言"这件事必须被说出来"。
        """
        ci = bootstrap_kappa_ci(AFTER_CALIBRATION)
        assert ci["point"] == 0.8175
        assert ci["point_band"] == "几乎完全一致"
        assert ci["spans_bands"] is True
        assert ci["ci_low"] < 0.8175
        assert "~" in ci["ci_band"]  # 区间跨档时报的是范围而不是单一档位

    def test_reports_degenerate_resamples_instead_of_hiding_them(self):
        """退化重采样（kappa 无定义）必须被计数上报，不能当 0 或 1 混进分布。

        类别极不平衡时（这里 24:1），大量重采样会抽不到少数类，
        期望一致率变成 1.0。把这些当 0 会压低下界，当 1 会抬高上界，
        两种都是在编数据。计数本身就是"kappa 不适用于这份数据"的信号。
        """
        pairs = _pairs(both_pass=24, manual_pass_auto_fail=0, manual_fail_auto_pass=1, both_fail=0)
        ci = bootstrap_kappa_ci(pairs)
        assert ci is not None
        assert ci["degenerate_resamples"] > 0
        assert ci["valid_resamples"] + ci["degenerate_resamples"] == ci["resamples"]

    def test_provenance_fields_present(self):
        """区间必须自带方法/种子/次数，否则事后无法复现。"""
        ci = bootstrap_kappa_ci(AFTER_CALIBRATION)
        assert ci["method"] == "percentile bootstrap"
        assert ci["confidence"] == 0.95
        assert ci["resamples"] == 2000
        assert isinstance(ci["seed"], int)
        assert ci["n"] == 25

    def test_larger_sample_narrows_the_interval(self):
        """同样的一致性结构、4 倍样本量，区间必须变窄。

        这条是 bootstrap 实现正确性的健全性检查：如果区间宽度对 n
        不敏感，说明重采样写错了。
        """
        narrow = bootstrap_kappa_ci(
            _pairs(both_pass=28, manual_pass_auto_fail=0, manual_fail_auto_pass=8, both_fail=64)
        )
        wide = bootstrap_kappa_ci(AFTER_CALIBRATION)
        assert (narrow["ci_high"] - narrow["ci_low"]) < (wide["ci_high"] - wide["ci_low"])


class TestCalibrationSuggestion:
    def test_flags_low_point_estimate_as_criteria_problem(self):
        """点估计就没过线 → 提示改判分标准。"""
        ci = bootstrap_kappa_ci(BEFORE_CALIBRATION)
        msg = build_calibration_suggestion(0.4582, ci, 25)
        assert msg is not None
        assert "复核判分标准" in msg

    def test_flags_passing_point_with_failing_lower_bound_as_sample_size_problem(self):
        """点估计过线但下界没过 → 提示补样本，而不是改口径。

        这两种情况的处理动作完全不同：一个是去标更多数据，
        一个是去改 prompt。旧门禁只看点估计，第二种情况直接静默放过。
        """
        ci = bootstrap_kappa_ci(AFTER_CALIBRATION)
        assert ci["ci_low"] < 0.7  # 前提：这份数据的下界确实没过线
        msg = build_calibration_suggestion(0.8175, ci, 25)
        assert msg is not None
        assert "置信区间下界" in msg
        assert "扩充人工标注样本量" in msg
        assert "复核判分标准" not in msg  # 不该误导成口径问题

    def test_silent_when_lower_bound_also_passes(self):
        ci = bootstrap_kappa_ci(
            _pairs(both_pass=40, manual_pass_auto_fail=0, manual_fail_auto_pass=1, both_fail=40)
        )
        assert ci["ci_low"] >= 0.7
        assert build_calibration_suggestion(ci["point"], ci, 81) is None

    def test_silent_below_min_sample_size(self):
        """样本太少时不出提示：此时任何结论都不成立，包括"口径有问题"。"""
        assert build_calibration_suggestion(0.3, None, 5) is None

    def test_none_kappa_produces_no_suggestion(self):
        assert build_calibration_suggestion(None, None, 25) is None


# ---------------------------------------------------------------------------
# 配对 bootstrap 的夹具：**逐行对齐**的真实数据（来自 eval_platform.db）
#
# 这里刻意不复用上面 `_pairs()` 的 2x2 展开夹具，原因见
# TestPairedKappaDelta.test_rejects_2x2_expanded_fixtures_as_misaligned：
# 按混淆矩阵展开会打散行对应关系，而配对检验的全部前提就是行对应。
# 三列自动判定对着同一批 25 条人工标注（人工标注只存在于任务 10 的行上）：
#   AUTO_T10 / AUTO_T9  = 校准前同一配置的两次完整跑（任务 9 是意外多跑的一轮）
#   AUTO_T13            = 校准后（阈值 0.7 + answer_relevancy criteria 修订）
# ---------------------------------------------------------------------------
_T, _F = True, False
MANUAL_25 = [_T] * 9 + [_F] * 16
AUTO_TASK10 = [_T, _T, _T, _T, _T, _T, _T, _F, _T,
               _F, _F, _T, _F, _F, _T, _F, _F, _T, _T, _F, _T, _T, _F, _F, _F]
AUTO_TASK13 = [_T, _T, _T, _T, _T, _T, _F, _F, _T,
               _F, _F, _F, _F, _F, _F, _F, _F, _F, _F, _F, _F, _F, _F, _F, _F]
# 与 TASK10 只差第 12 行——同一个配置、同一批样本，重跑一次就变了一行
AUTO_TASK9 = [_T, _T, _T, _T, _T, _T, _T, _F, _T,
              _F, _F, _T, _T, _F, _T, _F, _F, _T, _T, _F, _T, _T, _F, _F, _F]

BEFORE_ALIGNED = list(zip(MANUAL_25, AUTO_TASK10))
AFTER_ALIGNED = list(zip(MANUAL_25, AUTO_TASK13))
BEFORE_ALIGNED_RERUN = list(zip(MANUAL_25, AUTO_TASK9))


class TestPairedKappaDelta:
    """配对 bootstrap：检验的是两个 kappa 之**差**，不是两个区间的位置关系。

    这个类存在的原因是一个我自己犯过的统计错误。原先 `calibration_compare.py`
    的结论行是"两区间是否重叠"——重叠就宣布"提升尚未超出抽样噪声"。那是错的：
    重叠不蕴含不显著，而且这两个 kappa 对着同一批标注算，属于配对数据，
    共同的行难度波动在相减时会被消掉，所以配对差的区间可以窄得多。
    """

    def test_aligned_fixtures_reproduce_the_published_point_kappas(self):
        """前提校验：夹具本身必须复算出文档里的三个 kappa，否则后面全是空谈。"""
        assert cohens_kappa(BEFORE_ALIGNED)["kappa"] == 0.4582
        assert cohens_kappa(AFTER_ALIGNED)["kappa"] == 0.8175
        assert cohens_kappa(BEFORE_ALIGNED_RERUN)["kappa"] == 0.3939

    def test_reproduces_published_paired_delta_and_interval(self):
        """文档里写死的四个数字必须复现：Δ、双侧区间、单侧下界、P(Δ>0)。

        这四个数字散落在 9 份文档里（一页纸/STAR/HANDOFF/元评价×2/踩坑记录…）。
        它们变了就意味着所有对外结论要重写，所以钉在测试里。
        """
        r = paired_kappa_delta_ci(BEFORE_ALIGNED, AFTER_ALIGNED)
        assert r["delta"] == 0.3593
        assert r["baseline_kappa"] == 0.4582
        assert r["calibrated_kappa"] == 0.8175
        assert (r["ci_low"], r["ci_high"]) == (-0.0129, 0.6716)
        assert r["excludes_zero"] is False        # 双侧下界压着 0——不能说"已验证"
        assert r["one_sided_ci_low"] == 0.0565
        assert r["one_sided_excludes_zero"] is True   # 单侧为正——方向站得住
        assert r["p_delta_gt_zero"] == 0.9665

    def test_delta_equals_difference_of_point_estimates(self):
        """点估计的差必须就是两个 kappa 相减，不受重采样影响。"""
        r = paired_kappa_delta_ci(BEFORE_ALIGNED, AFTER_ALIGNED)
        assert r["delta"] == round(r["calibrated_kappa"] - r["baseline_kappa"], 4)

    def test_paired_interval_is_narrower_than_naive_independent_difference(self):
        """配对的意义就在这一条：同一份数据，配对区间显著窄于"两个独立区间相减"。

        naive 做法（把两个独立 CI 的端点相减）给出宽度 1.1117，配对给出 0.6845。
        换句话说，用独立区间去看"重不重叠"会白扔掉近 40% 的检验效力——
        这正是原先那个错误判据的代价。
        """
        before_ci = bootstrap_kappa_ci(BEFORE_ALIGNED)
        after_ci = bootstrap_kappa_ci(AFTER_ALIGNED)
        # 前提：两个独立区间确实重叠（原判据据此宣布"不显著"）
        assert before_ci["ci_high"] >= after_ci["ci_low"]

        naive_width = (after_ci["ci_high"] - before_ci["ci_low"]) - (
            after_ci["ci_low"] - before_ci["ci_high"]
        )
        paired = paired_kappa_delta_ci(BEFORE_ALIGNED, AFTER_ALIGNED)
        paired_width = paired["ci_high"] - paired["ci_low"]
        assert paired_width < naive_width
        assert round(paired_width, 4) == 0.6845

    def test_baseline_choice_flips_the_significance_verdict(self):
        """**最锋利的一条**：换一个同配置重跑的基线，显著性判定就翻。

        任务 9 与任务 10 是同一个校准前配置的两次完整跑，pass/fail 只差 1 行
        （第 12 行）。就这 1 行把基线 kappa 从 0.4582 挪到 0.3939，
        并让双侧区间从"含 0"变成"排除 0"。文档里用的恰好是更保守的基线，
        属于运气——这条测试把这份运气变成显式记录，避免后人（包括我）
        再把"排除 0"当成稳定结论。
        """
        with_task10 = paired_kappa_delta_ci(BEFORE_ALIGNED, AFTER_ALIGNED)
        with_task9 = paired_kappa_delta_ci(BEFORE_ALIGNED_RERUN, AFTER_ALIGNED)

        assert with_task10["excludes_zero"] is False
        assert with_task9["excludes_zero"] is True      # 判定翻转
        assert (with_task9["ci_low"], with_task9["ci_high"]) == (0.0565, 0.722)
        assert with_task9["delta"] == 0.4236

    def test_same_config_rerun_noise_floor_is_a_sizeable_share_of_the_effect(self):
        """同配置重跑的裁判噪声底 0.0643，占声称效应量 0.3593 的 17.9%。

        这个比例是"方向明确、量级待定"这句话的定量依据：当同一配置重跑两次
        就能差 0.064 kappa 时，声称 0.359 的提升"已被证明"是站不住的。
        """
        k10 = cohens_kappa(BEFORE_ALIGNED)["kappa"]
        k9 = cohens_kappa(BEFORE_ALIGNED_RERUN)["kappa"]
        noise_floor = round(abs(k10 - k9), 4)
        assert noise_floor == 0.0643

        # 两轮只在一行上判定不同
        diff_rows = [i for i in range(25) if AUTO_TASK10[i] != AUTO_TASK9[i]]
        assert diff_rows == [12]

        effect = paired_kappa_delta_ci(BEFORE_ALIGNED, AFTER_ALIGNED)["delta"]
        assert round(noise_floor / effect, 3) == 0.179

    def test_rejects_2x2_expanded_fixtures_as_misaligned(self):
        """按 2x2 混淆矩阵展开的夹具必须被拒绝——这是我实际踩到的坑。

        `BEFORE_CALIBRATION` / `AFTER_CALIBRATION` 两个夹具长度都是 25、
        各自的 kappa 也都对（0.4582 / 0.8175），拿去做配对 bootstrap
        会返回一个**像样但错误**的区间 [-0.0657, 0.7558]（真值 [-0.0129, 0.6716]）。
        原因：展开顺序把行对应关系打散了，第 i 项不再是同一行。

        一个测不出对齐错误的配对检验比没有检验更危险——它给的区间还更宽更
        "保守"，没人会怀疑。所以逐行校验人工列是硬校验。
        """
        with pytest.raises(ValueError, match="逐行对齐"):
            paired_kappa_delta_ci(BEFORE_CALIBRATION, AFTER_CALIBRATION)

    def test_error_message_names_the_offending_row_and_likely_cause(self):
        """报错必须指出第几行不对齐、以及最可能的原因，否则调用方无从下手。"""
        with pytest.raises(ValueError) as exc:
            paired_kappa_delta_ci(BEFORE_CALIBRATION, AFTER_CALIBRATION)
        msg = str(exc.value)
        assert "第 7 行" in msg          # 两个展开夹具首次分叉的位置
        assert "2×2" in msg              # 指向真正的成因

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="行数一致"):
            paired_kappa_delta_ci(BEFORE_ALIGNED, AFTER_ALIGNED[:20])

    def test_returns_none_below_min_sample_size(self):
        assert paired_kappa_delta_ci(BEFORE_ALIGNED[:8], AFTER_ALIGNED[:8]) is None

    def test_deterministic_across_calls(self):
        """固定种子：文档引用的区间必须任何时候都能复现。"""
        assert paired_kappa_delta_ci(BEFORE_ALIGNED, AFTER_ALIGNED) == paired_kappa_delta_ci(
            BEFORE_ALIGNED, AFTER_ALIGNED
        )

    def test_identical_inputs_give_zero_delta_and_zero_width_interval(self):
        """同一份判定与自己比：Δ 恒为 0，区间退化成一点，且不得报"排除 0"。"""
        r = paired_kappa_delta_ci(BEFORE_ALIGNED, BEFORE_ALIGNED)
        assert r["delta"] == 0.0
        assert (r["ci_low"], r["ci_high"]) == (0.0, 0.0)
        assert r["excludes_zero"] is False
        assert r["p_delta_gt_zero"] == 0.0

    def test_doubling_the_sample_narrows_the_interval_and_excludes_zero(self):
        """n=50 外推：同样的联合模式复制一份，双侧下界离开 0。

        这是文档里"扩样到 n=50"的**定量依据**（不是"样本越多越好"的泛泛之谈）。
        注意这是**功效计算**：它假设观测到的 25 行联合模式就是总体模式，
        回答的是"若效应真如观测这么大，多少样本能测出来"，
        **不是**"效应确实这么大"的证据。
        """
        n50_before = BEFORE_ALIGNED * 2
        n50_after = AFTER_ALIGNED * 2
        r25 = paired_kappa_delta_ci(BEFORE_ALIGNED, AFTER_ALIGNED)
        r50 = paired_kappa_delta_ci(n50_before, n50_after)

        assert r50["n"] == 50
        assert r50["delta"] == r25["delta"]          # 点估计不变，只有区间收窄
        assert (r50["ci_high"] - r50["ci_low"]) < (r25["ci_high"] - r25["ci_low"])
        assert r50["excludes_zero"] is True
        assert (r50["ci_low"], r50["ci_high"]) == (0.1166, 0.5884)

    def test_one_sided_bound_is_never_looser_than_two_sided(self):
        """单侧下界必须 >= 双侧下界，否则分位点算错了。"""
        for base in (BEFORE_ALIGNED, BEFORE_ALIGNED_RERUN):
            r = paired_kappa_delta_ci(base, AFTER_ALIGNED)
            assert r["one_sided_ci_low"] >= r["ci_low"]

    def test_counts_degenerate_resamples_instead_of_imputing_them(self):
        """任一侧 kappa 无定义的重采样整份丢弃并计数，不能给缺失侧补 0。

        补 0 会把"这个差不存在"伪装成"差很大"。类别极不平衡（24:1）时
        退化比例可达 1/3 以上，这个计数本身就是"区间不可信"的信号。
        """
        manual = [_T] * 24 + [_F]
        base = [_T] * 23 + [_F, _T]
        cal = [_T] * 24 + [_F]
        r = paired_kappa_delta_ci(list(zip(manual, base)), list(zip(manual, cal)))
        assert r is not None
        assert r["degenerate_resamples"] > 0
        assert r["valid_resamples"] + r["degenerate_resamples"] == r["resamples"]

    def test_provenance_fields_present(self):
        """区间自带方法/种子/次数/n，否则事后无法复现文档里的数字。"""
        r = paired_kappa_delta_ci(BEFORE_ALIGNED, AFTER_ALIGNED)
        assert r["method"] == "paired percentile bootstrap (同一 resample 重算两侧 kappa)"
        assert r["confidence"] == 0.95
        assert r["resamples"] == 2000
        assert isinstance(r["seed"], int)
        assert r["n"] == 25
