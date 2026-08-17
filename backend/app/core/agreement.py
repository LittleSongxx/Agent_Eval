"""人工-自动一致性统计：Cohen's kappa + Bootstrap 置信区间。

为什么需要这个模块：
  kappa 的点估计原先在 `api/report.py` 和 `scripts/calibration_compare.py`
  各写了一遍（同一套公式复制两份），而且只报点估计。n=25 时"kappa=0.8175"
  这个数字本身的抽样不确定性极大——换 25 条样本很可能落到 0.55 或 0.95。
  只报点估计等于把一个区间伪装成一个定论，这正是这个项目要检测的那类问题。

  所以这里做三件事：
  1. 公式收敛到一处，报告接口和校准脚本共用，杜绝两处口径漂移；
  2. 用 percentile bootstrap 给出置信区间，让"n 太小"这件事显式可见；
  3. 门禁改用 CI 下界而非点估计——点估计过线不代表结论稳。

Bootstrap 采用固定随机种子：同一份数据任何时候查询都得到同一个区间。
区间随页面刷新而变的报告是没法拿去做决策的。
"""

from __future__ import annotations

import random
import typing as t

# 固定种子：保证同一份标注数据每次计算得到完全相同的区间（可复现 > 每次微小随机）
BOOTSTRAP_SEED = 20260807
# 重采样次数：2000 次对 percentile CI 已经收敛，且 25 条数据下耗时 < 10ms
BOOTSTRAP_RESAMPLES = 2000
# kappa 少于该样本量时不报区间：bootstrap 无法凭空造出信息，
# n<10 时区间会宽到覆盖几乎整个值域，报出来只会误导
MIN_N_FOR_CI = 10

# Landis & Koch (1977) 分档，用于把裸数字翻译成结论强度
_LANDIS_KOCH_BANDS: list[tuple[float, str]] = [
    (0.0, "差"),
    (0.20, "轻微"),
    (0.40, "一般"),
    (0.60, "中等"),
    (0.80, "显著"),
    (1.01, "几乎完全一致"),
]


def interpret_kappa(kappa: float | None) -> str | None:
    """按 Landis & Koch 分档把 kappa 翻译成一致性强度。"""
    if kappa is None:
        return None
    if kappa < 0:
        return "差"
    for upper, label in _LANDIS_KOCH_BANDS:
        if kappa <= upper:
            return label
    return "几乎完全一致"


def cohens_kappa(pairs: t.Sequence[tuple[bool, bool]]) -> dict[str, t.Any]:
    """二分类 Cohen's kappa + 2x2 混淆矩阵。

    pairs 为 (人工通过, 自动通过) 的布尔对。kappa 在期望一致率为 1.0 时无定义
    （例如两个标注者全部判通过），此时返回 None 而不是硬凑一个 0 或 1。
    """
    total = len(pairs)
    if total == 0:
        return {
            "total": 0,
            "both_pass": 0,
            "manual_pass_auto_fail": 0,
            "manual_fail_auto_pass": 0,
            "both_fail": 0,
            "agreement_rate": None,
            "kappa": None,
        }

    both_pass = sum(1 for m, a in pairs if m and a)
    manual_pass_auto_fail = sum(1 for m, a in pairs if m and not a)
    manual_fail_auto_pass = sum(1 for m, a in pairs if not m and a)
    both_fail = sum(1 for m, a in pairs if not m and not a)

    observed = (both_pass + both_fail) / total
    expected = (
        (both_pass + manual_pass_auto_fail) * (both_pass + manual_fail_auto_pass)
        + (manual_fail_auto_pass + both_fail) * (manual_pass_auto_fail + both_fail)
    ) / (total * total)
    kappa = None if expected >= 1.0 else (observed - expected) / (1.0 - expected)

    return {
        "total": total,
        "both_pass": both_pass,
        "manual_pass_auto_fail": manual_pass_auto_fail,
        "manual_fail_auto_pass": manual_fail_auto_pass,
        "both_fail": both_fail,
        "agreement_rate": round(observed, 4),
        "kappa": round(kappa, 4) if kappa is not None else None,
    }


def bootstrap_kappa_ci(
    pairs: t.Sequence[tuple[bool, bool]],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict[str, t.Any] | None:
    """对 kappa 与一致率做 percentile bootstrap，返回置信区间。

    实现细节里有两处值得说明：

    1. **退化重采样**。25 条里若某次重采样恰好抽到全部同判，期望一致率为 1.0，
       kappa 无定义。这类样本被跳过并计入 `degenerate_resamples`——直接当 0
       会把区间下界人为压低，当 1 会人为抬高，两种都是编数据。退化比例高
       本身就是"类别太不平衡，kappa 不稳"的信号，所以它被报出来而不是藏掉。

    2. **区间跨档**。点估计落在"显著"档不代表结论是"显著"：若区间横跨
       中等~几乎完全一致，可下的结论只能是最弱的那一档。`spans_bands`
       和 `ci_band` 就是为了让这句话没法被含糊过去。
    """
    total = len(pairs)
    if total < MIN_N_FOR_CI:
        return None

    point = cohens_kappa(pairs)
    if point["kappa"] is None:
        return None

    rng = random.Random(seed)
    indices = range(total)
    kappa_samples: list[float] = []
    rate_samples: list[float] = []
    degenerate = 0

    for _ in range(resamples):
        drawn = [pairs[rng.choice(indices)] for _ in indices]
        stats = cohens_kappa(drawn)
        rate_samples.append(stats["agreement_rate"])
        if stats["kappa"] is None:
            degenerate += 1
            continue
        kappa_samples.append(stats["kappa"])

    if not kappa_samples:
        return None

    kappa_samples.sort()
    rate_samples.sort()
    ci_low, ci_high = _percentile_interval(kappa_samples, confidence)
    rate_low, rate_high = _percentile_interval(rate_samples, confidence)

    low_band = interpret_kappa(ci_low)
    high_band = interpret_kappa(ci_high)

    return {
        "point": point["kappa"],
        "point_band": interpret_kappa(point["kappa"]),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "ci_band": low_band if low_band == high_band else f"{low_band}~{high_band}",
        # 区间跨档时，能站得住的结论只有下界那一档
        "spans_bands": low_band != high_band,
        "agreement_rate_ci_low": round(rate_low, 4),
        "agreement_rate_ci_high": round(rate_high, 4),
        "n": total,
        "method": "percentile bootstrap",
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "valid_resamples": len(kappa_samples),
        # kappa 无定义的重采样次数：占比越高说明类别越不平衡，区间越不可信
        "degenerate_resamples": degenerate,
    }


def _percentile_interval(sorted_samples: list[float], confidence: float) -> tuple[float, float]:
    """已排序样本的 percentile 区间（下标取最近邻，样本量 2000 时误差可忽略）。"""
    n = len(sorted_samples)
    alpha = (1.0 - confidence) / 2.0
    low_idx = max(0, min(n - 1, int(round(alpha * (n - 1)))))
    high_idx = max(0, min(n - 1, int(round((1.0 - alpha) * (n - 1)))))
    return sorted_samples[low_idx], sorted_samples[high_idx]


def _percentile_at(sorted_samples: list[float], quantile: float) -> float:
    """已排序样本的单个分位点（下标最近邻，与 `_percentile_interval` 同口径）。"""
    n = len(sorted_samples)
    idx = max(0, min(n - 1, int(round(quantile * (n - 1)))))
    return sorted_samples[idx]


def _require_row_alignment(
    baseline_pairs: t.Sequence[tuple[bool, bool]],
    calibrated_pairs: t.Sequence[tuple[bool, bool]],
) -> None:
    """校验两侧逐行对齐：人工列必须完全相同，否则不是配对数据。

    配对检验的前提"两侧是同一批行"在函数内部无法完全验证——调用方完全可以
    传两份行序不同的数据进来。但有一个必要条件是可以廉价验证的：既然是同一批
    人工标注，**人工那一列逐行必须相等**，变的只该是自动判定那一列。

    这个校验不是防御性冗余，它抓的是一类真实的错误：把 2×2 混淆矩阵按
    `[(T,T)]*n + [(T,F)]*m + ...` 展开成两侧样本时，行对应关系会被展开顺序
    打散，而两侧长度依然相等。此时配对 bootstrap 会返回一个**看起来合理**的
    区间（实测比真值宽 0.14，方向还更保守），没有任何迹象提示它算错了。
    """
    for i, (base, cal) in enumerate(zip(baseline_pairs, calibrated_pairs)):
        if base[0] != cal[0]:
            raise ValueError(
                f"配对 bootstrap 要求两侧逐行对齐（同一批人工标注，只有自动判定变化），"
                f"但第 {i} 行的人工判定不同：baseline={base[0]} / calibrated={cal[0]}。"
                f"常见原因：用 2×2 混淆矩阵展开构造样本，行对应关系已被展开顺序打散。"
            )


def paired_kappa_delta_ci(
    baseline_pairs: t.Sequence[tuple[bool, bool]],
    calibrated_pairs: t.Sequence[tuple[bool, bool]],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict[str, t.Any] | None:
    """配对 bootstrap：两个 kappa **之差**的置信区间。

    为什么不能拿两个独立区间看重不重叠：
      校准前后这两个 kappa 是对着**同一批人工标注**算的——同样 25 行、同样的
      人工 pass/fail，只有自动判定那一列变了。这是配对数据。两个独立区间各自
      都很宽、几乎必然重叠，但**重叠不等于差异不显著**：行与行之间共同的难度
      波动（某行本来就难判）在相减时会被消掉，所以配对差的区间可以窄得多。
      反过来，独立区间不重叠也不足以直接断定显著。要检验的是差本身，因此必须
      **一次抽定行下标，在同一份 resample 上重算两个 kappa 再相减**。

    退化重采样（任一侧 kappa 无定义）整份丢弃，而不是把那一侧当 0：
      配对差的前提是两侧来自同一组行，缺一侧这个差就不存在，补值等于编数据。

    单侧与双侧都报：
      双侧下界是更严的判据，但"下界压在 0 上"和"方向站不住"是两件事，
      所以同时给出单侧下界与 P(Δ>0)，让结论强度可以被准确表述而不是二选一。

    为什么要逐行校验人工列（`_require_row_alignment`）：
      写完这个函数后我拿"2×2 混淆矩阵展开"的现成夹具去测它——
      `[(T,T)]*10 + [(T,F)]*1 + ...` 这种。两侧长度都是 25，长度校验过了，
      函数照样返回一个像样的区间 [-0.0657, 0.7558]。但那是**错的**：
      按矩阵展开后两侧的第 i 项已经不是同一行，配对关系被展开顺序打散了，
      真实行对齐数据给的是 [-0.0129, 0.6716]。
      **一个测不出对齐错误的配对检验，比没有检验更危险**——它会返回一个
      看起来合理、宽度还更保守的区间，没人会怀疑。所以这里把"两侧人工列
      必须逐行相等"变成硬校验：这是配对数据唯一可被廉价验证的必要条件
      （同一批标注 → 人工那一列不可能变），不满足就是调用方传错了数据。
    """
    total = len(baseline_pairs)
    if total != len(calibrated_pairs):
        raise ValueError(
            f"配对 bootstrap 要求两侧行数一致（同一批标注），"
            f"实际 baseline={total} / calibrated={len(calibrated_pairs)}"
        )
    _require_row_alignment(baseline_pairs, calibrated_pairs)
    if total < MIN_N_FOR_CI:
        return None

    base_point = cohens_kappa(baseline_pairs)
    cal_point = cohens_kappa(calibrated_pairs)
    if base_point["kappa"] is None or cal_point["kappa"] is None:
        return None

    rng = random.Random(seed)
    indices = range(total)
    deltas: list[float] = []
    degenerate = 0

    for _ in range(resamples):
        drawn = [rng.choice(indices) for _ in indices]
        # 同一份行下标喂给两侧——这就是"配对"的全部含义
        base_stats = cohens_kappa([baseline_pairs[i] for i in drawn])
        cal_stats = cohens_kappa([calibrated_pairs[i] for i in drawn])
        if base_stats["kappa"] is None or cal_stats["kappa"] is None:
            degenerate += 1
            continue
        deltas.append(cal_stats["kappa"] - base_stats["kappa"])

    if not deltas:
        return None

    deltas.sort()
    ci_low, ci_high = _percentile_interval(deltas, confidence)
    one_sided_low = _percentile_at(deltas, 1.0 - confidence)
    p_positive = sum(1 for d in deltas if d > 0) / len(deltas)

    return {
        "delta": round(cal_point["kappa"] - base_point["kappa"], 4),
        "baseline_kappa": base_point["kappa"],
        "calibrated_kappa": cal_point["kappa"],
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        # 双侧区间是否排除 0：这是"提升已验证"能不能说出口的判据
        "excludes_zero": ci_low > 0.0,
        "one_sided_ci_low": round(one_sided_low, 4),
        "one_sided_excludes_zero": one_sided_low > 0.0,
        "p_delta_gt_zero": round(p_positive, 4),
        "n": total,
        "method": "paired percentile bootstrap (同一 resample 重算两侧 kappa)",
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "valid_resamples": len(deltas),
        "degenerate_resamples": degenerate,
    }


def build_calibration_suggestion(
    point_kappa: float | None,
    ci: dict[str, t.Any] | None,
    total: int,
    gate: float = 0.7,
) -> str | None:
    """按 CI 下界而不是点估计出校准提示。

    原先门禁是 `point_kappa < 0.7`。问题在于 n=25 时点估计 0.82 完全可能对应
    下界 0.55——"过线"只是没被抽样噪声打到而已。所以这里分三种情况：
      - 下界也过线：不提示；
      - 点估计过线但下界没过：提示"样本量不足以支撑结论"，而不是"口径有问题"，
        因为这两者的处理动作完全不同（补标注 vs 改判分标准）；
      - 点估计就没过线：提示复核判分标准。
    """
    if point_kappa is None or total < MIN_N_FOR_CI:
        return None

    ci_low = ci.get("ci_low") if ci else None

    if point_kappa < gate:
        return (
            f"人工与自动评分一致性的 Cohen's kappa 为 {point_kappa}（< {gate}），"
            "建议复核判分标准（指标 prompt_override）与人工标注口径，修订后重新评测。"
        )

    if ci_low is not None and ci_low < gate:
        return (
            f"kappa 点估计 {point_kappa} 已达 {gate}，但 95% 置信区间下界为 {ci_low}"
            f"（n={total}），尚不足以断定一致性稳定达标。建议扩充人工标注样本量后复核，"
            "而非直接采信点估计。"
        )

    return None


def pairwise_annotator_kappa(
    labels_by_annotator: dict[str, dict[int, bool]],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict[str, t.Any]:
    """人-人一致性：所有标注者两两之间的 kappa + Bootstrap CI。

    为什么这个数比 judge-人 kappa 更重要：

        它是 judge-人 kappa 的**上界参照**。

    在两位人类之间只有 0.65 的任务上，judge 与其中一位达到 0.82 不是
    "judge 很准"，而是"judge 拟合了这一个人的偏好"——尤其当这个人的标注
    正好就是调 judge prompt 时用的那份。缺了这个上界，judge 的 kappa 是一个
    没有量纲的分数：不知道 0.82 算高还是算离谱。

    两处实现选择：

    1. 只用独立标注。仲裁记录由调用方（`core/annotation.labels_by_annotator`）
       提前剔除——仲裁者已看过双方答案，与其算一致性是循环论证。
    2. 每对标注者只在**双方都标过**的行上比较。交集之外的行没有可比信息，
       把缺失当"不一致"会凭空造出分歧。`overlap_n` 因此逐对报出。

    单标注者时返回 `insufficient_annotators=True`。这一支必须显式存在：
    "测不出来"和"一致性很好"是两回事，而缺省值最容易被读成后者。
    """
    annotators = sorted(labels_by_annotator.keys())
    pairs_out: list[dict[str, t.Any]] = []

    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            name_a, name_b = annotators[i], annotators[j]
            labels_a = labels_by_annotator[name_a]
            labels_b = labels_by_annotator[name_b]
            shared = sorted(set(labels_a) & set(labels_b))
            if not shared:
                continue

            pair_data = [(labels_a[row_id], labels_b[row_id]) for row_id in shared]
            stats = cohens_kappa(pair_data)
            # cohens_kappa 的键名是 manual_*/auto_*（它最初只服务人-自动比较）。
            # 人-人场景下数值含义不变，这里换成 a/b 的中性名字避免误读。
            pairs_out.append(
                {
                    "annotator_a": name_a,
                    "annotator_b": name_b,
                    "overlap_n": len(shared),
                    "kappa": stats["kappa"],
                    "band": interpret_kappa(stats["kappa"]),
                    "agreement_rate": stats["agreement_rate"],
                    "both_pass": stats["both_pass"],
                    "both_fail": stats["both_fail"],
                    "a_pass_b_fail": stats["manual_pass_auto_fail"],
                    "a_fail_b_pass": stats["manual_fail_auto_pass"],
                    "disagreement_count": (
                        stats["manual_pass_auto_fail"] + stats["manual_fail_auto_pass"]
                    ),
                    "kappa_ci": bootstrap_kappa_ci(
                        pair_data, resamples=resamples, seed=seed, confidence=confidence
                    ),
                }
            )

    kappas = [p["kappa"] for p in pairs_out if p["kappa"] is not None]
    # 上界取两两 kappa 的**最小值**而不是均值：judge 只要超过任意一对人类的
    # 一致性，"它拟合了某个人"的嫌疑就已经成立，用均值会把这个信号平均掉。
    ceiling = round(min(kappas), 4) if kappas else None

    return {
        "annotator_count": len(annotators),
        "annotators": annotators,
        "pairs": pairs_out,
        "mean_kappa": round(sum(kappas) / len(kappas), 4) if kappas else None,
        "min_kappa": ceiling,
        "ceiling_kappa": ceiling,
        "ceiling_band": interpret_kappa(ceiling),
        "insufficient_annotators": len(annotators) < 2,
        "note": (
            "只有 1 位标注者，人-人 kappa 无法计算：judge 的 kappa 目前没有上界参照。"
            if len(annotators) < 2
            else f"{len(pairs_out)} 对标注者两两比较，上界取最小 kappa。"
        ),
    }


def judge_ceiling_comparison(
    labels_by_annotator: dict[str, dict[int, bool]],
    judge_labels: dict[int, bool],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict[str, t.Any]:
    """把"judge 与 A 的一致性"和"B 与 A 的一致性"做**配对**比较。

    这个比较在构造上就是配对的，不需要额外造检验：两侧的参照列都是 A 的标注，
    逐行相同，于是可以直接复用 `paired_kappa_delta_ci`（它会校验参照列逐元素
    相等，对不齐直接抛错）。

        baseline   = [(A_i, B_i)]      —— 人-人
        calibrated = [(A_i, judge_i)]  —— judge-人
        Δ = judge-人 kappa − 人-人 kappa

    Δ 的置信区间下界 > 0 意味着 judge 比另一位人类更像 A。这**不是**好消息：
    在人类彼此都没那么一致的任务上，judge 越过人类天花板的常见解释是它拟合了
    A 的个人偏好，而不是它更接近事实。所以这一支的 verdict 是告警而非通过。

    为什么不用两个独立区间比大小：那正是本项目上一轮修掉的错误做法。
    两个宽区间几乎总是重叠，看着"无差异"，实际是检验功效被自己丢掉了。

    多重比较：k 位标注者产生 k*(k-1) 个有序对，每对做一次检验。这里**不做**
    Bonferroni 校正，而是把比较次数原样报出（`comparison_count`）由读数的人
    自己折算。理由：n=25 量级下校正后几乎不会有任何一项显著，用校正把信号抹平
    比把多重性摆出来更糟——后者至少是可讨论的。
    """
    annotators = sorted(labels_by_annotator.keys())
    comparisons: list[dict[str, t.Any]] = []

    for reference in annotators:
        for other in annotators:
            if reference == other:
                continue
            shared = sorted(
                set(labels_by_annotator[reference])
                & set(labels_by_annotator[other])
                & set(judge_labels)
            )
            if len(shared) < MIN_N_FOR_CI:
                continue

            human_pairs = [
                (labels_by_annotator[reference][r], labels_by_annotator[other][r])
                for r in shared
            ]
            judge_pairs = [
                (labels_by_annotator[reference][r], judge_labels[r]) for r in shared
            ]
            delta = paired_kappa_delta_ci(
                human_pairs,
                judge_pairs,
                resamples=resamples,
                seed=seed,
                confidence=confidence,
            )
            if delta is None:
                continue

            comparisons.append(
                {
                    "reference_annotator": reference,
                    "other_annotator": other,
                    "n": len(shared),
                    "human_human_kappa": delta["baseline_kappa"],
                    "judge_human_kappa": delta["calibrated_kappa"],
                    "delta": delta["delta"],
                    "delta_ci_low": delta["ci_low"],
                    "delta_ci_high": delta["ci_high"],
                    "judge_exceeds_human": bool(delta["ci_low"] > 0.0),
                }
            )

    exceeded = [c for c in comparisons if c["judge_exceeds_human"]]

    if len(annotators) < 2:
        status = "ceiling_unknown"
        note = (
            "只有 1 位标注者：人-人 kappa 无法计算，judge 的 kappa 没有上界参照。"
            "这是当前标注数据的硬限制，不是「judge 通过了」的结论。"
        )
    elif not comparisons:
        status = "insufficient_overlap"
        note = (
            f"标注者 {annotators} 之间与 judge 的三方共同标注行不足 "
            f"{MIN_N_FOR_CI} 条，无法给出区间。"
        )
    elif exceeded:
        status = "judge_above_ceiling"
        note = (
            f"{len(exceeded)}/{len(comparisons)} 项比较显示 judge 比另一位人类更贴近参照标注者"
            "（配对 Δ 下界 > 0）。这更可能是 judge 拟合了该标注者的个人偏好，"
            "而不是 judge 更接近事实——需要核查 judge prompt 是否用该标注者的标注调过。"
        )
    else:
        status = "within_ceiling"
        note = (
            f"judge-人 kappa 未显著超过人-人 kappa（{len(comparisons)} 项比较，"
            "配对区间均含 0）：judge 的一致性处在人类彼此一致性的量级内。"
        )

    return {
        "status": status,
        "note": note,
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "multiplicity_note": (
            f"共 {len(comparisons)} 次配对检验，未做多重比较校正——"
            "比较次数已列出，请据此折算显著性。"
        ),
    }
