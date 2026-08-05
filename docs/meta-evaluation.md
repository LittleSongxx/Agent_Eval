# 评测系统的元评价指南（Meta-Evaluation）

> 评测系统的输出要支撑上线决策，那么**评测系统本身必须先被评测**。
> 本文档回答三个问题：指标测得准吗（效度）、测得稳吗（信度）、能分开好坏吗（区分度），
> 并给出与平台功能一一对应的可执行实验设计。
> 配套工具：`backend/scripts/compare_with_ragas.py`（与 RAGAS 对照实验）。

## 0. 为什么需要元评价

LLM 应用评测的三个现实问题：

1. **LLM-as-Judge 不是真值**——同一行样本，Judge 可能给出不同分数，甚至与人工判断相左；
2. **自研指标没有背书**——不套 RAGAS/DeepEval 意味着"分数含义"需要自己证明；
3. **分数会被"用错"**——如果指标分不出好坏，任何基于它的优化决策都是盲的。

元评价把评测系统当作被测对象，回答"这套分数我该信几分"。

## 1. 四个评测层次

### 1.1 效度（Validity）：测的是不是"那个东西"

| 方法 | 说明 | 平台对应 |
|------|------|----------|
| **准则效度**（最硬） | 以人工专家标注为黄金标准，计算自动分与人工分的一致性：Pearson / Spearman 相关、一致率、Cohen's Kappa | 报告"人工一致性"统计（`manual_auto_agreement_rate`）；人工复核流程 |
| **结构效度** | 与业界成熟框架（RAGAS）在同一批样本上的分数相关性——不替代人工，但证明"与主流判断方向一致" | `scripts/compare_with_ragas.py` |
| **已知组区分** | 构造已知好坏样本，验证分数能区分（见 1.3 与错误注入实验） | 数据集 + 评测执行 |

> 三种证据强度：人工一致性 > 已知组区分 > 与 RAGAS 相关。面试/汇报时按此优先级组织证据。

### 1.2 信度（Reliability）：测得稳不稳

| 方法 | 说明 | 平台对应 |
|------|------|----------|
| **重测信度** | 同一行样本重复评分 N 次，标准差越小越稳定；标准差超阈值标记低置信度 | `EVAL_JUDGE_SAMPLES=N` 多采样 + 报告 `judge_std_mean` / `low_confidence_count` |
| **评分者间一致性** | 两个不同 Judge 模型评同一批样本，Cohen's Kappa / 一致性率 | 配置多个 LLM 配置，同一场景换 Judge 跑两遍，报告对比 |
| **偏置检查** | 位置偏置（换序互评）、冗长偏置（length-controlled win-rate）、自我偏好偏置（同族模型互评）、格式偏置 | 盲测的左右随机化（md5）；对照实验可人工构造换序样本 |

### 1.3 区分度（Discrimination）：能不能分开好坏

- **错误注入实验**（推荐先做，成本最低）：取正确样本，注入典型错误（幻觉、事实错误、答非所问），
  看对应指标分数是否显著下降。示例结果（真实运行数据见下文实验记录）：
  - 正确基线：faithfulness ≈ 1.0
  - 注入"元组可变"幻觉：faithfulness 显著下降
  - 注入无关回答：answer_relevancy / faithfulness 双低
- **AUC 分析**（进阶）：以人工标注的好坏为二分类真值，指标分数作为判分器，计算 AUC——衡量指标"排序能力"。

### 1.4 实用性（Pragmatics）：用得起吗

- **成本**：Judge token 用量与估算成本（报告 `cost` 字段）；确定性指标零成本兜底的意义在此体现。
- **延迟**：逐行耗时（报告 execution_time_ms）。
- **稳定性**：Judge 调用失败率（error_count）。
- **可解释性**：每条分数是否带可读理由（平台逐条保存 Judge 理由，人工可复核）。

## 2. 实验 SOP

### 实验 A：与 RAGAS 的结构效度对照（最低成本、最高性价比）

```bash
cd backend
conda activate eval
pip install ragas==0.3.7 langchain-openai   # 一次性
python -m scripts.compare_with_ragas --dataset "RAG 示例数据集" --limit 20
```

要求：
- 样本量建议 **n ≥ 20**（n < 10 时相关系数无统计意义，仅作流程验证）；
- 数据集字段必须包含 `user_input` / `response` / `retrieved_contexts` / `reference`；
- `answer_relevancy` 需要 OpenAI 兼容 `/embeddings` 接口（默认 `text-embedding-v3`，可 `RAGAS_EMBEDDING_MODEL` 覆盖）；
- 结论口径：**Pearson/Spearman > 0.8 = 排序判断高度一致；0.6~0.8 = 方向一致，可结合人工抽检校准绝对分**。

> 说明：RAGAS 工具链指标（tool_call_accuracy 等）为多轮消息格式，与平台的数据集格式
> 不同。Agent 场景的对照建议改用"人工黄金集一致性"（实验 B），或自行做格式转换。

### 实验 B：人工黄金集一致性（准则效度，Agent 场景推荐）

1. 选取 20~50 条 Agent 样本（含正确/错误工具调用、正确/幻觉回复）；
2. 平台跑一遍评测 → 报告页逐条标记人工复核（pass/fail）；
3. 查看报告"人工一致性"统计：一致率 ≥ 85% 视为通过；分歧样本逐个分析（badcase 归因）；
4. 有分歧 → 定位是"指标口径问题"还是"标注问题"，修改判分标准（prompt_override）后重跑。

### 实验 C：错误注入区分度

1. 用 RAG 数据集生成器或手工构造 N 组样本：每组 = 正确样本 + 幻觉注入版 + 无关回答版；
2. 同一场景跑评测，对比三组分数；
3. 验收标准：faithfulness（幻觉注入组）与 answer_relevancy（无关回答组）应显著低于基线组。

### 实验 D：重测信度（Judge 稳定性）

```bash
EVAL_JUDGE_SAMPLES=5 python run.py   # 开启 5 次采样
```

1. 对同一数据集跑两遍评测（或同一遍开启多采样）；
2. 报告 `judge_std_mean` < 0.05 视为稳定；`low_confidence_count` 高的指标优先人工复核；
3. 换一个 Judge 模型（如 qwen-max）跑同一任务，报告对比看分数分布是否一致。

## 3. 结果如何汇报（面试/评审口径）

- **不要只说"我们做了评测"**，要说"我们评测了评测"：
  > 自研 Judge 指标与 RAGAS 在同一批样本上的 Spearman 相关性 0.85（n=30）；
  > 错误注入实验中，幻觉样本的 faithfulness 从 1.0 降至 0.3，区分度显著；
  > 5 次采样标准差均值 0.02，评分稳定；与人工复核一致率 92%。
- 数字要真实、注明样本量；n 小就明说"流程验证级"，并给出扩大样本的计划。

## 4. 平台能力与元评价的映射一览

| 元评价需求 | 平台功能 |
|------------|----------|
| 人工黄金集 | 数据集管理 + 报告人工复核 |
| 自动 vs 人工一致性 | 报告 `manual_auto_agreement_rate` |
| 与 RAGAS 对照 | `scripts/compare_with_ragas.py` |
| 重测信度 | `EVAL_JUDGE_SAMPLES` 多采样 + `judge_std_mean` |
| 成本/延迟/失败率 | 报告 `cost` / `execution_time_ms` / `error_count` |
| 回归稳定性 | 报告对比（baseline vs current） |
| 盲测去偏 | 盲测 md5 随机化左右展示 |
| 位置偏置检测 | `EVAL_SWAP_CHECK` 换序互评（反转列表字段复评，报告一致性） |
| 评分者间一致性 | 多裁判面板（`judge_llm_config_ids`）+ 裁判间 MAD；报告 Cohen's kappa |
| 人工校准闭环 | 报告 kappa < 0.7 自动提示修订判分标准 |
| 检索侧区分度 | `scripts/retrieval_noise_experiment.py` 噪声注入 |
| 检索链路演示 | `scripts/bm25_mock_retriever.py` 零依赖中文 BM25 |
| 评测集防污染 | `scripts/check_dataset_contamination.py`（n-gram + embedding 双信号） |
| 双通道交叉验证 | 生成式相关性 / 断言级忠实度 与整体判定指标并存 |

---

## 5. 实测记录（2026-08，真实运行数据）

以下数字由 `scripts/compare_with_ragas.py` 与平台评测引擎在 Qwen3.7-plus
（OpenAI 兼容 MaaS 网关）上真实运行产生，`ragas_comparison_report.json`
为脚本输出原始报告。

### 5.1 区分度实验（错误注入，3 行样本）

| 指标 | 正确基线 | 幻觉注入（"元组可变"） | 无关回答 |
|------|---------|----------------------|---------|
| faithfulness | 1.0 | 0.0 | 0.0 |
| factual_correctness | 1.0 | 0.0 | 0.0 |
| answer_relevancy | 1.0 | 0.1 | 0.0 |
| context_recall / precision | 1.0 | 1.0（不受生成侧污染影响） | 1.0 / 1.0 |

结论：生成侧指标对幻觉/答非所问敏感，检索侧指标保持稳定——验证了
"检索质量与生成质量解耦"的分层设计。

### 5.2 结构效度对照（RAGAS 0.3.7，n=25 混合集：9 正确 / 8 幻觉注入 / 8 截断）

| 指标 | 平台均值 | RAGAS 均值 | Pearson | Spearman |
|------|---------|-----------|---------|----------|
| faithfulness | 0.712 | 0.722 | 0.584 | 0.686 |
| answer_relevancy | 0.700 | 0.800 | 0.351 | 0.292 |
| context_precision | 0.952 | 1.000 | 无方差 | 无方差 |
| context_recall | 0.972 | 1.000 | 无方差 | 无方差 |

解读与边界：

1. **faithfulness 方向一致**：均值几乎相同（0.71 vs 0.72），Spearman 0.69
   属于"排序方向一致"区间——在含坏样本的混合集上，双方对"哪些回答忠实"
   的排序判断一致，为自研指标提供实证背书。
2. **answer_relevancy 中度相关**：RAGAS 采用"从回答生成问题 → 与用户问题做
   语义相似度"，平台为整体判定，机制差异导致排序分歧，已记录为校准点。
3. **检索侧无方差（设计边界）**：合成数据集的 retrieved_contexts 直接取自
   源文档 chunk（= 完美检索），检索侧指标天然满分。检索侧的结构效度需要
   带真实检索噪声的数据集验证（后续实验方向）。
4. **绝对分不可直接对比**：RAGAS 的 faithfulness 为原子断言级拆解（逐句
   核对），平台为整体判定；粒度差异在实验 A（纯好样本集）中已观测到
   （平台 0.88 vs RAGAS 0.56），混合集中因坏样本占主导而收敛。
5. **天花板效应教训**：纯好样本集（n=25）上双方满分、相关系数无定义——
   证明评测集必须包含负样本，否则任何指标都"看起来很准"。
