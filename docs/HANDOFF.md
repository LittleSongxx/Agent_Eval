# 项目交接说明（给接手 AI 的完整背景 prompt）

> **用途**：把本项目（AI 评测平台）完整交接给另一位 AI 或开发者继续推进。
> 接手后请按「8. 验证清单」先确认现状，再按「6. 后续工作」的优先级推进。
> 本文件是唯一入口：先读本文件，再按需读它引用的文档。
>
> 快照时间：2026-08-06，dev 分支 HEAD = `e63b3cf`。

---

## 0. 项目是什么（一句话）

面向 RAG / AI Agent / 多轮对话的 **AI 应用评测平台**：评测内核为平台原生实现的
LLM-as-a-Judge（双层 prompt），直接调用 OpenAI 兼容模型（通义千问 Qwen Plus /
DashScope）返回评分与评判理由。**项目是求职作品集**，核心竞争力是「元评价体系」——
用实验证明平台自己的测量可信（区分度 / 结构效度 / 信度 / 实用性四维），而不是只堆功能。

## 1. 技术栈与环境

| 项 | 值 |
|---|---|
| 后端 | FastAPI + SQLAlchemy + SQLite，Python 3.13（anaconda），异步 + WebSocket |
| 前端 | React 18 + TypeScript + Ant Design 5 + Vite（Node 24） |
| 评测 LLM | Qwen Plus（DashScope，OpenAI 兼容）；`backend/.env` 已配好 `LLM_API_KEY` |
| 环境 | WSL2 Linux；后端端口 8000（`http://localhost:8000/docs`），前端 5173 |
| 测试 | `cd backend && python -m pytest tests/ -q`（**必须在 backend 目录下跑**，仓库根没有 tests/）|

关键环境变量（`backend/app/core/config.py`）：`LLM_API_KEY/LLM_MODEL/LLM_ENDPOINT`、
`SEED_ON_STARTUP`（默认 true，首启自动造种子数据）、`EVAL_JUDGE_SAMPLES=1`（多采样
信度）、`JUDGE_COT_MODE=False`（强制推理）、`EVAL_SWAP_CHECK=False`（换序互评）。

## 2. 当前状态（截至快照）

- **git**：dev 分支共 3 个大阶段提交（见 3.1），领先 origin/dev **2 个提交未 push**；
  main 分支停在 `b90c2ba`，dev 尚未合并回 main。
- **测试**：109 全部通过（含检索实验守门测试）。
- **服务**：启动验证过，`/docs`、`/api/datasets`（7 个数据集）、`/api/reports` 均 200。
- **工作区残留**：`backend/ragas_comparison_report_n10.json`、`_n25_clean.json`
  两个中间报告未入库（无引用，可直接删除）；`ragas_comparison_checkpoint.json`
  已 gitignore（断点续跑临时产物）。
- **代码层面无已知 bug**。剩余工作全部是「补证据 + 收尾」（见 6）。

## 3. 必须知道的架构要点（防「修坏」清单）

### 3.1 git 历史阶段速览

| 提交 | 内容 |
|---|---|
| `b90c2ba`（main） | 评测可靠性（加权总分/多采样/成本/WebSocket 线程安全/任务心跳回收）+ 元评价体系 + RAGAS 对照脚本 |
| `9eee48d`（dev） | 双通道指标（断言级忠实度/生成式相关性）+ Judge 工程化（强制 CoT/换序互评/多裁判面板+MAD/kappa 校准闭环）+ 数据集版本化 + 治理工具（BM25 mock/污染检测） |
| `d88da01` | 前端多裁判面板 UI + 审查修复 5 类 bug |
| `d60c15c` | claim 分解质量：保留意见性断言 + 空断言兜底改 0（与 RAGAS 行为一致） |
| `4c0847d` | **缺陷 A/B 修复 + 检索侧 BM25 实测**（详见 4） |
| `e63b3cf` | 新增 [开发历程踩坑与成果记录.md](开发历程踩坑与成果记录.md) |

### 3.2 评测内核链路（改这里之前先读这段）

```
评测任务 → 逐行取数据 → extract_eval_fields（注入运行时字段）
  → NativeBuiltinLLMMetric.ascore
      → _sample_payload(row_data, allow_fields=judge_fields)   ← 口径白名单
      → prompt 组装（系统层输出协议 + DB 业务判分标准）
      → LLM 调用（多采样/多裁判/换序互评在此注入）
      → 评分解析 + 聚合（均值/多数票/MAD/kappa）
```

**三条不可违反的规则（都有测试守门，改了就红）：**

1. **`judge_fields` ≠ `required_fields`**（`backend/app/core/prompt_manager.py` +
   `evaluation_engine.py`）。前者是「裁判可见范围」（17 个内置指标逐个显式声明），
   后者只是「字段必须存在才打分」的校验门槛。两者本就不相等：8 个 Agent/多轮指标
   要看 `response` 但不能列为必需（运行时才注入）；`context_precision` 要看
   `reference` 而它不声明必需。**任何「简化」把两者合并都会破坏 9/17 个指标**。
   新增内置指标必须同时声明两个字段集，结构不变量测试会检查
   `judge_fields ⊇ required_fields`。
2. **`_sample_payload` 的白名单分支会整段跳过兜底循环**——兜底循环（catch-all loop）
   是缺陷 A（标签泄露）和缺陷 B（口径越界）的共同载体，永远不要让它重新接管
   内置指标的下发。
3. **两个双通道指标不经过 `_sample_payload`**：`ClaimFaithfulnessMetric`（只读
   response/retrieved_contexts）和 `GenerativeAnswerRelevancyMetric`（只读
   user_input/response）。缺陷 A/B 的修复 diff 都不落在这两个类，任何改动不要
   顺手把它们接进 sample 链路，否则会引入它们原本免疫的污染。

### 3.3 其他关键点

- **双层 prompt**：系统层 prompt 管输出协议（JSON 结构），业务判分标准存
  `metric_definitions` 表，可按场景覆盖。改协议要同步改解析器。
- **防 fork 约束（硬性）**：平台内核**零依赖第三方评测框架**（RAGAS/DeepEval 只允许
  出现在 `backend/scripts/` 对照实验里，并明确标注为外部验证工具）。第三方兼容问题
  一律用项目内最小 shim 解决（见 `backend/scripts/_shims/` 的 RAGAS vertexai 案例）。
- **`_JUDGE_HIDDEN_FIELDS = {"generation_meta"}`**（evaluation_engine.py 内）：
  缺陷 A 的修复，黑名单永远并入 exclude_fields。
- **报告合并**：`backend/scripts/merge_n25_report.py` 把多个来源的对照报告按指标
  合并，每指标带 `_provenance`（来源文件/时间/缺陷修复状态）。**任何对照数字必须
  走这个脚本或同等溯源机制，禁止人工转抄**。
- **前端**：页面在 `frontend/src/pages/`（12 个页面）；评测执行页
  `EvaluationPage.tsx`（含附加裁判多选、克隆任务继承面板）；API 封装
  `frontend/src/services/api.ts`；类型 `frontend/src/types/index.ts`。

## 4. 实验结论速查（重要数字）

### 4.1 RAGAS 对照（n=25：9 正确/8 幻觉/8 截断，缺陷 A/B 双修复后）

| 指标 | 平台 | RAGAS | Pearson | Spearman | 结论 |
|---|---|---|---|---|---|
| faithfulness | 0.728 | 0.689 | 0.735 | 0.757 | 高度一致 |
| faithfulness_claim | 0.649 | 0.702 | 0.805 | 0.817 | 排序高度一致 |
| answer_relevancy | 0.744 | 0.809 | 0.373 | 0.239 | 弱相关（真实机制差异，非缺陷） |
| answer_relevancy_generative | 0.913 | 0.816 | 0.571 | 0.545 | 中度一致 |
| context_precision / recall | 1.000 | 1.000 | 无定义 | 无定义 | 逐行相等 |

- 检索侧**逐行等于 RAGAS**（分层偏离 0.275/0.213 → 0.000）——平台检索侧那点方差
  100% 是缺陷 B 伪影，无一是检索信号。
- answer_relevancy 弱相关原因：平台裁判对截断样本给两极 0/1；RAGAS 反推问题 +
  embedding **结构上无法惩罚截断**（给 0.60~0.98）。修复前的高相关是泄露掩盖的
  「对的分数、错的理由」。**这是机制差异，不是 bug，不要试图「修」掉它**。
- 双通道优势：截断样本偏离是整体版 **1/3.6**（0.063 vs 0.225）；修复后 Pearson
  +0.047→+0.071。

### 4.2 检索侧实测（真实 BM25 检索，25 条互异查询）

| 场景 | 库 | H@1 | MRR |
|---|---|---|---|
| 纯知识库 | 6 | 1.000 | 1.000 |
| 无关干扰 ×全部(26) | 32 | 0.960 | 0.980 |
| 同域难负例 ×全部(10) | 16 | 0.880 | 0.940 |
| 无关全部 + 难负例全部 | 42 | **0.840** | **0.913** |

- 无关干扰几乎不掉（主题可分）；**同域难负例才是真压力**（"7天无理由退货起算"
  输给"优选商城 5天…起算"，只差一个数字，bigram 不可分）。
- 关键教训：上一版噪声实验是**闭式循环**（noise_ids+reference_ids 拼列表，
  MRR≡1/(n+1) 与数据无关），已废弃。**守门测试 `test_ranking_is_data_dependent`
  防回归：金标正文换成无关文本后 H@1 必须归零。**

### 4.2a 检索侧结构效度：RAGAS NonLLM 规则版对标（✅ 2026-08-06）

同一批 BM25 检索输出喂给 RAGAS v0.4.3 NonLLMContextRecall / NonLLMContextPrecision
（Levenshtein 归一化相似度阈值 0.5，零 LLM 零 embedding，确定性 ↔ 确定性，跳过
LLM 判定差异干扰），实现 `scripts/compare_ragas_nonllm.py`（dump/score 两阶段
隔离环境），原始输出 `backend/ragas_nonllm_report.json`：

| 场景 | 平台 H@1 | 平台 MRR | RAGAS NonLLM recall | RAGAS NonLLM precision |
|---|---|---|---|---|
| 纯知识库 | 1.000 | 1.000 | 1.000 | 1.000 |
| 无关干扰 ×全部 | 0.960 | 0.980 | 1.000 | 0.980 |
| 同域难负例 ×全部 | 0.880 | 0.940 | 1.000 | 0.940 |
| 无关全部 + 难负例全部 | 0.840 | **0.913** | 1.000 | **0.913** |

- **逐场景 MRR ≡ NonLLM precision**：单金标下 RAGAS AP ≡ 1/rank ≡ 平台 MRR，
  两个独立实现（ID 级精确命中 vs 字符级相似度）交叉验证，检索侧结构效度成立。
- **H@3/H@5 饱和边界同样适用 NonLLM**：recall 恒 1.0 与平台 H@5 恒 1.0 同因
  （金标总在 top-5 内），共享同一检索输入，不构成"两个实现都错"。

### 4.2b 检索侧准则效度：人工黄金集（✅ 2026-08-06）

25 条查询逐条标注「期望命中的 chunk」（最重压力场景）：初标 + 用户人工逐条
复核 **25/25 一致**，人工审核确认为最终真值（`scripts/retrieval_goldset_annotations.json`，
provenance 含人工复核记录；标注表 `docs/检索侧黄金集标注表.md`；
评分 `scripts/compute_goldset_agreement.py` → `retrieval_goldset_agreement.json`）：

- 判定分布 **Y 23 / MULTI 2 / N 0**：人工审核未发现金标构造错误；
- 金标 ∈ 人工期望集 **25/25**——"每条查询唯一 source chunk"的合成构造成立；
- 金标口径与人工黄金集口径 **H@1 / MRR 完全一致（0.840 / 0.9133）**——检索器
  排序质量在人工真值下同样成立；
- 2 条多来源（行 6/19：验收后退款时限跨第 2 节与第 8 节重复），金标均 top-1，
  不影响评分；难负例（hardneg-0/4/6/7 等）人工全部排除，印证其"看似相关实则
  错误"的构造有效。

### 4.3 元评价四维状态

| 维度 | 状态 |
|---|---|
| 区分度 | ✅ 实测：错误注入敏感（幻觉/截断可分离）、检索侧难负例单调下降 |
| 结构效度 | ✅ 实测：生成侧与 RAGAS 对照（4.1 表）；检索侧 RAGAS NonLLM 规则版对标（4.2a，MRR ≡ precision 逐场景相等） |
| 信度 | ✅ **实测（2026-08-06）：mean_std 0.0193（<0.05 稳定判据），6 指标 judge_std_mean 0.000~0.039，低置信对 6/150 其中 5 个在截断类；成本 901k tokens ≈ ¥1.41**；重测信度（任务 9/10 双轮）Pearson 0.948~1.000、pass/fail 一致 24/25 |
| 准则效度 | ✅ **实测+校准闭环（2026-08-06/07）：25 条人工标注校准前一致率 0.72、kappa 0.4582（中等一致），分歧 6/7 为"自动过松"集中在截断/幻觉类；校准（阈值 0.5→0.7 + AR criteria 修订）后重跑 kappa 0.8175、一致率 0.92、自动过松清零（详情见 4.3a）** |
| 实用性 | ✅ 实测：成本统计/实时进度/低置信度标记/人工 kappa 闭环 |

### 4.3a 校准动作（2026-08-06 已执行，重跑验证已完成 2026-08-07）

分歧三层根因（业界调研定位，非拍脑袋）：
1. **截断类 faithfulness 高分 = 设计使然**（RAGAS 定义只测已说出的 claim 是否
   grounded，不测完整性）——**不要试图"修"它**；
2. **answer_relevancy 给截断满分 = 执行漏判**（criteria 无截断可观察定义）；
   row 7 误杀 = 常识外推（裁判脑补"完整答案还应包含…"，回答实与 REF 一致）；
3. **幻觉类 0.5 擦线 = 阈值偏低**（DeepEval faithfulness 默认 0.8 / AR 0.7）。

落地：场景 4 阈值 0.5→0.7（零成本模拟 kappa 0.4582→**0.7458**，0.85 更高但
0.9 误杀 correct 类，高阈值在 25 样本上是过拟合）；`builtin_answer_relevancy`
criteria 补截断可观察定义（省略号/未完成句结尾 → ≤0.3）+ 防格式误杀 + 禁止
常识外推（prompt_manager.py）。

**重跑验证（2026-08-07，任务 13，同 25 行 × 5 采样；任务 11/12 首跑被 pytest
TestClient 误杀后经 worker_pid 修复重建）**：一致率 0.72→**0.92**、kappa
0.4582→**0.8175**（超模拟预测 0.7458）；自动过松 6→**0**；幻觉类 0.75→1.0、
截断类 0.5→**1.0**；correct 类 0.889→0.778（新增 row 6 分歧）。判定翻转 7 行中
6 行与人工一致（幻觉 2 + 截断 4）。仍分歧 2 行（row 6/7）为 correct 类整体通道
误伤（AR 0.52/0.3 vs AR_gen 0.88/0.95），与双通道仲裁已知案例同源。阈值扫描
0.6~0.8 kappa 平台期恒 0.8175（0.7 非过拟合尖峰）。对比脚本：
`scripts/calibration_compare.py --before 10 --after 13`。**样本量边界：25 条
kappa 点估计 CI 宽（50 条 ±0.20），只能当方向信号，扩样后做部署决策。**

### 4.4 两个已修复缺陷（面试高频追问，回答口径）

- **缺陷 A（标签泄露）**：`generation_meta` 真值标签经兜底循环进 judge prompt
  （100 个里 64 个）→ 区分度/一致性虚高。修复：`_JUDGE_HIDDEN_FIELDS` 黑名单 +
  双层回归测试；关闭修复测试即失败（非空转证明）。
- **缺陷 B（口径越界）**：`required_fields` 只校验不裁剪 → 检索侧看见 response、
  answer_relevancy 看见 reference。A/B 实验证明因果（context_recall
  0.7875→1.0000 与 RAGAS 一致）；修复 = 拆出 `judge_fields` 口径 +
  `_sample_payload` 白名单模式。原定「按 required_fields 裁剪」方案会打断
  9/17 指标，被否决——这个「修复方案的方案评审」是项目叙事的一部分，别改口径。

## 5. 已知坑与陷阱（接手后别踩）

1. **路径**：脚本内引用项目路径一律 `Path(__file__).resolve().parent.parent`；
   曾发生 sys.path 指向 /tmp 导致 0 字节输出的教训。pytest 在 `backend/` 下跑。
2. **后台任务**：harness 环境 `sleep` 直接阻塞，用 until-loop 或 run_in_background；
   后台输出要记录 redirect 路径（曾因轮询错路径而找不到结果）。
3. **pgrep 自匹配**：`pgrep -f "uvicorn app.main:app"` 会匹配到自身进程，用端口
   curl 探测代替。
4. **报告 JSON**：`rows` 字段是计数不是列表（曾引发 TypeError）。
5. **中文文件名**：`docs/元评价实验与前沿对标分析报告.md`、`docs/开发历程踩坑与成果记录.md`
   在 shell 里要加引号；git 显示为转义形式属正常。
6. **别动的地方**：judge_fields 口径（3.2）、双通道指标链路（3.2）、
   `_shims/`（RAGAS 对照依赖）、`.gitignore` 里的 checkpoint。
7. **提交纪律**：只在用户明确要求时 commit/push；绝不提交 `.env`（`CLAUDE.md`
   硬性规定）；测试全绿才提交；提交信息遵循仓库既有 `fix:`/`feat:`/`docs:` 风格。

## 6. 后续工作（按优先级，每项给验证标准）

### P0 收尾（10 分钟，等用户确认）
- push dev（2 个提交）→ 是否合并回 main 由用户决定；
- 删除 `backend/ragas_comparison_report_n10.json` 与 `_n25_clean.json`；
- 8 个 SQLAlchemy/datetime 弃用警告可顺手清理（cosmetic，不紧急）。

### P1 信度实验（✅ 已完成 2026-08-06，数字见 4.3）
- 25 行混合集 × 6 指标 × 5 采样实测：mean_std **0.0193**、各指标 judge_std_mean
  0.000~0.039、低置信对 6/150（5 个在截断类——方差随样本难度单调上升）；
  全程零失败，成本 901k tokens ≈ ¥1.41。
- 元评价四维已全部有实测数据（meta-evaluation.md §5.4 / 元评价报告 §2.5）。

### P1 检索侧收尾
- **✅ 检索侧人工黄金集已完成（2026-08-06，25 条，见 4.2b）**：标注表
  docs/检索侧黄金集标注表.md（25 条查询 × BM25 top-5 含 chunk 文本，
  `scripts/make_goldset_worksheet.py` 生成），判定存
  `scripts/retrieval_goldset_annotations.json`（初标 + 用户人工逐条复核
  25/25 一致），评分脚本 `scripts/compute_goldset_agreement.py`；
  生成侧黄金集已实测（kappa 0.4582，见 4.3）
- **✅ RAGAS NonLLMContextPrecision/Recall 规则版对标已完成（2026-08-06，
  见 4.2a）**：确定性 ↔ 确定性，MRR ≡ NonLLM precision 逐场景相等，结构效度
  证据落地（自研阈值法已被证明不可校准，未重复造轮子）；
- 可选：NDCG（排序质量分层）。
- **验证标准**：✅ 检索侧元评价新增两个证据条目（区分度 BM25 实测 + 结构效度
  NonLLM 对标，见 4.2/4.2a）；✅ H@3/H@5 饱和边界已在文档中标注（4.2 解读、
  元评价报告 4.2.3-4.2.7）。

### P1 answer_relevancy 双通道仲裁（✅ 首轮分析完成 2026-08-06，校准后重跑待任务 12）
- 零 LLM 成本首轮分析（`scripts/dual_channel_arbitration.py --task 10`，任务 10
  双通道分数 + 25 条人工标签为真值，原始输出 `dual_channel_arbitration_report.json`）：
  - **分歧行（|AR−AR_gen| > 0.2，10 条）：用低通道 90% 一致（9/10）**，用高通道
    仅 20%——分歧行 9 条人工=fail 且低通道=fail（截断/幻觉类），唯一低通道错误
    是 row 7（外部知识污染误杀，生成式 0.95 正确）；
  - **min(两通道) 组合最优**：阈值 0.7 时一致率 0.68 / kappa 0.3939，优于整体
    通道（0.60/0.2733）与生成式通道（0.48/0.1425）；
  - 机制解释：整体通道对截断/答非所问敏感但被污染误伤；生成式通道温和但截断
    也高分（RAGAS 同机制）。取低通道 = 两机制取保守侧，恰好消掉两边的过松；
  - 边界：任务 10 为校准前数据（旧 criteria + 阈值 0.5 政策），kappa 绝对值
    偏低；**任务 13（校准后）已用同脚本重跑对比（见 5.6 / 4.3a 数字）**。
- **验证标准（已量化 2026-08-07）**：校准后分歧样本上 min 组合 vs 单通道 vs 人工
  的一致性提升对比（任务 13 + 任务 10 人工真值：min kappa 0.5033 > ar 0.4337，
  见 meta-evaluation §5.6）。

### P1 人工校准闭环（✅ 实测+校准已执行，重跑验证已完成，数字见 4.3/4.3a）
- 25 条人工标注 → kappa **0.4582** → 报告自动触发 < 0.7 校准建议，机制真实跑通；
- 校准已落地：阈值 0.5→0.7 + AR criteria 修订（4.3a）；
- **验证中**：任务 11 重跑同 25 行 → `python -m scripts.calibration_compare
  --before 10 --after 11` 对比校准前后 kappa / 一致率 / 分歧分布；
- 扩样 50-100 条人工标注（kappa CI 变窄）+ 校准集入库 + 复评脚本为后续。

### P2 功能（P1 已闭环）
- ✅ **CI 门禁（2026-08-07）**：`.github/workflows/ci.yml`——push/PR 触发后端
  pytest 全量（114 用例，含评测脚本冒烟 `tests/test_scripts_smoke.py`：所有
  scripts/*.py 必须 --help 正常）+ 前端 `tsc -b && vite build` 类型门禁；
  期间给 compute_goldset_agreement / make_goldset_worksheet 补了 argparse
  （此前无参数解析，--help 会被忽略并直接执行真实计算）。
- 待做：归因评测（CitationAccuracy）、轨迹评测（反模式检测）。

### 主线：面试材料化（与上述并行）
- 一页纸项目简介（技术亮点 + 数字）、三分钟 demo 脚本、把踩坑记录改写成
  STAR 故事。技术验证最终要换算成 offer，别省这部分。

## 7. 文档地图

| 文档 | 内容 |
|---|---|
| [开发历程踩坑与成果记录.md](开发历程踩坑与成果记录.md) | **先读**：全部阶段复盘 + 可迁移方法论 |
| [元评价实验与前沿对标分析报告.md](元评价实验与前沿对标分析报告.md) | 实验全景：缺陷 A/B、RAGAS 对照、检索侧实测、前沿调研、路线图 |
| [meta-evaluation.md](meta-evaluation.md) | 元评价方法论（四层体系 + 实验 SOP + 实测记录） |
| [eval-dataset-guide.md](eval-dataset-guide.md) | 评测集构建指南（分层覆盖 + 防污染） |
| [evaluation-guide.md](evaluation-guide.md) | AI 评测指标与场景实践指南 |
| [检索侧黄金集标注表.md](检索侧黄金集标注表.md) | 检索侧人工黄金集 25 条判定（初标 + 用户复核一致） |
| [面试材料_一页纸.md](面试材料_一页纸.md) | 面试一页纸：技术亮点 + 真实数字（校准后 kappa 0.8175 已回填，2026-08-07） |
| [面试材料_STAR.md](面试材料_STAR.md) | 4 个 STAR 故事（假实验识别/污染定位/校准闭环/交叉验证） |
| `backend/scripts/` | 实验脚本（compare_with_ragas / ab_response_leak / merge_n25_report / retrieval_noise_experiment / compare_ragas_nonllm / check_dataset_contamination / bm25_mock_retriever / calibration_compare / make_goldset_worksheet / compute_goldset_agreement / dual_channel_arbitration）+ `_fixtures/`（双干扰语料）+ `retrieval_goldset_annotations.json`（人工黄金集） |
| `backend/ragas_comparison_report.json` / `retrieval_bm25_report.json` / `ragas_nonllm_report.json` / `calibration_compare_report.json` / `retrieval_goldset_agreement.json` / `dual_channel_arbitration_report.json` | 对照/检索/校准/黄金集/仲裁实验原始输出（带 provenance / 逐条 detail） |

## 8. 接手验证清单（先做这 5 件事，再动任何代码）

1. `cd backend && python -m pytest tests/ -q` → 期望 **109 passed**；
2. 启动后端 `python -m uvicorn app.main:app --port 8000` → `/docs`、`/api/datasets` 200；
3. `python -m scripts.retrieval_noise_experiment --dataset "售后政策对照集（含坏样本）"`
   → 期望与 4.2 表一致（H@1 0.84 / MRR 0.913 于最重场景）；
4. `python -m scripts.merge_n25_report`（注意不带 `.py` 后缀，带后缀会报 ModuleNotFoundError）→ 期望 6 个指标、每指标带 `_provenance`；
5. 读「7. 文档地图」里前三个文档（本文件 + 两份元评价文档），确认对实验结论的
   理解与 4 一致后再规划下一步。

---

**给接手 AI 的最后提醒**：这个项目的价值不在功能数量，在「测量可信」这件事上的
诚实与严谨——闭式循环的废弃、口径的拆分、机制的如实记录、数字的 provenance，
这些是简历上的差异点，也是接手后唯一不能破坏的东西。
