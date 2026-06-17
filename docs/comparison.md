# Comparison / 与类似项目的差异

本文帮助外部用户理解 AI Evaluation Platform 与 Ragas、Easy Dataset、rag_eval 类项目的关系。这里的对比重点是产品定位和工程边界，不表示谁“更好”，而是解决的问题不同。

## Short Answer / 一句话结论

- **Ragas** 更像评测 SDK/库：适合在代码里快速接入指标、实验和数据生成。
- **Easy Dataset** 更像数据集生产与桌面工作台：强在文档解析、数据生成、微调数据导出和多语言 UI。
- **rag_eval 类项目** 多数更像示例工程或脚本化 RAG 评测流水线。
- **AI Evaluation Platform** 更像自托管评测工作台：强调数据集、场景、原生指标、接口实时评测、报告、人工复核和盲测闭环。

## Comparison Matrix / 对比矩阵

| 维度 | AI Evaluation Platform | Ragas | Easy Dataset | rag_eval 类项目 |
|------|------------------------|-------|--------------|-----------------|
| 主要形态 | Web 平台 + 后端评测服务 | Python SDK / toolkit | 桌面/应用式数据集工具 | 示例项目/脚本/小型包 |
| 核心目标 | 管理和执行 AI 应用评测闭环 | 在代码中评估 LLM 应用 | 从文档生成、清洗、管理数据集 | 快速演示 RAG 评测流程 |
| 数据集管理 UI | 有 | 通常需自行接入 | 强 | 通常较弱 |
| RAG 指标 | 内置并原生执行 | 强，生态成熟 | 有相关能力 | 常依赖 Ragas 或自定义 |
| Agent 指标 | 内置工具调用、任务完成等指标 | 新版本方向覆盖 Agent evals | 视版本能力而定 | 通常有限 |
| 多轮对话指标 | 内置 topic/turn/role/retention 类指标 | 可自定义或扩展 | 有对话数据生成能力 | 通常有限 |
| Endpoint 实时评测 | 一等能力，可保存 endpoint、请求模板、字段映射 | 需要工程接入 | 可按产品能力配置 | 通常手写脚本 |
| 报告和人工复核 | 内置报告、逐条理由、人工状态 | 需要自建 UI 或使用集成平台 | 有评测/盲测相关 UI | 通常输出文件 |
| 人工盲测 | 内置 LLM/Endpoint A/B 盲测 | 不是核心 UI 能力 | 新版本已有 Arena/盲测能力 | 通常没有 |
| 中文业务适配 | 默认中文 UI、中文指标说明、中文 Judge Prompt 友好 | 国际化 SDK | 多语言强 | 视项目而定 |
| 默认部署 | 自托管 FastAPI + React | 库安装 | 应用/客户端 | 本地脚本 |

## Advantages of This Project / 本项目优势

### 1. Evaluation Workflow, Not Only Metrics

本项目不仅计算指标，还管理评测生命周期：

```text
数据集 -> 场景模板 -> 指标配置 -> 执行任务 -> 报告 -> 人工复核 -> 对比决策
```

这对团队协作很重要，因为实际评测常常不是一次 `evaluate()` 调用，而是持续迭代数据集、Prompt、检索器、Agent 工具链和上线阈值。

### 2. Native Endpoint Evaluation

很多团队的被测对象不是一个 Python 函数，而是已经部署好的 Chat/RAG/Agent HTTP 服务。本项目把 endpoint 配置作为一等对象：

- 保存 URL、Authorization、Headers、请求体模板。
- 支持 JSON 与 SSE 流式接口。
- 支持通过字段映射抽取 `response`、`retrieved_contexts`、`tool_calls`。
- 创建评测任务时冻结 endpoint 快照，历史报告可复现。

### 3. Automatic + Human Evaluation Loop

自动评分适合规模化，人工判断适合上线决策。本项目把两者放在同一个工作流里：

- 自动指标输出分数和理由。
- 报告逐条支持人工复核状态和备注。
- 人工盲测支持模型/接口 A/B 主观对比。

### 4. Scenario-Oriented Metrics

本项目按应用场景组织指标：

- RAG：检索质量、生成可信度、回答质量、检索单测。
- Agent：任务完成、工具选择、参数正确性、步骤效率。
- 多轮对话：话题保持、轮次相关、对话完整、知识保持、角色遵守。

这比“把一堆指标平铺给用户”更适合评测工程落地。

### 5. Chinese-First Business Evaluation

项目默认文档、UI、预置场景和说明都偏中文业务团队使用。它更适合中文 RAG、客服 Agent、内部知识库、企业助手等场景。

## What This Project Is Not / 不是什么

- 不是 Ragas 的二次封装。
- 不承诺与 Ragas 同名指标逐项分数完全一致。
- 不是模型微调数据生产工具的完整替代。
- 不是开箱即用的 SaaS 多租户平台。
- 不是生产级密钥管理系统。

## When to Use Which / 选择建议

| 需求 | 推荐 |
|------|------|
| 只想在 Python 代码中快速评估 RAG 指标 | Ragas |
| 重点是从 PDF/DOCX/Markdown 生成微调或评测数据 | Easy Dataset |
| 想学习 RAG 评测最小示例或实验脚本 | rag_eval 类项目 |
| 想搭一个团队可用的自托管评测工作台 | AI Evaluation Platform |
| 想把线上 endpoint 直接纳入评测和报告 | AI Evaluation Platform |
| 想结合自动评分、人工复核和人工盲测 | AI Evaluation Platform 或 Easy Dataset 新版本，按你的数据生产/评测侧重点选择 |

## Relationship to Referenced Projects / 与参考项目的关系

本项目参考了业界常见评测概念和多个开源项目的产品思路，例如：

- Ragas 的 RAG 指标体系和 LLM application evaluation 思路。
- Easy Dataset 的数据集工作台和人机协同数据流程。
- rag_eval 类项目的轻量 RAG 评测实验方式。

但当前实现是平台原生执行器：后端不依赖 Ragas 或 Easy Dataset 作为运行时依赖，指标执行、Judge Prompt、报告结构和接口评测由本项目自行实现。

## English Summary

Use Ragas when you need a Python evaluation toolkit. Use Easy Dataset when dataset generation and curation is the center of gravity. Use rag_eval-style projects for minimal RAG evaluation examples. Use AI Evaluation Platform when you want a self-hosted evaluation workbench with datasets, scenarios, endpoint execution, native metrics, reports, manual review, and blind testing in one place.
