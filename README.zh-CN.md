# AI Evaluation Platform

[English](README.md) | [简体中文](README.zh-CN.md)

面向 RAG、AI Agent、多轮对话、LLM-as-a-Judge、接口评测、评测报告和人工盲测的自托管 AI 应用评测工作台。

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Backend](https://img.shields.io/badge/backend-FastAPI-009688.svg)](backend)
[![Frontend](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61dafb.svg)](frontend)
[![LLM](https://img.shields.io/badge/LLM-OpenAI--compatible-black.svg)](#配置项)

AI Evaluation Platform 帮助团队把大模型应用评测做成可复用的工程流程，而不是一次性的指标脚本。平台把数据集管理、评测场景、原生指标、OpenAI 兼容 Judge 模型、被测接口实时评测、报告分析、人工复核和 A/B 盲测串在一个 Web UI 里。

## 界面预览

### 评测实验工作台

在一个流程里完成数据集选择、场景配置、Judge 模型、被测接口、指标覆盖、调试验证和任务创建。

![评测实验工作台](docs/imgs/lab-workflow.png)

### 评测场景

内置 RAG、Agent、多轮对话等场景模板，支持调整指标、权重、阈值和 Prompt 覆盖配置。

![评测场景](docs/imgs/use%20cases.png)

### 自定义数据集

支持单轮样本、多轮对话、检索上下文、工具调用、参考答案和业务自定义字段。

![自定义数据集](docs/imgs/mul-dataset.png)

### LLM Judge 配置

通过 OpenAI 兼容接口配置 Judge 模型，可接入 Qwen、GPT 兼容服务或私有兼容网关。

![LLM Judge 配置](docs/imgs/LLM-judge.png)

### 评测执行

支持对已有输出做离线评测，也支持直接调用已保存的 Chat、RAG 或 Agent 接口做实时评测。

![评测执行](docs/imgs/eval.png)

### 评测报告

查看任务状态、通过率、指标摘要、报告对比入口和评测进度。

![评测报告](docs/imgs/report.png)

### 报告详情

逐条查看指标得分、Judge 理由、是否通过、接口调用 trace 和人工复核字段。

![报告详情](docs/imgs/reportDetail.png)

### 人工盲测

随机展示两个 LLM 或接口输出，收集人工投票并生成统计结果。

![人工盲测](docs/imgs/blind-test.png)

## 为什么做这个项目

很多评测工具只覆盖评测链路的一部分，比如指标计算、合成数据集生成或 Prompt 实验。本项目更偏向产品化评测工作台，目标是把 AI 应用上线前后需要的评测动作连起来。

- 从数据集、场景、指标到执行、报告、人工复核和盲测都有 Web UI。
- 支持离线评测，也支持直接调用 Chat、RAG、Agent 被测接口做实时评测。
- 内置 RAG、Agent、多轮对话指标，并支持自定义 Judge Prompt。
- 自动评测之外，提供人工盲测和人工复核，帮助团队做上线决策。
- 默认 SQLite 便于本地快速体验，也可以通过 SQLAlchemy `DATABASE_URL` 切换到 PostgreSQL 等关系型数据库。

相比 Ragas、rag_eval 这类偏指标库的项目，或偏数据集生产的工具，本项目更强调评测运营流程：收集样本、固化场景快照、运行可对比任务、查看逐条评分理由，并把自动评测与人工判断结合起来。

## 核心能力

| 能力 | 说明 |
|------|------|
| 数据集管理 | 支持单轮样本、多轮 conversation、RAG contexts、tool calls、CSV/JSON 导入导出 |
| RAG 评测 | Faithfulness、Context Recall、Context Precision、Answer Relevancy、HitRate@K、MRR 等 |
| Agent 评测 | Task Completion、Tool Correctness、Argument Correctness、Step Efficiency、Goal Accuracy |
| 多轮对话评测 | Topic Adherence、Turn Relevancy、Conversation Completeness、Knowledge Retention、Role Adherence |
| 接口实时评测 | 调用被测 endpoint，抽取 response、contexts、tool calls 后直接评测 |
| 人工盲测 | 支持 LLM vs LLM、Endpoint vs Endpoint、LLM vs Endpoint 主观对比 |
| 报告与复核 | 任务进度、汇总分、逐条评分理由、人工状态、报告对比 |
| OpenAI 兼容 Judge | 默认 DashScope Qwen Plus，可替换为任意 OpenAI 兼容 Chat 模型 |

## 核心场景

### RAG 评测

同时评估检索和生成质量：

- 检索质量：Context Recall、Context Precision、Contextual Relevancy。
- 生成可信度：Faithfulness、Factual Correctness。
- 回答质量：Answer Relevancy、Answer Completeness。
- 检索单测：当样本有文档 ID 时，可计算 HitRate@K 和 MRR。

### Agent 评测

评估 Agent 是否通过正确的工具调用完成任务：

- Task Completion 和 Goal Accuracy。
- Tool Correctness。
- Argument Correctness。
- Step Efficiency。

适合客服 Agent、任务执行 Agent、流程型 Agent 和工具调用链路。

### 多轮对话评测

评估多轮对话过程中的稳定性：

- Topic Adherence。
- Turn Relevancy。
- Conversation Completeness。
- Knowledge Retention。
- Role Adherence。

适合客服机器人、企业助手和连续问答系统。

### 接口 A/B 盲测

用人工偏好对比两个响应来源：

- LLM 配置 vs LLM 配置。
- Endpoint vs Endpoint。
- LLM 配置 vs Endpoint。

适合新旧模型、新旧 RAG 服务、Prompt 版本或 Agent 实现上线前对比。

## 评测产出

每次评测完成后会产出：

- 总体通过率和指标摘要。
- 逐条样本得分、通过状态、Judge 理由和耗时。
- 实时接口评测的 endpoint trace。
- 人工复核状态、分数、标签和备注。
- 用于回归检查的报告对比。

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | FastAPI、SQLAlchemy、Pydantic，默认 SQLite |
| 前端 | React 18、TypeScript、Ant Design 5、Vite |
| 评测引擎 | 原生指标执行器 + OpenAI 兼容 Judge Prompt |
| 默认 Judge 模型 | DashScope 兼容模式下的 Qwen Plus |

## 快速开始

### 环境要求

- Python 3.9+，推荐 Python 3.11+
- Node.js 18+
- `uv` 或 `pip`
- 一个 OpenAI 兼容模型 API Key，例如 DashScope

### 1. 配置后端

```bash
cd backend
cp .env.example .env
```

编辑 `backend/.env`：

```bash
LLM_API_KEY=your-api-key
LLM_ENDPOINT=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
```

安装依赖：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

启动后端：

```bash
python3 run.py
```

打开：

```text
http://localhost:8000/docs
```

### 2. 启动前端

```bash
cd frontend
npm install
npm run dev
```

打开：

```text
http://localhost:5173
```

## 典型流程

```text
1. 配置 Judge LLM
2. 创建或导入数据集
3. 选择 RAG、Agent 或多轮对话评测场景
4. 运行离线评测或接口实时评测
5. 查看报告摘要、逐条评分理由和失败样本
6. 结合人工复核或盲测结果做上线决策
```

## 文档

- [使用指南](docs/usage-guide.md)
- [评测指南](docs/evaluation-guide.md)
- [开源发布检查清单](docs/open-source-readiness.md)
- [数据库与存储选型](docs/database-and-storage.md)
- [同类项目对比](docs/comparison.md)

## 配置项

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | Judge 模型 API Key | 真实调用时必填 |
| `LLM_MODEL` | Judge 模型名称 | `qwen-plus` |
| `LLM_ENDPOINT` | OpenAI 兼容 API 地址 | DashScope 兼容模式地址 |
| `DATABASE_URL` | SQLAlchemy 数据库连接 | `sqlite:///./eval_platform.db` |
| `UPLOAD_DIR` | 上传文件目录 | `./uploads` |
| `CORS_ORIGINS` | 允许访问的前端域名 | `["http://localhost:5173"]` |
| `SEED_ON_STARTUP` | 启动时创建演示数据 | `true` |
| `RUN_EVAL_ON_CREATE` | 创建任务后立即执行评测 | `true` |

团队或生产环境建议使用 PostgreSQL：

```bash
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/ai_eval_platform
```

切换存储前建议阅读 [数据库与存储选型](docs/database-and-storage.md)。

## 开发

后端测试：

```bash
cd backend
source venv/bin/activate
python3 -m pytest tests/ -v
```

前端构建：

```bash
cd frontend
npm run build
```

## 安全说明

- 不要提交 `.env`、SQLite 数据库、上传文件、WAL/SHM 文件或本地 IDE 元数据。
- API Key 和 endpoint Authorization Header 会用于执行，但 API 响应中会做脱敏。
- 当前项目默认面向可信内部用户；如果暴露到非可信网络，请先补充认证、授权、审计日志和密钥加密。
- 发布已有 Git 历史前，建议扫描并清理历史提交里可能存在的数据库或密钥。

披露安全问题请参考 [SECURITY.md](SECURITY.md)。

## 交流与共建

如果这个项目对你的 RAG、AI Agent、多轮对话评测或 AI Infra 工作有帮助，欢迎给项目一个 Star，也欢迎关注后续更新、提交 Issue、贡献 PR 或分享你的真实评测场景。

作者长期从事一线互联网公司后端开发与架构工作，近几年持续关注并实践 AI Agent、AI Infra、LLM 应用评测和工程化落地。如果你也在做相关方向，欢迎交流想法、使用反馈和共建建议。

- 微信：`huangyiminghappy`
- 邮箱：`huangyiminghappy@gmail.com`

## License

Apache License 2.0. See [LICENSE](LICENSE).
