# AI Evaluation Platform

面向 RAG、AI Agent、多轮对话和接口 A/B 盲测的开源 AI 应用评测平台。平台提供数据集管理、评测场景配置、原生指标执行、OpenAI-compatible Judge LLM 打分、评测报告、人工复核和人工盲测工作流。

An open-source evaluation platform for RAG, AI Agent, multi-turn conversation, and endpoint A/B blind testing. It includes dataset management, scenario-based metric configuration, native metric execution, OpenAI-compatible judge models, reports, manual review, and human preference testing.

## What It Looks Like / 界面与效果

The platform is designed as an evaluation workbench rather than a metric-only SDK. A typical user can configure resources, run evaluations, inspect reports, and make release decisions without leaving the Web UI.

平台不是单纯的指标库，而是一个完整评测工作台。典型用户可以在 Web 界面里完成资源配置、评测执行、报告分析和上线决策。

| Area | UI Entry | What You Get |
|------|----------|--------------|
| Experiment Workbench / 评测实验工作台 | `/workbench` | One-page workflow for selecting dataset, scenario, judge model, endpoint target, metric overrides, debug run, and task creation |
| Dataset Management / 数据管理 | `/datasets` | Dataset schema, rows, CSV/JSON import/export, RAG generated samples, conversation/tool-call fields |
| Metric & Scenario Management / 指标与场景 | `/metrics`, `/scenarios` | Built-in metric catalog, scenario templates for RAG/Agent/multi-turn, per-scenario thresholds and prompt overrides |
| Endpoint Targets / 被测接口 | `/endpoint-targets` | Saved Chat/RAG/Agent endpoints with request templates, headers, response field mapping, and connectivity test |
| Evaluation Execution / 评测执行 | `/evaluations` | Offline evaluation or live endpoint evaluation, task progress, logs, clone/rerun workflow |
| Reports / 评测报告 | `/reports` | Summary score, pass rate, metric breakdown, row-level judge reasons, manual review, report comparison |
| Blind Test / 人工盲测 | `/blind-tests` | LLM vs LLM, endpoint vs endpoint, or LLM vs endpoint A/B blind comparison with human votes |

### Product Tour / 产品界面预览

**Evaluation Workbench / 评测实验工作台**

The workbench connects dataset selection, scenario configuration, judge model, endpoint target, debug validation, and task creation in one workflow.

![Evaluation workbench](docs/imgs/lab-workflow.png)

**Evaluation Scenarios / 评测场景**

Preset scenario templates cover RAG, Agent, and multi-turn conversation evaluation. Teams can adjust metrics, weights, thresholds, and prompt overrides.

![Evaluation scenarios](docs/imgs/use%20cases.png)

**Custom Dataset / 自定义数据集**

Datasets can model single-turn samples, multi-turn conversations, retrieved contexts, tool calls, reference answers, and custom business fields.

![Custom dataset](docs/imgs/mul-dataset.png)

**LLM Judge Configuration / LLM Judge 配置**

Judge models are configured through OpenAI-compatible endpoints, so the platform can use Qwen, GPT-compatible services, or private compatible gateways.

![LLM judge configuration](docs/imgs/LLM-judge.png)

**Evaluation Execution / 评测执行**

Run offline evaluation on existing outputs or live endpoint evaluation by calling a saved Chat/RAG/Agent API and mapping response fields into metrics.

![Evaluation execution](docs/imgs/eval.png)

**Reports / 评测报告**

Reports show task status, pass rate, metric summaries, report comparison entry points, and progress across evaluation runs.

![Evaluation report](docs/imgs/report.png)

**Report Detail / 报告详情**

Row-level details include metric scores, judge reasons, pass/fail status, endpoint trace, and manual review fields.

![Report detail](docs/imgs/reportDetail.png)

**Blind Test / 人工盲测**

Blind tests compare two LLM or endpoint outputs with randomized display order, human votes, and summary statistics.

![Blind test](docs/imgs/blind-test.png)

## Why This Project

很多评测工具只覆盖“库级指标计算”或“数据集生成”其中一段。本项目的定位是把评测工程里的关键环节连起来：

- 从数据集、场景、指标到执行报告都有 Web UI。
- 支持离线评测，也支持直接调用被测 Chat/RAG/Agent 接口做实时评测。
- 内置 RAG、Agent、多轮对话指标，并保留中文业务指标和自定义 Prompt 的空间。
- 自动评测之外，提供人工盲测和结果级人工复核，适合上线前决策。
- 默认使用 SQLite 便于快速体验，同时通过 SQLAlchemy `DATABASE_URL` 支持迁移到 PostgreSQL/MySQL 等服务型数据库。

Compared with library-first evaluation tools, this project is product-workflow-first: it helps teams collect samples, freeze scenario snapshots, run repeatable evaluations, inspect row-level reasons, and combine automatic scores with human decisions.

## Feature Highlights

| 能力 | 说明 |
|------|------|
| 数据集管理 | 支持单轮、多轮 conversation、RAG contexts、tool calls、CSV/JSON 导入导出 |
| RAG 评测 | Faithfulness、Context Recall、Context Precision、Answer Relevancy、HitRate@K、MRR 等 |
| Agent 评测 | Task Completion、Tool Correctness、Argument Correctness、Step Efficiency、Goal Accuracy |
| 多轮对话评测 | Topic Adherence、Turn Relevancy、Conversation Completeness、Knowledge Retention、Role Adherence |
| 接口实时评测 | 对被测 endpoint 发请求，抽取 response/contexts/tool calls 后直接评测 |
| 人工盲测 | 支持 LLM vs LLM、Endpoint vs Endpoint、LLM vs Endpoint 主观对比 |
| 报告与复核 | 任务进度、汇总分、逐条评分理由、人工状态、报告对比 |
| OpenAI 兼容 | 默认 DashScope Qwen Plus，可替换为任意 OpenAI-compatible Chat Completions 模型 |

## Core Scenarios / 核心场景

### RAG Evaluation / RAG 评测

Evaluate retrieval and generation together:

- Retrieval quality: Context Recall, Context Precision, Contextual Relevancy.
- Generation trust: Faithfulness, Factual Correctness.
- Answer quality: Answer Relevancy, Answer Completeness.
- Retrieval unit tests: HitRate@K and MRR when document IDs are available.

评估 RAG 时不仅看回答，还能拆开看检索是否召回、上下文是否有噪声、回答是否基于证据、是否覆盖标准答案。

### Agent Evaluation / Agent 评测

Evaluate whether an agent can complete tasks through tool use:

- Task Completion and Goal Accuracy.
- Tool Correctness.
- Argument Correctness.
- Step Efficiency.

适合客服 Agent、任务执行 Agent、工具调用链路等场景，重点判断工具有没有调对、参数是否正确、任务是否真的闭环。

### Multi-Turn Conversation Evaluation / 多轮对话评测

Evaluate conversation stability over turns:

- Topic Adherence.
- Turn Relevancy.
- Conversation Completeness.
- Knowledge Retention.
- Role Adherence.

适合客服机器人、企业助手、连续问答系统，重点看是否跑题、忘上下文、前后矛盾或越过角色边界。

### Endpoint A/B Blind Test / 接口 A/B 盲测

Compare two response sources with human preference:

- LLM config vs LLM config.
- Endpoint vs endpoint.
- LLM config vs endpoint.

适合新旧模型、新旧 RAG 服务、不同 Agent 版本上线前的主观体验对比。

## Result Output / 评测产出

Each completed evaluation produces:

- Overall pass rate and metric summary.
- Row-level score, pass/fail status, judge reason, and execution time.
- Endpoint trace for live endpoint evaluation.
- Manual review status, score, tags, and notes.
- Report comparison for regression checks.

一次评测完成后，你可以看到总体通过率、各指标均分、逐条样本评分理由、接口调用 trace、人工复核状态，以及两次报告之间的差异。

## Tech Stack

| Layer | Stack |
|-------|-------|
| Backend | FastAPI, SQLAlchemy, Pydantic, SQLite by default |
| Frontend | React 18, TypeScript, Ant Design 5, Vite |
| Evaluation engine | Native metric executors plus OpenAI-compatible judge prompts |
| Default judge model | Qwen Plus through DashScope compatible mode |

## Quick Start

### Prerequisites

- Python 3.9+，推荐 Python 3.11+
- Node.js 18+
- `uv` 或 `pip`
- 一个 OpenAI-compatible 模型 API Key，例如 DashScope

### 1. Configure Backend

```bash
cd backend
cp .env.example .env
```

Edit `backend/.env` and set:

```bash
LLM_API_KEY=your-api-key
LLM_ENDPOINT=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
```

Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start the backend:

```bash
python3 run.py
```

Health check:

```text
http://localhost:8000/api/health
```

Swagger API:

```text
http://localhost:8000/docs
```

### 2. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## Typical Workflow

```text
1. Configure Judge LLM
2. Create or import datasets
3. Select a scenario: RAG, Agent, or multi-turn conversation
4. Run offline evaluation or endpoint evaluation
5. Inspect report, row-level reasons, and failures
6. Use manual review or blind test for release decisions
```

## Documentation

- [Usage Guide](docs/usage-guide.md): 操作手册
- [Evaluation Guide](docs/evaluation-guide.md): 指标与场景实践
- [Open Source Readiness](docs/open-source-readiness.md): 开源发布检查清单
- [Database and Storage](docs/database-and-storage.md): 数据库存储选型建议
- [Comparison](docs/comparison.md): 与 Ragas、Easy Dataset、rag_eval 类工具的差异

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_API_KEY` | Judge model API key | Required for real LLM calls |
| `LLM_MODEL` | Judge model name | `qwen-plus` |
| `LLM_ENDPOINT` | OpenAI-compatible API endpoint | DashScope compatible endpoint |
| `DATABASE_URL` | SQLAlchemy database URL | `sqlite:///./eval_platform.db` |
| `UPLOAD_DIR` | Uploaded document directory | `./uploads` |
| `CORS_ORIGINS` | Allowed frontend origins | `["http://localhost:5173"]` |
| `SEED_ON_STARTUP` | Seed demo data at startup | `true` |
| `RUN_EVAL_ON_CREATE` | Run evaluation immediately after task creation | `true` |

For production or shared team environments, prefer PostgreSQL:

```bash
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/ai_eval_platform
```

See [Database and Storage](docs/database-and-storage.md) before changing the storage backend.

## Development

Backend tests:

```bash
cd backend
source venv/bin/activate
python3 -m pytest tests/ -v
```

Frontend type check and build:

```bash
cd frontend
npm run build
```

## Security Notes

- Never commit `.env`, SQLite databases, uploaded files, WAL/SHM files, or local IDE metadata.
- API keys and endpoint authorization headers are stored for execution but masked in API responses.
- The current project is designed for trusted internal users. Add authentication, authorization, audit logs, and secret encryption before exposing it to untrusted networks.
- Before publishing an existing Git history, scan and clean historical commits that may contain databases or secrets.

See [SECURITY.md](SECURITY.md) for disclosure guidance.

## English Summary

AI Evaluation Platform is a self-hosted evaluation workbench for LLM applications. It is especially useful when you need more than a metric library: dataset curation, scenario presets, endpoint execution, report inspection, manual review, and human blind testing in one workflow.

The default setup is intentionally lightweight with SQLite and seeded demo data. For team or production usage, run the backend behind authentication and move persistent storage to PostgreSQL or another managed relational database.

## Community / 交流

如果这个项目对你的 RAG、AI Agent、多轮对话评测或 AI Infra 工作有帮助，欢迎给项目一个 Star，也欢迎关注后续更新、提交 Issue、贡献 PR 或分享你的真实评测场景。

作者长期从事一线互联网公司后端开发与架构工作，近几年持续关注并实践 AI Agent、AI Infra、LLM 应用评测和工程化落地。如果你也在做相关方向，欢迎交流想法、使用反馈和共建建议。

- WeChat: `huangyiminghappy`
- Email: `huangyiminghappy@gmail.com`

If this project is useful for your RAG, AI Agent, multi-turn conversation evaluation, or AI infrastructure work, a Star would be greatly appreciated. You are also welcome to watch the repository, open issues, contribute pull requests, or share real evaluation use cases.

The author has years of hands-on backend engineering and architecture experience in large-scale internet systems, and has recently been working in AI Agent, AI infrastructure, LLM application evaluation, and production-oriented AI engineering. Feel free to reach out for technical discussions, feedback, or collaboration.

## License

Apache License 2.0. See [LICENSE](LICENSE).
