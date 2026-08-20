# AI Evaluation Platform

[English](README.md) | [简体中文](README.zh-CN.md)

Self-hosted AI evaluation workbench for RAG, AI Agents, multi-turn conversations, LLM-as-a-Judge, endpoint evaluation, reports, and human blind testing.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Backend](https://img.shields.io/badge/backend-FastAPI-009688.svg)](backend)
[![Frontend](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61dafb.svg)](frontend)
[![LLM](https://img.shields.io/badge/LLM-OpenAI--compatible-black.svg)](#configuration)

AI Evaluation Platform helps teams evaluate LLM applications as an engineering workflow instead of a one-off metric script. It connects dataset management, scenario presets, native metrics, OpenAI-compatible judge models, live endpoint evaluation, report analysis, manual review, and A/B blind testing in one Web UI.

## Screenshots

### Evaluation Workbench

Configure dataset, scenario, judge model, endpoint target, metric overrides, debug validation, and task creation in one workflow.

![Evaluation workbench](docs/imgs/lab-workflow.png)

### Evaluation Scenarios

Use built-in scenario templates for RAG, Agent, and multi-turn conversation evaluation. Teams can adjust metrics, weights, thresholds, and prompt overrides.

![Evaluation scenarios](docs/imgs/use%20cases.png)

### Custom Datasets

Model single-turn samples, multi-turn conversations, retrieved contexts, tool calls, reference answers, and custom business fields.

![Custom dataset](docs/imgs/mul-dataset.png)

### LLM Judge Configuration

Configure judge models through OpenAI-compatible endpoints, including Qwen, GPT-compatible services, and private compatible gateways.

![LLM judge configuration](docs/imgs/LLM-judge.png)

### Evaluation Execution

Run offline evaluation on existing outputs or live endpoint evaluation by calling saved Chat, RAG, or Agent APIs.

![Evaluation execution](docs/imgs/eval.png)

### Reports

Inspect task status, pass rate, metric summaries, report comparison entry points, and evaluation progress.

![Evaluation report](docs/imgs/report.png)

### Report Detail

Review row-level metric scores, judge reasons, pass/fail status, endpoint traces, and manual review fields.

![Report detail](docs/imgs/reportDetail.png)

### Human Blind Test

Compare two LLM or endpoint outputs with randomized display order, human votes, and summary statistics.

![Blind test](docs/imgs/blind-test.png)

## Why This Project

Many evaluation tools focus on only one part of the workflow: metric calculation, synthetic dataset generation, or prompt-level experiments. This project is designed as a product-oriented evaluation workbench.

- Web UI for datasets, scenarios, metrics, execution, reports, manual review, and blind tests.
- Offline evaluation and live endpoint evaluation for Chat, RAG, and Agent services.
- Built-in metrics for RAG, Agent, and multi-turn conversations, with custom judge prompts.
- Human blind testing and manual review for release decisions beyond automatic scores.
- Lightweight SQLite quick start, with SQLAlchemy `DATABASE_URL` support for PostgreSQL and other relational databases.

Compared with library-first tools such as Ragas, rag_eval, or dataset-focused tools, this platform emphasizes repeatable evaluation operations: collect samples, freeze scenario snapshots, run comparable tasks, inspect row-level reasons, and combine automatic judging with human decisions.

## Feature Highlights

| Area | What It Provides |
|------|------------------|
| Dataset management | Single-turn rows, multi-turn conversations, RAG contexts, tool calls, CSV/JSON import and export |
| RAG evaluation | Faithfulness, Context Recall, Context Precision, Answer Relevancy, HitRate@K, MRR, and more |
| Agent evaluation | Task Completion, Tool Correctness, Argument Correctness, Step Efficiency, Goal Accuracy |
| Multi-turn evaluation | Topic Adherence, Turn Relevancy, Conversation Completeness, Knowledge Retention, Role Adherence |
| Live endpoint evaluation | Call a target endpoint, extract response/contexts/tool calls, and evaluate the result directly |
| Human blind testing | Compare LLM vs LLM, endpoint vs endpoint, or LLM vs endpoint outputs |
| Reports and review | Task progress, summary score, row-level judge reasons, manual status, and report comparison |
| OpenAI-compatible judge | Default Qwen Plus through DashScope compatible mode; replaceable with any compatible chat model |
| Weighted total & judge stability | Weighted total score aggregated by scenario snapshot weights; optional multi-sample judging with per-metric std and low-confidence rows |
| Evaluation cost tracking | Per-row judge token usage recorded, total tokens and estimated cost shown in reports (unit price configurable) |
| Dual-channel metrics | Generative answer relevancy (reverse-question + semantic similarity) and claim-level faithfulness (atomic claim decomposition and verification), coexisting with holistic metrics for cross-validation |
| Judge engineering | Optional forced CoT, position-swap consistency check, multi-judge panel aggregation (mean/majority + inter-judge MAD) |
| Human calibration loop | Reports compute Cohen's kappa, Bootstrap intervals, and annotator-ceiling status; scores below 0.7 trigger review suggestions and pilot results are not presented as general accuracy |
| Dataset versioning | Version auto-increments on row changes and is frozen at task creation for traceability |
| Dataset governance tools | Contamination check (n-gram + embedding similarity), retrieval noise injection experiment, zero-dependency Chinese BM25 mock retriever |

## Core Scenarios

### RAG Evaluation

Evaluate retrieval and generation together:

- Retrieval quality: Context Recall, Context Precision, Contextual Relevancy.
- Generation trust: Faithfulness and Factual Correctness.
- Answer quality: Answer Relevancy and Answer Completeness.
- Retrieval unit checks: HitRate@K and MRR when document IDs are available.

### Agent Evaluation

Evaluate whether an Agent completes the task through correct tool use:

- Task Completion and Goal Accuracy.
- Tool Correctness.
- Argument Correctness.
- Step Efficiency.

This is useful for customer-service agents, task execution agents, workflow agents, and tool-calling systems.

### Multi-Turn Conversation Evaluation

Evaluate conversation stability across turns:

- Topic Adherence.
- Turn Relevancy.
- Conversation Completeness.
- Knowledge Retention.
- Role Adherence.

This is useful for customer support bots, enterprise assistants, and continuous Q&A systems.

### Endpoint A/B Blind Test

Compare two response sources with human preference:

- LLM config vs LLM config.
- Endpoint vs endpoint.
- LLM config vs endpoint.

This is useful before releasing a new model, RAG service, prompt version, or Agent implementation.

## Result Output

Each completed evaluation produces:

- Overall pass rate and metric summary.
- Row-level score, pass/fail status, judge reason, and execution time.
- Endpoint trace for live endpoint evaluation.
- Manual review status, score, tags, and notes.
- Report comparison for regression checks.

## Tech Stack

| Layer | Stack |
|-------|-------|
| Backend | FastAPI, SQLAlchemy, Pydantic, SQLite by default |
| Frontend | React 18, TypeScript, Ant Design 5, Vite |
| Evaluation engine | Native metric executors plus OpenAI-compatible judge prompts |
| Default judge model | Qwen Plus through DashScope compatible mode |

## Quick Start

### Prerequisites

- Python 3.9+, Python 3.11+ recommended
- Node.js 18+
- `uv` or `pip`
- An OpenAI-compatible model API key, such as DashScope

### 1. Configure Backend

```bash
cd backend
cp .env.example .env
```

Edit `backend/.env`:

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

Open:

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
1. Configure a Judge LLM
2. Create or import datasets
3. Select a scenario: RAG, Agent, or multi-turn conversation
4. Run offline evaluation or live endpoint evaluation
5. Inspect report summaries, row-level reasons, and failures
6. Use manual review or blind test results for release decisions
```

## Roadmap

The first public release focuses on a complete local evaluation workflow. Upcoming work will make the project easier to deploy, operate, and extend in team environments.

- v0.2: Docker Compose quick start, GitHub Actions CI, and release checklist automation.
- v0.3: Alembic migrations, richer PostgreSQL deployment guidance, and indexes for larger evaluation datasets.
- v0.4: Authentication, RBAC, audit logs, and encrypted storage for sensitive fields.
- v0.5: More built-in scenario templates, import/export examples, and benchmark datasets.
- Later: SDK/CLI integration, pluggable judge providers, observability integration, and public demo assets.

## Documentation

- [Usage Guide](docs/usage-guide.md)
- [Evaluation Guide](docs/evaluation-guide.md)
- [Open Source Readiness](docs/open-source-readiness.md)
- [Database and Storage](docs/database-and-storage.md)
- [Comparison](docs/comparison.md)
- [Changelog](CHANGELOG.md)

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

Frontend build:

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

## Community

If this project is useful for your RAG, AI Agent, multi-turn conversation evaluation, or AI infrastructure work, a Star would be greatly appreciated. You are also welcome to watch the repository, open issues, contribute pull requests, or share real evaluation use cases.


## License

Apache License 2.0. See [LICENSE](LICENSE).
