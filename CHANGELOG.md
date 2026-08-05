# Changelog

All notable changes to AI Evaluation Platform are documented in this file.

This project follows semantic versioning where practical. The early releases are expected to evolve quickly while the public APIs and deployment workflow stabilize.

## [Unreleased]

### Added

- **Weighted total score**: reports now include an overall score aggregated by scenario snapshot weights, giving experiments a single comparable number.
- **Judge reliability quantification**: optional multi-sample judging (`EVAL_JUDGE_SAMPLES`); reports expose per-metric sampling std and low-confidence row counts.
- **Cost tracking**: per-row judge token usage is recorded; reports show total tokens and an estimated cost (unit price configurable).
- **Manual agreement**: reports show agreement between automatic scores and human review conclusions for cross-validation.
- **WebSocket live progress**: evaluation logs and progress are pushed over WebSocket, with automatic polling fallback on disconnect.
- **Stale task recovery & endpoint retries**: tasks left running after a restart are marked failed; endpoint calls retry with backoff.
- Added evaluation dataset building guide (docs/eval-dataset-guide.md).
- Added meta-evaluation guide (docs/meta-evaluation.md) with a RAGAS comparison script (scripts/compare_with_ragas.py) and real experiment records.

- Added dual-channel metrics: generative answer relevancy (reverse-question + semantic similarity) and claim-level faithfulness (atomic claim decomposition & verification).
- Judge engineering: forced CoT (JUDGE_COT_MODE), position-swap consistency check (EVAL_SWAP_CHECK), multi-judge panel (judge_llm_config_ids) with mean/majority aggregation and inter-judge MAD.
- Human calibration loop: reports compute Cohen's kappa and suggest rubric revisions below 0.7.
- Dataset versioning: version auto-increments on row changes and is frozen at task creation.
- Dataset governance tools: contamination check (n-gram + embedding), retrieval noise injection experiment, zero-dependency Chinese BM25 mock retriever (backend/scripts/).

## [v0.1.0] - 2026-07-20

### Highlights

- First public open-source release of AI Evaluation Platform.
- Self-hosted evaluation workbench for RAG, AI Agents, multi-turn conversations, endpoint evaluation, LLM-as-a-Judge, reports, manual review, and human blind testing.
- English and Simplified Chinese README entry points with product screenshots.

### Included

- FastAPI backend with SQLAlchemy models and native evaluation executors.
- React 18, TypeScript, Ant Design 5, and Vite frontend.
- Dataset management for single-turn rows, multi-turn conversations, RAG contexts, tool calls, reference answers, and custom fields.
- Scenario-oriented metrics for RAG, Agent, and multi-turn conversation evaluation.
- OpenAI-compatible Judge LLM configuration, with DashScope Qwen Plus as the default example.
- Offline evaluation and live endpoint evaluation for saved Chat, RAG, and Agent endpoints.
- Report list, report detail, row-level judge reasons, manual review fields, and report comparison entry points.
- Human blind test workflow for LLM vs LLM, endpoint vs endpoint, and LLM vs endpoint comparisons.
- Open-source documentation for usage, evaluation concepts, storage decisions, security notes, contribution, and project comparison.

### Security and Storage

- `.env`, local databases, uploads, IDE metadata, and build artifacts are excluded from source control.
- API keys and endpoint authorization headers are masked in API responses.
- SQLite remains the default for local quick start; PostgreSQL is recommended for team or production deployments.

### Known Limitations

- Authentication, RBAC, audit logs, and encrypted secret storage are not included yet.
- Alembic migrations are not included yet; the current version uses SQLAlchemy table creation for quick start.
- Docker Compose and GitHub Actions CI are planned for a later release.

[v0.1.0]: #v010-2026-07-20
