# Changelog

All notable changes to AI Evaluation Platform are documented in this file.

This project follows semantic versioning where practical. The early releases are expected to evolve quickly while the public APIs and deployment workflow stabilize.

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

[v0.1.0]: https://github.com/huangyiminghappy/ai-eval-platform/releases/tag/v0.1.0
