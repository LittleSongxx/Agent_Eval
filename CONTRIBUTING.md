# Contributing

感谢你愿意参与 AI Evaluation Platform。这个项目关注 RAG、Agent、多轮对话和接口评测工程，欢迎提交 bug 修复、指标实现、文档、示例数据和部署改进。

## Development Setup

Backend:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 run.py
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Checks Before Pull Request

Run backend tests:

```bash
cd backend
source venv/bin/activate
python3 -m pytest tests/ -v
```

Run frontend build:

```bash
cd frontend
npm run build
```

## Coding Guidelines

- Keep backend code typed and PEP 8 friendly.
- Prefer FastAPI `HTTPException` for API errors.
- Keep API schemas in `backend/app/schemas/`.
- Keep frontend API calls in `frontend/src/services/api.ts`.
- Use React function components and hooks.
- Prefer existing Ant Design patterns over new UI abstractions.
- Do not commit runtime data: `.env`, `*.db`, uploads, IDE metadata, caches, or build artifacts.

## Adding A Metric

1. Add or update the metric definition model/config in `backend/app/models/metric_definition.py` if needed.
2. Implement scoring in `backend/app/core/evaluation_engine.py`.
3. Add schema or API changes if the metric needs new configuration.
4. Update `frontend/src/components/MetricConfigPanel.tsx` if the UI needs new controls.
5. Add focused tests in `backend/tests/`.
6. Update `docs/evaluation-guide.md`.

## Security

Do not open public issues containing API keys, authorization headers, uploaded private documents, customer data, or complete private evaluation reports. Follow [SECURITY.md](SECURITY.md).

## Commit Style

Use concise, descriptive commit messages, for example:

```text
feat(evaluation): add endpoint target authorization redaction
fix(dataset): preserve row_count after import
docs(readme): clarify PostgreSQL deployment path
```

## English Summary

Please keep changes scoped, include tests for backend behavior changes, run the backend test suite and frontend build, and never commit local secrets or runtime data.
