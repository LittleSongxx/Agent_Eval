# 版本日志

这里记录 AI Evaluation Platform 的重要版本变化。

项目会尽量遵循语义化版本。早期版本会围绕公开 API、部署流程和团队协作能力快速演进。

## [v0.1.0] - 2026-07-20

### 亮点

- AI Evaluation Platform 的首个公开开源版本。
- 面向 RAG、AI Agent、多轮对话、接口评测、LLM-as-a-Judge、评测报告、人工复核和人工盲测的自托管评测工作台。
- 提供英文和简体中文 README 入口，并补充产品运行截图。

### 已包含

- 基于 FastAPI、SQLAlchemy 和原生评测执行器的后端服务。
- 基于 React 18、TypeScript、Ant Design 5 和 Vite 的前端应用。
- 支持单轮样本、多轮对话、RAG contexts、tool calls、参考答案和业务自定义字段的数据集管理。
- 面向 RAG、Agent、多轮对话的场景化指标体系。
- OpenAI 兼容 Judge LLM 配置，默认示例使用 DashScope Qwen Plus。
- 支持离线评测，也支持调用已保存的 Chat、RAG、Agent endpoint 做实时评测。
- 支持报告列表、报告详情、逐条 Judge 理由、人工复核字段和报告对比入口。
- 支持 LLM vs LLM、Endpoint vs Endpoint、LLM vs Endpoint 的人工盲测。
- 补充使用指南、评测概念、存储选型、安全说明、贡献指南和同类项目对比等开源文档。

### 安全与存储

- `.env`、本地数据库、上传文件、IDE 元数据和构建产物已从源码管理中排除。
- API Key 和 endpoint Authorization Header 在 API 响应中会做脱敏。
- SQLite 保持为本地快速体验默认值；团队或生产环境建议使用 PostgreSQL。

### 已知限制

- 暂未包含登录认证、RBAC、审计日志和敏感字段加密存储。
- 暂未引入 Alembic migration；当前版本仍以 SQLAlchemy 自动建表支持快速启动。
- Docker Compose 和 GitHub Actions CI 会在后续版本补充。

[v0.1.0]: https://github.com/huangyiminghappy/ai-eval-platform/releases/tag/v0.1.0
