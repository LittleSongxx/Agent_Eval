# AI 评测平台

基于 Ragas 框架的 AI 应用评测平台，支持 RAG、AI Agent、多轮对话等场景的评测。

## 功能特性

- **LLM 配置中心**：集中管理 LLM API 配置，支持 OpenAI 兼容接口
- **数据管理中心**：自定义字段 Schema，支持 CSV/JSON 导入
- **场景与指标管理**：预置 RAG/Agent/多轮对话模板，支持自定义指标
- **评测执行引擎**：端到端 + 过程质量统一评测，异步执行
- **评测报告中心**：总览面板 + 逐条明细 + 历史对比

## 技术栈

- **后端**：FastAPI + SQLAlchemy + SQLite
- **前端**：React 18 + TypeScript + Ant Design 5 + Vite
- **评测内核**：Ragas (metrics, llms, prompt system)
- **LLM**：预配置通义千问 (Qwen Plus via DashScope)

## 快速开始

### 1. 启动后端

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

后端启动后访问 http://localhost:8000/docs 查看 API 文档。

### 2. 启动前端

```bash
cd frontend
npm install
npm run dev
```

前端启动后访问 http://localhost:5173 使用评测平台。

### 3. 预置数据

系统首次启动时自动创建：
- 通义千问 LLM 配置（DashScope 端点）
- 10 个内置评测指标
- 3 个预置场景模板（RAG/Agent/多轮对话）
- RAG 示例数据集（5 条 Python 问答数据）
- 一次已完成的示例评测（含报告）

## 项目结构

```
ai-eval-platform/
├── backend/
│   ├── app/
│   │   ├── api/                  # API 路由层
│   │   │   ├── llm_config.py     # LLM 配置 CRUD
│   │   │   ├── dataset.py        # 数据集管理 + 行操作 + 导入
│   │   │   ├── metric.py         # 指标定义管理
│   │   │   ├── scenario.py       # 场景管理（含预置模板）
│   │   │   ├── evaluation.py     # 评测任务创建与执行
│   │   │   └── report.py         # 评测报告查询
│   │   ├── core/
│   │   │   ├── config.py         # 应用配置（环境变量）
│   │   │   ├── database.py       # SQLAlchemy 引擎与会话
│   │   │   └── evaluation_engine.py  # Ragas 评测执行引擎
│   │   ├── models/               # SQLAlchemy ORM 模型
│   │   │   ├── llm_config.py     # LLMConfig 表
│   │   │   ├── dataset.py        # Dataset + DatasetRow 表
│   │   │   ├── metric_definition.py  # MetricDefinition 表
│   │   │   ├── scenario.py       # EvalScenario + ScenarioMetric 表
│   │   │   └── evaluation.py     # EvalTask + EvalRowResult 表
│   │   ├── schemas/              # Pydantic 请求/响应模型
│   │   │   ├── llm_config.py
│   │   │   ├── dataset.py
│   │   │   ├── metric_definition.py
│   │   │   ├── scenario.py
│   │   │   └── evaluation.py
│   │   ├── seed.py               # 预置数据填充
│   │   ├── main.py               # FastAPI 应用入口
│   │   └── services/             # 业务逻辑层（预留）
│   ├── tests/                    # 后端测试
│   ├── requirements.txt
│   └── run.py                    # 启动脚本
├── frontend/
│   └── src/
│       ├── pages/                # 页面组件
│       │   ├── LLMConfigPage.tsx       # LLM 配置页
│       │   ├── DatasetListPage.tsx     # 数据集列表页
│       │   ├── DatasetDetailPage.tsx   # 数据集详情页
│       │   ├── DatasetPage.tsx         # 数据集管理入口
│       │   ├── ScenarioListPage.tsx    # 场景列表页
│       │   ├── ScenarioPage.tsx        # 场景管理入口
│       │   ├── EvaluationPage.tsx      # 评测任务页
│       │   ├── ReportPage.tsx          # 报告列表页
│       │   └── ReportDetailPage.tsx    # 报告详情页
│       ├── services/
│       │   └── api.ts            # API 调用封装
│       ├── types/
│       │   └── index.ts          # TypeScript 类型定义
│       ├── components/           # 公共组件
│       ├── App.tsx               # 路由入口
│       └── main.tsx              # 应用入口
└── docs/
    └── usage-guide.md            # 使用指南
```

## 使用流程

1. 配置 LLM → 2. 准备数据集 → 3. 选择场景 → 4. 执行评测 → 5. 查看报告

详细使用文档见 [使用指南](docs/usage-guide.md)

## API 文档

启动后端后访问 http://localhost:8000/docs（Swagger UI）

## 开发

### 运行测试
```bash
cd backend
pytest tests/ -v
```

## 许可证

MIT
