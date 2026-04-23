# AI 评测平台

面向 RAG、AI Agent、多轮对话等场景的 AI 应用评测平台。评测内核由平台原生执行器负责，直接使用 OpenAI 兼容模型返回评分和评判理由。

## 快速启动（3 分钟）

### 前置条件

- Python 3.9+（推荐 3.11+）
- Node.js 18+
- uv（Python 包管理，可选但推荐）或 pip

### 第一步：启动后端

```bash
cd ai-eval-platform/backend

# 配置 API Key（必须）
# 编辑 .env 文件，将 LLM_API_KEY 替换为你的 DashScope API Key
# 默认已配置通义千问 qwen-plus 模型

# 安装依赖（二选一）
uv pip install -r requirements.txt --python venv/bin/python   # 用 uv（更快）
# 或
source venv/bin/activate && pip install -r requirements.txt    # 用 pip

# 启动
source venv/bin/activate
python3 run.py
```

看到以下输出表示启动成功：
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

验证：浏览器打开 http://localhost:8000/api/health ，看到 `{"status":"ok"}`

### 第二步：启动前端

新开一个终端：
```bash
cd ai-eval-platform/frontend
npm install    # 首次运行需要
npm run dev
```

看到以下输出表示启动成功：
```
  VITE v5.x.x  ready in xxx ms
  ➜  Local:   http://localhost:5173/
```

### 第三步：打开浏览器

访问 http://localhost:5173 ，你应该看到：

- 左侧导航栏：LLM 配置、数据管理、场景管理、评测执行、评测报告
- 系统已自动填充预置数据（LLM 配置、示例数据集、示例评测报告）

## 预置数据

首次启动后自动创建：

| 内容 | 说明 |
|------|------|
| Qwen Plus LLM 配置 | 从 `.env` 读取 DashScope 端点和 API Key |
| 10 个内置评测指标 | Faithfulness、Context Recall、Tool Call Accuracy 等 |
| 3 个预置场景模板 | RAG 评测、Agent 评测、多轮对话评测 |
| RAG 示例数据集 | 5 条 Python 问答数据（含 response 和 retrieved_contexts） |
| 示例评测报告 | 一次已完成的 RAG 评测（含逐条评分和理由） |

## 使用流程

```
1. 配置 LLM → 2. 准备数据集 → 3. 选择场景 → 4. 执行评测 → 5. 查看报告
```

详细操作指南见 [docs/usage-guide.md](docs/usage-guide.md)

## 开发

```bash
# 后端测试
cd backend && source venv/bin/activate && python3 -m pytest tests/ -v

# 前端类型检查
cd frontend && npx tsc --noEmit

# API 文档
# 后端启动后访问 http://localhost:8000/docs
```

## 技术栈

| 层 | 技术 |
|----|------|
| 后端 | FastAPI + SQLAlchemy + SQLite |
| 前端 | React 18 + TypeScript + Ant Design 5 + Vite |
| 评测内核 | 原生 Metric Executor + OpenAI 兼容 Judge |
| 评测 LLM | 通义千问 Qwen Plus (DashScope，OpenAI 兼容) |
