# Database and Storage / 数据库与存储选型

本文说明 AI Evaluation Platform 开源前后的数据库建议。结论先说：**SQLite 适合作为本地快速体验默认值，不建议作为多人协作或生产评测环境的长期存储。**

## Current State / 当前状态

当前后端通过 SQLAlchemy 访问数据库，默认：

```bash
DATABASE_URL=sqlite:///./eval_platform.db
```

这对开源体验很友好：

- 无需安装数据库服务。
- 一条命令即可启动后端并自动建表。
- 适合 Demo、个人评测、本地开发、CI 测试。

但评测平台的数据形态比普通 CRUD 更复杂：

- 数据集样本、评测结果、LLM Judge 理由、Endpoint trace 都可能快速增长。
- RAG 数据集生成会保存上传文件、切片、样本和中间日志。
- 后台任务会并发写入进度、逐条结果和报告汇总。
- LLM API Key、Endpoint Authorization 等敏感字段需要更完整的密钥管理。

因此，开源默认保留 SQLite，但团队部署建议换服务型数据库。

## Recommendation / 推荐方案

| 场景 | 推荐存储 | 原因 |
|------|----------|------|
| 本地体验、个人 Demo | SQLite | 零依赖、启动最快 |
| 小团队内网试用 | PostgreSQL | 并发写入更可靠，JSON 字段、索引、备份更成熟 |
| 生产级评测平台 | PostgreSQL + 对象存储 | 数据库保存结构化元数据，对象存储保存上传文档和大 trace |
| 大规模日志/Trace 分析 | PostgreSQL + ClickHouse/OpenSearch | 报告元数据与长文本/高频事件分层存储 |

首选升级路径：

```bash
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/ai_eval_platform
```

需要在 `backend/requirements.txt` 增加 PostgreSQL 驱动，例如：

```text
psycopg[binary]>=3.1.0
```

MySQL 也可以通过 SQLAlchemy 使用，但本项目大量使用 JSON 字段，PostgreSQL 的 JSONB、索引和迁移生态更合适。

## Why Not Replace SQLite Completely? / 为什么不完全移除 SQLite

开源项目的第一体验非常重要。评测平台本身已经依赖 Python、Node、LLM API，如果再强制用户安装 PostgreSQL，会显著抬高试用门槛。

更好的方式是：

1. SQLite 保持默认，负责“3 分钟跑起来”。
2. `DATABASE_URL` 支持 PostgreSQL/MySQL，负责“团队环境可扩展”。
3. 文档明确边界，避免用户误把本地默认配置当生产架构。

当前代码已经按数据库 URL 自动判断：只有 SQLite 会启用 `check_same_thread` 和 WAL PRAGMA；非 SQLite 会走通用 SQLAlchemy engine。

## Open Source Release Rules / 开源发布规则

发布仓库里不要包含：

- `*.db`
- `*.db-wal`
- `*.db-shm`
- `uploads/`
- `backend/uploads/`
- `frontend/tsconfig.tsbuildinfo`
- `.env`
- `.idea/`

已加入 Git 历史的数据库文件需要在公开前处理：

```bash
git rm --cached eval_platform.db test.db frontend/tsconfig.tsbuildinfo
```

如果历史提交中已经包含真实 API Key 或用户数据，建议用 `git filter-repo` 清理历史，或者创建一个干净的新公开仓库重新导入当前代码。

## Migration Plan / 迁移计划

当前项目使用 SQLAlchemy `create_all` 和轻量兼容补丁，适合早期 Demo，但不适合作为长期迁移系统。

开源后建议按阶段补齐：

| 阶段 | 目标 |
|------|------|
| Phase 1 | 保留 `create_all`，文档说明 SQLite/PostgreSQL 配置 |
| Phase 2 | 引入 Alembic，生成初始 migration，停止继续增加 runtime schema patch |
| Phase 3 | 对大表添加索引：`dataset_rows.dataset_id`、`eval_row_results.eval_task_id`、`created_at` 等 |
| Phase 4 | 引入对象存储抽象，把上传文档、长 trace、大报告附件从数据库拆出 |
| Phase 5 | 加密敏感字段，支持 KMS/环境变量密钥轮换 |

## Sensitive Data Model / 敏感数据处理

当前敏感字段包括：

- `llm_configs.api_key`
- `endpoint_targets.authorization`
- `eval_tasks.target_config.authorization`
- `blind_test_tasks.target_a.authorization`
- `blind_test_tasks.target_b.authorization`
- `rag_dataset_jobs.target_authorization`

开源前已做的保护：

- LLM API Key 只返回 masked 字段。
- Endpoint Authorization API 响应不回显原文。
- 评测任务和盲测任务响应会脱敏 target 配置。
- 编辑 Endpoint 时 Authorization 留空不会覆盖原值。

生产环境还应补充：

- 数据库字段级加密。
- 后端认证与 RBAC。
- 管理操作审计日志。
- Secret rotation 和过期策略。
- 上传文件的大小、类型、病毒扫描和生命周期管理。

## English Summary

SQLite is the right default for local open-source adoption, but PostgreSQL is the right default for shared or production deployments. Keep SQLite for quick start, make `DATABASE_URL` portable, add Alembic migrations before schema growth accelerates, and move large uploaded or generated artifacts to object storage when evaluation volume grows.
