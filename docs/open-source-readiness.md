# Open Source Readiness Checklist / 开源发布检查清单

这份清单用于把 AI Evaluation Platform 从本地项目整理成适合公开发布的开源仓库。

## Current Findings / 当前审计发现

本次仓库检查发现：

| 项目 | 状态 | 建议 |
|------|------|------|
| `backend/.env` | 本地存在，`LLM_API_KEY` 已脱敏为 `xxx` | 保持忽略，不要提交真实环境配置 |
| `backend/eval_platform.db` | 本地运行库可能曾包含真实 LLM API Key | 不提交；如曾公开过，轮换 Key |
| `eval_platform.db` | 曾被 Git 跟踪，且 schema 与当前 ORM 不完全一致 | 已从索引移除；公开前清理历史或新建公开仓库 |
| `test.db` | 曾被 Git 跟踪 | 已从索引移除 |
| `frontend/tsconfig.tsbuildinfo` | 曾被 Git 跟踪 | 已从索引移除 |
| `.idea/` | 曾被 Git 跟踪 | 已从索引移除 |
| `uploads/` | 本地包含 RAG 上传文件 | 保持忽略；不要发布用户文件 |
| API 响应中的 Authorization | Endpoint/Task/Blind Test 曾有回显风险 | 已增加响应脱敏 |

## Must Do Before Public Release / 发布前必须完成

- [ ] 确认本地 `backend/.env` 中没有真实 `LLM_API_KEY`，示例值使用 `xxx` 或空值。
- [ ] 确认 `git status --short` 中没有 `.env`、数据库、上传文件、IDE 文件。
- [ ] 清理 Git 历史中已提交过的数据库文件，或新建一个干净公开仓库。
- [ ] 选择并确认许可证。当前已放入 Apache-2.0。
- [ ] 检查 `README.md` 的模型供应商说明是否符合你的公开定位。
- [ ] 确认所有示例域名、Authorization、Headers 都是占位值或公开测试值。
- [ ] 运行后端测试和前端构建。
- [x] 创建首个 release tag，例如 `v0.1.0`。

## Recommended Repository Files / 推荐开源文件

已补充或建议保留：

- `README.md`: 中英文项目入口。
- `LICENSE`: 开源许可证。
- `SECURITY.md`: 安全披露说明。
- `CONTRIBUTING.md`: 贡献指南。
- `CHANGELOG.md`: 版本发布记录。
- `.env.example`: 环境变量样例。
- `.gitignore`: 忽略本地密钥、数据库、上传文件和构建产物。
- `docs/database-and-storage.md`: 存储架构建议。
- `docs/comparison.md`: 与类似项目对比。
- `docs/evaluation-guide.md`: 指标实践说明。
- `docs/usage-guide.md`: 操作手册。

后续可以补：

- `CODE_OF_CONDUCT.md`
- `docker-compose.yml`
- `backend/alembic/`
- GitHub issue templates
- GitHub Actions CI

## Security Review / 安全检查

### Secrets

需要保护的字段：

- LLM API Key
- Endpoint Authorization
- 上传的业务文档
- 评测样本和模型输出
- 评测报告中的原始回答、检索上下文、工具调用 trace

当前代码层面已做：

- LLM API Key masked response。
- Endpoint Authorization 不回显原文。
- Evaluation target config 脱敏。
- Blind test target config 脱敏。
- 编辑已保存 endpoint 时，Authorization 留空不会覆盖原值。

生产部署仍需：

- 登录认证。
- RBAC 权限。
- 数据库字段级加密。
- 审计日志。
- 上传文件访问控制。
- 网络层 TLS。
- 密钥轮换机制。

### Git History

如果你直接公开当前仓库历史，需要检查历史提交：

```bash
git log --all --oneline -- '*.db' 'backend/.env' '.env'
```

若历史里出现真实密钥或数据库，请用 `git filter-repo` 清理，或者创建干净仓库：

```bash
mkdir ../ai-eval-platform-public
rsync -a --exclude .git --exclude backend/.env --exclude '*.db' --exclude uploads ./ ../ai-eval-platform-public/
cd ../ai-eval-platform-public
git init
git add .
git commit -m "chore: initial open source release"
```

## Product Positioning / 开源定位

建议定位语：

中文：

> 一个面向 RAG、Agent、多轮对话和接口盲测的自托管 AI 应用评测平台，帮助团队把数据集、评测场景、自动打分、报告分析和人工决策连接成闭环。

English:

> A self-hosted evaluation workbench for RAG, Agent, multi-turn conversation, and endpoint blind testing, connecting datasets, scenario metrics, automated judging, reports, and human review into one workflow.

## Release Quality Bar / 发布质量门槛

建议首个开源版本至少满足：

- 后端 `pytest` 全部通过。
- 前端 `npm run build` 通过。
- 新用户按 README 能启动前后端。
- 不包含真实数据、密钥、上传文件和本地数据库。
- README 能解释“它和 Ragas/Easy Dataset/rag_eval 类项目有什么不同”。
- 文档明确 SQLite 只是本地默认值，生产建议 PostgreSQL。

## English Summary

Before going public, remove runtime data from Git, rotate any key that may have existed in a local database or `.env`, clean historical commits or start a fresh public repository, keep SQLite for quick start, document PostgreSQL for team usage, and ensure sensitive endpoint credentials are never returned by API responses.
