# Dev Environment

**用途**: 快速迭代、功能开发、本地调试。

## 配置策略

| 配置项 | 值 | 原因 |
|--------|-----|------|
| LLM | mock (默认) | 零成本、零延迟、可重复 |
| Tools | mock | 不依赖外部 API |
| Embedding | mock | 快速启动 |
| Eval | 小规模 (10 cases) | 快速反馈 |
| Feature flags | 全部可切换 | 方便测试新功能 |
| Database | Docker PostgreSQL + Redis | 可重置 |
| Vector store | 可选 (Milvus) | keyword retriever fallback 保证可用 |

## 启动命令

```bash
docker compose up -d postgres redis
cp .env.example .env
python scripts/init_db.py
.venv/bin/python -m uvicorn enterprise_agentic_rag.app.main:app --reload --port 8000
cd frontend && npm run dev
```

## 部署方式

- 本地手动启动
- 或 CI 自动部署到 dev 环境

## 允许的操作

- 随意修改代码
- 重置数据库
- 删除 collection
- 修改 prompt
- 切换任何 feature flag

## 禁止的操作

- 无（dev 环境完全自由）
