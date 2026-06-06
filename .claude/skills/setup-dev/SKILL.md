---
name: setup-dev
description: Use when setting up the development environment for the first time, or when the user asks to start/launch/bring up the full dev stack. Covers Docker services, database init, document ingestion, graph index building, and starting backend/frontend.
---

# Setup Dev Environment

## Overview

一键启动完整开发环境。按顺序执行：Docker 服务 → 健康检查 → 数据库初始化 → 文档入库 → 图谱索引 → 图谱构建 → 可选启动后端/前端。

## Quick Reference

| 步骤 | 命令 | 说明 |
|------|------|------|
| 1. Docker 服务 | `docker compose up -d` | 启动全部 9 个服务 |
| 2. 健康检查 | `bash scripts/healthcheck.sh` | 等待所有服务就绪 |
| 3. 数据库初始化 | `python scripts/init_db.py` | 建表 + 写入 demo 用户 |
| 4. 文档入库 | `python scripts/ingest_docs.py` | Markdown → ES + Milvus + MinIO |
| 5. 图谱索引 | `python scripts/build_graph_indexes.py` | Neo4j 约束和索引 |
| 6. 图谱构建 | `python scripts/ingest_graph.py --full` | 实体/关系抽取 → Neo4j |
| 7. 后端启动 | `uv run uvicorn enterprise_agentic_rag.app.main:app --reload --port 8000` | FastAPI 开发服务器 |
| 8. 前端启动 | `cd frontend && npm run dev` | Next.js 开发服务器 |

## Implementation

### 全量启动（推荐）

```bash
# 步骤 1-2: 基础设施
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag
bash scripts/start_dev.sh

# 步骤 3: 数据库
uv run python scripts/init_db.py

# 步骤 4-6: 数据管道
uv run python scripts/ingest_docs.py
uv run python scripts/build_graph_indexes.py
uv run python scripts/ingest_graph.py --full

# 步骤 7-8: 应用（两个终端分别运行）
uv run uvicorn enterprise_agentic_rag.app.main:app --reload --port 8000 &
cd frontend && npm run dev
```

### 仅基础设施（跳过数据管道）

如果只需要启动服务但不入库数据：
```bash
bash scripts/start_dev.sh
uv run python scripts/init_db.py
```

### 仅应用层（服务已在运行）

```bash
uv run uvicorn enterprise_agentic_rag.app.main:app --reload --port 8000 &
cd frontend && npm run dev
```

## 执行策略

执行时遵循以下原则：

1. **每一步都检查上一步是否成功**，失败则停止并报告
2. **健康检查要等所有 9 个服务都通过**（PostgreSQL、Redis、Milvus、MinIO、ES、Neo4j、Prometheus、Grafana、OTel Collector）
3. **图谱构建需要 Neo4j 可用 + 已有文档数据**，如果 `ingest_docs.py` 失败则跳过
4. **前端启动需要 `node_modules` 已安装**，如果不存在则先运行 `cd frontend && npm install`
5. **后端启动前确认 `.env` 存在**，不存在则从 `.env.example` 复制

## 常见问题

| 问题 | 解决 |
|------|------|
| Docker 服务启动失败 | `docker compose ps` 检查状态，`docker compose logs -f <service>` 查看日志 |
| Milvus 健康检查超时 | Milvus 启动较慢（start_period: 40s），多等一会 |
| ES 内存不足 | 编辑 `docker-compose.yml` 降低 `ES_JAVA_OPTS` 的 `-Xmx` |
| `ingest_docs.py` 报错 | 确认 `data/docs/` 目录存在且有 `.md` 文件 |
| `ingest_graph.py` 报错 | 确认 Neo4j 已启动且 `ENABLE_GRAPH_RAG=true` |
| 前端 `npm run dev` 失败 | 先运行 `cd frontend && npm install` |
