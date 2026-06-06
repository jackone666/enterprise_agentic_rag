---
name: reset-env
description: Use when the user wants to completely reset the development environment — wipe all Docker volumes and data, restart services, reinitialize database, and optionally re-ingest all documents and rebuild the knowledge graph. Use when the environment is corrupted, data is inconsistent, or a clean slate is needed.
---

# Reset Environment

## Overview

一键重置整个开发环境。比 `scripts/reset_dev.sh` 更完整：销毁 → 重建 → 数据恢复全流程。

## 流程

```
确认销毁 → docker compose down -v → 清理本地数据 → 
重新创建 .env → docker compose up -d → 健康检查 → 
init_db.py → ingest_docs.py → build_graph_indexes.py → ingest_graph.py --full
```

## Implementation

### 完整重置（推荐）

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag

# ⚠️ 先确认：这会删除所有数据
echo "⚠️  这将删除所有 Docker 卷和本地数据！"
echo "    按 Ctrl+C 取消，或回车继续..."
read -r

# 阶段 1: 销毁
echo "→ 停止服务并删除卷..."
docker compose down -v

echo "→ 清理本地数据..."
rm -rf data/db data/logs/events.jsonl data/eval/failed_cases.jsonl 2>/dev/null || true

# 阶段 2: 重建基础设施
echo "→ 重新创建 .env..."
cp .env.example .env

echo "→ 启动 Docker 服务..."
docker compose up -d

echo "→ 等待服务就绪（约 60-90s）..."
sleep 10
bash scripts/healthcheck.sh

# 阶段 3: 数据库
echo "→ 初始化数据库..."
uv run python scripts/init_db.py

# 阶段 4: 数据管道
echo "→ 文档入库..."
uv run python scripts/ingest_docs.py

echo "→ 构建图谱索引..."
uv run python scripts/build_graph_indexes.py

echo "→ 构建知识图谱..."
uv run python scripts/ingest_graph.py --full

echo ""
echo "✅ 重置完成！环境已恢复到初始状态。"
```

### 仅基础设施重置（保留 .env 和数据管道参数）

如果只想重启 Docker 服务，不重建数据：

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag
docker compose down -v
docker compose up -d
bash scripts/healthcheck.sh
```

## 执行策略

1. **必须先用醒目警告提示用户**，明确告知所有数据将被删除
2. **需要用户确认后才执行**，不要在无人值守时自动重置
3. `docker compose down -v` 会删除所有命名卷（pgdata、redisdata、milvusdata、miniodata、esdata、promdata、grafanadata、neo4j_data、neo4j_logs）
4. 清理本地数据时注意不要删除 `data/docs/`（这是源文档，不是运行时数据）
5. 健康检查通过后再执行后续数据管道步骤
6. 如果某个数据管道步骤失败，报告错误但继续后续步骤

## 常见问题

| 问题 | 解决 |
|------|------|
| `docker compose down -v` 卡住 | 某个容器未响应，`docker kill` 后重试 |
| 端口被占用 | `lsof -i :<port>` 查找占用进程，先释放端口 |
| 健康检查失败 | 给更多启动时间（某些服务启动慢），然后 `docker compose ps` 排查 |
| 重置后 `ingest_graph.py` 失败 | 确认 `ingest_docs.py` 先成功执行，Neo4j 已就绪 |
