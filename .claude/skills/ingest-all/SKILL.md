---
name: ingest-all
description: Use when the user wants to re-ingest all documents into the knowledge base — rebuild ES indexes, Milvus vectors, MinIO storage, and Neo4j knowledge graph. Use when documents have been updated, the index is stale, or after adding new markdown files to data/docs/.
---

# Ingest All Data

## Overview

全量数据入库管道：文档 → 切片 → ES 关键词索引 + Milvus 向量 + MinIO 对象存储 + Neo4j 知识图谱。

## 管道架构

```
data/docs/*.md
    │
    ▼
IngestionPipeline (ingest_docs.py)
    ├── MinIO  (原始文档存储)
    ├── ES     (IK 分词全文索引)
    └── Milvus (向量嵌入)
    │
    ▼
GraphIndexer (build_graph_indexes.py)
    └── Neo4j  (约束与索引)
    │
    ▼
GraphIndexer.build_graph (ingest_graph.py)
    ├── EntityExtractor   (实体抽取)
    ├── RelationExtractor (关系抽取)
    └── Neo4j  (节点 + 关系)
```

## Implementation

### 全量入库

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag

# 步骤 1: 确认所有依赖服务在线
bash scripts/healthcheck.sh || {
    echo "❌ 部分服务未就绪，请先启动: docker compose up -d"
    exit 1
}

# 步骤 2: 文档入库
echo "━━━ 第 1/3 步: 文档入库 ━━━"
uv run python scripts/ingest_docs.py

# 步骤 3: 图谱索引
echo "━━━ 第 2/3 步: 图谱索引 ━━━"
uv run python scripts/build_graph_indexes.py

# 步骤 4: 图谱构建
echo "━━━ 第 3/3 步: 知识图谱构建 ━━━"
uv run python scripts/ingest_graph.py --full

echo ""
echo "✅ 全量入库完成！"
```

### 增量更新（仅更新变化的文件）

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag
uv run python scripts/schedule_ingest.py --mode smart
```

### 单文件更新

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag
uv run python scripts/schedule_ingest.py --mode single --source <filename>.md
```

### 监听模式（实时自动更新）

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag
uv run python scripts/schedule_ingest.py --mode watch --interval 30
```

## 执行策略

1. **先检查依赖服务是否就绪**，ES、Milvus、MinIO、Neo4j 缺一不可
2. `ingest_docs.py` 必须最先执行（后续步骤依赖文档和切片数据）
3. `build_graph_indexes.py` 在 `ingest_graph.py` 之前执行（创建约束和索引）
4. `ingest_graph.py --full` 会清空已有图谱后重建，增量场景去掉 `--full`
5. 每一步失败时报告具体错误，不要静默跳过

## 常见问题

| 问题 | 解决 |
|------|------|
| ES 索引失败 | 检查 ES 是否在线: `curl http://localhost:9200/_cluster/health` |
| Milvus 写入失败 | 检查 collection 是否存在: `curl http://localhost:9091/healthz` |
| 图谱构建报"文档不存在" | 先运行 `ingest_docs.py`，确认 `data/docs/` 有 `.md` 文件 |
| `schedule_ingest.py` 锁冲突 | 删除 `data/.ingest.lock` 后重试 |
| 增量更新没检测到变化 | 检查文件 hash 是否变化（`schedule_ingest.py` 基于 SHA256） |
