---
name: health
description: Use when checking if all Docker services are healthy and reachable. Covers all 9 services (PostgreSQL, Redis, Milvus, MinIO, Elasticsearch, Neo4j, Prometheus, Grafana, OTel Collector). Use when the user asks to check service status, verify infrastructure, or diagnose connection issues.
---

# Health Check

## Overview

检查项目全部 9 个 Docker 服务的健康状态。比 `scripts/healthcheck.sh` 更完整（原脚本只覆盖 6 个，缺 ES、Neo4j、Milvus、OTel Collector）。

## Quick Reference

| 服务 | 端口 | 健康检查方式 |
|------|------|-------------|
| PostgreSQL | 5432 | `pg_isready -h localhost -p 5432 -U rag_user -d enterprise_rag` |
| Redis | 6379 | `redis-cli -h localhost -p 6379 ping` |
| Milvus | 19530 | `curl -fsS http://localhost:9091/healthz` |
| MinIO | 9000 | `curl -fsS http://localhost:9000/minio/health/live` |
| Elasticsearch | 9200 | `curl -fsS http://localhost:9200/_cluster/health` |
| Neo4j | 7687 | `curl -fsS -u neo4j:password http://localhost:7474` |
| Prometheus | 9090 | `curl -fsS http://localhost:9090/-/healthy` |
| Grafana | 3000 | `curl -fsS http://localhost:3000/api/health` |
| OTel Collector | 4317 | `curl -fsS http://localhost:13133/` (internal health endpoint) |

## Implementation

执行完整健康检查：

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag

echo "=============================================="
echo " Enterprise Agentic RAG — Health Check (Full)"
echo "=============================================="
echo ""

# PostgreSQL
printf "  %-20s ... " "PostgreSQL"
pg_isready -h localhost -p 5432 -U rag_user -d enterprise_rag -q 2>/dev/null && echo "✅ OK" || echo "❌ FAIL"

# Redis
printf "  %-20s ... " "Redis"
redis-cli -h localhost -p 6379 ping 2>/dev/null | grep -q PONG && echo "✅ OK" || echo "❌ FAIL"

# Milvus
printf "  %-20s ... " "Milvus"
curl -fsS http://localhost:9091/healthz 2>/dev/null | grep -q OK && echo "✅ OK" || echo "❌ FAIL"

# MinIO
printf "  %-20s ... " "MinIO"
curl -fsS http://localhost:9000/minio/health/live 2>/dev/null >/dev/null && echo "✅ OK" || echo "❌ FAIL"

# Elasticsearch
printf "  %-20s ... " "Elasticsearch"
curl -fsS http://localhost:9200/_cluster/health 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('status') in ('green','yellow') else 1)" 2>/dev/null && echo "✅ OK" || echo "❌ FAIL"

# Neo4j
printf "  %-20s ... " "Neo4j"
curl -fsS -u neo4j:password http://localhost:7474 2>/dev/null >/dev/null && echo "✅ OK" || echo "❌ FAIL"

# Prometheus
printf "  %-20s ... " "Prometheus"
curl -fsS http://localhost:9090/-/healthy 2>/dev/null >/dev/null && echo "✅ OK" || echo "❌ FAIL"

# Grafana
printf "  %-20s ... " "Grafana"
curl -fsS http://localhost:3000/api/health 2>/dev/null >/dev/null && echo "✅ OK" || echo "❌ FAIL"

# OTel Collector
printf "  %-20s ... " "OTel Collector"
wget --spider -q http://localhost:13133/ 2>/dev/null && echo "✅ OK" || echo "❌ FAIL"

echo ""
echo "Done. If any service is FAIL, check: docker compose ps"
```

## 注意事项

- ES 健康检查允许 `yellow` 状态（单节点集群，未分配副本分片是正常的）
- Milvus 健康端点使用 **9091**（内部管理端口），不是 19530（gRPC 端口）
- OTel Collector 健康检查用内部端口 **13133**，不是 4317/4318
- 如果 `.env` 中自定义了端口，需要用对应的环境变量值替代默认端口

## 常见问题

| 症状 | 原因 | 解决 |
|------|------|------|
| ES 始终 red | 磁盘空间不足 | `docker compose down -v` 清理卷，确保有 >5GB |
| Neo4j 连接拒绝 | 内存不足 | 降低 `NEO4J_dbms_memory_heap_max__size` |
| Milvus 超时 | 首次启动 etcd 初始化慢 | 等待 60s 后重试 |
| Grafana 401 | 认证凭据不对 | 检查 `GF_SECURITY_ADMIN_USER/PASSWORD` |
