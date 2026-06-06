---
name: debug-retrieval
description: Use when debugging the RAG retrieval pipeline for a specific query — trace what each retriever (keyword, vector, graph) returned, how results were fused and reranked, and identify why certain documents were or weren't retrieved. Use when the user asks "why didn't my query find X" or wants to inspect retrieval quality.
---

# Debug Retrieval

## Overview

对指定查询运行完整的 Graph-Augmented Hybrid RAG 检索管道，输出每个检索器的命中情况、融合权重、重排序结果和完整 trace，帮助定位检索问题。

## 检索管道

```
Query → Query Analysis → 
  ├── Keyword Retriever (ES BM25 + IK)
  ├── Vector Retriever (Milvus)
  └── Graph Retriever (Neo4j entity paths)
  → Fusion (weighted RRF) → Reranker (Cross-Encoder) → Top-K Results
```

## Implementation

### 单查询调试

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag
uv run python scripts/run_graph_rag.py "你的查询问题"
```

### 批量对比调试

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag

# 对比多个查询的检索效果
for query in \
  "HarmonyOS NEXT 如何申请权限？" \
  "Ability 和页面生命周期有什么关系？" \
  "错误码 15500000 是什么问题？"
do
  echo "=============================================="
  echo "Query: $query"
  echo "=============================================="
  uv run python scripts/run_graph_rag.py "$query" 2>&1 | head -60
  echo ""
done
```

### 编程式调试（更详细的 trace）

```python
# debug_retrieval.py — 放在项目根目录运行
import asyncio, json, sys
sys.path.insert(0, "src")
from enterprise_agentic_rag.rag.graph_rag_orchestrator import GraphRAGOrchestrator

async def debug(query: str):
    orch = GraphRAGOrchestrator()
    result = await orch.retrieve(query=query, top_k=5)
    trace = result.get("retrieval_trace", {})
    
    print(f"Query: {query}")
    print(f"Mode: {trace.get('mode')}")
    print(f"Retrievers: {trace.get('enabled_retrievers')}")
    print()
    
    # 各检索器命中数
    print("--- Hits ---")
    print(f"  keyword: {trace.get('keyword_hit_count', 0)}")
    print(f"  vector:  {trace.get('vector_hit_count', 0)}")
    print(f"  graph:   {trace.get('graph_hit_count', 0)}")
    print(f"  merged:  {trace.get('merged_count', 0)}")
    print(f"  reranked:{trace.get('reranked_count', 0)}")
    print()
    
    # 延迟分析
    print("--- Latency (ms) ---")
    print(f"  keyword: {trace.get('keyword_latency_ms', 0):.1f}")
    print(f"  vector:  {trace.get('vector_latency_ms', 0):.1f}")
    print(f"  graph:   {trace.get('graph_latency_ms', 0):.1f}")
    print(f"  fusion:  {trace.get('fusion_latency_ms', 0):.1f}")
    print(f"  total:   {trace.get('total_latency_ms', 0):.1f}")
    print()
    
    # 融合
    print(f"Fusion: {trace.get('fusion_method')}")
    print(f"Weights: {trace.get('fusion_weights')}")
    print()
    
    # 降级
    if trace.get('degraded_from'):
        print(f"⚠️  Degraded: {trace['degraded_from']} → {trace.get('degraded_to')}")
    print()
    
    # Top-5 结果
    docs = result.get("retrieved_docs", [])
    print(f"--- Top-{len(docs)} Results ---")
    for i, doc in enumerate(docs):
        score = doc.get('score', 0)
        sources = doc.get('matched_sources', [])
        content = doc.get('content', '')[:100]
        print(f"  [{i+1}] score={score:.4f} sources={sources}")
        print(f"      {content}...")
        paths = doc.get('graph_paths', [])
        if paths:
            for p in paths[:2]:
                print(f"      path: {' → '.join(p.get('path_entities', []))}")
    print()
    
    # 错误
    if trace.get('errors'):
        print("--- Errors ---")
        for e in trace['errors']:
            print(f"  ✗ {e}")
    
    # 全量 trace JSON
    print("\n--- Full Trace ---")
    print(json.dumps(trace, indent=2, ensure_ascii=False, default=str))

asyncio.run(debug("你的查询问题"))
```

## 分析维度

拿到 trace 后，按以下维度排查：

| 维度 | 看什么 | 问题信号 |
|------|--------|---------|
| 检索覆盖 | `keyword_hit_count` + `vector_hit_count` | 两者都为 0 → 文档缺失或索引问题 |
| 图谱贡献 | `graph_hit_count` + `graph_paths_count` | 始终为 0 → Neo4j 无相关数据 |
| 融合效果 | `merged_count` vs 各检索器命中数 | merged 远小于 sum → 重复结果多 |
| 重排序 | `reranked_count` + 结果 score | score 都很低 → 文档虽然命中但不相关 |
| 延迟瓶颈 | 各 `latency_ms` | 某个检索器特别慢 → 需要优化或降级 |
| 降级 | `degraded_from` / `degraded_to` | 频繁降级 → 服务不稳定 |

## 常见问题

| 问题 | 可能原因 | 解决 |
|------|---------|------|
| keyword_hit_count = 0 | ES 索引为空或 IK 分词不匹配 | `python scripts/ingest_docs.py` 重建索引 |
| vector_hit_count = 0 | Milvus collection 为空 | 检查 Milvus collection，重新入库 |
| graph_hit_count 始终为 0 | 图谱无数据或查询无实体链接 | `python scripts/ingest_graph.py --full` |
| merged_count 很低 | 多路检索结果重叠严重 | 调整 fusion 算法或增加检索多样性 |
| 某个检索器延迟 > 1s | 该服务负载高或网络延迟 | 检查对应 Docker 服务状态 |
| 降级频繁 | 外部服务不稳定 | 检查 `docker compose ps`，必要时重启 |
