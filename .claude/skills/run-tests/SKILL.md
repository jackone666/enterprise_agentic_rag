---
name: run-tests
description: Use when running the project test suite. Supports unit tests, integration tests, and full test runs with automatic dependency checking (Docker services for integration tests). Use when the user asks to run tests, check if tests pass, or verify changes.
---

# Run Tests

## Overview

运行项目测试套件。支持按类型筛选（单元测试 vs 集成测试），并在运行集成测试前自动检查外部服务依赖。

## 测试分类

| 分类 | pytest 标记 | 依赖 | 数量 |
|------|-----------|------|------|
| 单元测试 | 无标记 | 无 | ~540 |
| 集成测试 | `integration` | ES, Neo4j, Milvus 等 | ~25 |
| 全部 | — | 全部 Docker 服务 | ~565 |

## Implementation

### 仅单元测试（快速，无需外部服务）

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag
uv run pytest tests -q -m "not integration"
```

### 仅集成测试（需要 Docker 服务）

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag

# 先检查依赖服务
bash scripts/healthcheck.sh || {
    echo "⚠️  部分服务未就绪，集成测试可能被跳过"
}

uv run pytest tests -q -m "integration"
```

### 全量测试

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag

# 确保所有 Docker 服务在线
docker compose up -d
bash scripts/healthcheck.sh

# 运行全量测试
uv run pytest tests -q
```

### 带超时标记的测试

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag
uv run pytest tests -q -m "timeout" --timeout 30
```

### 单个测试文件

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag
uv run pytest tests/test_workflow.py -q
```

### 详细输出（查看 skipped 原因）

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag
uv run pytest tests -q -v --tb=short
```

## 执行策略

1. **全量测试前先检查服务状态**，如果 Docker 服务未启动，先运行 `docker compose up -d`
2. **如果只需要快速验证代码逻辑**，用 `-m "not integration"` 跳过集成测试
3. **3 个 graph retriever 集成测试会被跳过**是正常的：它们需要 Neo4j 中已有索引的 Entity 数据（`available` 检查）
4. **ES 集成测试会被跳过**是正常的：需要 ES 在线且索引非空
5. **测试结束后报告**：passed / failed / skipped 数量和原因

## 预期结果

- 全量测试: `562-565 passed, 3 skipped`
- 单元测试: `~540 passed, 0 skipped`
- 集成测试: `~22-25 passed, 3 skipped`（3 个 graph retriever 需要已索引图谱数据）

## 常见问题

| 问题 | 解决 |
|------|------|
| `Future exception was never retrieved` | 异步连接未正确清理的噪声，不影响测试结果 |
| ES 集成测试全 skipped | ES 在线但索引为空/缺失，正常降级行为 |
| Graph retriever 测试 skipped | 需要先运行 `ingest_graph.py --full` 写入图谱数据 |
| 测试卡住不动 | 可能是某个测试等待外部服务超时，用 `--timeout 30` 加超时保护 |
| `asyncio_mode` 相关错误 | 确保 `pytest-asyncio` 已安装: `uv pip install pytest-asyncio` |
