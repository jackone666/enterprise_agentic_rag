# Staging Environment

**用途**: 生产前验证、Eval Gate、Canary 测试。

## 配置策略

| 配置项 | 值 | 原因 |
|--------|-----|------|
| LLM | DeepSeek (真实) | 必须验证真实 LLM 行为 |
| Tools | HTTP Adapter → mock fallback | 测试 adapter 链路 |
| Embedding | Local (1024d) | 真实向量质量 |
| Eval | 完整 10 cases + Eval Gate | 全部阈值检查 |
| Database | Docker PostgreSQL + Redis | 真实存储 |
| Vector store | Milvus (必须) | 验证向量检索 |
| Rate limiter | 开启 (60/min) | 验证限流逻辑 |
| Feature flags | 与 production 对齐 | 提前发现问题 |

## 启动命令

```bash
docker compose up -d          # 全栈
python scripts/init_db.py
python scripts/ingest_docs.py
.venv/bin/python -m uvicorn enterprise_agentic_rag.app.main:app --port 8000
```

## Eval Gate 要求

部署到 staging 后**必须**运行 Eval Gate：

```bash
python -m enterprise_agentic_rag.evals.regression_eval
```

不通过 → 回滚 dev，**不允许**晋升到 production。

## Canary 要求

Staging 上可开启 canary 测试：

1. 5% 流量 → 观察 30min
2. 20% 流量 → 观察 60min
3. 50% 流量 → 观察 120min
4. 全部通过 → 可晋升 production

## 允许的操作

- 运行 Eval Gate
- Canary 测试
- 功能开关切换（非破坏性）

## 禁止的操作

- 直接修改 production 数据
- 跳过 Eval Gate
- 使用 production API keys
