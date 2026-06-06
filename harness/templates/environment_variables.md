# Environment Variables by Environment

| 变量 | Dev | Staging | Production |
|------|-----|---------|------------|
| `LLM_PROVIDER` | `mock` | `openai-compatible` | `openai-compatible` |
| `LLM_MODEL` | `mock` | `deepseek-chat` | `deepseek-chat` |
| `EMBEDDING_PROVIDER` | `mock` | `local` | `local` |
| `ENABLE_DESTRUCTIVE_TOOLS` | `true` | `false` | `false` |
| `ENABLE_SENSITIVE_TOOLS` | `true` | `true` | `true` (w/ confirmation) |
| `ENABLE_EVAL_GATE` | `false` | `true` | `true` |
| `ENABLE_REAL_LLM` | `false` | `true` | `true` |
| `LOG_LEVEL` | `DEBUG` | `INFO` | `WARNING` |
| `RETRIEVAL_K` | `5` | `5` | `5` |
| `RATE_LIMIT_PER_MINUTE` | `999` | `60` | `60` |
| `POSTGRES_HOST` | `localhost` | `localhost` | `prod-pg.internal` |
| `REDIS_HOST` | `localhost` | `localhost` | `prod-redis.internal` |
