# Production Environment

**用途**: 面向真实用户的服务。

## 配置策略

| 配置项 | 值 | 原因 |
|--------|-----|------|
| LLM | DeepSeek (真实) | 生产质量 |
| Tools | HTTP Adapter (真实) | 连接真实工单/用户系统 |
| Embedding | Local (1024d) | 生产向量 |
| Eval Gate | 强制开启 | 每次部署前必须通过 |
| Database | PostgreSQL + Redis (持久化) | 不丢数据 |
| Vector store | Milvus | 全量向量检索 |
| Rate limiter | 开启 (60/min per user) | 保护后端 |
| Sensitive tools | **禁止** | 防止误操作 |
| Feature flags | 只允许稳定 flag | 新 flag 需 staging 验证 |

## 部署前置条件

- [ ] Eval Gate **全部通过**
- [ ] Canary 测试 **全部通过**
- [ ] Release Manager **审批通过**
- [ ] Rollback Plan **已准备**
- [ ] Incident Runbook **已 review**
- [ ] 监控 Dashboard **已配置告警**

## 部署命令

```bash
# 仅 Release Manager 执行
export PRODUCTION=true
docker compose -f docker-compose.prod.yml up -d
python scripts/init_db.py
.venv/bin/python -m uvicorn enterprise_agentic_rag.app.main:app --port 8000
```

## 监控要求

| 指标 | 告警阈值 | 动作 |
|------|----------|------|
| fallback_rate | > 0.30 | 通知 oncall |
| error_rate | > 0.01 | 自动回滚 |
| p95_latency | > 3000ms | 通知 oncall |
| thumbs_down_rate | > 0.10 | 检查 prompt |
| verification_pass_rate | < 0.80 | 检查 verifier |

## 允许的操作

- 发布 stable 版本
- 紧急 hotfix（需 Release Manager 审批）
- 开启/关闭低风险 feature flag

## 禁止的操作

- 开启 sensitive tools
- 跳过 Eval Gate
- 无 rollback plan 的部署
- 直接修改数据库
- 使用 dev/staging API keys
