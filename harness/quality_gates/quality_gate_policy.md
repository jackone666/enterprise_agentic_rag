# Quality Gate Policy

## Gate 阻断策略

| Gate | Dev | Staging | Production | Override |
|------|-----|---------|------------|----------|
| CI Pipeline | Warning | Block | Block | No |
| Agent Eval Gate | Skip | Block | Block | Release Manager only |
| Prompt Eval Gate | Skip | Block (if prompt changed) | Block | Prompt Lead only |
| Security Gate (critical) | Block | Block | Block | No |
| Security Gate (medium) | Skip | Warning | Block | Security Lead only |
| Canary Monitor | Skip | Required | Required | No (auto-rollback) |
| Production Readiness Review | Skip | Skip | Required | No |

## Override 规则

### 允许人工 Override 的情况

1. **紧急 hotfix**（安全漏洞/数据丢失）
   - 需要 Release Manager + Security Lead 双审批
   - 事后 24h 内补齐 Eval Gate

2. **基础设施变更**（不涉及 Agent 逻辑）
   - CI/CD 配置、Dockerfile、监控配置
   - 需要 Platform Lead 审批

3. **文档/注释变更**
   - 自动通过所有 gate

### 永远不允许 Override 的情况

1. **Critical 安全漏洞** — security_gate 的 critical 级别不允许 override
2. **Production 环境** — 任何 gate 在 production 都不能被 skip
3. **未审批的 sensitive tools 变更**
4. **未通过 canary 就全量发布**
