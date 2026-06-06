# Feature Flag Strategy

## 设计原则

1. **零停机切换**: 所有 flag 通过环境变量控制，无需重新部署
2. **独立回滚**: 每个子系统有独立 flag，出问题只回滚问题部分
3. **生产保守**: production 默认关闭高风险 flag，staging 验证后开启
4. **指标驱动**: 每个 flag 关联到具体监控指标

## Flag 生命周期

```
dev → staging → canary 5% → 20% → 50% → production 100%
  ↓       ↓         ↓         ↓       ↓            ↓
 测试    验证     小流量      扩大    半量         全量
```

## 高风险 Flag 治理

以下 flags 在生产环境修改需要审批:

| Flag | 风险 | 审批人 |
|------|------|--------|
| `enable_real_llm` | High — 成本+延迟 | Engineering Lead |
| `enable_sensitive_tools` | Critical — 安全 | Security Lead |
| `enable_destructive_tools` | Critical — 安全 | Security Lead |
| `enable_llm_router` | High — 质量 | Prompt Lead |
| `enable_langgraph_v2_workflow` | High — 全链路 | Platform Lead |
| `primary_llm_provider` | High — 质量+成本 | Engineering Lead |

## Flag 修改后验证

修改 flag 后必须:
1. 运行 smoke test
2. 观察关联指标 15 分钟
3. 如指标异常 → 立即回滚 flag
