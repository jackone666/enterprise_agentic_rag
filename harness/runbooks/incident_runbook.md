# Incident Runbook

Agent 系统常见故障的诊断和止损流程。

---

## Incident 1: fallback_rate 升高

**告警**: `fallback_rate > 0.30` (normal: < 0.20)

**诊断**:
```bash
# 1. 查看最近 100 条请求的 fallback_reason 分布
curl http://localhost:8000/metrics | jq '.intent_distribution'

# 2. 检查各子系统状态
curl http://localhost:8000/health
docker compose ps

# 3. 查看最近的 tool_audit_logs
# (检查 PostgreSQL tool_audit_logs 表)
```

**常见原因**:
| 原因 | fallback_reason | 修复 |
|------|-----------------|------|
| Milvus 宕机 | `no_relevant_docs` | `docker compose up -d milvus` |
| Embedding 模型加载失败 | `no_relevant_docs` | 重启服务 |
| LLM API 限流 | `llm_failure` | 降低请求频率或切备用模型 |
| 外部工具 API 超时 | `tool_failure` | 工具 mock fallback |
| 意图分类异常 | `unknown_intent` | 切 keyword router |

**止损**:
```bash
# 紧急: 关闭所有非必要功能
export ENABLE_TOOL_CALLING=false
export LLM_PROVIDER=mock
export RETRIEVER_MODE=keyword
```

---

## Incident 2: verification_pass_rate 下降

**告警**: `verification_pass_rate < 0.70` (normal: >= 0.85)

**诊断**:
```bash
# 1. 检查最近失败案例的 verification_reason
curl -X POST http://localhost:8000/chat -d '{"query":"test","user_id":"u001"}' | jq '.verification_reason'

# 2. 检查是否为 LLM verifier 误判
export VERIFIER_MODE=rule_only
# 重新测试看是否恢复
```

**常见原因**:
| 原因 | 症状 | 修复 |
|------|------|------|
| LLM verifier prompt 过严 | false negative 增多 | `rule_only` 模式或调整 prompt |
| 检索结果质量差 | citation_present 失败 | 检查 RAG pipeline |
| 新 prompt 格式变化 | verifier 规则误判 | 更新 verifier 规则 |

**止损**:
```bash
export VERIFIER_MODE=rule_only
export VERIFICATION_STRICTNESS=lenient
```

---

## Incident 3: tool_success_rate 下降

**告警**: `tool_success_rate < 0.85` (normal: >= 0.90)

**诊断**:
```bash
# 1. 查看 circuit breaker 状态
# (检查日志中的 "熔断器已打开")

# 2. 测试目标 API
curl -s -o /dev/null -w "%{http_code}" $TICKET_API_BASE_URL/health

# 3. 查看 tool_audit_logs 中的错误
# (PostgreSQL tool_audit_logs 表)
```

**止损**:
```bash
# 强制所有工具走 mock
export TICKET_API_BASE_URL=""
export USER_PROFILE_API_BASE_URL=""
export SYSTEM_STATUS_API_BASE_URL=""
export ENABLE_SENSITIVE_TOOLS=false
```

---

## Incident 4: thumbs_down_rate 升高

**告警**: `thumbs_down_rate > 0.15` (normal: < 0.05)

**诊断**:
```bash
# 1. 查看最近 failed_cases
python scripts/export_failed_cases.py /tmp/review.jsonl
cat /tmp/review.jsonl | jq '.feedback_text'

# 2. 分析失败模式
# - 答案不相关 → RAG 问题
# - 答案错误 → LLM 问题
# - 无回答 → fallback 问题
```

**止损**:
```bash
# 回滚到上一个 stable prompt
export PROMPT_VERSION=v1.2
# 或完整回滚
export LLM_PROVIDER=mock
```

---

## Incident 5: latency 升高

**告警**: `p95_latency_ms > 3000` (normal: < 2000)

**诊断**:
```bash
# 1. 查看各节点耗时
curl http://localhost:8000/metrics | jq '.avg_latency_ms'

# 2. 检查是否有节点重试过多
# (查看 retry_count)

# 3. 检查 Milvus/Redis 延迟
```

**止损**:
```bash
# 降低复杂操作
export RETRIEVAL_K=3       # 减少检索量
export LLM_MAX_TOKENS=512  # 减少生成长度
export ENABLE_EVAL_GATE=false  # 暂时关闭评估
export VERIFIER_MODE=rule_only
```

---

## 通用 SOP

1. **确认告警**: 查看 Prometheus /metrics 确认异常指标
2. **快速止损**: 按对应 Incident 执行止损命令
3. **根因分析**: 查看日志 + PostgreSQL audit 表
4. **修复验证**: 修复后 `pytest` + `regression_eval` 确认
5. **事后复盘**: 记录 incident 到 `harness/runbooks/incidents/`
