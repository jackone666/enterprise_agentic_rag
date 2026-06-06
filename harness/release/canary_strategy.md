# Canary Release Strategy

## Canary 对象

以下资产可以独立灰度发布：

| 资产 | 灰度方式 | 回滚方式 |
|------|----------|----------|
| Code image | Docker tag 切换 | `git checkout <stable_tag>` |
| Prompt version | `PROMPT_VERSION` 环境变量 | 切回上一版本号 |
| Workflow version | `WORKFLOW_VERSION` / feature flag | `ENABLE_V2_WORKFLOW=false` |
| RAG config | `RAG_CONFIG_VERSION` | 切回旧 config |
| Retriever | `ENABLE_QDRANT_RETRIEVER` flag | 关 flag 走 keyword |
| Tool | `ENABLE_TOOL_CALLING` + per-tool flags | 关单个工具或全关 |
| Verifier | `VERIFIER_VERSION` | 切回 rule_only 模式 |
| LLM provider | `LLM_PROVIDER` | 切换 provider 或 model |

## 分流策略

使用 `session_id` 哈希分流：

```python
def is_canary(session_id: str, percent: int) -> bool:
    h = int(hashlib.md5(session_id.encode()).hexdigest()[:8], 16)
    return (h % 100) < percent
```

## 放量流程

```
Step 1: 5%  ──观察 30 min──→  fallback_rate, error_rate, p95_latency
Step 2: 20% ──观察 60 min──→  + thumbs_down_rate, verification_pass_rate
Step 3: 50% ──观察 120 min─→  + tool_success_rate, human_fallback_rate
Step 4: 100%                  all previous steps passed
```

## 放量条件（每个 Step 必须满足）

| 指标 | 允许范围 |
|------|----------|
| `error_rate` | ≤ baseline + 1% |
| `fallback_rate` | ≤ baseline + 5% |
| `human_fallback_rate` | ≤ baseline + 3% |
| `verification_pass_rate` | ≥ baseline - 3% |
| `tool_success_rate` | ≥ baseline - 3% |
| `thumbs_down_rate` | ≤ baseline + 2% |
| `p95_latency_ms` | ≤ baseline + 20% |

## 停止放量（任一触发即停止）

- Critical incident 发生
- 权限越权
- 幻觉答案被 verifier 误放
- `tool_success_rate` 大幅下降
- `thumbs_down_rate` 明显升高
- `p95_latency` 超阈值 3x

## 回滚动作（按异常类型）

| 异常类型 | 回滚动作 |
|----------|----------|
| Prompt 退化 | `PROMPT_VERSION` 切回上一版本 |
| RAG 检索下降 | `RETRIEVER_MODE=keyword` 或切旧 collection |
| Tool 失败 | `ENABLE_TOOL_CALLING=false` 或禁用单个工具 |
| LLM 故障 | `LLM_PROVIDER` 切换 fallback provider |
| Workflow 错误 | `WORKFLOW_VERSION` 回滚 |
| Verifier 误判 | `VERIFIER_MODE=rule_only` |

## 记录要求

每次 canary 必须记录：
- 灰度对象（code/prompt/workflow/rag/tool/llm/verifier）
- 每个 step 的指标快照
- 放量/停止决策和原因
- 最终结果（promoted / rolled back）
