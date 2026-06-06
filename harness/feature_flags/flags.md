# Feature Flags — Enterprise Agentic RAG

All flags read from environment variables. Zero-downtime toggle without redeploy.

---

## Flag Index

| Flag | Default | Risk | Rollback | Environments |
|------|---------|------|----------|-------------|
| `enable_langgraph_v2_workflow` | `false` | High | Set `false` | dev, staging |
| `enable_llm_router` | `false` | High | Set `false` | dev, staging |
| `enable_qdrant_retriever` | `true` | Medium | Set `false` (falls to keyword) | dev, staging, prod |
| `enable_tool_calling` | `true` | Medium | Set `false` | dev, staging, prod |
| `enable_sensitive_tools` | `false` | High | Set `false` | staging, prod |
| `enable_answer_verifier` | `true` | Low | Set `false` | dev, staging, prod |
| `enable_human_fallback` | `true` | Low | Set `false` | dev, staging, prod |
| `enable_online_feedback` | `true` | Low | Set `false` | dev, staging, prod |
| `enable_real_llm` | `false` | High | Set `false` (falls to mock) | dev, staging |
| `enable_eval_gate` | `true` | Low | Set `false` (skip in emergency) | staging, prod |

---

## Flag Details

### `enable_langgraph_v2_workflow`

- **作用**: 切换 LangGraph 工作流版本。v2 可能包含新的节点编排逻辑。
- **默认值**: `false`（使用 v1 稳定版）
- **风险**: 工作流变更影响全链路，包括路由、检索、生成、校验。
- **回滚**: 设置 `false` 即时回退到 v1。
- **适用环境**: dev 默认开启 v2 测试；staging 在 Eval Gate 通过后开启；production 谨慎。

### `enable_llm_router`

- **作用**: 使用 LLM 做意图分类（而非 keyword classifier）。
- **默认值**: `false`（使用 keyword 路由）
- **风险**: LLM 调用增加延迟和成本，分类可能不如 keyword 稳定。
- **回滚**: 设置 `false` 即时回退到 keyword classifier。
- **适用环境**: dev 测试用；staging 跑 Eval Gate 验证；production 在 intent_accuracy 达标后开启。

### `enable_qdrant_retriever`

- **作用**: 启用 Milvus 向量检索（关闭则只用 keyword retriever）。
- **默认值**: `true`
- **风险**: 向量检索依赖 Milvus 服务可用性。
- **回滚**: 设置 `false` 回退到纯 keyword 检索。
- **适用环境**: 所有环境。

### `enable_tool_calling`

- **作用**: 启用工具调用节点（ticket / user profile / system status）。
- **默认值**: `true`
- **风险**: 工具调用增加延迟，外部 API 可能不可用。
- **回滚**: 设置 `false` 禁用所有工具调用。
- **适用环境**: 所有环境。production 应保持开启。

### `enable_sensitive_tools`

- **作用**: 允许执行敏感/破坏性工具（如 create_ticket）。
- **默认值**: `false`
- **风险**: 误操作可能创建垃圾工单或修改用户数据。
- **回滚**: 设置 `false` 即时禁用。
- **适用环境**: **production 必须 `false`**。staging 测试用。

### `enable_answer_verifier`

- **作用**: 是否对生成答案执行校验（规则 + LLM）。
- **默认值**: `true`
- **风险**: 低。关闭后会降低 answer 质量但加速响应。
- **回滚**: 设置 `false`。
- **适用环境**: 所有环境。

### `enable_human_fallback`

- **作用**: 校验失败 / 检索为空 / LLM 失败时是否升级到人工。
- **默认值**: `true`
- **风险**: 低。关闭后失败请求不会升级，用户可能得到低质量回答。
- **回滚**: 设置 `false`。
- **适用环境**: 所有环境。

### `enable_online_feedback`

- **作用**: 是否在回答下方展示 👍👎 反馈按钮。
- **默认值**: `true`
- **风险**: 无。
- **回滚**: 设置 `false`。
- **适用环境**: 所有环境。

### `enable_real_llm`

- **作用**: 使用真实 LLM (DeepSeek) 还是 mock provider。
- **默认值**: `false` → 但在 production 必须为 `true`。
- **风险**: 真实 LLM 调用有成本、延迟、可用性风险。
- **回滚**: 设置 `false` 回退到 mock（紧急情况）。
- **适用环境**: dev 可 `false`；staging 测试用 `true`；production 始终 `true`。

### `enable_eval_gate`

- **作用**: 部署到 staging/production 前是否强制执行 Eval Gate。
- **默认值**: `true`
- **风险**: 无。仅影响部署流程。
- **回滚**: 紧急 hotfix 时可临时 `false` 跳过 Eval Gate。
- **适用环境**: staging 和 production。

---

### `enable_summary_memory`

- **作用**: 是否启用会话摘要生成。
- **默认值**: `true`
- **风险**: Low — 关闭后摘要为空但不影响回答。
- **回滚**: 设置 `false`。
- **适用环境**: 所有环境。

### `enable_checkpoint_memory`

- **作用**: 是否启用 LangGraph checkpoint 持久化（Redis）。
- **默认值**: `true`
- **风险**: Low — 关闭后无法恢复中断的工作流。
- **回滚**: 设置 `false`。
- **适用环境**: 所有环境。

### `enable_reranker`

- **作用**: 是否对检索结果进行重排序。
- **默认值**: `true`
- **风险**: Medium — 关闭会降低 top-k 精度但加速响应。
- **回滚**: 设置 `false`。
- **适用环境**: 所有环境。

### `enable_llm_verifier`

- **作用**: 使用 LLM 做答案校验（关闭则只用 rule-based）。
- **默认值**: `false`
- **风险**: Medium — LLM verifier 有成本但更准确。
- **回滚**: 设置 `false` 回退到 rule verifier。
- **适用环境**: dev 可 `false`；staging 测试用；production 视情况。

### `enable_groundedness_check`

- **作用**: 是否检查答案是否基于检索文档（接地检查）。
- **默认值**: `true`
- **风险**: Low。
- **回滚**: 设置 `false`。
- **适用环境**: 所有环境。

### `primary_llm_provider`

- **作用**: 主 LLM provider 标识（`openai-compatible` / `dashscope` / `mock`）。
- **默认值**: `openai-compatible`
- **风险**: High — 切换 provider 影响答案质量和延迟。
- **回滚**: 回退到 `mock` 或上一个 provider。
- **适用环境**: 所有环境。production 必须配置 fallback。

### `fallback_llm_provider`

- **作用**: LLM 调用失败时的备用 provider。
- **默认值**: `mock`
- **风险**: Low — 仅在主 provider 失败时触发。
- **回滚**: 设置 `mock` 确保永远可用。
- **适用环境**: 所有环境。

### `enable_data_flywheel`

- **作用**: 是否自动沉淀失败 case 到 PostgreSQL + JSONL。
- **默认值**: `true`
- **风险**: Low。
- **回滚**: 设置 `false`。
- **适用环境**: 所有环境。
