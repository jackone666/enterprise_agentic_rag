# Pipeline Variables

Harness Pipeline 使用的变量定义。

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `commit_sha` | Git commit SHA | `${GIT_COMMIT}` |
| `image_tag` | Docker image tag | `${commit_sha}` |
| `environment` | 目标环境 | `dev` |
| `workflow_version` | LangGraph workflow 版本 | `v1` |
| `rag_config_version` | RAG 参数版本 | `v1` |
| `milvus_collection_version` | Milvus collection 版本 | `enterprise_kb` |
| `router_prompt_version` | Router prompt 版本 | `v1` |
| `knowledge_prompt_version` | Knowledge agent prompt 版本 | `v1` |
| `verifier_prompt_version` | Verifier prompt 版本 | `v1` |
| `answer_prompt_version` | Answer prompt 版本 | `v1` |
| `enable_milvus_retriever` | 启用向量检索 | `true` |
| `enable_tool_calling` | 启用工具调用 | `true` |
| `enable_answer_verifier` | 启用答案校验 | `true` |
| `enable_real_llm` | 启用真实 LLM | `false` for dev |
| `canary_percent` | 灰度百分比 | `0` |
| `rollback_target_version` | 回滚目标版本 | previous stable tag |
| `eval_gate_enabled` | 启用 Eval Gate | `true` for staging/prod |
| `security_gate_enabled` | 启用 Security Gate | `true` for production |
