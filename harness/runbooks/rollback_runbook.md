# Rollback Runbook

Agent 系统回滚不只是 `git revert`，每个子系统独立回滚。

---

## 1. Prompt 异常回滚

**症状**: 答案质量下降、citation 率下降、意图分类错误增多。

```bash
# 方案 A: 切回上一版本 prompt
export PROMPT_VERSION=v1.2

# 方案 B: 降到 mock provider（紧急）
export LLM_PROVIDER=mock

# 验证
python -m enterprise_agentic_rag.evals.regression_eval
# 确认: citation_present_rate >= 0.95, intent_accuracy >= 0.90
```

## 2. RAG 检索异常回滚

**症状**: hit@k 下降、检索结果不相关、检索延迟升高。

```bash
# 方案 A: 切到 keyword-only 检索
export RETRIEVER_MODE=keyword
export ENABLE_QDRANT_RETRIEVER=false

# 方案 B: 切到上一个 collection
export MILVUS_COLLECTION=enterprise_kb_v1

# 方案 C: 降低 top_k（减少噪声）
export RETRIEVAL_K=3

# 验证
python -m enterprise_agentic_rag.evals.regression_eval
# 确认: hit@k >= 0.75, avg_retrieval_score >= 0.50
```

## 3. Tool 异常回滚

**症状**: tool_success_rate 下降、外部 API 超时、circuit breaker 打开。

```bash
# 方案 A: 强制 mock fallback
export TICKET_API_BASE_URL=""
export USER_PROFILE_API_BASE_URL=""
export SYSTEM_STATUS_API_BASE_URL=""

# 方案 B: 禁用特定工具
export TOOL_DISABLED_create_ticket=true

# 方案 C: 关闭工具调用
export ENABLE_TOOL_CALLING=false

# 验证
curl http://localhost:8000/metrics
# 确认: tool_success_rate >= 0.90
```

## 4. LLM 异常回滚

**症状**: LLM 调用失败、延迟暴增、费用超预算。

```bash
# 方案 A: 切模型
export LLM_MODEL=deepseek-chat  # 从 v3 降级

# 方案 B: 切 provider
export LLM_PROVIDER=mock  # 紧急：回退到 mock

# 方案 C: 降低参数
export GENERATION_TEMPERATURE=0.0
export LLM_MAX_RETRIES=1

# 验证
curl -X POST http://localhost:8000/chat -d '{"query":"test","user_id":"u001"}'
# 确认: HTTP 200, answer 非空
```

## 5. Workflow 异常回滚

**症状**: 节点执行顺序错误、routing 错误、fallback 率升高。

```bash
# 方案 A: feature flag 切回 v1
export ENABLE_LANGGRAPH_V2_WORKFLOW=false

# 方案 B: 完全回退到最小工作流
export WORKFLOW_MODE=minimal
# minimal: check_permission → retrieve → generate → finalize

# 验证
pytest tests/test_workflow.py -v
# 确认: 所有 workflow 测试通过
```

## 6. Verifier 异常回滚

**症状**: 过高 false negative、答案被错误拒绝。

```bash
# 方案 A: rule-only 模式
export VERIFIER_MODE=rule_only

# 方案 B: 降低严格度
export VERIFICATION_STRICTNESS=lenient

# 方案 C: 关闭校验
export ENABLE_ANSWER_VERIFIER=false

# 验证
# 确认: verification_pass_rate >= 0.85, false_pass_rate <= 0.05
```

## 7. 全量回滚（兜底）

```bash
# 回到上一个 stable tag
docker compose down
git checkout v1.5.0  # 上一个稳定版本
docker compose up -d
python scripts/init_db.py
python scripts/ingest_docs.py
.venv/bin/python -m uvicorn enterprise_agentic_rag.app.main:app --port 8000 &

# SLA: 10 分钟内完成
```
