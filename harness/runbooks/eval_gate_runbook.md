# Eval Gate Runbook

Eval Gate 失败后的根因分析和修复指南。

---

## 诊断流程

```text
Eval Gate FAILED
  ├─ check_rag_metrics FAILED?
  │    ├─ hit@k < 0.80  → RAG 问题
  │    ├─ recall@k < 0.75 → 检索覆盖不足
  │    └─ avg_retrieval_score < 0.60 → 向量质量或文档不足
  │
  ├─ check_answer_metrics FAILED?
  │    ├─ citation_present_rate < 0.95 → Prompt 问题
  │    ├─ groundedness < 0.85 → RAG + Prompt 联合问题
  │    └─ refusal_correctness < 0.90 → 兜底逻辑问题
  │
  └─ check_agent_metrics FAILED?
       ├─ verification_pass_rate < 0.85 → Verifier 问题
       ├─ fallback_rate > 0.25 → 全链路问题
       └─ tool_success_rate < 0.90 → Tool Adapter 问题
```

---

## Case 1: RAG 检索指标不达标

**症状**: hit@k / recall@k / avg_retrieval_score 低于阈值。

**诊断步骤**:
1. 检查 Milvus 服务状态: `curl localhost:19530/healthz`
2. 检查 collection 是否存在: `python scripts/ingest_docs.py`
3. 检查 embedding 模型是否正常加载
4. 检查 eval dataset 的 expected_sources 是否合理

**修复方案**:
- Milvus 不可用 → 启动 `docker compose up -d milvus`
- Collection 空 → 重新 `python scripts/ingest_docs.py`
- Embedding 维度不匹配 → 检查 `EMBEDDING_DIMENSIONS`
- expected_sources 过期 → 更新 `data/eval/regression_cases.jsonl`

---

## Case 2: 答案质量指标不达标

**症状**: citation_present_rate / groundedness 低于阈值。

**诊断步骤**:
1. 运行 `python -m enterprise_agentic_rag.evals.regression_eval` 查看具体失败 case
2. 检查生成答案是否包含 `[1]` `[2]` 引用标记
3. 检查 prompt_template 是否正确
4. 检查 knowledge_agent 是否正确组装 citations

**修复方案**:
- Prompt 缺少引用指令 → 更新 `context/prompt_builder.py`
- Knowledge agent 未返回 citations → 检查 `agents/knowledge_agent.py`
- LLM 生成未按格式 → 检查 LLM provider 的 temperature 和 prompt

---

## Case 3: Verifier 指标不达标

**症状**: verification_pass_rate < 0.85 或 false_pass_rate > 0.05。

**诊断步骤**:
1. 检查 verifier 的 rule 逻辑是否正确
2. 检查 LLM verifier 的 prompt 是否合理
3. 对比 rule 模式和 LLM 模式的结果差异

**修复方案**:
- 规则过严 → 调整 `agents/verifier_agent.py` 的阈值
- LLM verifier prompt 误判 → 更新 prompt
- 临时关闭 LLM verifier → `export VERIFIER_MODE=rule_only`

---

## Case 4: Tool 指标不达标

**症状**: tool_success_rate < 0.90。

**诊断步骤**:
1. 检查目标 API 是否可达: `curl $TICKET_API_BASE_URL/health`
2. 检查 API token 是否有效
3. 检查 circuit breaker 是否打开
4. 检查 retry 是否耗尽

**修复方案**:
- API 不可用 → 等待恢复或 `export TOOL_MODE=mock`
- Token 过期 → 更新 `TICKET_API_TOKEN`
- Circuit breaker open → 等待 30s 自动恢复或重启服务

---

## 通用修复 SOP

1. 确认失败指标和具体阈值差距
2. 查找对应的子系统（RAG / Prompt / Tool / Verifier）
3. 执行对应 Case 的诊断步骤
4. 应用修复
5. 重新运行 Eval Gate:
   ```bash
   python -m enterprise_agentic_rag.evals.regression_eval
   ```
6. 全部通过 → 标记 ready
