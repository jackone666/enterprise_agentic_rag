# Metric Thresholds — Agent Eval Gate

每个指标的含义、阈值设定依据、低于阈值时的处理方案。

---

## RAG 检索指标

| 指标 | 阈值 | 含义 | 为什么 | 低于阈值处理 |
|------|------|------|--------|------------|
| `hit_at_3` | ≥ 0.80 | Top-3 至少命中 1 个相关文档的比例 | 用户通常只看前 3 个结果 | 检查 Milvus collection / 重新入库 / 调整 top_k |
| `hit_at_5` | ≥ 0.85 | Top-5 至少命中 1 个相关文档 | QA 场景 top-5 足够 | 同上 + 考虑查询改写 |
| `recall_at_5` | ≥ 0.75 | 期望文档被检索到的比例 | 不能遗漏关键文档 | 检查 expected_sources 是否合理 / embedding 质量 |
| `mrr` | ≥ 0.70 | 第一个相关文档的排名的倒数均值 | 第一个结果最重要 | 考虑 reranker / 调整融合权重 |
| `avg_retrieval_score` | ≥ 0.60 | 检索结果平均相关度 | 太低说明向量质量差 | 检查 embedding 模型 / 文档质量 |
| `no_relevant_docs_rate` | ≤ 0.20 | 检索完全没命中 | 超过 20% 说明覆盖不足 | 补充知识库文档 / 优化查询改写 |

## 答案质量指标

| 指标 | 阈值 | 含义 | 为什么 | 低于阈值处理 |
|------|------|------|--------|------------|
| `citation_present_rate` | ≥ 0.95 | 答案包含引用标记的比例 | 无引用 = 不可信 | 检查 prompt 是否要求引用 / LLM 是否正确生成 |
| `groundedness` | ≥ 0.85 | 答案是否基于检索文档 | 幻觉检测的基础 | 检查 context 构建 / verifier prompt |
| `answer_relevance` | ≥ 0.85 | 答案是否回答问题 | 跑题是浪费 | 检查 knowledge_prompt / LLM temperature |
| `refusal_correctness` | ≥ 0.90 | 无文档时是否正确拒绝 | 不能胡说 | 检查 fallback 逻辑 |

## Agent 运行指标

| 指标 | 阈值 | 含义 | 为什么 | 低于阈值处理 |
|------|------|------|--------|------------|
| `verification_pass_rate` | ≥ 0.85 | 答案校验通过率 | 太低说明 verifier 或生成有问题 | 检查 verifier rules / LLM verifier prompt |
| `fallback_rate` | ≤ 0.25 | 触发兜底的比例 | 超过 1/4 就是系统问题 | 诊断 no_relevant_docs / tool_failure / llm_failure |
| `human_fallback_rate` | ≤ 0.15 | 升级到人工的比例 | 人工成本高 | 提高 answer 质量 / 减少 retrieval 失败 |
| `tool_success_rate` | ≥ 0.90 | 工具执行成功率 | 工具是关键链路 | 检查外部 API / circuit breaker / 重试策略 |
| `end_to_end_success_rate` | ≥ 0.90 | 端到端请求成功率 | 用户最关心的指标 | 根因分析 failed_cases |

## 严重性分级

| 级别 | 条件 | 动作 |
|------|------|------|
| **Critical** | `end_to_end_success_rate < 0.80` 或 `groundedness < 0.70` | 立即阻断 + 自动回滚 |
| **High** | 任一核心指标不达标 | 阻断 staging→production |
| **Medium** | 指标接近阈值（差距 < 0.05） | Warning + 建议改进 |
| **Low** | 非核心指标轻微下降 | 记录但不阻断 |
