# Production Readiness Review (PRR)

生产发布前必须完成的评审清单。

---

## 1. Code Readiness

- [ ] `pytest` — 100% pass (302/302)
- [ ] `npm run build` — 0 TypeScript errors
- [ ] `docker compose build` — all images built
- [ ] `docker compose config` — valid
- [ ] No critical dependency CVEs
- [ ] Commit SHA recorded

## 2. Agent Quality Readiness

- [ ] Agent Eval Gate — all thresholds passed
- [ ] Prompt Eval Gate — passed (if prompt files changed)
- [ ] RAG metrics: hit@3≥0.80, recall@5≥0.75, MRR≥0.70
- [ ] Answer metrics: citation≥0.95, groundedness≥0.85
- [ ] Verifier metrics: pass_rate≥0.85, false_pass≤0.05
- [ ] Tool contract tests passed (if adapter changed)

## 3. Runtime Readiness

- [ ] PostgreSQL healthy + 12 tables exist
- [ ] Redis healthy + ping OK
- [ ] Milvus healthy + collection exists
- [ ] MinIO healthy + bucket exists
- [ ] Prometheus healthy + scraping
- [ ] Grafana dashboard configured

## 4. Safety Readiness

- [ ] Permission guardrail enabled (`knowledge_search` enforced)
- [ ] Human fallback enabled (`enable_human_fallback=true`)
- [ ] Sensitive tools require confirmation
- [ ] Destructive tools disabled (`ENABLE_DESTRUCTIVE_TOOLS=false`)
- [ ] Rate limiter active (Redis or fail-open memory mode)

## 5. Observability Readiness

- [ ] `GET /metrics` → JSON valid
- [ ] `GET /prometheus_metrics` → Prometheus format
- [ ] `trace_id` present in every `/chat` response
- [ ] `node_events` recorded (JSONL + PostgreSQL)
- [ ] `tool_events` recorded
- [ ] `llm_events` recorded
- [ ] `retrieval_events` recorded

## 6. Rollback Readiness

- [ ] Previous stable image tag: `${PREVIOUS_STABLE_TAG}`
- [ ] Previous stable prompt versions known (router/knowledge/verifier/answer)
- [ ] Previous stable workflow version: `${PREVIOUS_WORKFLOW_VERSION}`
- [ ] Previous stable rag config: `${PREVIOUS_RAG_CONFIG}`
- [ ] Fallback LLM provider configured
- [ ] Tool disable flags available per tool
- [ ] Rollback runbook reviewed by oncall

## 7. Approval

| Role | Required | Signed |
|------|----------|--------|
| Engineering Lead | Yes | ________ |
| Product Owner | If user-facing change | ________ |
| Security Lead | If security-relevant | ________ |
| Release Manager | Yes | ________ |

---

## PRR Outcome

| Outcome | Action |
|---------|--------|
| All ✅ | Proceed to production deployment |
| Minor issues | Fix + re-run PRR |
| Critical issues | **BLOCK** — do not deploy |
