# Release Checklist

上线前逐项确认。

---

## Pre-Release Checklist

### Code & Build
- [ ] `pytest` all passed (302/302)
- [ ] `cd frontend && npm run build` passed (0 TypeScript errors)
- [ ] `docker compose build` succeeded
- [ ] `docker compose config` valid
- [ ] No critical dependency CVEs

### Database & Infra
- [ ] `docker compose up -d` all services healthy
- [ ] `python scripts/init_db.py` completed
- [ ] `python scripts/ingest_docs.py` completed
- [ ] PostgreSQL 12 tables verified
- [ ] Redis ping OK
- [ ] Milvus collection ready
- [ ] MinIO bucket exists

### Agent Quality
- [ ] Agent Eval Gate passed (all thresholds)
- [ ] Prompt Eval Gate passed (if prompt changed)
- [ ] RAG metrics: hit@3≥0.80, recall@5≥0.75, MRR≥0.70
- [ ] Answer metrics: citation≥0.95, groundedness≥0.85, refusal≥0.90
- [ ] Agent metrics: verify≥0.85, fallback≤0.25, tool≥0.90

### Safety
- [ ] Security gate passed — no critical issues
- [ ] Destructive tools disabled (`ENABLE_DESTRUCTIVE_TOOLS=false`)
- [ ] Sensitive tools require confirmation
- [ ] Permission check enabled
- [ ] Human fallback enabled
- [ ] No API keys in source code
- [ ] `.env` is gitignored

### Observability
- [ ] `GET /health` → 200
- [ ] `GET /metrics` → JSON valid
- [ ] `GET /prometheus_metrics` → Prometheus format
- [ ] `POST /chat` → returns trace_id + node_events
- [ ] Node events recorded in JSONL + PostgreSQL
- [ ] Tool audit logs recorded

### Smoke Test
- [ ] `/health` → healthy
- [ ] `/chat` → returns answer with citations
- [ ] `/metrics` → returns JSON
- [ ] `/feedback` → accepts thumbs
- [ ] Frontend http://localhost:5173 accessible

### Canary & Rollback
- [ ] Canary plan defined: 5%→20%→50%→100%
- [ ] Canary metrics configured
- [ ] Rollback plan defined for code/prompt/workflow/retriever/LLM/tool/verifier
- [ ] Previous stable image tag known
- [ ] Previous stable prompt versions known
- [ ] Fallback LLM provider configured
- [ ] Tool disable flags available

### Approval
- [ ] Engineering lead approved
- [ ] Product owner approved (if user-facing change)
- [ ] Security lead approved (if security-relevant change)
- [ ] Release Manager signed off

### Post-Release
- [ ] Monitor canary metrics for 60 minutes
- [ ] Check fallback_rate within baseline
- [ ] Check thumbs_down_rate within baseline
- [ ] Verify no regression in eval metrics
- [ ] Notify #deployments channel
- [ ] Update runbook if new failure modes discovered
