# AGENTS.md

面向开发者门户 / 企业技术支持 / 智能客服入口的多智能体 Agentic RAG 系统，基于 LangGraph 实现的主从 Agent 架构，支持深度意图识别、混合检索、代码生成/执行与答案校验。

## Setup commands

- Install deps: `uv sync` (or `poetry install`)
- Start dev:    `uv run uvicorn enterprise_agentic_rag.app.main:app --reload --port 8000`
- Build:        `uv build`
- Test:         `uv run pytest`
- Lint:         `uv run ruff check .`
- Typecheck:    `uv run mypy .`            # optional, code-analysis extra
- Frontend dev: `cd frontend && npm run dev`
- Widget dev:   `cd widget && npm run dev`
- All services: `docker compose up -d && ./scripts/healthcheck.sh`

## Project layout

- `src/enterprise_agentic_rag/` — Python backend (FastAPI + LangGraph multi-agent)
  - `app/` — FastAPI routes (`/chat`, `/chat/stream`, `/feedback`)
  - `agent/` / `agents/` — Deep Intent classifier + 5 Agent implementations
  - `rag/` — Retrieval engine (vector, keyword, graph, fusion, reranker, cache)
  - `workflows/` — Intent-aware RAG workflows (hybrid/graph/error/code)
  - `graph/` — LangGraph state (72 fields) + workflow builder + 16 nodes
  - `memory/` — Multi-tier memory (short/summary/user/long-term)
  - `context/` — Context management, token budget, prompt registry, conflict detection
  - `llm/` — LLM provider abstraction (OpenAI / DashScope / vLLM / mock)
  - `tools/` — Tool system with safe/sensitive tier classification
  - `observability/` — JSONL tracing, OTel integration, Prometheus metrics
  - `evals/` — RAG evaluation, regression eval, agent decision eval (22 cases)
  - `recovery/` — Fallback policy, retry, threshold calibration
- `frontend/` — Developer console (React + TypeScript + Vite + Tailwind)
- `widget/` — End-user chat widget (full-page + floating embed mode)
- `technical_deep_dive/` — 12 interview-prep docs + `主题/` subdirectory
- `tests/` — 20+ pytest files (unit + integration)
- `scripts/` — Dev ops scripts (healthcheck, data init, eval gate)
- `deploy/` — Docker, Prometheus alerts
- `harness/` — Legacy harness config (not Mavis; see `.harness/` for Mavis agents)

## Code style

- Python: ruff (`E, F, I, N, W, UP`), line-length=120, target=py311
- TypeScript: strict mode (`tsconfig`), Vite + Tailwind v4 + CVA
- Tests: pytest-asyncio, `asyncio_mode = auto`, `pytest.ini_options` markers
- Run `uv run ruff check . --fix` before committing
- No secrets in `.env` — all secrets via environment variables

## Testing instructions

- Unit tests: `uv run pytest -m "not integration"` (no Docker services needed)
- Integration tests: `uv run pytest -m integration` (requires Docker services)
- Full suite: `uv run pytest`
- Eval Gate: `python scripts/run_eval_gate.py --threshold 0.7`

## PR & commit conventions

- Branch from `main`; never push to it directly
- Commit message: conventional commits (`feat:` / `fix:` / `docs:` / `refactor:`)
- Eval Gate must be green before opening a PR

## Security

- Never commit secrets — `.env` is in `.gitignore`
- Code execution tier: safe (direct) / sensitive (user confirm) — see `tools/` base tool design
- Production: no dev-mode fallbacks (memory mocks, fail-open rate limiting)