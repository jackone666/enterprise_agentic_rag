# Code Standards

Shared conventions for all developers on this project. Inherit from here — don't duplicate in individual agent.md bodies.

## Python

- **Ruff** (`E, F, I, N, W, UP`), line-length 120, target Python 3.11
- Run `uv run ruff check . --fix` before committing
- **TypedDict for all state** — never use bare `dict` for AgentState fields
- **Async-first**: use `async def` for all I/O (DB, Redis, LLM calls)
- **No secrets**: read from `os.getenv` or Pydantic Settings, never hardcode

## TypeScript / React

- Strict TypeScript; no `any`
- CVA for component variant logic
- Tailwind v4 — use `@apply` sparingly; prefer utility classes directly in JSX
- SSE streaming: always handle `error` and `end` events; never assume clean stream

## Testing

- Every new feature needs a test in `tests/test_*.py`
- Unit tests: `uv run pytest -m "not integration"` — no Docker required
- Integration tests require `docker compose up -d`
- Eval gate (8 cases) must pass before opening a PR

## Git workflow

- Branch from `main`; PR against `main`
- Conventional commits: `feat:` / `fix:` / `docs:` / `refactor:` / `test:` / `chore:`
- Never force-push `main`

## Key file locations

| What | Path |
|---|---|
| AgentState (72 fields) | `src/enterprise_agentic_rag/graph/state.py` |
| LangGraph workflow builder | `src/enterprise_agentic_rag/graph/builder.py` |
| LangGraph nodes (16) | `src/enterprise_agentic_rag/graph/nodes/` |
| MasterAgent routing | `src/enterprise_agentic_rag/agents/master_agent.py` |
| RAG fusion | `src/enterprise_agentic_rag/rag/fusion.py` |
| Eval gate | `scripts/run_eval_gate.py` |
| Docker services | `docker-compose.yml` |
| Interview docs | `technical_deep_dive/` + `technical_deep_dive/主题/` |