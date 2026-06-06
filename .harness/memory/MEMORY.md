# Team Memory

Shared cross-rein knowledge that persists across sessions.

## Project facts (verified)

- Project name: Enterprise Agentic RAG — 开发者智能客服 Agent 平台
- Default branch: `main`
- 5 internal agents: Master / Tool / Knowledge / Code / Verifier (implementation detail, not Mavis reins)
- AgentState: 72 TypedDict fields in `graph/state.py`
- 16 LangGraph nodes in `graph/nodes/`
- 4 intent-aware RAG workflows in `workflows/`
- 22 agent decision eval cases in `evals/agent_decision_eval.py`
- 8 eval gate cases; must pass before opening a PR
- v3.1 refactor: `retrieval/` merged into `rag/`, `graph/workflow.py` split into `builder.py` + `nodes/`
- Semantic cache: dual-layer (SHA256 exact + embedding similarity ≥ 0.92)
- Cross-Encoder: `pdurugyan/qwen3-reranker-0.6b-q8_0` via Ollama
- Interview docs: 12 top-level + 12 topic docs in `technical_deep_dive/`

## Team roster

| Rein | Owner | Responsibility |
|---|---|---|
| `developer` | — | Python backend, LangGraph, FastAPI |
| `frontend-dev` | — | React console + widget |
| `tester` | — | pytest, eval gate, regression |
| `rag-expert` | — | RAG pipeline, fusion, reranker, graph rag |
| `agent-expert` | — | MasterAgent, AgentState, node graph |

## Gotchas

- Don't confuse the 5 internal agents with Mavis harness reins — the harness reins are AI coding agents, the 5 internal agents are runtime components
- `max_graph_steps = 18` is an upper bound; typical requests take 5-8 steps
- Neo4j unavailable → `hybrid_only` mode automatically (no graph augmentation)
- Production: dev-mode memory fallbacks (MockRepository, MemoryVectorStore) are disabled