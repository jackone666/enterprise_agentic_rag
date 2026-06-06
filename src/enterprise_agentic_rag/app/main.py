"""FastAPI application for the Enterprise Agentic RAG system.

Start with:
    uvicorn enterprise_agentic_rag.app.main:app --reload

Endpoints:
    GET  /health    — health check
    POST /chat      — process a user query
    GET  /metrics   — current observability metrics snapshot
    POST /feedback  — submit thumbs_up/down feedback (Data Flywheel)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Load .env before any enterprise_agentic_rag imports
_env = Path(__file__).resolve().parents[3] / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
        if _k and _k not in os.environ:
            os.environ[_k] = _v

import asyncio
import json

from fastapi import FastAPI, File, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from enterprise_agentic_rag.config.settings import get_settings
from enterprise_agentic_rag.evals.online_feedback import FeedbackHandler, FeedbackRecord
from enterprise_agentic_rag.graph.workflow import build_workflow
from enterprise_agentic_rag.observability.metrics import get_metrics_collector, get_prometheus_metrics
from enterprise_agentic_rag.observability.tracer import get_tracer

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Enterprise Agentic RAG",
    description="Multi-agent RAG QA system for enterprise knowledge bases.",
    version="0.1.0",
)

# CORS — allow frontend dev server + widget
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",  # developer console
        "http://localhost:5174", "http://127.0.0.1:5174",  # customer widget
        "*",  # allow embedded widget from any origin
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_workflow = build_workflow()
_tracer = get_tracer()
_metrics = get_metrics_collector()
_feedback_handler = FeedbackHandler()

# In-memory store: trace_id → last result (for feedback auto-capture)
_recent_results: dict[str, dict[str, Any]] = {}
_MAX_RECENT = 1000  # cap to prevent unbounded memory growth


def _store_result(trace_id: str, result: dict[str, Any]) -> None:
    """Store a recent result for feedback auto-capture, capping memory."""
    if len(_recent_results) >= _MAX_RECENT:
        # Evict oldest key
        oldest = next(iter(_recent_results))
        del _recent_results[oldest]
    _recent_results[trace_id] = result


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Incoming chat request."""

    query: str = Field(..., description="User question", min_length=1)
    user_id: str = Field(default="anonymous", description="Caller identifier")
    session_id: str = Field(default="default", description="Session identifier")
    deep_thinking: bool = Field(default=True, description="Enable chain-of-thought reasoning visibility")


class ChatResponse(BaseModel):
    """Chat response returned to the client."""

    answer: str = Field(..., description="Final answer text")
    citations: list[dict[str, Any]] = Field(default_factory=list)
    intent: str = Field(default="unknown")
    need_human: bool = Field(default=False)
    verified: bool = Field(default=True)
    verification_reason: str = Field(default="")
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    tool_errors: list[str] = Field(default_factory=list)
    # RAG
    retrieved_docs: list[dict[str, Any]] = Field(default_factory=list)
    # Memory / context
    chat_history_count: int = Field(default=0)
    session_summary: str = Field(default="")
    memory_ckpt_id: str = Field(default="")
    token_budget: dict[str, Any] = Field(default_factory=dict)
    # Fallback / recovery
    fallback_reason: str = Field(default="")
    recovery_action: str = Field(default="")
    recoverable: bool = Field(default=True)
    retry_count: dict[str, int] = Field(default_factory=dict)
    retry_history: list[dict[str, Any]] = Field(default_factory=list)
    # Observability
    trace_id: str = Field(default="")
    retrieval_events: list[dict[str, Any]] = Field(default_factory=list)
    verification_events: list[dict[str, Any]] = Field(default_factory=list)
    # Evaluation
    auto_captured: bool = Field(default=False)
    eval_result: dict[str, Any] = Field(default_factory=dict)
    pipeline_trace: dict[str, Any] = Field(default_factory=dict)
    retrieval_backend: str = Field(default="keyword")


class FeedbackRequest(BaseModel):
    """User feedback submission."""

    trace_id: str = Field(..., description="Trace ID from /chat response")
    session_id: str = Field(default="", description="Session ID")
    thumbs_up: bool = Field(default=True, description="User satisfaction")
    feedback_text: str = Field(default="", description="Optional feedback text")
    user_id: str = Field(default="anonymous", description="Caller identifier")


class FeedbackResponse(BaseModel):
    """Feedback processing result."""

    received: bool = Field(default=True)
    auto_captured: bool = Field(default=False)
    reason: str = Field(default="")


class MetricsResponse(BaseModel):
    """Observability metrics snapshot."""

    metrics: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Return the current observability metrics snapshot (JSON)."""
    return MetricsResponse(metrics=_metrics.snapshot())


@app.get("/prometheus_metrics")
async def prometheus_metrics() -> Response:
    """Return Prometheus-format metrics (text/plain)."""
    return Response(content=get_prometheus_metrics(), media_type="text/plain")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a user query through the full agentic RAG pipeline."""
    trace_id = _tracer.new_trace()
    settings = get_settings()

    # Rate limiter — fail-open: Redis down → request passes through
    from enterprise_agentic_rag.middleware.rate_limiter import get_rate_limiter
    limiter = get_rate_limiter()
    rate_key = request.user_id or "anonymous"
    if not limiter.is_allowed(rate_key):
        return ChatResponse(
            answer="请求过于频繁，请稍后再试。",
            intent="rate_limited",
            need_human=False,
            trace_id=trace_id,
        )

    initial_state: dict[str, Any] = {
        "query": request.query,
        "user_id": request.user_id,
        "session_id": request.session_id,
        "trace_id": trace_id,
        "tool_events": [],
        "retrieval_events": [],
        "verification_events": [],
    }

    try:
        result = await asyncio.wait_for(
            _workflow.ainvoke(initial_state),
            timeout=settings.app.request_timeout_seconds,
        )
    except TimeoutError:
        result = {
            **initial_state,
            "final_answer": "请求处理超时，已为您转人工处理。",
            "intent": "timeout",
            "need_human": True,
            "verified": False,
            "fallback_reason": "request_timeout",
        }

    # Store for potential feedback
    _store_result(trace_id, result)

    # Auto-capture failed cases (need_human, not verified, fallback)
    auto_captured = False
    auto_reason = FeedbackHandler._auto_capture_reason(result)
    if auto_reason:
        _feedback_handler.process_feedback(
            FeedbackRecord(
                trace_id=trace_id,
                session_id=request.session_id,
                thumbs_up=True,  # neutral — not user-initiated
                user_id=request.user_id,
            ),
            result=result,
        )
        auto_captured = True

    # Build pipeline trace for frontend visualization
    pipeline_trace = _build_pipeline_trace(result)

    # Submit to async evaluator (non-blocking) — bad cases auto-persist to PG
    eval_result = {}
    try:
        from enterprise_agentic_rag.evals.async_evaluator import EvalJob, get_async_evaluator
        async_eval = get_async_evaluator()
        async_eval.submit(EvalJob(
            query=request.query,
            answer=result.get("final_answer", ""),
            retrieved_docs=result.get("retrieved_docs", []),
            trace_id=trace_id,
            session_id=request.session_id,
            user_id=request.user_id,
        ))
        # Return cached result if available from prior request
        cached = async_eval.get_result(trace_id)
        if cached:
            eval_result = cached
    except Exception:
        pass

    return ChatResponse(
        answer=result.get("final_answer", ""),
        citations=result.get("citations", []),
        intent=result.get("intent", "unknown"),
        need_human=result.get("need_human", False),
        verified=result.get("verified", True),
        verification_reason=result.get("verification_reason", ""),
        tool_results=result.get("tool_results", []),
        tool_errors=result.get("tool_errors", []),
        retrieved_docs=result.get("retrieved_docs", []),
        chat_history_count=len(result.get("chat_history", [])),
        session_summary=result.get("session_summary", ""),
        memory_ckpt_id=result.get("memory_ckpt_id", ""),
        token_budget=result.get("token_budget", {}),
        fallback_reason=result.get("fallback_reason", ""),
        recovery_action=result.get("recovery_action", ""),
        recoverable=result.get("recoverable", True),
        retry_count=result.get("retry_count", {}),
        retry_history=result.get("retry_history", []),
        trace_id=result.get("trace_id", trace_id),
        retrieval_events=result.get("retrieval_events", []),
        verification_events=result.get("verification_events", []),
        auto_captured=auto_captured,
        eval_result=eval_result,
        pipeline_trace=pipeline_trace,
        retrieval_backend=result.get("retrieval_backend", "keyword"),
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Process a user query through the RAG pipeline with SSE streaming.

    Streams pipeline node events as they execute, allowing the frontend
    to show real-time progress. When deep_thinking is enabled, streams
    chain-of-thought reasoning separately from the final answer.

    SSE Event types:
        start       — request received
        node_end    — a workflow node completed execution
        thinking    — chain-of-thought reasoning chunk (only when deep_thinking=true)
        answer_chunk — answer text chunk (streamed as generated)
        done        — final response with full answer, citations, etc.
        error       — timeout or exception
        end         — stream terminated
    """

    async def event_generator():
        trace_id = _tracer.new_trace()
        settings = get_settings()
        initial_state: dict[str, Any] = {
            "query": request.query,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "trace_id": trace_id,
            "tool_events": [],
            "retrieval_events": [],
            "verification_events": [],
            "deep_thinking": request.deep_thinking,
        }

        # Stream start event
        yield f"data: {json.dumps({'type': 'start', 'trace_id': trace_id, 'query': request.query[:100]}, ensure_ascii=False)}\n\n"

        try:
            result: dict[str, Any] = initial_state
            prev_answer_len = 0
            prev_thinking_len = 0

            async with asyncio.timeout(settings.app.request_timeout_seconds):
                async for state_update in _workflow.astream(initial_state, stream_mode="values"):
                    if isinstance(state_update, dict):
                        result = state_update

                        # Stream thinking (CoT) when deep thinking is enabled
                        if request.deep_thinking:
                            thinking = result.get("deep_thinking_content", result.get("thinking_trace", ""))
                            if thinking and len(thinking) > prev_thinking_len:
                                new_content = thinking[prev_thinking_len:]
                                prev_thinking_len = len(thinking)
                                yield f"data: {json.dumps({'type': 'thinking', 'thinking_content': new_content}, ensure_ascii=False)}\n\n"

                        # Stream answer chunks
                        draft = result.get("draft_answer", result.get("final_answer", ""))
                        if draft and len(draft) > prev_answer_len:
                            new_chunk = draft[prev_answer_len:]
                            prev_answer_len = len(draft)
                            yield f"data: {json.dumps({'type': 'answer_chunk', 'content': new_chunk}, ensure_ascii=False)}\n\n"

            _store_result(trace_id, result)

            # Build final done event with complete response
            final_answer = result.get("final_answer", result.get("draft_answer", ""))
            # Include thinking trace in done event for clients that joined late
            thinking_trace = ""
            if request.deep_thinking:
                thinking_trace = result.get("deep_thinking_content", result.get("thinking_trace", ""))

            yield f"data: {json.dumps({'type': 'done', 'answer': final_answer, 'trace_id': trace_id, 'citations': result.get('citations', []), 'verified': result.get('verified', True), 'intent': result.get('intent', 'unknown'), 'thinking': thinking_trace}, ensure_ascii=False)}\n\n"

        except TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'message': '请求处理超时，已停止执行', 'trace_id': trace_id}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _safe_summary(output: dict[str, Any], node_name: str) -> dict[str, Any]:
    """Extract a safe summary from node output for streaming."""
    summary: dict[str, Any] = {"node": node_name}
    if node_name == "retrieve_knowledge":
        summary["docs_found"] = len(output.get("retrieved_docs", []))
        summary["mode"] = output.get("retrieval_mode", "")
    elif node_name == "generate_answer":
        summary["answer_length"] = len(output.get("draft_answer", ""))
    elif node_name == "verify_answer":
        summary["verified"] = output.get("verified", False)
    elif node_name == "deep_intent_recognition":
        summary["intent"] = output.get("intent", "")
        summary["confidence"] = output.get("deep_intent_confidence", 0)
    elif node_name == "call_tools":
        summary["tools_called"] = len(output.get("tool_calls", []))
        summary["tool_errors"] = len(output.get("tool_errors", []))
    return summary


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Submit user feedback (thumbs_up/down) for a previous /chat response.

    When thumbs_down is given, the case is automatically captured to
    data/eval/failed_cases.jsonl for the Data Flywheel.
    """
    # Look up the prior result
    prior_result = _recent_results.get(request.trace_id, None)

    fb = FeedbackRecord(
        trace_id=request.trace_id,
        session_id=request.session_id,
        thumbs_up=request.thumbs_up,
        feedback_text=request.feedback_text,
        user_id=request.user_id,
    )

    outcome = _feedback_handler.process_feedback(fb, result=prior_result)

    return FeedbackResponse(
        received=True,
        auto_captured=outcome["auto_captured"],
        reason=outcome["reason"],
    )


def _build_pipeline_trace(result: dict[str, Any]) -> dict[str, Any]:
    """Build structured pipeline trace for frontend visualization."""
    return {
        "total_latency_ms": 0.0,
        "node_count": 0,
        "steps": [],
        "backend": result.get("retrieval_backend", "keyword"),
    }


# ---------------------------------------------------------------------------
# Widget API — customer-facing suggestions
# ---------------------------------------------------------------------------
class SuggestionItem(BaseModel):
    """A suggested question for the smart customer service widget."""

    id: str = Field(..., description="Unique suggestion identifier")
    label: str = Field(..., description="Short category label")
    question: str = Field(..., description="The suggested question text")
    icon: str = Field(default="", description="Optional emoji icon")


class SuggestionsResponse(BaseModel):
    """List of suggested questions for the widget welcome screen."""

    suggestions: list[SuggestionItem] = Field(default_factory=list)


# Default suggestions — can be overridden via admin API or config
_DEFAULT_SUGGESTIONS: list[SuggestionItem] = [
    SuggestionItem(id="develop", label="开发入门", question="鸿蒙应用开发如何入门？", icon="💻"),
    SuggestionItem(id="upgrade", label="系统升级", question="如何升级HarmonyOS 6？", icon="⬆️"),
    SuggestionItem(id="api", label="API 使用", question="HarmonyOS 网络请求 API 怎么用？", icon="🔌"),
    SuggestionItem(id="error", label="错误排查", question="应用闪退怎么排查？", icon="🔧"),
    SuggestionItem(id="distribute", label="应用分发", question="如何发布应用到华为应用市场？", icon="📦"),
    SuggestionItem(id="permission", label="权限管理", question="HarmonyOS 动态权限怎么申请？", icon="🔐"),
    SuggestionItem(id="lifecycle", label="生命周期", question="ArkUI 页面生命周期是怎样的？", icon="🔄"),
    SuggestionItem(id="migration", label="版本迁移", question="从 API 9 迁移到 API 12 需要注意什么？", icon="📋"),
]


@app.get("/api/suggestions", response_model=SuggestionsResponse)
async def get_suggestions() -> SuggestionsResponse:
    """Return suggested questions for the customer-facing chatbot welcome screen.

    Can be customized by updating _DEFAULT_SUGGESTIONS or loading from config.
    """
    return SuggestionsResponse(suggestions=_DEFAULT_SUGGESTIONS)


# ---------------------------------------------------------------------------
# Admin routes — document management
# ---------------------------------------------------------------------------
@app.post("/admin/ingest")
async def admin_ingest() -> dict[str, Any]:
    """Trigger ingestion of local data/docs/*.md into MinIO + Milvus."""
    from enterprise_agentic_rag.rag.ingestion import IngestionPipeline

    pipeline = IngestionPipeline()
    report = pipeline.run()
    return {
        "total_docs": report.total_docs,
        "total_chunks": report.total_chunks,
        "minio_uploaded": report.minio_uploaded,
        "milvus_upserted": report.milvus_upserted,
        "errors": report.errors,
        "duration_ms": report.duration_ms,
    }


@app.get("/admin/docs")
async def admin_list_docs() -> dict[str, Any]:
    """List documents stored in MinIO."""
    from enterprise_agentic_rag.rag.minio_store import MinIOStore

    store = MinIOStore()
    docs = store.list_documents()
    return {"count": len(docs), "documents": docs}


@app.post("/admin/upload")
async def admin_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a document file → MinIO → chunk → embed → Milvus."""
    import os
    import tempfile

    # Save uploaded file to temp location
    suffix = os.path.splitext(file.filename or "upload.txt")[1] or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Upload to MinIO
        from enterprise_agentic_rag.rag.minio_store import MinIOStore
        store = MinIOStore()
        obj_name = file.filename or "uploaded_doc"
        minio_result = store.upload_document(tmp_path, object_name=obj_name)

        # Ingest: chunk + embed + Milvus
        from enterprise_agentic_rag.rag.embedding_provider import get_embedding_provider
        from enterprise_agentic_rag.rag.milvus_store import MilvusStore
        from enterprise_agentic_rag.rag.preprocessing import preprocess_document
        from enterprise_agentic_rag.rag.splitter import split_text

        text = content.decode("utf-8", errors="replace")
        meta = preprocess_document(text)

        chunks = split_text(text, chunk_size=500)
        chunk_dicts = [
            {"source": obj_name, "chunk_index": str(i), "content": c,
             "chunk_id": f"{obj_name}_{i}", "title": meta.get("title", "")}
            for i, c in enumerate(chunks)
        ]

        ep = get_embedding_provider()
        vecs = ep.embed([c["content"] for c in chunk_dicts])

        ms = MilvusStore(vector_size=ep.vector_size)
        milvus_count = ms.upsert_chunks(chunk_dicts, vecs)

        return {
            "success": True,
            "filename": file.filename,
            "minio_uploaded": minio_result is not None,
            "chunks": len(chunks),
            "milvus_indexed": milvus_count,
            "sections": len(meta.get("sections", [])),
            "tables": len(meta.get("tables", [])),
        }
    finally:
        os.unlink(tmp_path)


@app.get("/admin/bad-cases")
async def admin_bad_cases(limit: int = 50, source: str = "") -> dict[str, Any]:
    """Query bad cases + feedback from PostgreSQL for monitoring dashboard."""
    from sqlalchemy import text

    from enterprise_agentic_rag.storage.database import get_db_manager

    dbm = get_db_manager()
    if not await dbm.check_connection():
        return {"bad_cases": [], "stats": {"total": 0, "source_distribution": {}}, "alerts": []}

    async with dbm.session() as sess:
        where = "WHERE source = :source" if source else ""
        rows = await sess.execute(
            text(f"SELECT trace_id, query, reason, source, created_at FROM failed_cases {where} ORDER BY created_at DESC LIMIT :limit"),
            {"source": source, "limit": limit} if source else {"limit": limit},
        )
        cases = [dict(r._mapping) for r in rows.fetchall()]

        stats = await sess.execute(text("SELECT source, COUNT(*) as cnt FROM failed_cases GROUP BY source"))
        source_dist = {r.source: r.cnt for r in stats.fetchall()}

        total = await sess.execute(text("SELECT COUNT(*) FROM failed_cases"))

    # Build alerts from metrics
    from enterprise_agentic_rag.observability.metrics import get_metrics_collector
    m = get_metrics_collector().snapshot()
    alerts = []
    if m["fallback_rate"] > 0.25:
        alerts.append({"level": "warning", "msg": f"兜底率过高: {m['fallback_rate']:.1%}"})
    if m["human_fallback_rate"] > 0.15:
        alerts.append({"level": "warning", "msg": f"人工升级率过高: {m['human_fallback_rate']:.1%}"})
    if m["tool_success_rate"] < 0.9 and m.get("tool_attempts", 0) > 0:
        alerts.append({"level": "error", "msg": f"工具成功率过低: {m['tool_success_rate']:.1%}"})
    if m["verification_pass_rate"] < 0.85 and m.get("verification_attempts", 0) > 0:
        alerts.append({"level": "warning", "msg": f"校验通过率过低: {m['verification_pass_rate']:.1%}"})

    return {
        "bad_cases": cases,
        "stats": {"total": total.scalar(), "source_distribution": source_dist},
        "alerts": alerts,
    }
