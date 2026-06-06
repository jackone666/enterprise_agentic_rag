"""Repository layer — encapsulates common read/write operations.

All methods accept an async session.  Higher-level code uses these
repos so raw SQL never appears in business logic.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from enterprise_agentic_rag.storage.models import (
    EvalCaseModel,
    FailedCaseModel,
    FeedbackModel,
    LongTermMemoryModel,
    MessageModel,
    QALogModel,
    SessionModel,
    ToolAuditLogModel,
    UserModel,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ===================================================================
# Users
# ===================================================================
async def upsert_user(
    sess: AsyncSession,
    user_id: str,
    role: str = "basic",
    department: str = "",
    permissions: list[str] | None = None,
    name: str = "",
    email: str = "",
) -> UserModel:
    result = await sess.execute(select(UserModel).where(UserModel.user_id == user_id))
    user = result.scalar_one_or_none()
    now = _utcnow()
    if user is None:
        user = UserModel(
            user_id=user_id,
            name=name or f"用户{user_id}",
            role=role,
            department=department,
            email=email,
            permissions=json.dumps(permissions or []),
            created_at=now,
            updated_at=now,
        )
        sess.add(user)
    else:
        if role:
            user.role = role
        if department:
            user.department = department
        user.updated_at = now
    await sess.flush()
    return user


async def get_user(sess: AsyncSession, user_id: str) -> dict[str, Any] | None:
    result = await sess.execute(select(UserModel).where(UserModel.user_id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        return None
    return _user_to_dict(user)


def _user_to_dict(u: UserModel) -> dict[str, Any]:
    perms = []
    try:
        perms = json.loads(u.permissions)
    except (json.JSONDecodeError, TypeError):
        pass
    return {
        "user_id": u.user_id,
        "name": u.name,
        "role": u.role,
        "department": u.department,
        "email": u.email,
        "permissions": perms,
        "preferred_language": u.preferred_language,
    }


# ===================================================================
# Sessions
# ===================================================================
async def upsert_session(
    sess: AsyncSession,
    session_id: str,
    user_id: str = "",
    summary: str = "",
) -> SessionModel:
    result = await sess.execute(
        select(SessionModel).where(SessionModel.session_id == session_id)
    )
    model = result.scalar_one_or_none()
    now = _utcnow()
    if model is None:
        model = SessionModel(
            session_id=session_id,
            user_id=user_id,
            summary=summary,
            created_at=now,
            updated_at=now,
        )
        sess.add(model)
    else:
        if summary:
            model.summary = summary
        model.updated_at = now
    await sess.flush()
    return model


async def get_session_summary(sess: AsyncSession, session_id: str) -> str:
    result = await sess.execute(
        select(SessionModel.summary).where(SessionModel.session_id == session_id)
    )
    val = result.scalar_one_or_none()
    return val or ""


# ===================================================================
# Messages
# ===================================================================
async def insert_message(
    sess: AsyncSession,
    session_id: str,
    role: str,
    content: str,
    intent: str = "",
    metadata: dict[str, Any] | None = None,
) -> MessageModel:
    msg = MessageModel(
        session_id=session_id,
        role=role,
        content=content,
        intent=intent,
        metadata_=json.dumps(metadata or {}),
    )
    sess.add(msg)
    await sess.flush()
    return msg


async def get_messages(
    sess: AsyncSession,
    session_id: str,
    last_n: int | None = None,
) -> list[dict[str, Any]]:
    stmt = (
        select(MessageModel)
        .where(MessageModel.session_id == session_id)
        .order_by(MessageModel.created_at.asc())
    )
    if last_n:
        # Fetch all then take last_n (simpler than subquery for sqlite compat)
        pass
    result = await sess.execute(stmt)
    rows = result.scalars().all()
    if last_n:
        rows = rows[-last_n:]
    return [
        {"role": r.role, "content": r.content, "intent": r.intent}
        for r in rows
    ]


# ===================================================================
# QA Logs
# ===================================================================
async def insert_qa_log(
    sess: AsyncSession,
    trace_id: str,
    session_id: str = "",
    user_id: str = "",
    query: str = "",
    answer: str = "",
    intent: str = "",
    citations: list[dict[str, Any]] | None = None,
    verified: bool = True,
    need_human: bool = False,
    fallback_reason: str = "",
    latency_ms: float = 0.0,
) -> QALogModel:
    log = QALogModel(
        trace_id=trace_id,
        session_id=session_id,
        user_id=user_id,
        query=query,
        answer=answer,
        intent=intent,
        citations=json.dumps(citations or []),
        verified=verified,
        need_human=need_human,
        fallback_reason=fallback_reason,
        latency_ms=latency_ms,
    )
    sess.add(log)
    await sess.flush()
    return log


# ===================================================================
# Tool audit logs
# ===================================================================
async def insert_tool_audit_log(
    sess: AsyncSession,
    trace_id: str,
    session_id: str = "",
    user_id: str = "",
    tool_name: str = "",
    input_summary: str = "",
    output_summary: str = "",
    success: bool = True,
    error: str = "",
    latency_ms: float = 0.0,
) -> ToolAuditLogModel:
    log = ToolAuditLogModel(
        trace_id=trace_id,
        session_id=session_id,
        user_id=user_id,
        tool_name=tool_name,
        input_summary=input_summary,
        output_summary=output_summary,
        success=success,
        error=error,
        latency_ms=latency_ms,
    )
    sess.add(log)
    await sess.flush()
    return log


# ===================================================================
# Feedback
# ===================================================================
async def insert_feedback(
    sess: AsyncSession,
    trace_id: str,
    session_id: str = "",
    user_id: str = "",
    thumbs_up: bool = True,
    feedback_text: str = "",
) -> FeedbackModel:
    fb = FeedbackModel(
        trace_id=trace_id,
        session_id=session_id,
        user_id=user_id,
        thumbs_up=thumbs_up,
        feedback_text=feedback_text,
    )
    sess.add(fb)
    await sess.flush()
    return fb


# ===================================================================
# Long-Term Memories
# ===================================================================
async def upsert_long_term_memory(
    sess: AsyncSession,
    memory_id: str,
    user_id: str,
    content: str = "",
    importance: float = 0.0,
    memory_type: str = "episodic",
    source_session: str = "",
    source_turn: int = 0,
    metadata: dict[str, Any] | None = None,
) -> LongTermMemoryModel:
    result = await sess.execute(
        select(LongTermMemoryModel).where(LongTermMemoryModel.memory_id == memory_id)
    )
    model = result.scalar_one_or_none()
    now = _utcnow()
    if model is None:
        model = LongTermMemoryModel(
            memory_id=memory_id,
            user_id=user_id,
            content=content,
            importance=importance,
            memory_type=memory_type,
            source_session=source_session,
            source_turn=source_turn,
            metadata_=json.dumps(metadata or {}, ensure_ascii=False),
            created_at=now,
            accessed_at=now,
        )
        sess.add(model)
    else:
        model.content = content
        model.importance = importance
        model.memory_type = memory_type
        model.metadata_ = json.dumps(metadata or {}, ensure_ascii=False)
        model.accessed_at = now
    await sess.flush()
    return model


async def get_long_term_memories(
    sess: AsyncSession,
    user_id: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    stmt = (
        select(LongTermMemoryModel)
        .where(LongTermMemoryModel.user_id == user_id)
        .order_by(LongTermMemoryModel.accessed_at.desc())
        .limit(limit)
    )
    result = await sess.execute(stmt)
    rows = result.scalars().all()
    return [_ltm_to_dict(r) for r in rows]


async def get_long_term_memory_by_id(
    sess: AsyncSession,
    memory_id: str,
) -> dict[str, Any] | None:
    result = await sess.execute(
        select(LongTermMemoryModel).where(LongTermMemoryModel.memory_id == memory_id)
    )
    model = result.scalar_one_or_none()
    if model is None:
        return None
    return _ltm_to_dict(model)


async def delete_long_term_memory(
    sess: AsyncSession,
    memory_id: str,
) -> bool:
    result = await sess.execute(
        select(LongTermMemoryModel).where(LongTermMemoryModel.memory_id == memory_id)
    )
    model = result.scalar_one_or_none()
    if model is None:
        return False
    await sess.delete(model)
    await sess.flush()
    return True


async def delete_user_long_term_memories(
    sess: AsyncSession,
    user_id: str,
) -> int:
    result = await sess.execute(
        select(LongTermMemoryModel).where(LongTermMemoryModel.user_id == user_id)
    )
    rows = result.scalars().all()
    count = len(rows)
    for r in rows:
        await sess.delete(r)
    if count:
        await sess.flush()
    return count


def _ltm_to_dict(m: LongTermMemoryModel) -> dict[str, Any]:
    try:
        metadata = json.loads(m.metadata_ or "{}")
    except (json.JSONDecodeError, TypeError):
        metadata = {}
    return {
        "memory_id": m.memory_id,
        "user_id": m.user_id,
        "content": m.content,
        "importance": m.importance,
        "memory_type": m.memory_type,
        "source_session": m.source_session,
        "source_turn": m.source_turn,
        "metadata": metadata,
        "created_at": m.created_at.isoformat() if m.created_at else "",
        "accessed_at": m.accessed_at.isoformat() if m.accessed_at else "",
    }


# ===================================================================
# Eval cases
# ===================================================================
async def insert_eval_case(
    sess: AsyncSession,
    query: str,
    expected_intent: str = "",
    expected_sources: list[str] | None = None,
    expected_answer_keywords: list[str] | None = None,
    difficulty: str = "medium",
    source: str = "manual",
) -> EvalCaseModel:
    ec = EvalCaseModel(
        query=query,
        expected_intent=expected_intent,
        expected_sources=json.dumps(expected_sources or []),
        expected_answer_keywords=json.dumps(expected_answer_keywords or []),
        difficulty=difficulty,
        source=source,
    )
    sess.add(ec)
    await sess.flush()
    return ec


# ===================================================================
# Failed cases
# ===================================================================
async def insert_failed_case(
    sess: AsyncSession,
    trace_id: str = "",
    session_id: str = "",
    query: str = "",
    reason: str = "",
    source: str = "auto",
    payload: dict[str, Any] | None = None,
) -> FailedCaseModel:
    fc = FailedCaseModel(
        trace_id=trace_id,
        session_id=session_id,
        query=query,
        reason=reason,
        source=source,
        payload=json.dumps(payload or {}),
    )
    sess.add(fc)
    await sess.flush()
    return fc


# ===================================================================
# Convenience wrapper
# ===================================================================
class Repository:
    """Thin wrapper that delegates to individual functions above.

    Injected with a db_manager so callers don't manage sessions directly.
    """

    def __init__(self, db_manager: Any = None) -> None:
        from enterprise_agentic_rag.storage.database import get_db_manager

        self._dbm = db_manager or get_db_manager()

    async def available(self) -> bool:
        return await self._dbm.check_connection()

    # Delegates
    async def upsert_user(self, **kw: Any) -> Any:
        async with self._dbm.session() as sess:
            return await upsert_user(sess, **kw)

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        async with self._dbm.session() as sess:
            return await get_user(sess, user_id)

    async def upsert_session(self, **kw: Any) -> Any:
        async with self._dbm.session() as sess:
            return await upsert_session(sess, **kw)

    async def get_session_summary(self, session_id: str) -> str:
        async with self._dbm.session() as sess:
            return await get_session_summary(sess, session_id)

    async def insert_message(self, **kw: Any) -> Any:
        async with self._dbm.session() as sess:
            return await insert_message(sess, **kw)

    async def get_messages(self, session_id: str, last_n: int | None = None) -> list[dict[str, Any]]:
        async with self._dbm.session() as sess:
            return await get_messages(sess, session_id, last_n)

    async def insert_qa_log(self, **kw: Any) -> Any:
        async with self._dbm.session() as sess:
            return await insert_qa_log(sess, **kw)

    async def insert_tool_audit_log(self, **kw: Any) -> Any:
        async with self._dbm.session() as sess:
            return await insert_tool_audit_log(sess, **kw)

    async def insert_feedback(self, **kw: Any) -> Any:
        async with self._dbm.session() as sess:
            return await insert_feedback(sess, **kw)

    # Long-Term Memory delegates
    async def upsert_long_term_memory(self, **kw: Any) -> Any:
        async with self._dbm.session() as sess:
            return await upsert_long_term_memory(sess, **kw)

    async def get_long_term_memories(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        async with self._dbm.session() as sess:
            return await get_long_term_memories(sess, user_id, limit)

    async def delete_long_term_memory(self, memory_id: str) -> bool:
        async with self._dbm.session() as sess:
            return await delete_long_term_memory(sess, memory_id)

    async def delete_user_long_term_memories(self, user_id: str) -> int:
        async with self._dbm.session() as sess:
            return await delete_user_long_term_memories(sess, user_id)

    async def insert_eval_case(self, **kw: Any) -> Any:
        async with self._dbm.session() as sess:
            return await insert_eval_case(sess, **kw)

    async def insert_failed_case(self, **kw: Any) -> Any:
        async with self._dbm.session() as sess:
            return await insert_failed_case(sess, **kw)
