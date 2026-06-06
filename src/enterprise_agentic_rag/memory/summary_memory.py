"""Session summary — PostgreSQL + in-memory cache.

Summary stored in PostgreSQL sessions.summary.
Redis caches the summary for fast read access.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

REDIS_CACHE_TTL = 86400  # 24h


@dataclass
class SessionSummary:
    session_id: str
    summary: str
    key_topics: list[str] = field(default_factory=list)
    turn_count: int = 0


class SummaryMemory:
    """Manages session summaries — PostgreSQL primary, Redis cache, in-memory fallback."""

    def __init__(self, compress_threshold: int = 6) -> None:
        self.threshold = compress_threshold
        self._store: dict[str, SessionSummary] = {}  # in-memory fallback
        self._redis = None
        self._redis_available: bool | None = None
        self._repo = None

    # ------------------------------------------------------------------
    # Lazy clients
    # ------------------------------------------------------------------
    @property
    def redis(self):
        if self._redis is None and self._redis_available is not False:
            try:
                import redis.asyncio as aioredis
                from enterprise_agentic_rag.config.settings import get_settings
                s = get_settings()
                self._redis = aioredis.from_url(s.redis.connection_url, decode_responses=True)
                self._redis_available = True
            except Exception:
                self._redis_available = False
                self._redis = None
        return self._redis

    @property
    def repo(self):
        if self._repo is None:
            try:
                from enterprise_agentic_rag.storage.repositories import Repository
                self._repo = Repository()
            except Exception:
                self._repo = False
        return self._repo if self._repo is not False else None

    # ------------------------------------------------------------------
    # Update summary
    # ------------------------------------------------------------------
    def update_summary(
        self, session_id: str, turns: list[dict]
    ) -> SessionSummary:
        # Generate summary
        entry = self._generate_summary(session_id, turns)

        # In-memory fallback
        self._store[session_id] = entry

        # Try PostgreSQL
        self._save_pg(session_id, entry.summary)

        # Try Redis cache
        self._cache_redis(session_id, entry)

        return entry

    def get_summary(self, session_id: str) -> str:
        # In-memory fallback
        if session_id in self._store:
            return self._store[session_id].summary

        # Try Redis cache first
        cached = self._get_redis_cache(session_id)
        if cached:
            return cached

        # Try PostgreSQL
        pg_summary = self._get_pg_summary(session_id)
        return pg_summary or ""

    def clear_session(self, session_id: str) -> None:
        self._store.pop(session_id, None)
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if not loop.is_running() and self.redis:
                loop.run_until_complete(self.redis.delete(self._cache_key(session_id)))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Summary generation (LLM-first, heuristic fallback)
    # ------------------------------------------------------------------
    def _generate_summary(self, session_id: str, turns: list[dict]) -> SessionSummary:
        if len(turns) < self.threshold:
            existing = self._store.get(
                session_id, SessionSummary(session_id=session_id, summary="")
            )
            return existing

        # Try LLM summarization first
        try:
            from enterprise_agentic_rag.llm.provider_factory import get_llm_provider

            provider = get_llm_provider()
            if provider.provider_name != "mock":
                import asyncio

                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    prompt = self._build_summary_prompt(turns)
                    response = loop.run_until_complete(
                        provider.generate(prompt, temperature=0.3, max_tokens=512)
                    )
                    if response.success and response.content.strip():
                        summary_text = response.content.strip()
                        topics = self._extract_topics_from_summary(summary_text, turns)
                        return SessionSummary(
                            session_id=session_id,
                            summary=summary_text,
                            key_topics=topics,
                            turn_count=len(turns),
                        )
        except Exception:
            logger.warning("LLM summary failed, falling back to heuristic", exc_info=True)

        # Fallback to heuristic
        return self._generate_summary_heuristic(session_id, turns)

    def _generate_summary_heuristic(
        self, session_id: str, turns: list[dict]
    ) -> SessionSummary:
        """Original keyword-based heuristic summarisation — kept as fallback."""
        user_msgs = [t["content"] for t in turns if t.get("role") == "user"]
        assistant_msgs = [t["content"] for t in turns if t.get("role") == "assistant"]

        topics: set[str] = set()
        for msg in user_msgs:
            for kw in ("错误", "API", "密码", "权限", "工单", "SDK", "配置", "部署"):
                if kw in msg:
                    topics.add(kw)

        first_q = user_msgs[0][:80] if user_msgs else ""
        last_a = assistant_msgs[-1][:120] if assistant_msgs else ""

        summary_text = (
            f"[会话摘要] 共 {len(turns)} 轮对话。\n"
            f"首个问题: {first_q}\n"
            f"最后回答: {last_a}\n"
            f"涉及主题: {', '.join(sorted(topics)) if topics else '通用'}"
        )

        return SessionSummary(
            session_id=session_id,
            summary=summary_text,
            key_topics=sorted(topics),
            turn_count=len(turns),
        )

    @staticmethod
    def _build_summary_prompt(turns: list[dict]) -> str:
        """Build a Chinese-language summary prompt (system instruction embedded)."""
        conversation_text = "\n".join(
            f"[{t['role']}] {t['content'][:300]}" for t in turns
        )
        return (
            "你是一个对话摘要助手。请根据以下对话历史生成简洁的会话摘要。\n"
            "要求：\n"
            "1. 用 2-3 句话概括核心内容\n"
            "2. 提取关键主题词（逗号分隔）\n"
            "3. 如果有未解决的问题，请注明\n\n"
            f"对话历史（共 {len(turns)} 轮）：\n{conversation_text}\n\n"
            "请输出：\n摘要：\n主题："
        )

    @staticmethod
    def _extract_topics_from_summary(
        summary_text: str, turns: list[dict]
    ) -> list[str]:
        """Parse topic keywords from LLM-generated summary, with keyword-scan fallback."""
        topics: list[str] = []

        # Try to find "主题：" line in the LLM output
        for line in summary_text.split("\n"):
            line = line.strip()
            if line.startswith("主题") or line.startswith("关键词"):
                parts = line.split("：", 1) if "：" in line else line.split(":", 1)
                if len(parts) == 2:
                    topics = [t.strip() for t in parts[1].replace("、", ",").split(",") if t.strip()]
                    break

        # Fallback: keyword scan from original turns
        if not topics:
            user_msgs = [t["content"] for t in turns if t.get("role") == "user"]
            seen: set[str] = set()
            for msg in user_msgs:
                for kw in ("错误", "API", "密码", "权限", "工单", "SDK", "配置", "部署"):
                    if kw in msg and kw not in seen:
                        topics.append(kw)
                        seen.add(kw)

        return topics

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------
    def _save_pg(self, session_id: str, summary: str) -> None:
        repo = self.repo
        if repo is None:
            return
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(repo.upsert_session(
                    session_id=session_id, summary=summary
                ))
        except Exception:
            pass

    def _get_pg_summary(self, session_id: str) -> str | None:
        repo = self.repo
        if repo is None:
            return None
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                return loop.run_until_complete(repo.get_session_summary(session_id))
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Redis cache
    # ------------------------------------------------------------------
    def _cache_redis(self, session_id: str, entry: SessionSummary) -> None:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if not loop.is_running() and self.redis:
                loop.run_until_complete(
                    self.redis.set(
                        self._cache_key(session_id),
                        json.dumps({"summary": entry.summary, "turn_count": entry.turn_count},
                                   ensure_ascii=False),
                        ex=REDIS_CACHE_TTL,
                    )
                )
        except Exception:
            pass

    def _get_redis_cache(self, session_id: str) -> str | None:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if not loop.is_running() and self.redis:
                val = loop.run_until_complete(self.redis.get(self._cache_key(session_id)))
                if val:
                    return json.loads(val).get("summary", "")
        except Exception:
            pass
        return None

    @staticmethod
    def _cache_key(session_id: str) -> str:
        return f"summary:{session_id}"
