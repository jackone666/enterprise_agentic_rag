"""Answer verifier — claim-level verification with LLM and rule-based fallback.

Upgraded from holistic verification to fine-grained claim-level verification:
1. Decompose answer into atomic claims
2. Verify each claim against source documents
3. Integrate conflict detection from retrieved documents
4. LLM-first with rule-based fallback

Reference:
    TECHNICAL_DEEP_DIVE.md §35.4 — Claim-level Verification
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def verify_answer(
    draft_answer: str,
    citations: list[dict[str, Any]],
    retrieved_docs: list[dict[str, Any]],
    use_claim_level: bool = True,
) -> tuple[bool, str]:
    """Verify answer — sync compatibility wrapper.

    Production workflow uses ``verify_answer_async`` so LLM verification can run
    inside FastAPI's event loop. This wrapper keeps older scripts/tests working.
    """
    # ── Priority 1: Claim-level verification ──
    if use_claim_level and draft_answer.strip():
        try:
            from enterprise_agentic_rag.agents.claim_verifier import verify_answer_with_claims
            verified, reason, claim_result = verify_answer_with_claims(
                draft_answer, citations, retrieved_docs, use_llm=True,
            )
            if claim_result is not None:
                logger.info(
                    "Claim-level verification: %d/%d grounded, hallucination_rate=%.2f",
                    claim_result.grounded_claims, claim_result.total_claims,
                    claim_result.hallucination_rate,
                )
            return verified, reason
        except Exception as exc:
            logger.debug("Claim-level verification failed, falling back: %s", exc)

    # ── Priority 2: LLM-based verification ──
    from enterprise_agentic_rag.llm.provider_factory import get_llm_provider
    provider = get_llm_provider()

    if provider.provider_name != "mock" and draft_answer.strip():
        result = _verify_with_llm(provider, draft_answer, citations, retrieved_docs)
        if result is not None:
            return result

    # ── Priority 3: Rule-based fallback ──
    return _verify_rules(draft_answer, citations, retrieved_docs)


async def verify_answer_async(
    draft_answer: str,
    citations: list[dict[str, Any]],
    retrieved_docs: list[dict[str, Any]],
    use_claim_level: bool = True,
) -> tuple[bool, str]:
    """Verify answer asynchronously — true LLM-capable production path."""
    if use_claim_level and draft_answer.strip():
        try:
            from enterprise_agentic_rag.agents.claim_verifier import verify_answer_with_claims
            # Current claim verifier is sync and rule-heavy; run it in a worker
            # thread so it cannot block the request event loop.
            import asyncio
            verified, reason, claim_result = await asyncio.to_thread(
                verify_answer_with_claims,
                draft_answer,
                citations,
                retrieved_docs,
                False,
            )
            if claim_result is not None:
                logger.info(
                    "Claim-level verification: %d/%d grounded, hallucination_rate=%.2f",
                    claim_result.grounded_claims, claim_result.total_claims,
                    claim_result.hallucination_rate,
                )
            from enterprise_agentic_rag.config.settings import get_settings
            if not verified and get_settings().runtime.is_production:
                return verified, reason
        except Exception as exc:
            logger.debug("Claim-level verification failed, falling back: %s", exc)

    from enterprise_agentic_rag.llm.provider_factory import get_llm_provider
    provider = get_llm_provider()

    if provider.provider_name != "mock" and draft_answer.strip():
        result = await _verify_with_llm_async(provider, draft_answer, citations, retrieved_docs)
        if result is not None:
            return result

    return _verify_rules(draft_answer, citations, retrieved_docs)


def _verify_rules(draft: str, citations: list, docs: list) -> tuple[bool, str]:
    """Rule-based verification — always available fallback."""
    reasons: list[str] = []

    if not draft or not draft.strip():
        reasons.append("生成的答案为空")
        return False, "; ".join(reasons)

    if not docs:
        reasons.append("未检索到任何相关文档，答案缺乏依据")
        return False, "; ".join(reasons)

    # Check for noise documents
    scores = [d.get("score", 0) for d in docs]
    max_score = max(scores, default=0)
    if len(scores) >= 2:
        avg_score = sum(scores) / len(scores)
        if avg_score > 0 and (max_score / avg_score) < 1.1 and max_score < 0.02:
            reasons.append("检索结果均为低相关度噪音，无法作为答案依据")

    # Citation check
    has_cit = any(m in draft for m in ("参考来源", "引用", "[1]", "citation"))
    if not has_cit and citations:
        reasons.append("答案正文缺少引用标记（已自动补充）")
    if not has_cit and not citations:
        reasons.append("答案缺少引用来源，不可信")

    # Length check
    if len(draft.strip()) < 10:
        reasons.append("答案过短，可能未完整回答问题")

    # Conflict check (from conflict_detector)
    try:
        from enterprise_agentic_rag.context.conflict_detector import detect_conflicts
        conflict_report = detect_conflicts(docs)
        if conflict_report.has_conflicts:
            reasons.append(
                f"发现 {len(conflict_report.conflicts)} 处证据冲突，"
                f"涉及 {conflict_report.conflicting_docs} 个文档"
            )
    except Exception:
        pass  # Conflict detection is best-effort

    verified = len(reasons) <= 1
    return verified, "; ".join(reasons) if reasons else "所有检查通过"


def _verify_with_llm(provider, answer: str, citations: list, docs: list) -> tuple[bool, str] | None:
    """LLM-based verification with conflict awareness."""
    import asyncio
    import json

    # Build conflict context
    conflict_context = ""
    try:
        from enterprise_agentic_rag.context.conflict_detector import detect_conflicts
        conflict_report = detect_conflicts(docs)
        if conflict_report.has_conflicts:
            conflict_context = (
                f"\n\n⚠️ 检测到 {len(conflict_report.conflicts)} 处证据冲突。"
                f"请特别检查答案是否选择了可靠的证据来源。"
            )
    except Exception:
        pass

    prompt = (
        "你是一个企业级答案校验器。请从以下维度判断草稿答案是否可靠：\n"
        "1. 答案是否基于提供的参考文档？\n"
        "2. 是否包含幻觉信息（文档中未提及的事实）？\n"
        "3. 引用标记是否正确？\n"
        "4. 答案是否完整？\n"
        "5. 是否有多文档冲突未处理？\n"
        f"{conflict_context}\n"
        f"草稿答案:\n{answer}\n\n参考文档数: {len(docs)}, 引用数: {len(citations)}\n\n"
        '返回JSON: {"verified": true/false, "reason": "..."}'
    )
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return None
        resp = loop.run_until_complete(provider.generate(prompt, temperature=0.0, max_tokens=512))
        if resp.success:
            data = json.loads(resp.content)
            return data.get("verified", True), data.get("reason", "LLM verification")
    except Exception:
        pass
    return None


async def _verify_with_llm_async(provider, answer: str, citations: list, docs: list) -> tuple[bool, str] | None:
    """Async LLM-based verification with conflict awareness."""
    import json

    conflict_context = ""
    try:
        from enterprise_agentic_rag.context.conflict_detector import detect_conflicts
        conflict_report = detect_conflicts(docs)
        if conflict_report.has_conflicts:
            conflict_context = (
                f"\n\n检测到 {len(conflict_report.conflicts)} 处证据冲突。"
                f"请特别检查答案是否选择了可靠的证据来源。"
            )
    except Exception:
        pass

    prompt = (
        "你是一个企业级答案校验器。请从以下维度判断草稿答案是否可靠：\n"
        "1. 答案是否基于提供的参考文档？\n"
        "2. 是否包含幻觉信息（文档中未提及的事实）？\n"
        "3. 引用标记是否正确？\n"
        "4. 答案是否完整？\n"
        "5. 是否有多文档冲突未处理？\n"
        f"{conflict_context}\n"
        f"草稿答案:\n{answer}\n\n参考文档数: {len(docs)}, 引用数: {len(citations)}\n\n"
        '返回JSON: {"verified": true/false, "reason": "..."}'
    )
    try:
        resp = await provider.generate(prompt, temperature=0.0, max_tokens=512)
        if resp.success:
            data = json.loads(resp.content)
            return bool(data.get("verified", True)), data.get("reason", "LLM verification")
    except Exception:
        pass
    return None
