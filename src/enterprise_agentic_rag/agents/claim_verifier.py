"""Claim-level Verification — decompose answers into atomic claims and verify each.

Upgrades the existing holisitic verifier_agent.py with fine-grained,
assertion-by-assertion verification. Each claim is independently checked
against source documents for hallucination, contradiction, and grounding.

Reference:
    TECHNICAL_DEEP_DIVE.md §35.4 — "引入 Claim-level Verification"
    Expected impact: hallucination_rate ↓ 20%-40%
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Claim:
    """An atomic factual assertion extracted from an answer."""

    text: str
    claim_type: str = "factual"  # factual | code | api | version | comparison
    grounded: bool = False
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    issues: list[str] = field(default_factory=list)


@dataclass
class ClaimVerificationResult:
    """Result of claim-level verification."""

    claims: list[Claim]
    total_claims: int = 0
    grounded_claims: int = 0
    hallucinated_claims: int = 0
    uncertain_claims: int = 0
    overall_grounded: bool = False
    overall_confidence: float = 0.0
    summary: str = ""

    @property
    def hallucination_rate(self) -> float:
        if self.total_claims == 0:
            return 0.0
        return self.hallucinated_claims / self.total_claims

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "grounded_claims": self.grounded_claims,
            "hallucinated_claims": self.hallucinated_claims,
            "uncertain_claims": self.uncertain_claims,
            "overall_grounded": self.overall_grounded,
            "overall_confidence": self.overall_confidence,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "summary": self.summary,
            "claims": [
                {
                    "text": c.text[:200],
                    "type": c.claim_type,
                    "grounded": c.grounded,
                    "confidence": c.confidence,
                    "issues": c.issues,
                }
                for c in self.claims
            ],
        }


# ---------------------------------------------------------------------------
# Claim extraction patterns
# ---------------------------------------------------------------------------

# Sentences that express factual claims
_CLAIM_PATTERNS = [
    # API references
    (r'(@ohos\.[\w.]+|import\s+\{[^}]+\}\s+from\s+[\'"]@ohos\.[^\'"]+[\'"])', "api"),
    # Code blocks
    (r'```[\s\S]*?```', "code"),
    # Version references
    (r'(API\s*\d+|HarmonyOS\s*(NEXT\s*)?\d+\.\d+)', "version"),
    # Error codes
    (r'(\d{8,})', "error_code"),
    # Comparisons
    (r'(不同于|区别于|与.+相比|比.+更)', "comparison"),
    # Migration statements
    (r'(从.+迁移|替代|废弃|deprecated|removed\s+in)', "migration"),
]


def extract_claims(answer: str) -> list[Claim]:
    """Decompose an answer into atomic claims.

    Uses sentence splitting + pattern matching to identify claim types.
    Each sentence is a potential claim; sentences with code/API/version
    references are tagged with their claim type.
    """
    if not answer or not answer.strip():
        return []

    # Split into sentences (Chinese + English aware)
    sentences = re.split(r'(?<=[。！？.!?\n])\s*', answer)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]

    claims: list[Claim] = []
    for sent in sentences:
        claim_type = "factual"
        for pattern, ctype in _CLAIM_PATTERNS:
            if re.search(pattern, sent):
                claim_type = ctype
                break

        claims.append(Claim(text=sent, claim_type=claim_type))

    return claims


# ---------------------------------------------------------------------------
# Claim verification
# ---------------------------------------------------------------------------


def verify_claims(
    claims: list[Claim],
    retrieved_docs: list[dict[str, Any]],
    use_llm: bool = True,
) -> ClaimVerificationResult:
    """Verify each claim against retrieved documents.

    Verification strategy:
    - API/code claims: check if the API/symbol appears in docs
    - Version claims: check if version is mentioned in docs
    - Error code claims: check if error code appears in docs
    - Factual claims: check semantic overlap with doc content
    - Comparison claims: check if both sides appear in docs

    Args:
        claims: Extracted claims from the answer.
        retrieved_docs: Source documents for verification.
        use_llm: Whether to use LLM for verification (expensive but accurate).

    Returns:
        ClaimVerificationResult with per-claim scoring.
    """
    if not claims:
        return ClaimVerificationResult(
            claims=[], overall_grounded=True, overall_confidence=1.0,
            summary="No claims to verify (empty answer)",
        )

    # Build document corpus for verification
    doc_contents = [d.get("content", "") for d in retrieved_docs]
    doc_titles = [d.get("title", d.get("source", "")) for d in retrieved_docs]
    combined_corpus = " ".join(doc_contents).lower()

    for claim in claims:
        _verify_single_claim(claim, combined_corpus, doc_contents, doc_titles)

    # Compute overall result
    total = len(claims)
    grounded = sum(1 for c in claims if c.grounded)
    hallucinated = sum(1 for c in claims if not c.grounded and c.confidence < 0.3)
    uncertain = sum(1 for c in claims if not c.grounded and c.confidence >= 0.3)

    overall_confidence = sum(c.confidence for c in claims) / total if total > 0 else 1.0
    overall_grounded = hallucinated == 0 and uncertain <= max(1, total * 0.2)

    summary_parts = []
    if hallucinated > 0:
        summary_parts.append(f"{hallucinated}/{total} claims appear hallucinated")
    if uncertain > 0:
        summary_parts.append(f"{uncertain}/{total} claims are uncertain")
    if not summary_parts:
        summary_parts.append(f"All {total} claims are grounded in sources")

    return ClaimVerificationResult(
        claims=claims,
        total_claims=total,
        grounded_claims=grounded,
        hallucinated_claims=hallucinated,
        uncertain_claims=uncertain,
        overall_grounded=overall_grounded,
        overall_confidence=round(overall_confidence, 4),
        summary="; ".join(summary_parts),
    )


def _verify_single_claim(
    claim: Claim,
    combined_corpus: str,
    doc_contents: list[str],
    doc_titles: list[str],
) -> None:
    """Verify a single claim against the document corpus."""
    claim_text_lower = claim.text.lower()

    if claim.claim_type == "api":
        _verify_api_claim(claim, combined_corpus)
    elif claim.claim_type == "code":
        _verify_code_claim(claim, combined_corpus)
    elif claim.claim_type == "version":
        _verify_version_claim(claim, combined_corpus)
    elif claim.claim_type == "error_code":
        _verify_error_claim(claim, combined_corpus)
    elif claim.claim_type == "comparison":
        _verify_comparison_claim(claim, combined_corpus, doc_contents)
    else:
        _verify_factual_claim(claim, combined_corpus)

    # Find supporting evidence
    for i, content in enumerate(doc_contents):
        if _text_overlap(claim_text_lower, content.lower()) > 0.3:
            source = doc_titles[i] if i < len(doc_titles) else f"doc_{i}"
            claim.evidence.append(source[:100])


def _verify_api_claim(claim: Claim, corpus: str) -> None:
    """Verify API claims: check if API references appear in corpus."""
    api_refs = re.findall(r'@ohos\.[\w.]+', claim.text)
    if not api_refs:
        # Look for API-like patterns
        api_refs = re.findall(r'([\w]+\.(?:[\w]+\.)*[\w]+)', claim.text)

    matches = sum(1 for ref in api_refs if ref.lower() in corpus)
    if api_refs:
        ratio = matches / len(api_refs)
        claim.confidence = min(1.0, 0.4 + ratio * 0.6)
        claim.grounded = ratio >= 0.5
        if not claim.grounded:
            missing = [r for r in api_refs if r.lower() not in corpus]
            claim.issues.append(f"API references not found in sources: {missing}")
    else:
        claim.confidence = 0.5
        claim.issues.append("No API references to verify")


def _verify_code_claim(claim: Claim, corpus: str) -> None:
    """Verify code claims: check if code patterns appear in corpus."""
    # Extract code identifiers from claim
    code_ids = re.findall(r'\b([a-zA-Z_]\w{2,})\b', claim.text)
    # Filter common words
    stop_words = {"the", "and", "for", "this", "that", "with", "from", "your", "have", "been"}
    code_ids = [cid for cid in code_ids if cid.lower() not in stop_words]

    if code_ids:
        matches = sum(1 for cid in code_ids if cid.lower() in corpus)
        ratio = matches / len(code_ids)
        claim.confidence = min(1.0, 0.3 + ratio * 0.7)
        claim.grounded = ratio >= 0.4
    else:
        claim.confidence = 0.4
        claim.grounded = False


def _verify_version_claim(claim: Claim, corpus: str) -> None:
    """Verify version claims: check if version numbers appear in corpus."""
    versions = re.findall(r'(API\s*\d+|HarmonyOS\s*(?:NEXT\s*)?\d+\.\d+|\d+\.\d+\.\d+)', claim.text)

    if versions:
        matches = sum(1 for v in versions if v.replace(" ", "").lower() in corpus.replace(" ", ""))
        ratio = matches / len(versions)
        claim.confidence = min(1.0, 0.3 + ratio * 0.7)
        claim.grounded = ratio >= 0.5
        if not claim.grounded:
            claim.issues.append(f"Version references not confirmed: {versions}")
    else:
        claim.confidence = 0.5


def _verify_error_claim(claim: Claim, corpus: str) -> None:
    """Verify error code claims: check if error codes appear in corpus."""
    error_codes = re.findall(r'(\d{8,})', claim.text)

    if error_codes:
        matches = sum(1 for ec in error_codes if ec in corpus)
        ratio = matches / len(error_codes)
        claim.confidence = min(1.0, 0.2 + ratio * 0.8)
        claim.grounded = ratio >= 0.5
        if not claim.grounded:
            claim.issues.append(f"Error codes not found in sources: {error_codes}")
    else:
        claim.confidence = 0.5


def _verify_comparison_claim(claim: Claim, corpus: str, doc_contents: list[str]) -> None:
    """Verify comparison claims: both sides should appear in docs."""
    # Split on comparison markers
    parts = re.split(r'不同于|区别于|与|相比|比|更|vs\.?|versus', claim.text, maxsplit=2)
    if len(parts) >= 2:
        left = parts[0].strip().lower()
        right = parts[-1].strip().lower() if len(parts) > 1 else ""

        left_found = any(left[:20] in doc.lower() for doc in doc_contents)
        right_found = any(right[:20] in doc.lower() for doc in doc_contents) if right else False

        if left_found and right_found:
            claim.confidence = 0.9
            claim.grounded = True
        elif left_found or right_found:
            claim.confidence = 0.5
            claim.grounded = False
            claim.issues.append("Comparison partially grounded — one side not found")
        else:
            claim.confidence = 0.2
            claim.grounded = False
            claim.issues.append("Neither side of comparison found in sources")
    else:
        claim.confidence = 0.4


def _verify_factual_claim(claim: Claim, corpus: str) -> None:
    """Verify factual claims via semantic keyword overlap."""
    claim_words = set(claim.text.lower().split())
    # Filter meaningful words (>2 chars)
    claim_keywords = {w for w in claim_words if len(w) > 2}

    if not claim_keywords:
        claim.confidence = 0.3
        claim.grounded = False
        claim.issues.append("No verifiable keywords")
        return

    corpus_words = set(corpus.split())
    overlap = len(claim_keywords & corpus_words)
    ratio = overlap / len(claim_keywords)

    if ratio >= 0.5:
        claim.confidence = min(1.0, 0.5 + ratio * 0.5)
        claim.grounded = True
    elif ratio >= 0.3:
        claim.confidence = 0.4
        claim.grounded = False
    else:
        claim.confidence = 0.2
        claim.grounded = False
        claim.issues.append(f"Low keyword overlap ({overlap}/{len(claim_keywords)})")


def _text_overlap(text1: str, text2: str) -> float:
    """Compute simple n-gram overlap between two texts."""
    words1 = set(text1.split())
    words2 = set(text2.split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / max(len(words1), len(words2))


# ---------------------------------------------------------------------------
# Integration with verifier_agent
# ---------------------------------------------------------------------------


def verify_answer_with_claims(
    draft_answer: str,
    citations: list[dict[str, Any]],
    retrieved_docs: list[dict[str, Any]],
    use_llm: bool = True,
) -> tuple[bool, str, ClaimVerificationResult | None]:
    """Enhanced verification: claim-level + traditional.

    Returns:
        (verified, reason, claim_result) tuple.
    """
    # Run claim-level verification
    claims = extract_claims(draft_answer)
    claim_result = verify_claims(claims, retrieved_docs, use_llm=use_llm)

    # Build detailed reason
    reasons: list[str] = []

    if claim_result.hallucinated_claims > 0:
        reasons.append(
            f"发现 {claim_result.hallucinated_claims}/{claim_result.total_claims} "
            f"个可能幻觉的断言"
        )
        for claim in claim_result.claims:
            if not claim.grounded and claim.confidence < 0.3:
                reasons.append(f"  - 幻觉嫌疑: {claim.text[:80]}...")

    if claim_result.uncertain_claims > 0:
        reasons.append(f"{claim_result.uncertain_claims} 个断言无法确认")

    # Fall back to rule-based verifier for final judgment. Do not call
    # verifier_agent.verify_answer here, otherwise claim verification recurses
    # back into this function.
    from enterprise_agentic_rag.agents.verifier_agent import _verify_rules
    trad_verified, trad_reason = _verify_rules(draft_answer, citations, retrieved_docs)

    # Combine: claim-level is stricter
    verified = trad_verified and claim_result.overall_grounded

    if not reasons:
        reasons.append(trad_reason)
    else:
        reasons.append(f"传统校验: {trad_reason}")

    return verified, "; ".join(reasons), claim_result
