"""Conflict Evidence Detector — detect and annotate contradictory evidence across sources.

When multiple documents contain conflicting information about the same topic,
this module detects the conflicts and annotates them so the answer generator
can either:
1. Surface the conflict to the user (transparency)
2. Choose the most authoritative source (source authority ranking)
3. Defer to human (high-stakes conflicts)

Reference:
    TECHNICAL_DEEP_DIVE.md §33.3 — "证据冲突处理还不够强"
    Expected impact: verification_fail_rate ↓ 10%-20%
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConflictGroup:
    """A group of documents that conflict on a specific topic."""

    topic: str
    documents: list[dict[str, Any]] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    severity: str = "low"  # low | medium | high | critical
    resolution: str = ""  # Suggested resolution


@dataclass
class ConflictReport:
    """Complete conflict analysis for a set of retrieved documents."""

    conflicts: list[ConflictGroup] = field(default_factory=list)
    total_docs: int = 0
    consistent_docs: int = 0
    conflicting_docs: int = 0
    has_conflicts: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_docs": self.total_docs,
            "consistent_docs": self.consistent_docs,
            "conflicting_docs": self.conflicting_docs,
            "has_conflicts": self.has_conflicts,
            "conflicts": [
                {
                    "topic": c.topic,
                    "doc_sources": [d.get("source", "unknown") for d in c.documents],
                    "claims": c.claims,
                    "severity": c.severity,
                    "resolution": c.resolution,
                }
                for c in self.conflicts
            ],
        }


# ---------------------------------------------------------------------------
# Topic extraction for conflict detection
# ---------------------------------------------------------------------------


def detect_conflicts(
    documents: list[dict[str, Any]],
    min_overlap: float = 0.3,
) -> ConflictReport:
    """Detect contradictory information across retrieved documents.

    Strategy:
    1. Group documents by topic (using title/source/content overlap)
    2. Within each topic group, compare factual claims
    3. Flag contradictions based on:
       - API version disagreements (e.g., "API 12" vs "API 11")
       - Opposite statements (negation patterns)
       - Mutually exclusive recommendations
       - Stale vs current version references

    Args:
        documents: Retrieved documents to analyze.
        min_overlap: Minimum topic overlap to consider documents as same-topic.

    Returns:
        ConflictReport with detected conflicts and resolution suggestions.
    """
    if len(documents) < 2:
        return ConflictReport(
            total_docs=len(documents),
            consistent_docs=len(documents),
        )

    report = ConflictReport(total_docs=len(documents))

    # ── Topic extraction ──
    doc_topics = []
    for doc in documents:
        topics = _extract_topics(doc)
        doc_topics.append((doc, topics))

    # ── Find document pairs on same topic ──
    seen_pairs: set[tuple[int, int]] = set()

    for i in range(len(doc_topics)):
        for j in range(i + 1, len(doc_topics)):
            doc_a, topics_a = doc_topics[i]
            doc_b, topics_b = doc_topics[j]

            common_topics = topics_a & topics_b
            if not common_topics:
                continue

            pair_key = (i, j)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Check for actual contradiction
            conflicts = _check_contradictions(doc_a, doc_b, common_topics)
            if conflicts:
                report.conflicts.extend(conflicts)

    # ── Compute statistics ──
    conflicting_indices: set[int] = set()
    for conflict in report.conflicts:
        for doc in conflict.documents:
            for idx, d in enumerate(documents):
                if d.get("source") == doc.get("source"):
                    conflicting_indices.add(idx)

    report.conflicting_docs = len(conflicting_indices)
    report.consistent_docs = report.total_docs - report.conflicting_docs
    report.has_conflicts = len(report.conflicts) > 0

    logger.info(
        "Conflict detection: %d docs, %d conflicts found (%d docs involved)",
        report.total_docs, len(report.conflicts), report.conflicting_docs,
    )

    return report


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------


def _extract_topics(doc: dict[str, Any]) -> set[str]:
    """Extract topics from a document.

    Topics are derived from:
    - API references (@ohos.*)
    - Version references (API 12, HarmonyOS N.N)
    - Error codes (8+ digit numbers)
    - Component names (capitalized identifiers)
    - Headers/titles
    """
    content = doc.get("content", "")
    title = doc.get("title", doc.get("source", ""))
    text = f"{title} {content[:1000]}"

    topics: set[str] = set()

    # API references
    api_refs = re.findall(r'@ohos\.[\w.]+', text)
    for ref in api_refs:
        topics.add(f"api:{ref}")

    # Version references
    versions = re.findall(r'(API\s*\d{1,2}|HarmonyOS\s*(?:NEXT\s*)?\d+\.\d+)', text)
    for v in versions:
        topics.add(f"version:{v.strip()}")

    # Error codes
    error_codes = re.findall(r'\b(\d{8,})\b', text)
    for ec in error_codes[:5]:
        topics.add(f"error:{ec}")

    # Component/module names (PascalCase identifiers)
    components = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', text)
    for comp in components[:5]:
        topics.add(f"component:{comp}")

    # Key phrases (3+ word noun phrases from title)
    title_words = [w for w in title.split() if len(w) > 2]
    for i in range(len(title_words) - 2):
        phrase = " ".join(title_words[i:i + 3])
        if len(phrase) > 10:
            topics.add(f"topic:{phrase}")

    return topics


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------


def _check_contradictions(
    doc_a: dict[str, Any],
    doc_b: dict[str, Any],
    common_topics: set[str],
) -> list[ConflictGroup]:
    """Check for contradictions between two documents on the same topics."""
    content_a = doc_a.get("content", "").lower()
    content_b = doc_b.get("content", "").lower()
    source_a = doc_a.get("source", "unknown")
    source_b = doc_b.get("source", "unknown")

    conflicts: list[ConflictGroup] = []

    # ── Check 1: Version disagreements ──
    version_conflict = _check_version_conflict(content_a, content_b, common_topics)
    if version_conflict:
        version_conflict.documents = [doc_a, doc_b]
        conflicts.append(version_conflict)

    # ── Check 2: Negation / opposite statements ──
    negation_conflict = _check_negation_conflict(content_a, content_b, source_a, source_b)
    if negation_conflict:
        negation_conflict.documents = [doc_a, doc_b]
        conflicts.append(negation_conflict)

    # ── Check 3: API availability disagreement ──
    api_conflict = _check_api_availability_conflict(content_a, content_b)
    if api_conflict:
        api_conflict.documents = [doc_a, doc_b]
        conflicts.append(api_conflict)

    # ── Check 4: Recommendation conflict ──
    rec_conflict = _check_recommendation_conflict(content_a, content_b)
    if rec_conflict:
        rec_conflict.documents = [doc_a, doc_b]
        conflicts.append(rec_conflict)

    return conflicts


def _check_version_conflict(
    content_a: str, content_b: str, topics: set[str],
) -> ConflictGroup | None:
    """Check if two docs reference different API versions for the same API."""
    versions_a = set(re.findall(r'API\s*(\d{1,2})', content_a))
    versions_b = set(re.findall(r'API\s*(\d{1,2})', content_b))

    # Extract the API being discussed
    api_topic = None
    for topic in topics:
        if topic.startswith("api:"):
            api_topic = topic[4:]
            break

    if versions_a and versions_b and versions_a != versions_b:
        return ConflictGroup(
            topic=api_topic or "API compatibility",
            claims=[
                f"Document references API version(s): {versions_a}",
                f"Another document references API version(s): {versions_b}",
            ],
            severity="high",
            resolution=(
                "Prefer the newer API version unless the user explicitly asks about "
                "an older version. Note the version difference in the answer."
            ),
        )

    return None


def _check_negation_conflict(
    content_a: str, content_b: str, source_a: str, source_b: str,
) -> ConflictGroup | None:
    """Check if two docs make opposite claims on the same topic."""
    # Look for negation patterns in one but not the other
    negation_patterns = [
        r'(not\s+(?:supported|available|recommended|possible|allowed))',
        r'(不支持|不可用|不推荐|不允许|已废弃|已移除)',
    ]

    for pattern in negation_patterns:
        matches_a = set(re.findall(pattern, content_a, re.IGNORECASE))
        matches_b = set(re.findall(pattern, content_b, re.IGNORECASE))

        if matches_a and not matches_b:
            # Doc A says "not supported" but Doc B doesn't mention the limitation
            # Check if Doc B implies support
            positive_indicators = re.findall(
                r'(支持|可用|推荐|supported|available|recommended)',
                content_b, re.IGNORECASE,
            )
            if positive_indicators:
                return ConflictGroup(
                    topic="API support status",
                    claims=[
                        f"{source_a}: indicates NOT supported ({list(matches_a)[0]})",
                        f"{source_b}: implies supported ({positive_indicators[0]})",
                    ],
                    severity="high",
                    resolution=(
                        "Flag this conflict to the user. Prefer the more recent or "
                        "official documentation source."
                    ),
                )

    return None


def _check_api_availability_conflict(
    content_a: str, content_b: str,
) -> ConflictGroup | None:
    """Check if one doc uses an API that another says is deprecated."""
    deprecated_in_a = set(re.findall(
        r'(deprecated|废弃|removed\s+in|已移除)\s*(?:in\s*)?(?:API\s*)?(\d+)',
        content_a, re.IGNORECASE,
    ))

    api_usage_in_b = set(re.findall(r'@ohos\.[\w.]+', content_b))

    if deprecated_in_a and api_usage_in_b:
        return ConflictGroup(
            topic="API deprecation",
            claims=[
                f"Document indicates deprecation/removal: {deprecated_in_a}",
                f"Another document uses potentially deprecated APIs: {list(api_usage_in_b)[:3]}",
            ],
            severity="critical",
            resolution=(
                "The API may be deprecated in newer versions. Recommend the replacement "
                "API if available, or note the version constraint."
            ),
        )

    return None


def _check_recommendation_conflict(
    content_a: str, content_b: str,
) -> ConflictGroup | None:
    """Check if two docs recommend different approaches for the same task."""
    rec_patterns = [
        r'(recommend|建议|推荐|最佳实践|best\s+practice)\s+(?:is\s+)?(?:to\s+)?(\w[\w\s]{5,50})',
    ]

    recs_a: set[str] = set()
    recs_b: set[str] = set()

    for pattern in rec_patterns:
        for match in re.finditer(pattern, content_a, re.IGNORECASE):
            recs_a.add(match.group(2).strip().lower())
        for match in re.finditer(pattern, content_b, re.IGNORECASE):
            recs_b.add(match.group(2).strip().lower())

    if recs_a and recs_b and not (recs_a & recs_b):
        return ConflictGroup(
            topic="Best practice recommendation",
            claims=[
                f"Recommendation: {list(recs_a)[:2]}",
                f"Alternative recommendation: {list(recs_b)[:2]}",
            ],
            severity="medium",
            resolution=(
                "Present both approaches and let the user choose based on their "
                "specific requirements. Note any version constraints."
            ),
        )

    return None


# ---------------------------------------------------------------------------
# Conflict annotation for prompt building
# ---------------------------------------------------------------------------


def annotate_docs_with_conflicts(
    documents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], ConflictReport]:
    """Annotate documents with conflict information for prompt building.

    Adds `conflict_info` field to documents that are involved in conflicts.

    Returns:
        (annotated_docs, conflict_report) tuple.
    """
    report = detect_conflicts(documents)

    if not report.has_conflicts:
        return documents, report

    # Build source → conflict mapping
    source_conflicts: dict[str, list[ConflictGroup]] = {}
    for conflict in report.conflicts:
        for doc in conflict.documents:
            source = doc.get("source", "")
            if source not in source_conflicts:
                source_conflicts[source] = []
            source_conflicts[source].append(conflict)

    # Annotate documents
    annotated = []
    for doc in documents:
        doc_copy = dict(doc)
        source = doc.get("source", "")
        if source in source_conflicts:
            doc_copy["conflict_info"] = {
                "has_conflict": True,
                "conflicts": [
                    {
                        "topic": c.topic,
                        "severity": c.severity,
                        "resolution": c.resolution,
                    }
                    for c in source_conflicts[source]
                ],
            }
        annotated.append(doc_copy)

    return annotated, report


def build_conflict_warning(conflict_report: ConflictReport) -> str:
    """Build a user-facing conflict warning string for inclusion in prompts."""
    if not conflict_report.has_conflicts:
        return ""

    lines = ["\n⚠️ **证据冲突警告**：检索到的文档中存在以下信息冲突：\n"]
    for i, conflict in enumerate(conflict_report.conflicts, 1):
        lines.append(f"{i}. **{conflict.topic}** (严重程度: {conflict.severity})")
        for claim in conflict.claims:
            lines.append(f"   - {claim}")
        if conflict.resolution:
            lines.append(f"   → 建议处理: {conflict.resolution}")
        lines.append("")

    return "\n".join(lines)
