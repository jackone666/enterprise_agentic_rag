"""Relation extractor — extract relationships between entities from chunk content.

Rule-based relation detection. No online API calls.

Relations detected:
- RELATED_TO (default)
- DEPENDS_ON
- CALLS
- BELONGS_TO
- CAUSES
- FIXES
- PART_OF
- HAS_LIFECYCLE
- AFFECTS
- IMPORTS (code-level)
- EXTENDS (code-level)
- IMPLEMENTS (code-level)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from enterprise_agentic_rag.rag.graph.entity_extractor import Entity

logger = logging.getLogger(__name__)


# ===========================================================================
# Relation detection rules
# ===========================================================================

# Each rule: (relation_type, [list of regex patterns])
_RELATION_RULES: list[tuple[str, list[str]]] = [
    ("CAUSES", [
        r"(\w+)\s*(导致|造成|引起|引发|触发)\s*(\w+)",
        r"(\w+)\s*(因为|由于|原因是)\s*(\w+)",
        r"(\w+)\s*(causes?|leads?\s+to|results?\s+in|triggers?)\s*(\w+)",
        r"(白屏|黑屏|闪退|崩溃|报错|异常)\s*(因为|由于|原因是)\s*(\w+)",
        r"(\w+)\s*(错误|异常|失败|超时|崩溃|卡死)\s*(因为|由于)\s*(\w+)",
        r"(\w+)\s*(导致|造成)\s*(白屏|黑屏|闪退|崩溃|报错|卡顿)",
    ]),
    ("FIXES", [
        r"(\w+)\s*(修复|解决|修正|处理)\s*(\w+)",
        r"(\w+)\s*(fixes?|resolves?|solves?)\s*(\w+)",
        r"(通过|使用|调用)\s*(\w+)\s*(修复|解决)\s*(\w+)",
        r"(\w+)\s*修复\s*(错误码|问题|Bug|bug)\s*(\w+)",
    ]),
    ("DEPENDS_ON", [
        r"(\w+)\s*(依赖|需要|必须|要求|取决于)\s*(\w+)",
        r"(\w+)\s*(depends?\s+on|requires?|needs?)\s*(\w+)",
        r"(\w+)\s*(import|require)\s+.*\s+from\s+['\"](\w+)['\"]",
        r"(\w+)\s*依赖于\s*(\w+)",
        r"配置\s*(\w+)\s*(依赖|需要)\s*(\w+)",
    ]),
    ("CALLS", [
        r"(\w+)\s*(调用|执行|触发|启动)\s*(\w+)",
        r"(\w+)\s*(calls?|invokes?|executes?)\s*(\w+)",
        r"(\w+)\.(\w+)\(",  # method calls: obj.method()
        r"(\w+)\s*调用\s*(\w+)\s*(方法|函数|接口|API)",
        r"import\s+\{[^}]*(\w+)[^}]*\}\s+from\s+['\"]@ohos\.(\w+)['\"]",
    ]),
    ("BELONGS_TO", [
        r"(\w+)\s*(属于|归属于|是.*的一部分)\s*(\w+)",
        r"(\w+)\s*(belongs?\s+to|is\s+part\s+of)\s*(\w+)",
        r"(\w+)\s*位于\s*(\w+)\s*(模块|包|目录)",
        r"import\s+.*\s+from\s+['\"]@ohos\.(\w+)['\"]",
    ]),
    ("PART_OF", [
        r"(\w+)\s*(包含|包括|由.*组成)\s*(\w+)",
        r"(\w+)\s*(contains?|includes?|consists?\s+of)\s*(\w+)",
        r"(\w+)\s*(模块|组件|功能)\s*(包含|包括)\s*(\w+)",
        r"(\w+)\s*是\s*(\w+)\s*的\s*(一部分|子模块|组件)",
    ]),
    ("HAS_LIFECYCLE", [
        r"(\w+)\s*生命周期\s*(包括|包含|有)\s*(\w+)",
        r"(\w+)\s*(onCreate|onDestroy|onStart|onStop|onWindowStageCreate)\b",
        r"(\w+)\s*(的生命周期|生命周期方法|回调|callback)",
        r"(\w+Ability)\s*(的|拥有|有)\s*(onCreate|onWindowStageCreate|生命周期)",
    ]),
    ("AFFECTS", [
        r"(\w+)\s*(影响|波及|涉及)\s*(\w+)",
        r"(\w+)\s*(affects?|impacts?|influences?)\s*(\w+)",
        r"(\w+)\s*(修改|更新|变更)\s*后?\s*(影响|波及)\s*(\w+)",
        r"(\w+)\s*变更\s*(会影响|会波及)\s*(\w+)",
    ]),
    ("RELATED_TO", [
        r"(\w+)\s*(和|与|跟)\s*(\w+)\s*(相关|有关|关联|有关系|相关吗)",
        r"(\w+)\s*(和|与|跟)\s*(\w+)\s*(有什么关系|的关联|的调用链)",
        r"(\w+)\s*(is\s+)?related\s+to\s+(\w+)",
        r"(\w+)\s+→\s+(\w+)",
        r"(\w+)\s*>\s*(\w+)",
        r"(\w+)\s*->\s*(\w+)",
    ]),
    # Code-level relations (extracted from code blocks)
    ("IMPORTS", [
        r"import\s+\{[^}]*(\w+)[^}]*\}\s+from\s+['\"]@ohos\.(\w+)['\"]",
        r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]",
        r"require\(['\"]([^'\"]+)['\"]\)",
        r"from\s+(\S+)\s+import\s+(\w+)",
    ]),
    ("EXTENDS", [
        r"class\s+(\w+)\s+extends\s+(\w+)",
        r"(\w+)\s+extends\s+(\w+)",
    ]),
    ("IMPLEMENTS", [
        r"class\s+(\w+)\s+implements\s+([\w\s,]+)",
        r"(\w+)\s+implements\s+([\w\s,]+)",
    ]),
]


@dataclass
class Relation:
    """Extracted relation between two entities."""

    source_entity: Entity
    target_entity: Entity
    relation_type: str  # One of RELATION_TYPES
    weight: float = 1.0
    evidence_chunk_id: str = ""
    context_snippet: str = ""


def extract_relations(
    entities: list[Entity],
    chunk_content: str = "",
    chunk_id: str = "",
) -> list[Relation]:
    """Extract relations between entities from chunk content.

    Two strategies:
    1. Pattern-based: regex patterns that explicitly capture entity pairs.
    2. Co-occurrence: entities in the same chunk that match relation patterns.

    Args:
        entities: List of Entity objects extracted from this chunk.
        chunk_content: The chunk text content.
        chunk_id: The chunk identifier for evidence tracking.

    Returns:
        List of Relation objects.
    """
    relations: list[Relation] = []
    entity_by_name: dict[str, Entity] = {}
    for e in entities:
        entity_by_name[e.normalized_name] = e
        entity_by_name[e.name.lower()] = e
        entity_by_name[e.name] = e

    # Strategy 1: Pattern-based extraction
    for rel_type, patterns in _RELATION_RULES:
        for pattern in patterns:
            try:
                for m in re.finditer(pattern, chunk_content, re.IGNORECASE | re.MULTILINE):
                    rel = _try_extract_relation(m, rel_type, entities, entity_by_name, chunk_id, chunk_content)
                    if rel:
                        relations.append(rel)
            except re.error:
                continue

    # Strategy 2: Co-occurrence fallback
    # If we have 2+ entities in this chunk but few explicit relations,
    # add RELATED_TO for entity pairs that appear close together
    if len(entities) >= 2 and len(relations) <= len(entities):
        cooccur = _extract_cooccurrence_relations(entities, chunk_content, chunk_id)
        relations.extend(cooccur)

    # Deduplicate by (source, target, relation_type)
    seen: set[tuple[str, str, str]] = set()
    deduped: list[Relation] = []
    for r in relations:
        key = (r.source_entity.normalized_name, r.target_entity.normalized_name, r.relation_type)
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return deduped


def _try_extract_relation(
    match: re.Match,
    rel_type: str,
    entities: list[Entity],
    entity_by_name: dict[str, Entity],
    chunk_id: str,
    content: str,
) -> Relation | None:
    """Try to extract a relation from a regex match.

    Searches capture groups for known entity names.
    """
    # Collect all captured group values
    candidates: list[str] = []
    for i in range(1, match.lastindex + 1 if match.lastindex else 2):
        g = match.group(i)
        if g and len(g) >= 2:
            candidates.append(g.strip())

    # Find source and target entities among candidates
    source: Entity | None = None
    target: Entity | None = None

    for cand in candidates:
        ent = entity_by_name.get(cand.lower()) or entity_by_name.get(cand)
        if ent:
            if source is None:
                source = ent
            elif target is None and ent.normalized_name != source.normalized_name:
                target = ent
                break

    if source is None or target is None:
        return None

    # Get context snippet
    start = max(0, match.start() - 30)
    end = min(len(content), match.end() + 30)
    snippet = content[start:end].replace("\n", " ").strip()

    return Relation(
        source_entity=source,
        target_entity=target,
        relation_type=rel_type,
        weight=1.0,
        evidence_chunk_id=chunk_id,
        context_snippet=snippet,
    )


def _extract_cooccurrence_relations(
    entities: list[Entity],
    content: str,
    chunk_id: str,
    max_distance: int = 300,
) -> list[Relation]:
    """Create RELATED_TO relations for entities co-occurring within max_distance chars.

    Fallback strategy when explicit patterns don't capture all relations.
    """
    if len(entities) < 2:
        return []

    # Sort entities by position in content
    positioned: list[tuple[int, Entity]] = []
    for e in entities:
        pos = content.lower().find(e.name.lower())
        if pos >= 0:
            positioned.append((pos, e))

    positioned.sort(key=lambda x: x[0])

    relations: list[Relation] = []
    for i in range(len(positioned)):
        for j in range(i + 1, len(positioned)):
            pos_i, ent_i = positioned[i]
            pos_j, ent_j = positioned[j]
            distance = pos_j - pos_i

            if distance > max_distance:
                break

            # Only link different entity types to avoid noise
            if ent_i.type == ent_j.type:
                continue

            weight = max(0.1, 1.0 - distance / max_distance)

            relations.append(Relation(
                source_entity=ent_i,
                target_entity=ent_j,
                relation_type="RELATED_TO",
                weight=round(weight, 2),
                evidence_chunk_id=chunk_id,
                context_snippet="",
            ))

    return relations


def extract_relations_from_chunks(
    chunks: list[dict],
    entities_by_chunk: dict[str, list[Entity]] | None = None,
) -> list[Relation]:
    """Extract all relations across all chunks.

    Args:
        chunks: List of chunk dicts.
        entities_by_chunk: Pre-extracted entities keyed by chunk_id.
                           If None, entities are extracted on-the-fly.

    Returns:
        Flat list of all Relation objects.
    """
    from enterprise_agentic_rag.rag.graph.entity_extractor import extract_entities_from_chunk

    all_relations: list[Relation] = []

    for ch in chunks:
        content = ch.get("content", "")
        chunk_id = ch.get("chunk_id", "")

        if entities_by_chunk and chunk_id in entities_by_chunk:
            entities = entities_by_chunk[chunk_id]
        else:
            entities = extract_entities_from_chunk(content, chunk_id=chunk_id)

        if not entities:
            continue

        relations = extract_relations(entities, chunk_content=content, chunk_id=chunk_id)
        all_relations.extend(relations)

    return all_relations
