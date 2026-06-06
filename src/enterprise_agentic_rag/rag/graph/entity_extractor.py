"""Entity extractor — rule-based extraction of named entities from chunk content.

Detects: API, CLASS, FUNCTION, ERROR_CODE, COMPONENT, MODULE, CONFIG,
         LIFECYCLE, CONCEPT

No online API calls. Pure regex + keyword matching.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ===========================================================================
# Entity type detection patterns
# ===========================================================================

_PATTERNS: dict[str, list[str]] = {
    "ERROR_CODE": [
        # Numeric error codes: "9568321", "401", "500", "AUTH_12345"
        r"\b\d{5,10}\b",
        r"\bERR[A-Z_]+\b",
        r"\bAUTH_\w+\b",
        r"\b[A-Z]{2,6}_\d{3,8}\b",
        r"\b错误码\s*[:：]?\s*\d+\b",
        r"\bError\s*Code\s*[:：]?\s*\d+\b",
    ],
    "API": [
        r"@ohos\.\w+(?:\.\w+)*",
        r"\b[A-Z]\w*API\b",
        r"\b\w+接口\b",
        r"\bAPI\s+\w+\b",
        r"\bREST\w*\s+API\b",
        r"\b\w+\.\w+\(\)",  # function calls
        r"\bimport\s+\{[^}]+\}\s+from\s+['\"][^'\"]+['\"]",
    ],
    "FUNCTION": [
        r"\b(onCreate|onDestroy|onStart|onStop|onResume|onPause)\b",
        r"\b(onWindowStageCreate|onWindowStageDestroy)\b",
        r"\b(onForeground|onBackground)\b",
        r"\b(onPageShow|onPageHide|onBackPress)\b",
        r"\b\w+\(.*?\)\s*\{",  # function definitions
        r"\b(function|async function)\s+(\w+)",
        r"\b(def\s+\w+\()",
    ],
    "CLASS": [
        r"\bclass\s+(\w+)",
        r"\b(Ability|EntryAbility|UIAbility|ServiceAbility|DataAbility)\b",
        r"\bextends\s+(\w+)",
        r"\bimplements\s+(\w+)",
        r"\b@Component\b",
        r"\b@Entry\b",
        r"\b@Component\s*\n\s*struct\s+(\w+)",
    ],
    "CONFIG": [
        r"\b[A-Z_]{3,30}\s*=\s*",
        r"\b(config|Config|CONFIG)\w*\b",
        r"\b(\.env|\.json|\.yaml|\.xml)\b",
        r"\b配置\w*\s*[:：]",
        r"\bmodule\.json5?\b",
        r"\bapp\.json5?\b",
    ],
    "MODULE": [
        r"\b@ohos\.\w+\b",
        r"\bimport\s+.*\s+from\s+['\"]@ohos\.\w+['\"]",
        r"\b(ohos\.\w+(?:\.\w+)*)\b",
        r"\bnpm\s+(install|i)\s+\S+",
    ],
    "LIFECYCLE": [
        r"\b生命周期\b",
        r"\blifecycle\b",
        r"\bLifeCycle\b",
        r"\b(onCreate|onDestroy|onStart|onStop|onWindowStageCreate)\b",
        r"\b(AbilityLifecycleCallback)\b",
        r"\b(UIAbility onCreate onDestroy)\b",
    ],
    "COMPONENT": [
        r"\b@Component\b",
        r"\b@Entry\b",
        r"\bstruct\s+(\w+)",
        r"\b(Button|Text|Image|List|Column|Row|Flex)\b",
        r"\bbuild\(\)\s*\{",
        r"\b@Builder\b",
    ],
    "CONCEPT": [
        r"\b(ArkTS|ArkUI|Stage模型|FA模型)\b",
        r"\b(页面跳转|路由|导航)\b",
        r"\b(HAP|HAR|HSP)\b",
        r"\b(签名|证书|权限)\b",
        r"\b(白屏|黑屏|闪退|崩溃)\b",
        r"\b(性能优化|内存泄漏|卡顿)\b",
    ],
}

# Entity name extractors per type
_NAME_CAPTURE: dict[str, int] = {
    "CLASS": 1,
    "FUNCTION": 2,
    "COMPONENT": 1,
    "CONCEPT": 0,
}


@dataclass
class Entity:
    """Extracted entity from chunk content."""

    name: str
    type: str  # One of ENTITY_TYPES
    normalized_name: str = ""
    chunk_id: str = ""
    doc_id: str = ""
    context_snippet: str = ""  # Surrounding text for evidence
    confidence: float = 1.0

    def __post_init__(self):
        if not self.normalized_name:
            self.normalized_name = self.name.lower().strip()


def extract_entities_from_chunk(
    content: str,
    chunk_id: str = "",
    doc_id: str = "",
) -> list[Entity]:
    """Extract all entities from a single chunk.

    Includes both text-level entities (regex) and code-level symbols
    (AST or enhanced regex from code blocks).

    Args:
        content: Chunk text content.
        chunk_id: The chunk identifier for provenance.
        doc_id: The parent document identifier.

    Returns:
        List of extracted Entity objects (deduplicated within chunk).
    """
    from enterprise_agentic_rag.config.settings import get_settings

    entities: list[Entity] = []
    seen: set[tuple[str, str]] = set()  # (normalized_name, type)

    # --- Step 1: Text-level regex entity extraction ---
    for entity_type, patterns in _PATTERNS.items():
        for pattern in patterns:
            try:
                matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
                for m in matches:
                    # Extract entity name
                    name = _extract_name(m, entity_type, pattern, content)
                    if not name or len(name) < 2:
                        continue
                    name = name.strip()

                    normalized = name.lower().strip()
                    key = (normalized, entity_type)
                    if key in seen:
                        continue
                    seen.add(key)

                    # Get context snippet (±40 chars around match)
                    start = max(0, m.start() - 40)
                    end = min(len(content), m.end() + 40)
                    snippet = content[start:end].replace("\n", " ").strip()

                    entities.append(Entity(
                        name=name,
                        type=entity_type,
                        normalized_name=normalized,
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        context_snippet=snippet,
                    ))
            except re.error:
                logger.debug("Invalid regex pattern skipped: %s", pattern)
                continue

    # --- Step 2: Code block symbol extraction ---
    settings = get_settings()
    if settings.code_analysis.enable_ast_parsing:
        try:
            from enterprise_agentic_rag.rag.graph.code_symbol_extractor import (
                extract_code_blocks,
                extract_code_symbols_from_chunk,
            )

            # Check if chunk has code blocks
            code_blocks = extract_code_blocks(content)
            if code_blocks:
                # Tag chunk as containing code (used by fusion.py boost)
                code_symbols = extract_code_symbols_from_chunk(
                    content,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                )
                for sym in code_symbols:
                    key = (sym.normalized_name, sym.type)
                    if key in seen:
                        continue
                    seen.add(key)

                    entities.append(Entity(
                        name=sym.name,
                        type=sym.type,
                        normalized_name=sym.normalized_name,
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        context_snippet=sym.source_code[:120],
                        confidence=sym.confidence,
                    ))

                # Also tag CODE_BLOCK entity for retrieval boost
                for cb in code_blocks:
                    cb_key = (cb.language.lower(), "CODE_BLOCK")
                    if cb_key not in seen:
                        seen.add(cb_key)
                        entities.append(Entity(
                            name=cb.language,
                            type="CODE_BLOCK",
                            normalized_name=cb.language.lower(),
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            context_snippet=cb.code_text[:120],
                            confidence=1.0,
                        ))
        except Exception:
            # Code symbol extraction is non-critical — don't crash on failure
            logger.debug("Code symbol extraction failed for chunk %s", chunk_id)

    return entities


def _extract_name(
    match: re.Match,
    entity_type: str,
    pattern: str,
    content: str,
) -> str:
    """Extract the entity name from a regex match.

    Uses pre-configured capture groups or falls back to the full match.
    """
    # Try capture group 1 first
    if match.lastindex and match.lastindex >= 1:
        name = match.group(1)
        if name and len(name) >= 2:
            # Clean up: remove trailing punctuation from class/function names
            name = re.sub(r'[;:,{}\[\]()\'"]+$', '', name).strip()
            return name

    # For patterns without capture groups, use the full match
    full = match.group(0).strip()
    # Clean prefixes
    full = re.sub(r'^(import\s+|from\s+|class\s+|function\s+|struct\s+)', '', full).strip()
    # Clean suffixes
    full = re.sub(r'[;:,{}\[\]()\'"]+$', '', full).strip()
    return full


def extract_entities_from_chunks(
    chunks: list[dict],
) -> list[Entity]:
    """Extract entities from a list of chunk dicts.

    Args:
        chunks: List of chunk dicts with ``chunk_id``, ``content``, ``doc_id``.

    Returns:
        Flat list of all extracted Entity objects.
    """
    all_entities: list[Entity] = []
    for ch in chunks:
        content = ch.get("content", "")
        chunk_id = ch.get("chunk_id", "")
        doc_id = ch.get("doc_id", ch.get("source", ""))
        entities = extract_entities_from_chunk(content, chunk_id=chunk_id, doc_id=doc_id)
        all_entities.extend(entities)
    return all_entities
