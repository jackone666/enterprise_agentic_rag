"""Entity extractor for HarmonyOS developer queries.

Extracts structured entities from natural language queries:
- APIs (@ohos.* modules)
- ArkUI components
- Ability types
- Error patterns
- Version numbers
- File names
- Migration relationships
"""

from __future__ import annotations

import re
from typing import Any

from enterprise_agentic_rag.agents.deep_intent.schema import DeepIntentEntities

# ===========================================================================
# Entity pattern definitions
# ===========================================================================

# API modules — @ohos.* patterns
_API_PATTERNS = [
    r"@ohos\.\w+(?:\.\w+)*",           # @ohos.net.http, @ohos.data.preferences
    r"@system\.\w+",                    # @system.capability
    r"@kit\.\w+(?:\.\w+)*",            # @kit.AbilityKit
]

# ArkUI components
_COMPONENT_NAMES = {
    "Navigation", "Router", "Text", "Button", "List", "Grid",
    "Tabs", "Web", "Image", "Column", "Row", "Stack", "Flex",
    "Scroll", "Swiper", "TextInput", "TextArea", "Checkbox",
    "Radio", "Toggle", "Slider", "Progress", "Rating", "Picker",
    "DatePicker", "TimePicker", "TextPicker", "AlphabetIndexer",
    "MenuItem", "ContextMenu", "Popup", "AlertDialog", "ActionSheet",
    "CustomDialog", "LoadingProgress", "Search", "SideBarContainer",
    "Navigation", "NavPathStack", "NavDestination", "NavRouter",
    "Refresh", "ListItem", "ListItemGroup", "GridItem", "WaterFlow",
    "XComponent", "Canvas", "Video", "XComponent",
    "RelativeContainer", "GridRow", "GridCol", "Badge", "Counter",
    "CalendarPicker", "DataPanel", "Gauge", "Marquee", "QRCode",
    "Panel", "RichText", "Span", "SymbolGlyph",
}

# Ability types
_ABILITY_PATTERNS = [
    r"UIAbility",
    r"ExtensionAbility",
    r"ServiceExtensionAbility",
    r"DataShareExtensionAbility",
    r"FormExtensionAbility",
    r"InputMethodExtensionAbility",
    r"WindowExtensionAbility",
    r"StaticSubscriberExtensionAbility",
    r"Want",
    r"AbilityStage",
    r"AbilityConstant",
]

# Error patterns
_ERROR_PATTERNS = [
    r"BusinessError\s*\d*",
    r"permission\s+denied",
    r"Cannot\s+find\s+module",
    r"hvigor\s+ERROR",
    r"compile\s+failed",
    r"runtime\s+error",
    r"crash",
    r"SIGABRT",
    r"SIGSEGV",
    r"ArkTS:ERROR",
    r"ArkCompiler",
    r"TypeError",
    r"ReferenceError",
    r"SyntaxError",
    r"RangeError",
    r"URIError",
    r"白屏",
    r"黑屏",
    r"闪退",
    r"卡顿",
    r"OOM",
    r"内存泄漏",
    r"死锁",
    r"ANR",
    r"启动失败",
    r"安装失败",
    r"签名错误",
]

# Version patterns
_VERSION_PATTERNS = [
    r"API\s*(?:Level\s*)?(\d{1,2})",
    r"HarmonyOS\s*(?:NEXT\s*)?(\d+\.\d+(?:\.\d+)?)",
    r"OpenHarmony\s*(\d+\.\d+(?:\.\d+)?)",
    r"HarmonyOS\s*NEXT",
    r"SDK\s*(\d+)",
    r"NEXT\s*(?:版本|version)?",
]

# File patterns
_FILE_PATTERNS = [
    "module.json5",
    "Index.ets",
    "EntryAbility.ets",
    "build-profile.json5",
    "hvigorfile.ts",
    "hvigorw.bat",
    "oh-package.json5",
    "AppScope",
    "src/main/ets",
    "src/main/resources",
    "app.json5",
    "hap",
    "hsp",
]

# Migration relationships
_MIGRATION_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    ("Router", "Navigation", re.compile(r"Router.*迁移.*Navigation|Router.*替换.*Navigation|Router.*升级.*Navigation|Router.*改为.*Navigation")),
    ("FA模型", "Stage模型", re.compile(r"FA.*迁移.*Stage|FA.*替换.*Stage|FA.*升级.*Stage|FA.*改为.*Stage")),
    ("JS", "ArkTS", re.compile(r"JS.*迁移.*ArkTS|JS.*替换.*ArkTS|JavaScript.*迁移|JS.*转为.*ArkTS")),
    ("Java", "ArkTS", re.compile(r"Java.*迁移.*ArkTS|Java.*替换.*ArkTS|Java.*升级.*ArkTS")),
    ("旧API", "新API", re.compile(r"废弃.*API|替换.*API|旧.*API.*新|deprecated.*API")),
    ("DataAbility", "DataShare", re.compile(r"DataAbility.*迁移|DataAbility.*替换|DataAbility.*升级|DataAbility.*废弃")),
    ("XComponent", "XComponent", re.compile(r"XComponent.*迁移|XComponent.*升级")),
]


# ===========================================================================
# Main extract function
# ===========================================================================


def extract_entities(query: str) -> dict[str, Any]:
    """Extract all entities from a user query.

    Args:
        query: Raw user query string.

    Returns:
        Dict that can be passed to ``DeepIntentEntities.from_dict()``.
    """
    query_lower = query.lower()

    return {
        "apis": _extract_apis(query, query_lower),
        "components": _extract_components(query, query_lower),
        "errors": _extract_errors(query, query_lower),
        "api_levels": _extract_api_levels(query, query_lower),
        "versions": _extract_versions(query, query_lower),
        "files": _extract_files(query, query_lower),
        "migration_from": _extract_migration(query, query_lower).get("from"),
        "migration_to": _extract_migration(query, query_lower).get("to"),
    }


def extract_entities_typed(query: str) -> DeepIntentEntities:
    """Extract entities and return typed DeepIntentEntities."""
    d = extract_entities(query)
    return DeepIntentEntities.from_dict(d)


# ===========================================================================
# Individual extractors
# ===========================================================================


def _extract_apis(query: str, query_lower: str) -> list[str]:
    apis: list[str] = []
    for pattern in _API_PATTERNS:
        matches = re.findall(pattern, query, re.IGNORECASE)
        apis.extend(matches)

    # Also detect API mentions without @ prefix
    extra_apis = [
        "ohos.net.http", "ohos.router", "ohos.data.preferences",
        "ohos.file.fs", "ohos.notificationManager", "ohos.abilityAccessCtrl",
        "ohos.app.ability.UIAbility", "ohos.multimedia.media",
        "ohos.security.huks", "ohos.account.osAccount",
        "ohos.bundle.bundleManager", "ohos.distributedHardware.deviceManager",
        "ohos.telephony.radio", "ohos.wifiManager",
    ]
    for api in extra_apis:
        # Check if any part of the api path appears in the query
        parts = api.split(".")
        for part in parts:
            if len(part) > 3 and part.lower() in query_lower and api not in apis:
                # Check that a significant portion matches
                if sum(1 for p in parts if p.lower() in query_lower) >= 2:
                    apis.append(api)
                    break

    # Deduplicate
    seen = set()
    result = []
    for a in apis:
        normalized = a.lower().lstrip("@")
        if normalized not in seen:
            seen.add(normalized)
            result.append(a)
    return result


def _extract_components(query: str, query_lower: str) -> list[str]:
    components: list[str] = []
    for comp in _COMPONENT_NAMES:
        # Match as whole word to avoid partial matches
        if re.search(r'\b' + re.escape(comp) + r'\b', query):
            components.append(comp)
        elif comp.lower() in query_lower:
            # For CJK context, also check without word boundaries
            components.append(comp)
    # Deduplicate preserving order
    seen = set()
    result = []
    for c in components:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def _extract_errors(query: str, query_lower: str) -> list[str]:
    errors: list[str] = []
    for pattern in _ERROR_PATTERNS:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for m in matches:
            if isinstance(m, tuple):
                m = "".join(str(x) for x in m if x)
            if m and m.strip():
                errors.append(m.strip())
    # Deduplicate
    seen = set()
    result = []
    for e in errors:
        if e.lower() not in seen:
            seen.add(e.lower())
            result.append(e)
    return result


def _extract_api_levels(query: str, query_lower: str) -> list[str]:
    levels: list[str] = []
    # Match "API 9", "API Level 12", "API12"
    for m in re.finditer(r'API\s*(?:Level\s*)?(\d{1,2})', query, re.IGNORECASE):
        levels.append(f"API {m.group(1)}")
    # Match API level in text like "api9", "api level 11"
    for m in re.finditer(r'\bapi\s*(?:level\s*)?(\d{1,2})\b', query_lower):
        level_str = f"API {m.group(1)}"
        if level_str not in levels:
            levels.append(level_str)
    return levels


def _extract_versions(query: str, query_lower: str) -> list[str]:
    versions: list[str] = []
    if "harmonyos next" in query_lower:
        versions.append("HarmonyOS NEXT")
    for m in re.finditer(r'HarmonyOS\s+(\d+\.\d+(?:\.\d+)?)', query, re.IGNORECASE):
        versions.append(f"HarmonyOS {m.group(1)}")
    for m in re.finditer(r'OpenHarmony\s+(\d+\.\d+(?:\.\d+)?)', query, re.IGNORECASE):
        versions.append(f"OpenHarmony {m.group(1)}")
    for m in re.finditer(r'SDK\s*(\d+)', query, re.IGNORECASE):
        versions.append(f"SDK {m.group(1)}")
    return versions


def _extract_files(query: str, query_lower: str) -> list[str]:
    files: list[str] = []
    for pattern in _FILE_PATTERNS:
        if pattern.lower() in query_lower:
            files.append(pattern)
    return files


def _extract_migration(query: str, query_lower: str) -> dict[str, str | None]:
    """Detect migration relationships in query.

    Returns dict with 'from' and 'to' keys, or None values if no migration detected.
    """
    for from_tech, to_tech, pattern in _MIGRATION_PATTERNS:
        if pattern.search(query):
            return {"from": from_tech, "to": to_tech}

    # Generic migration detection: extract "X 迁移到 Y" or "从 X 迁移到 Y"
    generic = re.search(
        r'(?:从|把|将)?\s*(\w+(?:\s*\w+)?)\s*(?:迁移|升级|替换|改成|转(?:换)?为|转向)\s*(?:到|为|成)?\s*(\w+(?:\s*\w+)?)',
        query,
    )
    if generic:
        from_tech = generic.group(1).strip()
        to_tech = generic.group(2).strip()
        return {"from": from_tech, "to": to_tech}

    return {"from": None, "to": None}
