"""Rule-based deep intent recognition for HarmonyOS developer queries.

Provides fast, deterministic intent signals as input to the LLM classifier.
Not the final result — always combined with LLM deep intent classification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ===========================================================================
# Rule result container
# ===========================================================================


@dataclass
class RuleIntentResult:
    """Result of rule-based intent analysis.

    This is intermediate output — never the final DeepIntentResult.
    """

    candidate_intents: list[str] = field(default_factory=list)
    """Candidate primary intents detected by rules."""

    signals: dict[str, list[str]] = field(default_factory=dict)
    """Signals that triggered each intent, e.g. {'error_diagnosis': ['报错', 'error']}."""

    scenario_hints: list[str] = field(default_factory=list)
    """Scenario hints detected, e.g. ['white_screen', 'permission_error']."""

    suggested_tools: list[str] = field(default_factory=list)
    """Tools suggested by rule matching."""

    suggested_mode: str = ""
    """Suggested retrieval mode based on rules."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_intents": self.candidate_intents,
            "signals": self.signals,
            "scenario_hints": self.scenario_hints,
            "suggested_tools": self.suggested_tools,
            "suggested_mode": self.suggested_mode,
        }


# ===========================================================================
# Intent detection rules (Section 4 of spec)
# ===========================================================================

# Rule 1: Error diagnosis + Project debug
_ERROR_DIAGNOSIS_KEYWORDS = re.compile(
    r"报错|错误|异常|失败|error|failed|exception|denied|"
    r"hvigor|crash|白屏|黑屏|闪退|卡顿|OOM|ANR|"
    r"不工作|不行|无法|超时|401|403|404|500|502|503|"
    r"崩溃|宕机|挂掉|起不来|打不开|没反应|"
    r"SIGABRT|SIGSEGV|TypeError|ReferenceError|SyntaxError|"
    r"RangeError|URIError|BusinessError|ArkTS:ERROR|"
    r"compile.*failed|runtime.*error|install.*fail|"
    r"启动.*闪退|启动.*黑屏|启动.*白屏|启动.*卡住",
    re.IGNORECASE,
)

_PROJECT_DEBUG_KEYWORDS = re.compile(
    r"项目|页面|启动|首页|运行|工程|模块|编译|构建|build|"
    r"打包|签名|发布|部署|安装|真机|模拟器|设备|预览|"
    r"DevEco|IDE|调试|debug|运行不了|跑不起来|"
    r"打不开页面|进不去|点了没反应",
    re.IGNORECASE,
)

# Rule 2: Code generation
_CODE_GENERATION_KEYWORDS = re.compile(
    r"写一个|生成|实现|封装|demo|示例代码|代码|"
    r"怎么写|如何写|怎么实现|怎么封装|怎么生成|"
    r"帮我写|帮我实现|帮我生成|给我.*代码|"
    r"一段代码|代码片段|snippet|example|"
    r"完整代码|参考代码|代码示例|sample|"
    r"模板|template|脚手架|scaffold",
    re.IGNORECASE,
)

# Rule 3: Migration
_MIGRATION_KEYWORDS = re.compile(
    r"迁移|升级|替换|改成|废弃|deprecated|"
    r"从.*到|换成|替代|取代|改为|转为|"
    r"不再维护|不再支持|下架|移除|淘汰|"
    r"迁移到|升级到|替换为|替换成|"
    r"迁移方案|升级方案|迁移步骤|"
    r"新版本|旧.*API|新.*API",
    re.IGNORECASE,
)

# Rule 4: API usage
_API_USAGE_KEYWORDS = re.compile(
    r"怎么用|如何使用|参数|返回值|接口|组件|API|"
    r"使用方法|调用方式|怎么调|如何调|怎么发|如何发|"
    r"传入什么|返回什么|参数说明|接口说明|"
    r"有哪些参数|参数列表|返回值类型|"
    r"怎么调用|如何调用|调用示例|怎么请求|如何请求|"
    r"调用方法|用法|使用方式|使用方法|"
    r"配置项|配置参数|配置方法|"
    r"GET\s*请求|POST\s*请求|HTTP\s*请求|"
    r"@ohos\.\w+",
    re.IGNORECASE,
)

# Note: _API_USAGE_KEYWORDS matches @ohos.* patterns directly.

# Rule 5: Concept QA
_CONCEPT_QA_KEYWORDS = re.compile(
    r"是什么|区别|原理|机制|生命周期|"
    r"什么是|什么意思|含义|概念|"
    r"有什么不同|和.*区别|和.*有什么不同|"
    r"作用|功能|用途|介绍|说明|概述|"
    r"架构|设计.*理念|设计.*思想|"
    r"为什么.*要|为什么要|"
    r"关系|联系|关联|"
    r"优点|缺点|优势|劣势|适用场景",
    re.IGNORECASE,
)

# ===========================================================================
# Scenario detection
# ===========================================================================

_SCENARIO_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("white_screen", re.compile(r"白屏|white\s*screen|页面.*白|空白.*页", re.IGNORECASE)),
    ("black_screen", re.compile(r"黑屏|black\s*screen|页面.*黑", re.IGNORECASE)),
    ("crash", re.compile(r"闪退|crash|崩溃|SIGABRT|SIGSEGV|ANR", re.IGNORECASE)),
    ("permission_error", re.compile(r"permission\s*denied|权限.*拒绝|权限.*错误|无权限|没权限|权限.*失败", re.IGNORECASE)),
    ("build_error", re.compile(r"编译.*错误|编译.*失败|build.*fail|hvigor.*ERROR|compile.*error", re.IGNORECASE)),
    ("install_error", re.compile(r"安装.*失败|install.*fail|签名.*错误|打包.*失败", re.IGNORECASE)),
    ("network_error", re.compile(r"网络.*错误|网络.*异常|http.*error|请求.*失败|timeout|超时|连接.*失败", re.IGNORECASE)),
    ("router_to_navigation", re.compile(r"Router.*Navigation|router.*navigation|路由.*迁移|路由.*升级", re.IGNORECASE)),
    ("fa_to_stage", re.compile(r"FA.*Stage|fa.*stage|FA模型.*Stage|元服务.*迁移", re.IGNORECASE)),
    ("js_to_arkts", re.compile(r"JS.*ArkTS|js.*arkts|JavaScript.*ArkTS|JS.*迁移", re.IGNORECASE)),
    ("api_deprecation", re.compile(r"deprecated|废弃|不再.*维护|API.*变更|API.*变化|接口.*变更", re.IGNORECASE)),
    ("lifecycle", re.compile(r"生命周期|lifecycle|onCreate|onDestroy|onForeground|onBackground", re.IGNORECASE)),
    ("data_persistence", re.compile(r"数据.*存储|持久化|preferences|database|KV|关系型|数据库", re.IGNORECASE)),
    ("ui_layout", re.compile(r"布局|layout|排列|对齐|居中|flex|Column|Row|Stack", re.IGNORECASE)),
    ("navigation", re.compile(r"导航|跳转|Navigation|Router|页面.*跳转|路由", re.IGNORECASE)),
    ("notification", re.compile(r"通知|Notification|notification|推送|push", re.IGNORECASE)),
    ("distributed", re.compile(r"分布式|distributed|跨设备|多设备|协同|continuation", re.IGNORECASE)),
]


# ===========================================================================
# Main rule function
# ===========================================================================


def rule_based_intent(query: str) -> RuleIntentResult:
    """Apply rule-based intent detection.

    This is the first stage of deep intent recognition.
    Results are fed into the LLM classifier for refinement.

    Args:
        query: Raw user query string.

    Returns:
        RuleIntentResult with candidate intents, signals, scenario hints,
        suggested tools, and suggested retrieval mode.
    """
    result = RuleIntentResult()
    signals: dict[str, list[str]] = {}

    # ── Rule 1: Error diagnosis / Project debug ──
    error_matches = _ERROR_DIAGNOSIS_KEYWORDS.findall(query)
    project_matches = _PROJECT_DEBUG_KEYWORDS.findall(query)

    if error_matches:
        signals["error_diagnosis"] = list(set(error_matches))
        result.candidate_intents.append("error_diagnosis")

    if project_matches:
        signals["project_debug"] = list(set(project_matches))
        if "error_diagnosis" in signals:
            # Both error + project context → project_debug as primary
            result.candidate_intents.append("project_debug")
            if "error_diagnosis" in result.candidate_intents:
                result.candidate_intents.remove("error_diagnosis")
                result.candidate_intents.append("error_diagnosis")
        else:
            result.candidate_intents.append("project_debug")

    # ── Rule 2: Code generation ──
    code_matches = _CODE_GENERATION_KEYWORDS.findall(query)
    if code_matches:
        signals["code_generation"] = list(set(code_matches))
        result.candidate_intents.append("code_generation")

    # If code_generation is detected along with project_debug,
    # code_generation should take priority (user wants to write code)
    if "code_generation" in signals and "project_debug" in result.candidate_intents:
        # Move code_generation to front
        result.candidate_intents = ["code_generation"] + [
            i for i in result.candidate_intents if i != "code_generation"
        ]

    # ── Rule 3: Migration ──
    migration_matches = _MIGRATION_KEYWORDS.findall(query)
    if migration_matches:
        signals["migration"] = list(set(migration_matches))
        result.candidate_intents.append("migration")

    # ── Rule 4: API usage ──
    api_matches = _API_USAGE_KEYWORDS.findall(query)
    if api_matches:
        signals["api_usage"] = list(set(api_matches))
        result.candidate_intents.append("api_usage")

    # ── Rule 5: Concept QA ──
    concept_matches = _CONCEPT_QA_KEYWORDS.findall(query)
    if concept_matches:
        signals["concept_qa"] = list(set(concept_matches))
        result.candidate_intents.append("concept_qa")

    # ── Default fallback ──
    if not result.candidate_intents:
        result.candidate_intents.append("concept_qa")
        signals["concept_qa"] = ["default_fallback"]

    result.signals = signals

    # ── Scenario detection ──
    for scenario_name, pattern in _SCENARIO_PATTERNS:
        if pattern.search(query):
            result.scenario_hints.append(scenario_name)

    # ── Suggested tools ──
    result.suggested_tools = _suggest_tools(result)

    # ── Suggested mode ──
    result.suggested_mode = _suggest_mode(result)

    return result


# ===========================================================================
# Tool suggestion
# ===========================================================================


def _suggest_tools(result: RuleIntentResult) -> list[str]:
    """Suggest tools based on candidate intents and scenarios."""
    tools: list[str] = []
    primary = result.candidate_intents[0] if result.candidate_intents else ""
    secondary = result.candidate_intents[1:] if len(result.candidate_intents) > 1 else []

    # Always include basic search tools
    tools.extend(["keyword_search", "vector_search"])

    if primary == "error_diagnosis":
        tools.append("error_diagnosis_search")
        tools.append("official_doc_search")
        if "permission_error" in result.scenario_hints:
            tools.append("version_compatibility_check")
        if any(s in result.scenario_hints for s in ("build_error", "install_error")):
            tools.append("ticket_search")

    elif primary == "code_generation":
        tools.append("sample_code_search")
        tools.append("api_reference_search")
        tools.append("official_doc_search")
        tools.append("code_review")

    elif primary == "migration":
        tools.append("graph_search")
        tools.append("official_doc_search")
        tools.append("api_reference_search")

    elif primary == "api_usage":
        tools.append("api_reference_search")
        tools.append("sample_code_search")
        tools.append("official_doc_search")

    elif primary == "concept_qa":
        tools.append("official_doc_search")
        tools.append("hybrid_rag_search")

    # Deduplicate preserving order
    seen = set()
    unique = []
    for t in tools:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _suggest_mode(result: RuleIntentResult) -> str:
    """Suggest retrieval mode based on candidate intents."""
    primary = result.candidate_intents[0] if result.candidate_intents else ""

    if primary == "migration":
        return "graph_first"
    elif primary == "code_generation":
        return "parallel"
    elif primary == "api_usage":
        return "parallel"
    else:
        return "hybrid_only"
