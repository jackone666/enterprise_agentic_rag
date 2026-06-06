"""Comprehensive tests for the tool system."""

import pytest

from enterprise_agentic_rag.tools.base import BaseTool, ToolResult
from enterprise_agentic_rag.tools.executor import ToolExecutor
from enterprise_agentic_rag.tools.policies import PolicyDecision, ToolTier, evaluate_policy
from enterprise_agentic_rag.tools.registry import ToolRegistry
from enterprise_agentic_rag.tools.system_status_tool import GetErrorCodeDetailTool, GetSystemStatusTool
from enterprise_agentic_rag.tools.ticket_tool import CreateTicketTool, QueryTicketTool
from enterprise_agentic_rag.tools.user_profile_tool import GetUserProfileTool


# ============================================================================
# Failing tool (for testing error handling)
# ============================================================================
class _FailingTool(BaseTool):
    name: str = "failing_tool"
    description: str = "Always raises."
    is_sensitive: bool = False
    required_permissions: list[str] = ["read"]
    max_retries: int = 2

    async def execute(self, **kwargs):  # type: ignore[override]
        raise RuntimeError("simulated tool crash")


class _SensitiveReadTool(BaseTool):
    name: str = "sensitive_read"
    description: str = "A sensitive tool."
    is_sensitive: bool = True
    required_permissions: list[str] = ["read", "admin"]

    async def execute(self, **kwargs):  # type: ignore[override]
        return ToolResult(success=True, output={"data": "secret"})


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register_many([
        QueryTicketTool(),
        CreateTicketTool(),
        GetUserProfileTool(),
        GetSystemStatusTool(),
        GetErrorCodeDetailTool(),
        _FailingTool(),
        _SensitiveReadTool(),
    ])
    return r


@pytest.fixture
def executor(registry: ToolRegistry) -> ToolExecutor:
    return ToolExecutor(registry)


# ============================================================================
# Registry tests
# ============================================================================
class TestToolRegistry:
    def test_register_and_get(self, registry: ToolRegistry) -> None:
        tool = registry.get("query_ticket")
        assert tool.name == "query_ticket"
        assert tool.is_sensitive is False

    def test_register_duplicate_raises(self, registry: ToolRegistry) -> None:
        with pytest.raises(ValueError, match="already registered"):
            registry.register(QueryTicketTool())

    def test_get_missing_raises(self, registry: ToolRegistry) -> None:
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent_tool")

    def test_list_tools_returns_descriptions(self, registry: ToolRegistry) -> None:
        descs = registry.list_tools()
        assert len(descs) >= 5
        names = {d["name"] for d in descs}
        assert "query_ticket" in names

    def test_tool_names_property(self, registry: ToolRegistry) -> None:
        assert "get_system_status" in registry.tool_names

    def test_count(self, registry: ToolRegistry) -> None:
        assert registry.count >= 5


# ============================================================================
# Policy tests
# ============================================================================
class TestToolPolicy:
    def test_allowed_when_permissions_match(self) -> None:
        result = evaluate_policy(
            tool_name="query_ticket",
            tier=ToolTier.SAFE,
            required_permissions=["read"],
            user_permissions=["read", "write"],
        )
        assert result.decision == PolicyDecision.ALLOWED

    def test_denied_when_missing_permission(self) -> None:
        result = evaluate_policy(
            tool_name="create_ticket",
            tier=ToolTier.SAFE,
            required_permissions=["write"],
            user_permissions=["read"],
        )
        assert result.decision == PolicyDecision.DENIED
        assert "write" in result.reason

    def test_pending_when_sensitive_and_no_skip(self) -> None:
        result = evaluate_policy(
            tool_name="create_ticket",
            tier=ToolTier.SENSITIVE,
            required_permissions=["write"],
            user_permissions=["write"],
            skip_confirmation=False,
        )
        assert result.decision == PolicyDecision.PENDING

    def test_allowed_when_sensitive_but_skip(self) -> None:
        result = evaluate_policy(
            tool_name="create_ticket",
            tier=ToolTier.SENSITIVE,
            required_permissions=["write"],
            user_permissions=["write"],
            skip_confirmation=True,
        )
        assert result.decision == PolicyDecision.ALLOWED

    def test_denied_no_permissions(self) -> None:
        result = evaluate_policy(
            tool_name="admin_tool",
            tier=ToolTier.SAFE,
            required_permissions=["admin"],
            user_permissions=[],
        )
        assert result.decision == PolicyDecision.DENIED

    def test_destructive_denied_by_default(self) -> None:
        result = evaluate_policy(
            tool_name="delete_all",
            tier=ToolTier.DESTRUCTIVE,
            required_permissions=["admin"],
            user_permissions=["admin"],
        )
        assert result.decision == PolicyDecision.DENIED
        assert "ENABLE_DESTRUCTIVE_TOOLS" in result.reason


# ============================================================================
# Executor tests
# ============================================================================
class TestToolExecutor:
    @pytest.mark.asyncio
    async def test_execute_safe_tool_succeeds(self, executor: ToolExecutor) -> None:
        result = await executor.execute(
            tool_name="query_ticket",
            params={"ticket_id": "TKT-001"},
            user_permissions=["read"],
        )
        assert result.success is True
        assert result.tool_name == "query_ticket"
        assert result.latency_ms >= 0
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_execute_returns_error_for_missing_tool(self, executor: ToolExecutor) -> None:
        result = await executor.execute(
            tool_name="not_a_tool",
            params={},
            user_permissions=["read"],
        )
        assert result.success is False
        assert "not found" in result.error.lower() or "未注册" in result.error

    @pytest.mark.asyncio
    async def test_execute_denied_insufficient_permissions(self, executor: ToolExecutor) -> None:
        result = await executor.execute(
            tool_name="create_ticket",
            params={"user_id": "u001", "issue": "test"},
            user_permissions=["read"],  # missing 'write'
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_sensitive_pending_confirmation(self, executor: ToolExecutor) -> None:
        result = await executor.execute(
            tool_name="create_ticket",
            params={"user_id": "u001", "issue": "test"},
            user_permissions=["write"],
            skip_confirmation=False,
        )
        assert result.success is False
        assert "需要确认" in result.error or "确认" in result.error

    @pytest.mark.asyncio
    async def test_execute_sensitive_with_skip(self, executor: ToolExecutor) -> None:
        result = await executor.execute(
            tool_name="create_ticket",
            params={"user_id": "u001", "issue": "test"},
            user_permissions=["write"],
            skip_confirmation=True,
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_failing_tool_returns_error(self, executor: ToolExecutor) -> None:
        result = await executor.execute(
            tool_name="failing_tool",
            params={},
            user_permissions=["read"],
        )
        assert result.success is False
        assert "simulated tool crash" in result.error

    @pytest.mark.asyncio
    async def test_execute_missing_params_returns_error(self, executor: ToolExecutor) -> None:
        result = await executor.execute(
            tool_name="query_ticket",
            params={},
            user_permissions=["read"],
        )
        assert result.success is False
        assert "ticket_id" in result.error.lower()


# ============================================================================
# Individual tool tests
# ============================================================================
class TestQueryTicketTool:
    @pytest.mark.asyncio
    async def test_existing_ticket(self) -> None:
        tool = QueryTicketTool()
        result = await tool.execute(ticket_id="TKT-001")
        assert result.success is True
        assert result.output["found"] is True
        assert result.output["ticket"]["status"] == "open"

    @pytest.mark.asyncio
    async def test_nonexistent_ticket(self) -> None:
        tool = QueryTicketTool()
        result = await tool.execute(ticket_id="TKT-999")
        assert result.success is True
        assert result.output["found"] is False

    @pytest.mark.asyncio
    async def test_missing_ticket_id(self) -> None:
        tool = QueryTicketTool()
        result = await tool.execute()
        assert result.success is False


class TestCreateTicketTool:
    @pytest.mark.asyncio
    async def test_create_ticket_succeeds(self) -> None:
        tool = CreateTicketTool()
        result = await tool.execute(user_id="u001", issue="测试问题")
        assert result.success is True
        assert result.output["status"] == "open"
        assert "TKT-" in result.output["ticket_id"]

    @pytest.mark.asyncio
    async def test_missing_params(self) -> None:
        tool = CreateTicketTool()
        result = await tool.execute(user_id="u001")
        assert result.success is False


class TestGetUserProfileTool:
    @pytest.mark.asyncio
    async def test_existing_user(self) -> None:
        tool = GetUserProfileTool()
        result = await tool.execute(user_id="u001")
        assert result.success is True
        assert result.output["found"] is True
        assert result.output["profile"]["name"] == "张三"

    @pytest.mark.asyncio
    async def test_nonexistent_user(self) -> None:
        tool = GetUserProfileTool()
        result = await tool.execute(user_id="u999")
        assert result.success is True
        assert result.output["found"] is False


class TestSystemStatusTool:
    @pytest.mark.asyncio
    async def test_returns_all_services(self) -> None:
        tool = GetSystemStatusTool()
        result = await tool.execute()
        assert result.success is True
        assert "api_gateway" in result.output["services"]


class TestErrorCodeDetailTool:
    @pytest.mark.asyncio
    async def test_known_error_code(self) -> None:
        tool = GetErrorCodeDetailTool()
        result = await tool.execute(error_code="AUTH_401")
        assert result.success is True
        assert result.output["found"] is True
        assert "API Key" in result.output["error_detail"]["description"]

    @pytest.mark.asyncio
    async def test_unknown_error_code(self) -> None:
        tool = GetErrorCodeDetailTool()
        result = await tool.execute(error_code="ZZZ_999")
        assert result.success is True
        assert result.output["found"] is False

    @pytest.mark.asyncio
    async def test_missing_error_code(self) -> None:
        tool = GetErrorCodeDetailTool()
        result = await tool.execute()
        assert result.success is False


# ============================================================================
# Integration: tool_agent selection logic
# ============================================================================
class TestToolAgentSelection:
    @pytest.mark.asyncio
    async def test_troubleshooting_selects_system_status(self) -> None:
        from enterprise_agentic_rag.agents.tool_agent import _select_tools

        calls = _select_tools("我的 API 调用一直报 AUTH_401 错误", "troubleshooting")
        tool_names = {c[0] for c in calls}
        assert "get_system_status" in tool_names

    @pytest.mark.asyncio
    async def test_ticket_query_selects_ticket_lookup(self) -> None:
        from enterprise_agentic_rag.agents.tool_agent import _select_tools

        calls = _select_tools("查询工单 TKT-001 的处理进度", "ticket_query")
        tool_names = {c[0] for c in calls}
        assert "query_ticket" in tool_names

    @pytest.mark.asyncio
    async def test_call_tools_handles_permission_denial(self) -> None:
        from enterprise_agentic_rag.agents.tool_agent import call_tools

        results, tc, errors, pending = await call_tools(
            query="查询工单 TKT-001 和系统状态",
            intent="ticket_query",
            user_id="u001",
            user_permissions=["read"],  # query_ticket is 'read', OK
        )
        # query_ticket should succeed (needs 'read')
        # get_system_status should succeed (needs 'read')
        assert len(results) >= 1


# ============================================================================
# Workflow integration test
# ============================================================================
class TestWorkflowWithTools:
    @pytest.mark.asyncio
    async def test_troubleshooting_triggers_tools(self) -> None:
        from enterprise_agentic_rag.graph.workflow import build_workflow

        graph = build_workflow()
        result = await graph.ainvoke({
            "query": "我的 SDK 接入报 AUTH_401 错误，怎么办？",
            "user_id": "u001",
            "session_id": "s-tool-01",
        })

        # System status and error code tools should have been called
        tool_results = result.get("tool_results", [])
        tool_names = {tr["tool_name"] for tr in tool_results}
        assert "get_system_status" in tool_names, f"Expected get_system_status in {tool_names}"
        assert "get_error_code_detail" in tool_names, f"Expected get_error_code_detail in {tool_names}"

        # Should still produce a final answer
        assert len(result.get("final_answer", "")) > 0

    @pytest.mark.asyncio
    async def test_tool_errors_dont_crash_workflow(self) -> None:
        from enterprise_agentic_rag.graph.workflow import build_workflow

        graph = build_workflow()
        # Query that triggers ticket lookup but with insufficient permissions for write tools
        result = await graph.ainvoke({
            "query": "查询工单 TKT-001",
            "user_id": "u002",
            "session_id": "s-tool-02",
        })

        # Even if some tools fail, the workflow should complete
        assert "final_answer" in result
        assert result.get("need_human") is not None

    @pytest.mark.asyncio
    async def test_general_question_skips_tools(self) -> None:
        from enterprise_agentic_rag.graph.workflow import build_workflow

        graph = build_workflow()
        result = await graph.ainvoke({
            "query": "你好，今天天气怎么样？",
            "user_id": "u001",
            "session_id": "s-tool-03",
        })

        # General questions should not trigger tools
        tool_results = result.get("tool_results", [])
        assert tool_results == [], f"Expected no tools, got {tool_results}"

    @pytest.mark.asyncio
    async def test_tool_audit_logs_present(self) -> None:
        from enterprise_agentic_rag.graph.workflow import build_workflow

        graph = build_workflow()
        result = await graph.ainvoke({
            "query": "系统状态怎么样？AUTH_401",
            "user_id": "u001",
            "session_id": "s-tool-04",
        })

        audit_logs = result.get("tool_audit_logs", [])
        if result.get("tool_results"):
            assert len(audit_logs) > 0, "Audit logs should be present when tools are called"

    @pytest.mark.asyncio
    async def test_chat_response_includes_tool_results(self) -> None:
        from enterprise_agentic_rag.app.main import ChatRequest, chat

        request = ChatRequest(
            query="系统状态怎么样？AUTH_401 错误",
            user_id="u001",
            session_id="s-tool-05",
        )
        response = await chat(request)

        # Response should have tool_results when tools were triggered
        # (may be empty if tools don't match)
        assert hasattr(response, "tool_results")
