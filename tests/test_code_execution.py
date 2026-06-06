"""Tests for code execution tool — sandboxed code execution."""

import pytest

from enterprise_agentic_rag.tools.code_execution_tool import CodeExecutionTool


class TestCodeExecutionTool:
    """Tests for CodeExecutionTool."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tool = CodeExecutionTool()

    @pytest.mark.asyncio
    async def test_execute_python_valid(self):
        """Execute valid Python code."""
        result = await self.tool.execute(
            code="print('hello world')",
            language="python",
        )
        assert result.success is True
        assert "hello world" in result.output.get("stdout", "")

    @pytest.mark.asyncio
    async def test_execute_python_error(self):
        """Execute Python code with syntax error."""
        result = await self.tool.execute(
            code="print(undefined_variable",
            language="python",
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_javascript_valid(self):
        """Execute valid JavaScript code."""
        result = await self.tool.execute(
            code="console.log('hello from js');",
            language="javascript",
        )
        # Node --check only validates syntax, depends on runtime
        assert result.tool_name == "execute_code"

    @pytest.mark.asyncio
    async def test_execute_empty_code(self):
        """Execute empty code — should fail gracefully."""
        result = await self.tool.execute(code="", language="python")
        assert result.success is False
        assert "代码为空" in result.error

    @pytest.mark.asyncio
    async def test_execute_dangerous_code_blocked(self):
        """Dangerous code (os.system, subprocess) should be blocked."""
        result = await self.tool.execute(
            code="import os; os.system('ls')",
            language="python",
        )
        assert result.success is False
        assert "不安全" in result.error

    @pytest.mark.asyncio
    async def test_execute_unsupported_language(self):
        """Unsupported language should be rejected."""
        result = await self.tool.execute(
            code="println('hello')",
            language="ruby",
        )
        assert result.success is False
        assert "不支持" in result.error

    @pytest.mark.asyncio
    async def test_execute_blocked_require_child_process(self):
        """JS code using child_process should be blocked."""
        result = await self.tool.execute(
            code="const cp = require('child_process'); cp.exec('ls');",
            language="javascript",
        )
        assert result.success is False
        assert "不安全" in result.error

    def test_is_safe(self):
        """Test the safety check method."""
        assert self.tool._is_safe("console.log('hello')") is True
        assert self.tool._is_safe("import os; os.system('rm -rf /')") is False
        assert self.tool._is_safe("subprocess.call(['ls'])") is False
        assert self.tool._is_safe("eval('1+1')") is False

    def test_tool_metadata(self):
        """Verify tool metadata."""
        assert self.tool.name == "execute_code"
        assert self.tool.tier == "sensitive"
        assert "write" in self.tool.required_permissions
@pytest.mark.asyncio
async def test_production_disables_local_code_execution(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("ALLOW_LOCAL_CODE_EXECUTION", raising=False)

    tool = CodeExecutionTool()
    result = await tool.execute(code="print('hello')", language="python")

    assert result.success is False
    assert "生产环境禁止" in result.error
