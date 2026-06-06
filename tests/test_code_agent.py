"""Tests for code agent — code generation, template fallback, and symbol extraction."""


from enterprise_agentic_rag.agents.code_agent import (
    _detect_language,
    _extract_symbols_from_docs,
    _format_symbols_for_prompt,
    generate_code,
)


class TestCodeAgent:
    """Tests for code_agent module."""

    def test_detect_language_typescript(self):
        """Detect TypeScript/ArkTS from query keywords."""
        assert _detect_language("怎么调用@ohos.ability API") == "typescript"
        assert _detect_language("ArkTS 怎么写") == "typescript"
        assert _detect_language("HarmonyOS 开发") == "typescript"

    def test_detect_language_python(self):
        """Detect Python from query keywords."""
        assert _detect_language("python 怎么调用 API") == "python"
        assert _detect_language("用 pytest 写测试") == "python"

    def test_detect_language_default(self):
        """Default to typescript for this project's domain."""
        assert _detect_language("怎么调用 API") == "typescript"

    def test_generate_code_no_docs(self):
        """Fallback when no retrieved docs available."""
        result = generate_code("怎么调用 API", [], "typescript")
        assert result["success"] is False
        assert "未检索到相关文档" in result.get("error", "")

    def test_generate_code_with_docs_has_code_blocks(self):
        """Extract code blocks from retrieved docs."""
        docs = [
            {
                "source": "api_doc.md",
                "content": (
                    "## API Usage\n\n"
                    "```typescript\n"
                    "import { abilityManager } from '@ohos.app.ability';\n"
                    "const result = abilityManager.queryAbilities();\n"
                    "```\n"
                ),
                "score": 0.9,
                "chunk_id": "c1",
            },
        ]
        result = generate_code("怎么调用 abilityManager", docs)
        assert result["success"] is True
        assert "import" in result["code_snippet"]
        assert "abilityManager" in result["code_snippet"]
        assert result["language"] == "typescript"

    def test_generate_code_template_fallback(self):
        """Template fallback when docs don't contain code blocks."""
        docs = [
            {
                "source": "api_doc.md",
                "content": "API 文档说明，不包含代码块",
                "score": 0.8,
                "chunk_id": "c1",
            },
        ]
        result = generate_code("怎么调用 API", docs, "typescript")
        assert result["success"] is False
        assert "模板代码" in result.get("error", "")
        assert "TODO" in result["code_snippet"]

    def test_generate_code_python_language(self):
        """Generate Python code template."""
        docs = [
            {
                "source": "api_doc.md",
                "content": "API documentation text",
                "score": 0.7,
                "chunk_id": "c1",
            },
        ]
        result = generate_code("怎么用 Python 调用", docs, "python")
        assert result["language"] == "python"
        assert "def example_usage" in result["code_snippet"]

    def test_generate_code_citations(self):
        """Return proper citations with result."""
        docs = [
            {
                "source": "api.md",
                "content": (
                    "```typescript\n"
                    "import { abilityManager } from '@ohos.app.ability';\n"
                    "const result = abilityManager.queryAbilities();\n"
                    "console.log('Result:', result);\n"
                    "```"
                ),
                "score": 0.95,
                "chunk_id": "c1",
            },
        ]
        result = generate_code("test", docs)
        assert len(result.get("citations", [])) > 0


# ===========================================================================
# Symbol extraction tests
# ===========================================================================


class TestSymbolExtraction:
    """Test code symbol extraction from retrieved docs."""

    def test_extract_symbols_from_docs_with_code(self):
        docs = [
            {
                "source": "api_doc.md",
                "content": (
                    "## API\n\n"
                    "```typescript\n"
                    "import { abilityManager } from '@ohos.app.ability';\n"
                    "class MyAbility extends UIAbility {\n"
                    "  onCreate() {\n"
                    "    abilityManager.queryAbilities();\n"
                    "  }\n"
                    "}\n"
                    "```\n"
                ),
                "score": 0.9,
            },
        ]
        result = _extract_symbols_from_docs(docs, "typescript")
        assert isinstance(result, dict)
        assert result.get("total_count", 0) > 0

    def test_extract_symbols_empty_docs(self):
        result = _extract_symbols_from_docs([], "typescript")
        assert result == {}

    def test_extract_symbols_no_code(self):
        docs = [
            {
                "source": "doc.md",
                "content": "纯文本说明，没有任何代码块",
                "score": 0.8,
            },
        ]
        result = _extract_symbols_from_docs(docs, "typescript")
        assert result == {}

    def test_format_symbols_for_prompt(self):
        symbols_info = {
            "symbols": {
                "imports": [{"name": "import { abilityManager } from '@ohos.app.ability'", "normalized_name": "import { abilitymanager"}],
                "functions": [{"name": "onCreate", "normalized_name": "oncreate"}],
                "classes": [{"name": "MyAbility", "normalized_name": "myability"}],
                "method_calls": [{"name": "abilityManager.queryAbilities()", "normalized_name": "abilitymanager.queryabilities"}],
                "types": [],
            },
            "total_count": 4,
        }
        formatted = _format_symbols_for_prompt(symbols_info, "typescript")
        assert "参考文档中提取的代码符号" in formatted
        assert "abilityManager" in formatted
        assert "onCreate" in formatted
        assert "MyAbility" in formatted

    def test_format_symbols_empty(self):
        assert _format_symbols_for_prompt({}, "typescript") == ""
        assert _format_symbols_for_prompt({"symbols": {}}, "typescript") == ""

    def test_generate_code_integrates_symbols(self):
        """Code generation successfully runs with symbols (smoke test)."""
        docs = [
            {
                "source": "api.md",
                "content": (
                    "```typescript\n"
                    "import { abilityManager } from '@ohos.app.ability';\n"
                    "const result = abilityManager.queryAbilities();\n"
                    "```"
                ),
                "score": 0.95,
                "chunk_id": "c1",
            },
        ]
        # Should succeed with code blocks extracted
        result = generate_code("call abilityManager", docs)
        assert "code_snippet" in result
        assert "language" in result
