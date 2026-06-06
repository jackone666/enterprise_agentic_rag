"""Tests for code symbol extraction — AST and regex symbol extraction from code blocks."""

import pytest

from enterprise_agentic_rag.rag.graph.code_symbol_extractor import (
    CodeBlock,
    CodeSymbol,
    detect_code_density,
    extract_code_blocks,
    extract_symbols_from_code,
    chunk_has_code,
)


class TestCodeBlockDetection:
    """Tests for markdown code block detection."""

    def test_extract_typescript_block(self):
        """Extract a TypeScript code block."""
        content = """Some text here.

```typescript
import { ability } from '@ohos.app';
console.log('hello');
```

More text."""
        blocks = extract_code_blocks(content)
        assert len(blocks) == 1
        assert blocks[0].language == "typescript"
        assert "import" in blocks[0].code_text
        assert blocks[0].confidence == 1.0

    def test_extract_multiple_blocks(self):
        """Extract multiple code blocks of different languages."""
        content = """```python
print('hello')
```

```typescript
console.log('world');
```"""
        blocks = extract_code_blocks(content)
        assert len(blocks) == 2
        assert blocks[0].language == "python"
        assert blocks[1].language == "typescript"

    def test_extract_no_code_blocks(self):
        """Return empty list when no code blocks exist."""
        content = "Just plain text without any code blocks."
        blocks = extract_code_blocks(content)
        assert len(blocks) == 0

    def test_extract_unnamed_fence(self):
        """Code fence without language specifier."""
        content = """```
some code here
```"""
        blocks = extract_code_blocks(content)
        assert len(blocks) == 1
        assert blocks[0].language == "text"

    def test_chunk_has_code(self):
        """Quick check for code block presence."""
        assert chunk_has_code("text ```js\ncode\n``` more") is True
        assert chunk_has_code("plain text") is False

    def test_detect_code_density(self):
        """Calculate code proportion in chunk."""
        # Half code, half text
        content = "text " * 5 + "```\n" + "code " * 5 + "\n```"
        density = detect_code_density(content)
        assert 0.3 < density < 0.7  # roughly half

    def test_detect_code_density_no_code(self):
        """Zero density when no code."""
        assert detect_code_density("plain text") == 0.0

    def test_detect_code_density_empty(self):
        """Zero density for empty content."""
        assert detect_code_density("") == 0.0


class TestSymbolExtraction:
    """Tests for symbol extraction from code."""

    def test_extract_typescript_class(self):
        """Extract class from TypeScript code via regex fallback."""
        code = "class MyAbility extends UIAbility {\n  onCreate() {\n    super.onCreate();\n  }\n}"
        symbols = extract_symbols_from_code(code, "typescript")
        # Should find class and function
        types_found = {s.type for s in symbols}
        assert "CLASS" in types_found or "FUNCTION" in types_found

    def test_extract_python_imports(self):
        """Extract imports from Python code."""
        code = "import os\nfrom typing import Any, Dict\n\ndef main():\n    print('hello')"
        symbols = extract_symbols_from_code(code, "python")
        import_symbols = [s for s in symbols if s.type == "IMPORT"]
        assert len(import_symbols) >= 1

    def test_extract_python_class_and_function(self):
        """Extract class and function from Python code."""
        code = "class MyClass:\n    def my_method(self):\n        pass"
        symbols = extract_symbols_from_code(code, "python")
        types = {s.type for s in symbols}
        assert "CLASS" in types
        assert "FUNCTION" in types

    def test_extract_interface_from_typescript(self):
        """Extract interface from TypeScript code via regex."""
        code = "interface MyInterface {\n  name: string;\n  age: number;\n}"
        symbols = extract_symbols_from_code(code, "typescript")
        types = {s.type for s in symbols}
        assert "INTERFACE" in types

    def test_regex_fallback_for_unknown_language(self):
        """Use regex fallback for languages without AST support."""
        code = "class Foo { function bar() {} }"
        symbols = extract_symbols_from_code(code, "ruby")
        # Should still use regex fallback
        assert len(symbols) >= 0  # No crash

    def test_empty_code(self):
        """Handle empty code gracefully."""
        symbols = extract_symbols_from_code("", "typescript")
        assert len(symbols) == 0

    def test_code_symbol_deduplication(self):
        """Duplicate symbols are deduplicated."""
        code = "console.log('a'); console.log('b');"
        symbols = extract_symbols_from_code(code, "typescript")
        # console.log should appear only once
        method_calls = [s for s in symbols if s.type == "METHOD_CALL" and s.name == "console.log"]
        assert len(method_calls) <= 1
