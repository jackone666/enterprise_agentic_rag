"""Code symbol extractor — AST-level extraction from code blocks.

Detects code blocks in markdown content and extracts structured symbols
using tree-sitter (AST) or enhanced regex as fallback.

Symbol types: IMPORT, METHOD_CALL, PROPERTY, TYPE, INTERFACE, CODE_BLOCK
(in addition to the existing CLASS, FUNCTION, MODULE etc. from entity_extractor).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try importing tree-sitter (optional dependency)
try:
    import tree_sitter
    import tree_sitter_typescript

    _TREE_SITTER_AVAILABLE = True
    _TS_LANGUAGE = tree_sitter.Language(tree_sitter_typescript.language_typescript())
    _TS_PARSER = tree_sitter.Parser()
    _TS_PARSER.set_language(_TS_LANGUAGE)
except ImportError:
    _TREE_SITTER_AVAILABLE = False
    _TS_LANGUAGE = None
    _TS_PARSER = None


# ===========================================================================
# Data structures
# ===========================================================================


@dataclass
class CodeBlock:
    """A detected code block in markdown content."""

    language: str
    start_line: int  # line offset within the chunk
    end_line: int    # line offset within the chunk
    code_text: str
    confidence: float = 1.0  # 1.0 for markdown-fenced, lower for heuristic


@dataclass
class CodeSymbol:
    """A symbol extracted from code via AST or regex analysis."""

    name: str
    type: str  # IMPORT, METHOD_CALL, PROPERTY, TYPE, INTERFACE, CLASS, FUNCTION, MODULE
    normalized_name: str = ""
    source_code: str = ""     # The code snippet that defined this symbol
    line_number: int = 0      # Line within the code block
    code_block_idx: int = 0   # Index of the parent CodeBlock
    confidence: float = 1.0   # 1.0 for AST, 0.7 for regex

    def __post_init__(self):
        if not self.normalized_name:
            self.normalized_name = self.name.lower().strip()


# ===========================================================================
# Code block detection
# ===========================================================================

_CODE_FENCE_RE = re.compile(
    r"```(\w*)\s*\n(.*?)```",
    re.DOTALL,
)


def extract_code_blocks(content: str) -> list[CodeBlock]:
    """Extract code blocks from markdown fenced code regions.

    Args:
        content: Full chunk text content.

    Returns:
        List of detected CodeBlock objects. Empty list if no code blocks found.
    """
    blocks: list[CodeBlock] = []

    # Determine line offsets
    lines = content.split("\n")
    # Build cumulative line offset map
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line) + 1  # +1 for newline

    def _offset_to_line(char_offset: int) -> int:
        for i, start in enumerate(line_starts):
            if i + 1 < len(line_starts) and char_offset < line_starts[i + 1]:
                return i + 1
        return len(lines)

    for i, m in enumerate(_CODE_FENCE_RE.finditer(content)):
        language = m.group(1).strip().lower() or "text"
        code_text = m.group(2)
        start_line = _offset_to_line(m.start())
        end_line = _offset_to_line(m.end())
        blocks.append(CodeBlock(
            language=language,
            start_line=start_line,
            end_line=end_line,
            code_text=code_text,
            confidence=1.0,
        ))

    return blocks


def detect_code_density(content: str) -> float:
    """Calculate the proportion of content that is code.

    Returns a float in [0.0, 1.0] representing the fraction of characters
    contained within code fences.
    """
    if not content:
        return 0.0
    code_chars = sum(m.end() - m.start() for m in _CODE_FENCE_RE.finditer(content))
    return min(1.0, code_chars / len(content))


def chunk_has_code(content: str) -> bool:
    """Quick check whether a chunk contains any code blocks."""
    return _CODE_FENCE_RE.search(content) is not None


# ===========================================================================
# AST-level symbol extraction
# ===========================================================================


def extract_symbols_from_code(
    code: str,
    language: str,
    code_block_idx: int = 0,
) -> list[CodeSymbol]:
    """Extract symbols from code using AST or regex.

    Args:
        code: The code text to analyze.
        language: Programming language (typescript, javascript, python, etc.).
        code_block_idx: Index of this code block in the chunk (for provenance).

    Returns:
        List of extracted CodeSymbol objects.
    """
    lang_lower = language.lower()

    # Try tree-sitter for TypeScript/JavaScript/ArkTS
    if _TREE_SITTER_AVAILABLE and lang_lower in ("typescript", "javascript", "ts", "js", "arkts", "ets"):
        try:
            return _extract_typescript_symbols_ast(code, code_block_idx)
        except Exception as exc:
            logger.debug("tree-sitter AST extraction failed, falling back to regex: %s", exc)

    # Try Python ast
    if lang_lower in ("python", "py"):
        try:
            return _extract_python_symbols_ast(code, code_block_idx)
        except Exception as exc:
            logger.debug("Python AST extraction failed, falling back to regex: %s", exc)

    # Fallback to regex for all languages
    return _extract_symbols_regex(code, language, code_block_idx)


# ===========================================================================
# TypeScript/JavaScript AST extraction (tree-sitter)
# ===========================================================================


def _ts_get_text(node, source_code: str) -> str:
    """Safely extract the text of a tree-sitter node."""
    return source_code[node.start_byte:node.end_byte].strip()


def _ts_get_line(node) -> int:
    """Get the start line of a tree-sitter node (1-based)."""
    return node.start_point[0] + 1


def _extract_typescript_symbols_ast(code: str, code_block_idx: int = 0) -> list[CodeSymbol]:
    """Extract symbols from TypeScript/JavaScript code using tree-sitter AST."""
    if _TS_PARSER is None:
        return []

    tree = _TS_PARSER.parse(bytes(code, "utf-8"))
    root = tree.root_node
    symbols: list[CodeSymbol] = []

    def _walk(node):
        # Import declarations
        if node.type == "import_statement":
            import_text = _ts_get_text(node, code)
            # Extract module from: import { X, Y } from 'module'
            module_match = re.search(r"from\s+['\"]([^'\"]+)['\"]", import_text)
            if module_match:
                symbols.append(CodeSymbol(
                    name=module_match.group(1),
                    type="IMPORT",
                    source_code=import_text,
                    line_number=_ts_get_line(node),
                    code_block_idx=code_block_idx,
                    confidence=1.0,
                ))

        # Class declarations
        elif node.type == "class_declaration":
            for child in node.children:
                if child.type == "identifier":
                    name = _ts_get_text(child, code)
                    # Check for extends
                    extends = ""
                    for c in node.children:
                        if c.type == "class_heritage":
                            heritage_text = _ts_get_text(c, code)
                            extends_match = re.search(r"extends\s+(\w+)", heritage_text)
                            if extends_match:
                                extends = extends_match.group(1)
                            implements_match = re.findall(r"implements\s+([\w\s,]+)", heritage_text)
                    symbols.append(CodeSymbol(
                        name=name,
                        type="CLASS",
                        source_code=_ts_get_text(node, code),
                        line_number=_ts_get_line(node),
                        code_block_idx=code_block_idx,
                        confidence=1.0,
                    ))

        # Function/method declarations
        elif node.type in ("function_declaration", "method_definition"):
            name = ""
            for child in node.children:
                if child.type == "identifier":
                    name = _ts_get_text(child, code)
                    break
                elif child.type == "property_identifier":
                    name = _ts_get_text(child, code)
                    break
            if name:
                symbols.append(CodeSymbol(
                    name=name,
                    type="FUNCTION",
                    source_code=_ts_get_text(node, code),
                    line_number=_ts_get_line(node),
                    code_block_idx=code_block_idx,
                    confidence=1.0,
                ))

        # Interface declarations
        elif node.type == "interface_declaration":
            for child in node.children:
                if child.type == "type_identifier":
                    name = _ts_get_text(child, code)
                    symbols.append(CodeSymbol(
                        name=name,
                        type="INTERFACE",
                        source_code=_ts_get_text(node, code),
                        line_number=_ts_get_line(node),
                        code_block_idx=code_block_idx,
                        confidence=1.0,
                    ))
                    break

        # Type alias
        elif node.type == "type_alias_declaration":
            for child in node.children:
                if child.type == "type_identifier":
                    name = _ts_get_text(child, code)
                    symbols.append(CodeSymbol(
                        name=name,
                        type="TYPE",
                        source_code=_ts_get_text(node, code),
                        line_number=_ts_get_line(node),
                        code_block_idx=code_block_idx,
                        confidence=1.0,
                    ))
                    break

        # Call expressions (method calls)
        elif node.type == "call_expression":
            # Get the function being called
            func = node.child_by_field_name("function")
            if func:
                func_text = _ts_get_text(func, code)
                # Only record meaningful calls (not single-char, not literal)
                if len(func_text) > 1 and not func_text.startswith('"') and not func_text.startswith("'"):
                    symbols.append(CodeSymbol(
                        name=func_text,
                        type="METHOD_CALL",
                        source_code=_ts_get_text(node, code),
                        line_number=_ts_get_line(node),
                        code_block_idx=code_block_idx,
                        confidence=0.9,
                    ))

        # Property access (obj.prop)
        elif node.type == "member_expression":
            prop = node.child_by_field_name("property")
            obj = node.child_by_field_name("object")
            if prop and obj:
                obj_text = _ts_get_text(obj, code)
                prop_text = _ts_get_text(prop, code)
                if obj_text and prop_text:
                    symbols.append(CodeSymbol(
                        name=f"{obj_text}.{prop_text}",
                        type="PROPERTY",
                        source_code=_ts_get_text(node, code),
                        line_number=_ts_get_line(node),
                        code_block_idx=code_block_idx,
                        confidence=0.85,
                    ))

        # Recurse
        for child in node.children:
            _walk(child)

    _walk(root)

    # Deduplicate by (normalized_name, type)
    seen: set[tuple[str, str]] = set()
    deduped: list[CodeSymbol] = []
    for sym in symbols:
        key = (sym.normalized_name, sym.type)
        if key not in seen:
            seen.add(key)
            deduped.append(sym)

    return deduped


# ===========================================================================
# Python AST extraction (stdlib ast)
# ===========================================================================


def _extract_python_symbols_ast(code: str, code_block_idx: int = 0) -> list[CodeSymbol]:
    """Extract symbols from Python code using the stdlib ast module."""
    import ast as py_ast

    symbols: list[CodeSymbol] = []

    try:
        tree = py_ast.parse(code)
    except SyntaxError:
        return []

    for node in py_ast.walk(tree):
        if isinstance(node, py_ast.Import):
            for alias in node.names:
                symbols.append(CodeSymbol(
                    name=alias.name,
                    type="IMPORT",
                    source_code=f"import {alias.name}",
                    line_number=node.lineno,
                    code_block_idx=code_block_idx,
                    confidence=1.0,
                ))
        elif isinstance(node, py_ast.ImportFrom):
            if node.module:
                symbols.append(CodeSymbol(
                    name=node.module,
                    type="IMPORT",
                    source_code=f"from {node.module} import ...",
                    line_number=node.lineno,
                    code_block_idx=code_block_idx,
                    confidence=1.0,
                ))
        elif isinstance(node, py_ast.ClassDef):
            bases = [py_ast.unparse(b) for b in node.bases] if hasattr(py_ast, 'unparse') else []
            symbols.append(CodeSymbol(
                name=node.name,
                type="CLASS",
                source_code=f"class {node.name}",
                line_number=node.lineno,
                code_block_idx=code_block_idx,
                confidence=1.0,
            ))
        elif isinstance(node, py_ast.FunctionDef):
            # Skip __dunder__ methods to reduce noise
            symbols.append(CodeSymbol(
                name=node.name,
                type="FUNCTION",
                source_code=f"def {node.name}(...)",
                line_number=node.lineno,
                code_block_idx=code_block_idx,
                confidence=1.0,
            ))
        elif isinstance(node, py_ast.Call):
            if isinstance(node.func, py_ast.Name):
                symbols.append(CodeSymbol(
                    name=node.func.id,
                    type="METHOD_CALL",
                    source_code=f"{node.func.id}(...)",
                    line_number=node.lineno,
                    code_block_idx=code_block_idx,
                    confidence=0.9,
                ))

    return symbols


# ===========================================================================
# Regex fallback symbol extraction
# ===========================================================================

_CODE_SYMBOL_PATTERNS: dict[str, list[str]] = {
    "IMPORT": [
        r"import\s+\{[^}]+\}\s+from\s+['\"]([^'\"]+)['\"]",
        r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]",
        r"require\(['\"]([^'\"]+)['\"]\)",
        r"from\s+(\S+)\s+import\s+",
    ],
    "CLASS": [
        r"class\s+(\w+)(?:\s+extends\s+(\w+))?",
        r"@Component\s*(?:\n|.)*?struct\s+(\w+)",
    ],
    "FUNCTION": [
        r"(?:async\s+)?function\s+(\w+)\s*\(",
        r"(?:public|private|protected|static)?\s*(\w+)\s*\([^)]*\)\s*\{",
        r"def\s+(\w+)\s*\(",
    ],
    "INTERFACE": [
        r"interface\s+(\w+)",
    ],
    "TYPE": [
        r"type\s+(\w+)\s*=",
    ],
    "METHOD_CALL": [
        r"(\w+\.\w+)\s*\(",
        r"new\s+(\w+)\s*\(",
    ],
}


def _extract_symbols_regex(code: str, language: str, code_block_idx: int = 0) -> list[CodeSymbol]:
    """Extract symbols from code using enhanced regex patterns.

    Used as fallback when tree-sitter or stdlib ast are unavailable.
    """
    symbols: list[CodeSymbol] = []
    seen: set[tuple[str, str]] = set()

    for sym_type, patterns in _CODE_SYMBOL_PATTERNS.items():
        for pattern in patterns:
            try:
                for m in re.finditer(pattern, code, re.MULTILINE):
                    # Try capture groups — prefer group(1)
                    name = ""
                    if m.lastindex and m.lastindex >= 1:
                        name = m.group(1)
                    if not name or len(name) < 2:
                        continue
                    name = name.strip()

                    key = (name.lower(), sym_type)
                    if key in seen:
                        continue
                    seen.add(key)

                    # Estimate line number from match position
                    line_num = code[:m.start()].count("\n") + 1

                    symbols.append(CodeSymbol(
                        name=name,
                        type=sym_type,
                        source_code=m.group(0).strip()[:120],
                        line_number=line_num,
                        code_block_idx=code_block_idx,
                        confidence=0.7,  # Regex is less reliable than AST
                    ))
            except re.error:
                continue

    return symbols


# ===========================================================================
# High-level extraction from chunks
# ===========================================================================


def extract_code_symbols_from_chunk(
    content: str,
    chunk_id: str = "",
    doc_id: str = "",
) -> list[CodeSymbol]:
    """Extract code symbols from a chunk containing markdown with code blocks.

    1. Detect code blocks in the content
    2. For each code block, extract symbols via AST or regex
    3. Return all found symbols

    Args:
        content: Full chunk text (may contain markdown + code blocks).
        chunk_id: Chunk identifier for provenance.
        doc_id: Parent document identifier.

    Returns:
        List of extracted CodeSymbol objects.
    """
    code_blocks = extract_code_blocks(content)
    if not code_blocks:
        return []

    all_symbols: list[CodeSymbol] = []
    for i, cb in enumerate(code_blocks):
        symbols = extract_symbols_from_code(cb.code_text, cb.language, code_block_idx=i)
        all_symbols.extend(symbols)

    return all_symbols
