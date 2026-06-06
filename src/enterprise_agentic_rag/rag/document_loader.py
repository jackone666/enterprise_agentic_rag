"""Load markdown documents from the data/docs/ directory."""

from __future__ import annotations

from pathlib import Path


def _get_docs_dir() -> Path:
    """Return the absolute path to the data/docs directory.

    Resolves relative to this file's location so it works regardless of CWD.
    """
    return Path(__file__).resolve().parent.parent / "data" / "docs"


def load_markdown_files(docs_dir: Path | None = None) -> list[dict[str, str]]:
    """Load all .md files from the docs directory.

    Args:
        docs_dir: Optional override path. Defaults to the bundled data/docs/.

    Returns:
        List of dicts with ``filename``, ``source``, and ``content`` keys.
    """
    docs_dir = docs_dir or _get_docs_dir()
    documents: list[dict[str, str]] = []

    if not docs_dir.exists():
        return documents

    for md_path in sorted(docs_dir.glob("*.md")):
        content = md_path.read_text(encoding="utf-8")
        if not content.strip():
            continue
        documents.append({
            "filename": md_path.name,
            "source": str(md_path),
            "content": content,
        })

    return documents
