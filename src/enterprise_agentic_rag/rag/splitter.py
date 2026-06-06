"""Simple text splitter for markdown documents.

Splits on double-newlines (paragraphs) and enforces a max chunk size.
"""

from __future__ import annotations


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
) -> list[str]:
    """Split text into chunks by paragraphs, merging small ones.

    Args:
        text: Raw markdown text.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Not used in this simple splitter (placeholder).

    Returns:
        List of text chunks.
    """
    _ = chunk_overlap  # reserved for future use

    # Split on double newlines (paragraph boundaries)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        # If adding this paragraph exceeds chunk_size and we already have content,
        # finalize the current chunk
        if current_len + len(para) > chunk_size and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

        # If a single paragraph exceeds chunk_size, split it further on single newlines
        if len(para) > chunk_size:
            lines = para.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if current_len + len(line) > chunk_size and current:
                    chunks.append("\n".join(current))
                    current = []
                    current_len = 0
                current.append(line)
                current_len += len(line)
        else:
            current.append(para)
            current_len += len(para)

    # Don't forget the last chunk
    if current:
        chunks.append("\n\n".join(current))

    return chunks


def split_documents(
    documents: list[dict[str, str]],
    chunk_size: int = 500,
) -> list[dict[str, str]]:
    """Split a list of loaded documents into chunks.

    Args:
        documents: List from :func:`document_loader.load_markdown_files`.
        chunk_size: Target max characters per chunk.

    Returns:
        List of chunk dicts with ``source``, ``chunk_index``, and ``content``.
    """
    chunks: list[dict[str, str]] = []

    for doc in documents:
        source = doc["filename"]
        text_chunks = split_text(doc["content"], chunk_size=chunk_size)
        for idx, chunk_text in enumerate(text_chunks):
            chunks.append({
                "source": source,
                "chunk_index": str(idx),
                "content": chunk_text,
            })

    return chunks
