"""Tests for RAG pipeline components."""

import pytest

from enterprise_agentic_rag.rag.document_loader import load_markdown_files
from enterprise_agentic_rag.rag.retriever import KeywordRetriever
from enterprise_agentic_rag.rag.splitter import split_documents, split_text


class TestDocumentLoader:
    """Test document loading."""

    def test_loads_sample_docs(self) -> None:
        """Should load the bundled markdown files."""
        docs = load_markdown_files()
        assert len(docs) >= 1
        filenames = {d["filename"] for d in docs}
        assert "sample_faq.md" in filenames

    def test_each_doc_has_content(self) -> None:
        """Every document should have non-empty content."""
        docs = load_markdown_files()
        for d in docs:
            assert d["content"].strip(), f"{d['filename']} has empty content"


class TestSplitter:
    """Test text splitting."""

    def test_split_paragraphs(self) -> None:
        """Multi-paragraph text should be split into chunks."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = split_text(text, chunk_size=500)
        assert len(chunks) >= 1

    def test_split_documents(self) -> None:
        """Loaded documents should produce chunks."""
        docs = load_markdown_files()
        chunks = split_documents(docs, chunk_size=300)
        assert len(chunks) >= len(docs)  # at least one chunk per doc

    def test_empty_text(self) -> None:
        """Empty text should produce no chunks."""
        assert split_text("") == []


class TestRetriever:
    """Test keyword retriever."""

    @pytest.fixture(autouse=True)
    def _retriever(self) -> None:
        self.retriever = KeywordRetriever(chunk_size=500, top_k=3)

    def test_search_returns_results(self) -> None:
        """A relevant query should return results."""
        results = self.retriever.search("AUTH_401 错误")
        assert len(results) > 0, "Should find results about AUTH_401"

    def test_search_result_structure(self) -> None:
        """Results should have content, source, and score."""
        results = self.retriever.search("API 认证")
        for r in results:
            assert "content" in r
            assert "source" in r
            assert "score" in r
            assert isinstance(r["score"], float)

    def test_irrelevant_query(self) -> None:
        """A query with no keyword overlap should return empty list."""
        results = self.retriever.search("xyzzy_plugh_not_there_42")
        assert results == []
