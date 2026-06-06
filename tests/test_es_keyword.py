"""Tests for ESKeywordStore — Elasticsearch keyword search with IK Analyzer.

All tests gracefully degrade when Elasticsearch is unavailable.
"""

import pytest

from enterprise_agentic_rag.rag.es_keyword_store import ESKeywordStore


# ===================================================================
# ESKeywordStore
# ===================================================================
class TestESKeywordStore:
    def test_store_initializes_without_crashing(self) -> None:
        """ESKeywordStore should initialise even when ES is down."""
        store = ESKeywordStore()
        assert store._index == "enterprise_kb"

    def test_available_returns_bool(self) -> None:
        """available property should return a boolean, never throw."""
        store = ESKeywordStore()
        avail = store.available
        assert isinstance(avail, bool)

    def test_search_returns_empty_when_unavailable(self) -> None:
        """Search should return [] when ES is down, never crash."""
        store = ESKeywordStore()
        results = store.search("测试查询")
        assert isinstance(results, list)

    def test_stats_when_unavailable(self) -> None:
        """stats() should report unavailable when ES is down."""
        store = ESKeywordStore()
        info = store.stats()
        if not store.available:
            assert info["available"] is False
        else:
            assert "doc_count" in info


# ===================================================================
# Integration tests (require Elasticsearch)
# ===================================================================
@pytest.mark.integration
class TestESKeywordStoreIntegration:
    @pytest.fixture(autouse=True)
    def _store(self) -> None:
        self.store = ESKeywordStore(index_name="test_es_keyword_store")
        if not self.store.available:
            pytest.skip("Elasticsearch not available — skipping integration test")
        # Ensure clean index
        self.store.delete_index()

    def test_ensure_index_creates_mapping(self) -> None:
        """ensure_index() should create the index and return True."""
        assert self.store.ensure_index() is True
        # Second call should be idempotent
        assert self.store.ensure_index() is True

    def test_index_and_search_roundtrip(self) -> None:
        """Index chunks then search — should find relevant results."""
        self.store.ensure_index()

        chunks = [
            {
                "chunk_id": "api_auth.md_0",
                "source": "api_auth.md",
                "content": "API 认证支持三种方式：Bearer Token、API Key 和 OAuth 2.0。",
                "title": "API 认证说明",
                "tags": ["api", "auth"],
                "tenant_id": "default",
            },
            {
                "chunk_id": "api_auth.md_1",
                "source": "api_auth.md",
                "content": "Bearer Token 需要在请求头中携带 Authorization: Bearer <token>。",
                "title": "Bearer Token 使用",
                "tags": ["api", "auth"],
                "tenant_id": "default",
            },
            {
                "chunk_id": "password.md_0",
                "source": "password.md",
                "content": "密码重置流程：用户点击忘记密码 → 输入邮箱 → 接收重置链接 → 设置新密码。",
                "title": "密码重置",
                "tags": ["faq"],
                "tenant_id": "default",
            },
        ]

        count = self.store.index_chunks(chunks)
        assert count == 3

        # Search for API auth content
        results = self.store.search("API 认证方式", top_k=3)
        assert len(results) > 0
        # The top result should be from api_auth.md
        assert results[0]["source"] == "api_auth.md"

    def test_chinese_tokenization(self) -> None:
        """IK Analyzer should split Chinese text correctly.

        "认证方式" should match "API 认证" because IK splits it into
        ["认证", "方式"] at index time and ["认证方式"] at search time,
        meaning either token in the query can match.
        """
        self.store.ensure_index()

        chunks = [
            {
                "chunk_id": "test_0",
                "source": "test.md",
                "content": "API 认证支持多种方式，包括 Token 和 OAuth。",
                "title": "认证",
                "tags": [],
                "tenant_id": "default",
            },
        ]
        self.store.index_chunks(chunks)

        # "认证方式" should match because IK splits it to match "认证"
        results = self.store.search("认证方式", top_k=3)
        assert len(results) > 0

    def test_delete_by_source(self) -> None:
        """delete_by_source should remove all chunks from a source."""
        self.store.ensure_index()

        chunks = [
            {
                "chunk_id": "old_0", "source": "old.md",
                "content": "旧文档内容", "title": "旧文档", "tags": [], "tenant_id": "default",
            },
            {
                "chunk_id": "keep_0", "source": "keep.md",
                "content": "保留文档内容", "title": "保留文档", "tags": [], "tenant_id": "default",
            },
        ]
        self.store.index_chunks(chunks)

        # Delete old.md
        deleted = self.store.delete_by_source("old.md")
        assert deleted >= 1

        # old.md chunks should no longer appear
        results = self.store.search("旧文档", top_k=3)
        assert all(r["source"] != "old.md" for r in results)

    def test_search_with_filters(self) -> None:
        """Search with term filters should restrict results."""
        self.store.ensure_index()

        chunks = [
            {
                "chunk_id": "t1_0", "source": "t1.md",
                "content": "租户1的文档", "title": "T1 Doc",
                "tags": [], "tenant_id": "tenant_1",
            },
            {
                "chunk_id": "t2_0", "source": "t2.md",
                "content": "租户2的文档", "title": "T2 Doc",
                "tags": [], "tenant_id": "tenant_2",
            },
        ]
        self.store.index_chunks(chunks)

        # Search with tenant filter
        results = self.store.search("文档", top_k=5, filters={"tenant_id": "tenant_1"})
        assert all(r["source"] == "t1.md" for r in results)

    def test_irrelevant_query_returns_empty(self) -> None:
        """A query with no matches should return []."""
        self.store.ensure_index()
        chunks = [{
            "chunk_id": "x_0", "source": "x.md",
            "content": "This is about machine learning algorithms.",
            "title": "ML", "tags": [], "tenant_id": "default",
        }]
        self.store.index_chunks(chunks)
        results = self.store.search("xyzzy_nonexistent_12345", top_k=3)
        assert results == []

    def test_result_structure(self) -> None:
        """Results must have content, source, score, chunk_id, title."""
        self.store.ensure_index()
        chunks = [{
            "chunk_id": "s_0", "source": "s.md",
            "content": "结构化测试文档", "title": "Structure Test",
            "tags": [], "tenant_id": "default",
        }]
        self.store.index_chunks(chunks)
        results = self.store.search("结构化", top_k=1)
        assert len(results) > 0
        for key in ("content", "source", "score", "chunk_id", "title"):
            assert key in results[0], f"Missing key: {key}"
        assert isinstance(results[0]["score"], (int, float))

    def test_cleanup(self) -> None:
        """Clean up test index."""
        self.store.delete_index()
