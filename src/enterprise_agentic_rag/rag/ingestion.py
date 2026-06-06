"""Document ingestion pipeline.

Flow:
1. Load markdown files from data/docs/
2. Upload originals to MinIO
3. Two-layer dedup (SHA256 + near-duplicate)
4. Split into chunks
5. Generate embeddings
6. Index chunks to Elasticsearch (IK Analyzer keyword search)
7. Upsert chunks + vectors into Milvus
8. Produce an ingestion report
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from enterprise_agentic_rag.rag.document_loader import load_markdown_files
from enterprise_agentic_rag.rag.embedding_provider import get_embedding_provider
from enterprise_agentic_rag.rag.es_keyword_store import ESKeywordStore
from enterprise_agentic_rag.rag.milvus_store import MilvusStore
from enterprise_agentic_rag.rag.minio_store import MinIOStore
from enterprise_agentic_rag.rag.splitter import split_documents


@dataclass
class IngestionReport:
    total_docs: int = 0
    total_chunks: int = 0
    minio_uploaded: int = 0
    es_indexed: int = 0
    milvus_upserted: int = 0
    graph_entities: int = 0
    graph_relations: int = 0
    graph_indexed: bool = False
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.total_chunks > 0 and not self.errors


class IngestionPipeline:
    """Orchestrates the full ingestion flow."""

    def __init__(
        self,
        milvus: MilvusStore | None = None,
        minio: MinIOStore | None = None,
        es_store: ESKeywordStore | None = None,
        vector_size: int = 768,
    ) -> None:
        self.milvus = milvus or MilvusStore(vector_size=vector_size)
        self.minio = minio or MinIOStore()
        self._es_store = es_store or ESKeywordStore()
        self.embedder = get_embedding_provider()

    def run(self, docs_dir: str | None = None) -> IngestionReport:
        t0 = time.time()
        report = IngestionReport()

        # 1. Load documents
        raw_docs = load_markdown_files()
        report.total_docs = len(raw_docs)
        if not raw_docs:
            report.errors.append("No markdown files found in data/docs/")
            report.duration_ms = (time.time() - t0) * 1000
            return report

        # 2. Upload to MinIO
        for doc in raw_docs:
            fname = doc.get("filename", "")
            fpath = doc.get("source", "")
            if fname and fpath:
                result = self.minio.upload_document(fpath, object_name=fname)
                if result:
                    report.minio_uploaded += 1
                else:
                    # MinIO may be down — non-fatal
                    pass

        # 3. Two-layer dedup: SHA256 exact + embedding near-duplicate
        import hashlib

        from enterprise_agentic_rag.rag.near_dedup import NearDedupIndex, sample_doc_embedding

        dedup_index = NearDedupIndex(threshold=0.95)
        seen_hashes: set[str] = set()
        deduped: list[dict] = []
        near_dup_count = 0

        for doc in raw_docs:
            content = doc["content"]
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

            # Layer 1: exact hash
            if content_hash in seen_hashes:
                continue

            # Layer 2: near-duplicate via embedding
            doc_vec = sample_doc_embedding(content, self.embedder)
            is_dup, _, sim = dedup_index.detect(doc_vec)
            if is_dup:
                near_dup_count += 1
                continue

            seen_hashes.add(content_hash)
            doc["content_hash"] = content_hash
            dedup_index.register(
                doc.get("filename", "unknown"),
                content,
                embedding=doc_vec,
                metadata={"source": doc.get("source", ""), "size": len(content)},
            )
            deduped.append(doc)

        if len(deduped) < len(raw_docs):
            report.errors.append(
                f"Dedup: {len(raw_docs) - len(deduped)} removed "
                f"(hash={len(raw_docs) - len(deduped) - near_dup_count}, near={near_dup_count})"
            )
        raw_docs = deduped

        # 4. Split into chunks with metadata injection
        chunks = split_documents(raw_docs, chunk_size=500)
        for ch in chunks:
            ch["tenant_id"] = "default"
            ch["doc_id"] = ch.get("source", "unknown")
            ch["content_hash"] = hashlib.sha256(ch["content"].encode()).hexdigest()[:16]
        report.total_chunks = len(chunks)

        # 5. Embed
        texts = [c["content"] for c in chunks]
        try:
            vectors = self.embedder.embed(texts)
        except Exception as exc:
            report.errors.append(f"Embedding failed: {exc}")
            report.duration_ms = (time.time() - t0) * 1000
            return report

        # Assign chunk_ids before indexing (needed by both ES and Milvus)
        for i, ch in enumerate(chunks):
            ch["chunk_id"] = f"{ch.get('source', 'doc')}_{ch.get('chunk_index', i)}"

        # 6. Index to Elasticsearch (IK Analyzer keyword search)
        if self._es_store and self._es_store.available:
            try:
                self._es_store.ensure_index()
                report.es_indexed = self._es_store.index_chunks(chunks)
            except Exception as exc:
                report.errors.append(f"ES index failed: {exc}")
        else:
            report.errors.append("Elasticsearch unavailable — keyword search affected")

        # 7. Upsert to Milvus with tenant metadata
        if self.milvus.available:
            report.milvus_upserted = self.milvus.upsert_chunks(chunks, vectors)
        else:
            report.errors.append("Milvus unavailable — vectors not stored")

        # 8. Graph indexing (Neo4j) — optional, non-blocking
        _build_graph_for_ingestion(chunks, raw_docs, report)

        report.duration_ms = round((time.time() - t0) * 1000, 2)
        return report

    # ------------------------------------------------------------------
    # Incremental update (real-time single-document refresh)
    # ------------------------------------------------------------------
    def incremental_update(self, source: str) -> IngestionReport:
        """Re-ingest a single source document (delete-then-reindex).

        Useful for real-time updates when a specific document changes.
        The delete-before-reindex approach guarantees no stale chunks remain.

        Args:
            source: Source filename or path (e.g. ``sample_api_doc.md``).

        Returns:
            IngestionReport for this single-document update.
        """
        t0 = time.time()
        report = IngestionReport()

        # 1. Find the document in data/docs/
        raw_docs = load_markdown_files()
        target = None
        for doc in raw_docs:
            fname = doc.get("filename", "")
            fpath = doc.get("source", "")
            if fname == source or fpath.endswith(source):
                target = doc
                break

        if target is None:
            report.errors.append(f"Source not found: {source}")
            report.duration_ms = (time.time() - t0) * 1000
            return report

        report.total_docs = 1

        # 2. Delete old chunks from ES
        if self._es_store and self._es_store.available:
            self._es_store.ensure_index()
            self._es_store.delete_by_source(source)

        # 3. Delete old vectors from Milvus
        if self.milvus.available:
            self.milvus.delete_by_source(source)

        # 4. Re-upload to MinIO
        fname = target.get("filename", "")
        fpath = target.get("source", "")
        if fname and fpath:
            self.minio.upload_document(fpath, object_name=fname)
            report.minio_uploaded = 1

        # 5. Split into chunks
        import hashlib
        chunks = split_documents([target], chunk_size=500)
        for ch in chunks:
            ch["tenant_id"] = "default"
            ch["doc_id"] = ch.get("source", "unknown")
            ch["content_hash"] = hashlib.sha256(ch["content"].encode()).hexdigest()[:16]
        report.total_chunks = len(chunks)

        # 6. Embed
        texts = [c["content"] for c in chunks]
        try:
            vectors = self.embedder.embed(texts)
        except Exception as exc:
            report.errors.append(f"Embedding failed: {exc}")
            report.duration_ms = (time.time() - t0) * 1000
            return report

        # 7. Assign chunk_ids
        for i, ch in enumerate(chunks):
            ch["chunk_id"] = f"{ch.get('source', 'doc')}_{ch.get('chunk_index', i)}"

        # 8. Re-index to ES
        if self._es_store and self._es_store.available:
            self._es_store.ensure_index()
            report.es_indexed = self._es_store.index_chunks(chunks)

        # 9. Re-upsert to Milvus
        if self.milvus.available:
            report.milvus_upserted = self.milvus.upsert_chunks(chunks, vectors)

        report.duration_ms = round((time.time() - t0) * 1000, 2)
        return report

    # ------------------------------------------------------------------
    # Change detection
    # ------------------------------------------------------------------
    @staticmethod
    def detect_changes(docs_dir: str | None = None) -> list[str]:
        """Detect which source files have changed since last ingestion.

        Compares current SHA256 content hashes against existing chunks
        in the stores (ES or in-memory).

        Returns:
            List of source filenames that are new or changed.
        """
        import hashlib
        raw_docs = load_markdown_files()
        if not raw_docs:
            return []

        # Collect current content hashes per source
        current_hashes: dict[str, str] = {}
        for doc in raw_docs:
            fname = doc.get("filename", "")
            ch = hashlib.sha256(doc["content"].encode()).hexdigest()[:16]
            current_hashes[fname] = ch

        # Check against ES if available
        changed: list[str] = []
        es_store = ESKeywordStore()
        if es_store.available:
            for fname, h in current_hashes.items():
                try:
                    existing = es_store.search(
                        h, top_k=1,
                        filters={"source": fname}
                    )
                    if not existing:
                        changed.append(fname)
                except Exception:
                    changed.append(fname)
        else:
            # Without ES, rely on near-duplicate detection
            changed = list(current_hashes.keys())

        return changed

    # ------------------------------------------------------------------
    # Hybrid update (detect + incremental for changed files)
    # ------------------------------------------------------------------
    def smart_update(self, docs_dir: str | None = None) -> IngestionReport:
        """Smart hybrid update — detect changed files and incrementally update.

        Only re-ingests files that have changed. Falls back to full
        ingestion if detection isn't available.

        Returns:
            Combined IngestionReport covering all updated files.
        """
        t0 = time.time()
        changed = self.detect_changes()

        if not changed:
            report = IngestionReport()
            report.errors.append("No changes detected — up to date")
            report.duration_ms = (time.time() - t0) * 1000
            return report

        # If more than 50% changed, do full re-ingestion
        raw_docs = load_markdown_files()
        if len(changed) > len(raw_docs) * 0.5:
            return self.run()

        # Incremental update per changed file
        combined = IngestionReport()
        for src in changed:
            sub = self.incremental_update(src)
            combined.total_docs += sub.total_docs
            combined.total_chunks += sub.total_chunks
            combined.minio_uploaded += sub.minio_uploaded
            combined.es_indexed += sub.es_indexed
            combined.milvus_upserted += sub.milvus_upserted
            combined.errors.extend(sub.errors)
        combined.duration_ms = round((time.time() - t0) * 1000, 2)
        return combined


# ===========================================================================
# Graph indexing helper — called from ingestion pipeline
# ===========================================================================


def _build_graph_for_ingestion(
    chunks: list[dict],
    docs: list[dict],
    report: IngestionReport,
) -> None:
    """Build knowledge graph from chunks after main ingestion completes.

    This is a non-blocking step. Graph indexing failure does NOT affect
    keyword + vector ingestion — errors are recorded in the report but
    the main ingestion continues.

    Args:
        chunks: All chunks from ingestion.
        docs: All documents from ingestion.
        report: IngestionReport to update with graph stats.
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        from enterprise_agentic_rag.config.settings import get_settings
        settings = get_settings()
        if not settings.graph_rag.enabled:
            logger.info("Graph RAG disabled — skipping graph indexing")
            return
    except Exception:
        return

    try:
        from enterprise_agentic_rag.rag.graph.graph_indexer import GraphIndexer
        indexer = GraphIndexer()

        if not indexer.available:
            report.errors.append("Graph indexing skipped: Neo4j unavailable")
            logger.info("Graph indexing skipped: Neo4j unavailable")
            return

        # Build full graph
        graph_report = indexer.build_graph(docs=docs, chunks=chunks)

        report.graph_entities = graph_report.entity_count
        report.graph_relations = graph_report.relation_count
        report.graph_indexed = graph_report.nodes_created > 0

        if graph_report.errors:
            report.errors.extend(graph_report.errors)

        logger.info(
            "Graph indexing complete: %d entities, %d relations, %d nodes",
            graph_report.entity_count,
            graph_report.relation_count,
            graph_report.nodes_created,
        )
    except ImportError:
        report.errors.append("Graph indexing skipped: neo4j package not installed")
        logger.debug("neo4j package not installed — skipping graph indexing")
    except Exception as exc:
        report.errors.append(f"Graph indexing failed (non-blocking): {exc}")
        logger.warning("Graph indexing failed (non-blocking): %s", exc)
