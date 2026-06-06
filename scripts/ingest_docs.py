#!/usr/bin/env python3
"""Ingest local markdown documents into Qdrant + MinIO.

Usage:
    python scripts/ingest_docs.py

Requires:
    - Qdrant running (docker compose up -d qdrant)
    - MinIO running (docker compose up -d minio)
    - .env configured

If Qdrant or MinIO are unavailable, ingestion is skipped gracefully.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Load .env first ---
_project_root = Path(__file__).resolve().parents[1]
_env_file = _project_root / ".env"
if _env_file.exists():
    with open(_env_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

sys.path.insert(0, str(_project_root))

from enterprise_agentic_rag.rag.ingestion import IngestionPipeline

print("=" * 60)
print(" Enterprise Agentic RAG — Document Ingestion")
print("=" * 60)

pipeline = IngestionPipeline()
report = pipeline.run()

print(f"\n总文档数: {report.total_docs}")
print(f"总切片数: {report.total_chunks}")
print(f"MinIO 上传: {report.minio_uploaded}")
print(f"ES 索引: {report.es_indexed}")
print(f"Milvus 写入: {report.milvus_upserted}")
print(f"耗时: {report.duration_ms}ms")

if report.errors:
    print(f"\n⚠ 警告:")
    for e in report.errors:
        print(f"  - {e}")
else:
    print("\n✓ Ingestion complete!")
