#!/usr/bin/env python3
"""RAG knowledge base scheduled/real-time update manager.

Usage modes:

  1. Full re-ingestion:
     python scripts/schedule_ingest.py --mode full

  2. Smart incremental (detect changes → update only changed files):
     python scripts/schedule_ingest.py --mode smart

  3. Single document update:
     python scripts/schedule_ingest.py --mode single --source sample_api_doc.md

  4. Watch mode (file system watcher, real-time):
     python scripts/schedule_ingest.py --mode watch

For cron-based scheduled ingestion, add to crontab:
  # Every 2 hours smart update
  0 */2 * * * cd /path/to/project && python scripts/schedule_ingest.py --mode smart

  # Daily full re-ingestion at 3am
  0 3 * * * cd /path/to/project && python scripts/schedule_ingest.py --mode full

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │                     RAG Update Strategy                       │
  │                                                              │
  │  Real-time (watch mode)          Scheduled (cron)            │
  │  ┌────────────────────┐         ┌──────────────────────┐     │
  │  │ FileSystemWatcher   │         │ schedule_ingest.py   │     │
  │  │ (watchdog / polling)│         │ --mode smart/full    │     │
  │  └────────┬───────────┘         └──────────┬───────────┘     │
  │           │                                │                 │
  │           ▼                                ▼                 │
  │  ┌────────────────────────────────────────────────────────┐  │
  │  │              IngestionPipeline                          │  │
  │  │                                                        │  │
  │  │  incremental_update(source)    ← single doc refresh     │  │
  │  │  smart_update()                ← detect + incremental   │  │
  │  │  run()                         ← full re-ingestion      │  │
  │  └────────────────────────────────────────────────────────┘  │
  │           │                                │                 │
  │           ▼                                ▼                 │
  │  ┌─────────────┐                  ┌──────────────────┐      │
  │  │ ES IK Index │                  │ Milvus Vector DB │      │
  │  │ (delete_by_ │                  │ (delete_by_      │      │
  │  │  source +   │                  │  source +        │      │
  │  │  reindex)   │                  │  re-upsert)      │      │
  │  └─────────────┘                  └──────────────────┘      │
  └──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Project root resolution (works from any CWD)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def acquire_lock(lock_file: Path) -> bool:
    """Acquire a file-based lock to prevent concurrent ingestion runs.

    Returns True if lock is acquired, False if another instance is running.
    """
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    fp = open(lock_file, "w")
    try:
        fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fp.write(str(os.getpid()))
        fp.flush()
        # Keep fp open to hold the lock
        return True
    except (IOError, OSError):
        fp.close()
        return False


def get_file_hash(filepath: str) -> str:
    """SHA256 hash of file content (first 16 hex chars)."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def full_ingestion() -> None:
    """Full re-ingestion — reload, dedup, split, embed, index all docs."""
    from enterprise_agentic_rag.rag.ingestion import IngestionPipeline

    print(f"[{datetime.now().isoformat()}] Starting FULL ingestion...")
    t0 = time.time()
    pipeline = IngestionPipeline()
    report = pipeline.run()
    elapsed = time.time() - t0

    _print_report(report, elapsed)
    _save_state("full", report)


def smart_ingestion() -> None:
    """Smart incremental — only re-ingest changed files."""
    from enterprise_agentic_rag.rag.ingestion import IngestionPipeline

    print(f"[{datetime.now().isoformat()}] Starting SMART (incremental) ingestion...")
    t0 = time.time()
    pipeline = IngestionPipeline()
    report = pipeline.smart_update()
    elapsed = time.time() - t0

    _print_report(report, elapsed)
    _save_state("smart", report)


def single_update(source: str) -> None:
    """Update a single document in-place."""
    from enterprise_agentic_rag.rag.ingestion import IngestionPipeline

    print(f"[{datetime.now().isoformat()}] Updating single source: {source}")
    t0 = time.time()
    pipeline = IngestionPipeline()
    report = pipeline.incremental_update(source)
    elapsed = time.time() - t0

    _print_report(report, elapsed)


def watch_mode(interval: int = 30) -> None:
    """Polling-based file watcher for real-time updates.

    Every `interval` seconds, checks data/docs/ for changed files
    and triggers incremental updates.

    Args:
        interval: Polling interval in seconds.
    """
    from enterprise_agentic_rag.rag.document_loader import load_markdown_files
    from enterprise_agentic_rag.rag.ingestion import IngestionPipeline

    print(f"[{datetime.now().isoformat()}] Watch mode started "
          f"(polling every {interval}s). Press Ctrl+C to stop.")

    # Build initial hash map
    file_hashes: dict[str, str] = {}
    docs = load_markdown_files()
    for doc in docs:
        fname = doc.get("filename", "")
        fpath = doc.get("source", "")
        if fname and fpath and os.path.isfile(fpath):
            file_hashes[fname] = get_file_hash(fpath)

    pipeline = IngestionPipeline()
    print(f"  Watching {len(file_hashes)} files...")

    try:
        while True:
            time.sleep(interval)
            current_docs = load_markdown_files()
            current_files: set[str] = set()

            for doc in current_docs:
                fname = doc.get("filename", "")
                fpath = doc.get("source", "")
                if not fname or not fpath or not os.path.isfile(fpath):
                    continue
                current_files.add(fname)

                new_hash = get_file_hash(fpath)
                old_hash = file_hashes.get(fname, "")

                if old_hash and new_hash != old_hash:
                    print(f"  🔄 [{datetime.now().strftime('%H:%M:%S')}] "
                          f"Change detected: {fname}")
                    report = pipeline.incremental_update(fname)
                    if report.success:
                        print(f"     ✅ Updated ({report.total_chunks} chunks, "
                              f"{report.duration_ms:.0f}ms)")
                    else:
                        print(f"     ⚠️  Errors: {'; '.join(report.errors)}")
                elif not old_hash:
                    print(f"  🆕 [{datetime.now().strftime('%H:%M:%S')}] "
                          f"New file: {fname}")
                    report = pipeline.incremental_update(fname)
                    if report.success:
                        print(f"     ✅ Indexed ({report.total_chunks} chunks, "
                              f"{report.duration_ms:.0f}ms)")

                file_hashes[fname] = new_hash

            # Detect deleted files
            deleted = set(file_hashes.keys()) - current_files
            for fname in deleted:
                print(f"  🗑️  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"Removed: {fname}")
                del file_hashes[fname]

    except KeyboardInterrupt:
        print(f"\n[{datetime.now().isoformat()}] Watch mode stopped.")


def _print_report(report, elapsed_sec: float) -> None:
    """Print a formatted ingestion report."""
    status = "✅ SUCCESS" if report.success else "⚠️  PARTIAL"
    print(f"\n{'='*60}")
    print(f"  Ingestion Report — {status}")
    print(f"{'='*60}")
    print(f"  Documents:     {report.total_docs}")
    print(f"  Chunks:        {report.total_chunks}")
    print(f"  MinIO uploads: {report.minio_uploaded}")
    print(f"  ES indexed:    {report.es_indexed}")
    print(f"  Milvus upsert: {report.milvus_upserted}")
    print(f"  Duration:      {elapsed_sec:.2f}s ({report.duration_ms:.0f}ms)")
    if report.errors:
        print(f"  Errors ({len(report.errors)}):")
        for err in report.errors:
            print(f"    - {err}")
    print(f"{'='*60}\n")


def _save_state(mode: str, report) -> None:
    """Save ingestion state to data/ for audit trail."""
    state_file = _PROJECT_ROOT / "data" / "ingestion_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "total_docs": report.total_docs,
        "total_chunks": report.total_chunks,
        "es_indexed": getattr(report, "es_indexed", 0),
        "milvus_upserted": getattr(report, "milvus_upserted", 0),
        "duration_ms": report.duration_ms,
        "success": report.success,
        "errors": report.errors,
    }

    history: list = []
    if state_file.exists():
        try:
            history = json.loads(state_file.read_text())
        except json.JSONDecodeError:
            pass

    history.append(entry)
    # Keep last 100 runs
    if len(history) > 100:
        history = history[-100:]

    state_file.write_text(json.dumps(history, indent=2, ensure_ascii=False))


# ===================================================================
# CLI entry point
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG knowledge base update scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode full          Full re-ingestion of all docs
  %(prog)s --mode smart         Detect changes, update only changed files
  %(prog)s --mode single --source sample_api_doc.md
  %(prog)s --mode watch --interval 30    Watch for file changes every 30s
        """,
    )
    parser.add_argument(
        "--mode", choices=["full", "smart", "single", "watch"],
        default="smart",
        help="Update mode (default: smart)",
    )
    parser.add_argument(
        "--source",
        help="Source filename for single-document update",
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Polling interval in seconds for watch mode (default: 30)",
    )
    parser.add_argument(
        "--lock-file",
        default=str(_PROJECT_ROOT / "data" / ".ingest.lock"),
        help="Lock file to prevent concurrent runs",
    )

    args = parser.parse_args()

    # Prevent concurrent runs via file lock
    lock_path = Path(args.lock_file)
    if not acquire_lock(lock_path):
        print("⚠️  Another ingestion process is already running. Exiting.")
        sys.exit(0)

    try:
        if args.mode == "full":
            full_ingestion()
        elif args.mode == "smart":
            smart_ingestion()
        elif args.mode == "single":
            if not args.source:
                print("❌ --source is required for single mode")
                sys.exit(1)
            single_update(args.source)
        elif args.mode == "watch":
            watch_mode(interval=args.interval)
    finally:
        # Clean up lock file
        if lock_path.exists():
            lock_path.unlink()


if __name__ == "__main__":
    main()
