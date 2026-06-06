#!/usr/bin/env python3
"""Migrate local file data into PostgreSQL and Redis.

One-shot script that imports:
  1. data/logs/events.jsonl      → PostgreSQL (node/retrieval/verification/llm events)
  2. data/eval/failed_cases.jsonl → PostgreSQL (failed_cases table)
  3. data/eval/regression_cases.jsonl → PostgreSQL (eval_cases table)
  4. data/near_dedup_index.json   → Redis (dedup:* keys)

Usage:
    python scripts/migrate_local_to_middleware.py [--dry-run] [--skip-redis] [--skip-events]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parents[1]
_env_file = _project_root / ".env"
if _env_file.exists():
    with open(_env_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
sys.path.insert(0, str(_project_root))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _progress(desc: str, idx: int, total: int, width: int = 40) -> None:
    """Simple inline progress bar (no external dependency)."""
    pct = (idx + 1) / total if total else 1.0
    bar = "█" * int(pct * width)
    gap = "░" * (width - len(bar))
    print(f"\r  {desc} [{bar}{gap}] {idx+1}/{total} ({pct:.0%})", end="", flush=True)


def _read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping malformed lines."""
    records: list[dict] = []
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


# ---------------------------------------------------------------------------
# 1. events.jsonl → PostgreSQL
# ---------------------------------------------------------------------------
async def _migrate_events(dry_run: bool = False) -> dict:
    """Read data/logs/events.jsonl and insert into PG event tables."""
    from enterprise_agentic_rag.storage.database import get_db_manager
    from enterprise_agentic_rag.storage.models import (
        NodeEventModel, RetrievalEventModel,
        VerificationEventModel, LLMEventModel,
    )

    path = str(_project_root / "data" / "logs" / "events.jsonl")
    records = _read_jsonl(path)
    stats = {"total": len(records), "imported": 0, "skipped": 0, "errors": 0}

    if not records:
        print("  (no events to migrate)")
        return stats

    dbm = get_db_manager()
    if not await dbm.check_connection():
        print("  ✗ PostgreSQL unavailable — skipping events migration")
        return stats

    model_map = {
        "node_events": NodeEventModel,
        "retrieval_events": RetrievalEventModel,
        "verification_events": VerificationEventModel,
        "llm_events": LLMEventModel,
    }

    # Route event_type → table
    def _route(etype: str) -> str | None:
        if etype in ("node_start", "node_end"):
            return "node_events"
        if etype == "retrieval":
            return "retrieval_events"
        if etype == "verification":
            return "verification_events"
        if etype in ("llm_call", "llm_failure"):
            return "llm_events"
        return None

    if dry_run:
        table_counts: dict[str, int] = {}
        for r in records:
            tbl = _route(r.get("event_type", ""))
            if tbl:
                table_counts[tbl] = table_counts.get(tbl, 0) + 1
            else:
                stats["skipped"] += 1
        print(f"  [dry-run] Would import: {table_counts}")
        stats["imported"] = sum(table_counts.values())
        return stats

    batch_size = 500
    batch: list[tuple[str, dict]] = []

    for i, r in enumerate(records):
        etype = r.get("event_type", "")
        tbl = _route(etype)
        if tbl is None:
            stats["skipped"] += 1
            continue

        batch.append((tbl, r))

        if len(batch) >= batch_size or i == len(records) - 1:
            try:
                async with dbm.session() as sess:
                    for table_name, event in batch:
                        model_cls = model_map[table_name]
                        instance = model_cls(
                            trace_id=event.get("trace_id", ""),
                            session_id=event.get("session_id", ""),
                            user_id=event.get("user_id", ""),
                            event_type=event.get("event_type", ""),
                            node_name=event.get("node_name", event.get("tool_name", "")),
                            input_summary=str(event.get("input_summary", ""))[:500],
                            output_summary=str(event.get("output_summary", ""))[:500],
                            latency_ms=event.get("latency_ms", 0.0),
                            success=event.get("success", True),
                            error=str(event.get("error", ""))[:500],
                            meta_json=json.dumps(event, ensure_ascii=False, default=str),
                        )
                        sess.add(instance)
                    await sess.commit()
                stats["imported"] += len(batch)
            except Exception as exc:
                stats["errors"] += len(batch)
                print(f"\n  ⚠ Batch insert error: {exc}")
            batch.clear()

        if (i + 1) % batch_size == 0:
            _progress("events.jsonl", i, len(records))

    _progress("events.jsonl", len(records), len(records))
    print()
    return stats


# ---------------------------------------------------------------------------
# 2. failed_cases.jsonl → PostgreSQL
# ---------------------------------------------------------------------------
async def _migrate_failed_cases(dry_run: bool = False) -> dict:
    """Read data/eval/failed_cases.jsonl and insert into PG failed_cases table."""
    from enterprise_agentic_rag.storage.database import get_db_manager
    from enterprise_agentic_rag.storage.models import FailedCaseModel

    path = str(_project_root / "data" / "eval" / "failed_cases.jsonl")
    records = _read_jsonl(path)
    stats = {"total": len(records), "imported": 0, "errors": 0}

    if not records:
        print("  (no failed cases to migrate)")
        return stats

    dbm = get_db_manager()
    if not await dbm.check_connection():
        print("  ✗ PostgreSQL unavailable — skipping failed_cases migration")
        return stats

    if dry_run:
        print(f"  [dry-run] Would import {len(records)} failed cases")
        stats["imported"] = len(records)
        return stats

    try:
        async with dbm.session() as sess:
            for i, r in enumerate(records):
                instance = FailedCaseModel(
                    trace_id=r.get("trace_id", ""),
                    session_id=r.get("session_id", ""),
                    query=r.get("query", ""),
                    reason=r.get("fallback_reason", ""),
                    source=r.get("source", "import"),
                    payload=json.dumps(r.get("metadata", {}), ensure_ascii=False),
                )
                sess.add(instance)
                if (i + 1) % 200 == 0:
                    _progress("failed_cases.jsonl", i, len(records))
            await sess.commit()
            stats["imported"] = len(records)
    except Exception as exc:
        stats["errors"] = len(records)
        print(f"\n  ✗ Insert error: {exc}")

    _progress("failed_cases.jsonl", len(records), len(records))
    print()
    return stats


# ---------------------------------------------------------------------------
# 3. regression_cases.jsonl → PostgreSQL (thin wrapper)
# ---------------------------------------------------------------------------
async def _migrate_regression_cases(dry_run: bool = False) -> dict:
    """Read data/eval/regression_cases.jsonl and insert into PG eval_cases table."""
    from enterprise_agentic_rag.storage.database import get_db_manager
    from enterprise_agentic_rag.storage.repositories import insert_eval_case

    path = str(_project_root / "data" / "eval" / "regression_cases.jsonl")
    records = _read_jsonl(path)
    stats = {"total": len(records), "imported": 0, "errors": 0, "skipped": 0}

    if not records:
        print("  (no regression cases to migrate)")
        return stats

    dbm = get_db_manager()
    if not await dbm.check_connection():
        print("  ✗ PostgreSQL unavailable — skipping regression cases migration")
        return stats

    if dry_run:
        print(f"  [dry-run] Would import {len(records)} regression cases")
        stats["imported"] = len(records)
        return stats

    try:
        async with dbm.session() as sess:
            for i, r in enumerate(records):
                if not r.get("query"):
                    stats["skipped"] += 1
                    continue
                await insert_eval_case(
                    sess,
                    query=r.get("query", ""),
                    expected_intent=r.get("expected_intent", ""),
                    expected_sources=r.get("expected_sources", []),
                    expected_answer_keywords=r.get("expected_answer_keywords", []),
                    difficulty=r.get("difficulty", "medium"),
                    source="migration",
                )
            await sess.commit()
            stats["imported"] = len(records) - stats["skipped"]
    except Exception as exc:
        stats["errors"] = len(records)
        print(f"\n  ✗ Insert error: {exc}")

    print(f"  ✓ {stats['imported']} imported, {stats['skipped']} skipped")
    return stats


# ---------------------------------------------------------------------------
# 4. near_dedup_index.json → Redis
# ---------------------------------------------------------------------------
def _migrate_near_dedup(dry_run: bool = False) -> dict:
    """Read data/near_dedup_index.json and write to Redis dedup keys."""
    import redis as redis_lib
    from enterprise_agentic_rag.config.settings import get_settings

    path = str(_project_root / "data" / "near_dedup_index.json")
    stats = {"total": 0, "imported": 0, "errors": 0}

    if not os.path.exists(path):
        print("  (no near_dedup_index to migrate)")
        return stats

    with open(path, encoding="utf-8") as fh:
        try:
            fingerprints = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"  ✗ Invalid JSON: {exc}")
            stats["errors"] = 1
            return stats

    doc_ids = list(fingerprints.keys())
    stats["total"] = len(doc_ids)

    if not doc_ids:
        print("  (empty index)")
        return stats

    # Connect to Redis
    try:
        s = get_settings()
        r = redis_lib.from_url(s.redis.connection_url, decode_responses=True)
        r.ping()
    except Exception as exc:
        print(f"  ✗ Redis unavailable: {exc}")
        return stats

    if dry_run:
        print(f"  [dry-run] Would import {len(doc_ids)} dedup fingerprints to Redis")
        stats["imported"] = len(doc_ids)
        return stats

    DEDUP_DOC = "dedup:doc"
    DEDUP_HASH = "dedup:hash"
    DEDUP_ALL = "dedup:all"
    TTL = 60 * 60 * 24 * 30  # 30 days

    imported = 0
    for i, (doc_id, fp) in enumerate(fingerprints.items()):
        try:
            content_hash = fp.get("content_hash", "")
            emb = fp.get("embedding", [])
            meta = fp.get("metadata", {})

            # Convert embedding floats for JSON serialization
            if isinstance(emb, list) and emb and isinstance(emb[0], float):
                emb_str = json.dumps(emb, ensure_ascii=False)
            else:
                emb_str = json.dumps(emb if isinstance(emb, list) else [], ensure_ascii=False)

            pipe = r.pipeline()
            pipe.hset(
                f"{DEDUP_DOC}:{doc_id}",
                mapping={
                    "content_hash": content_hash,
                    "embedding": emb_str,
                    "metadata": json.dumps(meta, ensure_ascii=False),
                },
            )
            pipe.expire(f"{DEDUP_DOC}:{doc_id}", TTL)
            if content_hash:
                pipe.sadd(f"{DEDUP_HASH}:{content_hash}", doc_id)
                pipe.expire(f"{DEDUP_HASH}:{content_hash}", TTL)
            pipe.sadd(DEDUP_ALL, doc_id)
            pipe.expire(DEDUP_ALL, TTL)
            pipe.execute()
            imported += 1
        except Exception as exc:
            stats["errors"] += 1
            if stats["errors"] <= 3:
                print(f"\n  ⚠ doc_id={doc_id}: {exc}")

        if (i + 1) % 100 == 0:
            _progress("near_dedup_index.json", i, len(doc_ids))

    stats["imported"] = imported
    _progress("near_dedup_index.json", len(doc_ids), len(doc_ids))
    print()
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    dry_run = "--dry-run" in sys.argv
    skip_redis = "--skip-redis" in sys.argv
    skip_events = "--skip-events" in sys.argv

    title = "DRY RUN" if dry_run else "MIGRATION"
    print("=" * 60)
    print(f" Enterprise Agentic RAG — Local → Middleware {title}")
    print("=" * 60)

    all_stats: dict[str, dict] = {}

    # --------------------------------------------------
    # Step 1: events.jsonl → PostgreSQL
    # --------------------------------------------------
    if not skip_events:
        print("\n[1/4] events.jsonl → PostgreSQL event tables")
        print("-" * 40)
        all_stats["events"] = await _migrate_events(dry_run)
    else:
        print("\n[1/4] events.jsonl → SKIPPED (--skip-events)")

    # --------------------------------------------------
    # Step 2: failed_cases.jsonl → PostgreSQL
    # --------------------------------------------------
    print("\n[2/4] failed_cases.jsonl → PostgreSQL failed_cases")
    print("-" * 40)
    all_stats["failed_cases"] = await _migrate_failed_cases(dry_run)

    # --------------------------------------------------
    # Step 3: regression_cases.jsonl → PostgreSQL
    # --------------------------------------------------
    print("\n[3/4] regression_cases.jsonl → PostgreSQL eval_cases")
    print("-" * 40)
    all_stats["regression_cases"] = await _migrate_regression_cases(dry_run)

    # --------------------------------------------------
    # Step 4: near_dedup_index.json → Redis
    # --------------------------------------------------
    if not skip_redis:
        print("\n[4/4] near_dedup_index.json → Redis dedup keys")
        print("-" * 40)
        all_stats["near_dedup"] = _migrate_near_dedup(dry_run)
    else:
        print("\n[4/4] near_dedup_index.json → SKIPPED (--skip-redis)")

    # --------------------------------------------------
    # Report
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print(" Migration Summary")
    print("=" * 60)
    grand_total = 0
    grand_imported = 0
    for name, s in all_stats.items():
        total = s.get("total", 0)
        imported = s.get("imported", 0)
        skipped = s.get("skipped", 0)
        errors = s.get("errors", 0)
        grand_total += total
        grand_imported += imported
        line = f"  {name:<24s} → {imported:>6d} / {total:>6d}"
        if skipped:
            line += f"  ({skipped} skipped)"
        if errors:
            line += f"  ⚠ {errors} errors"
        print(line)

    print("-" * 40)
    print(f"  {'TOTAL':24s} → {grand_imported:>6d} / {grand_total:>6d}")

    if dry_run:
        print("\n  ⓘ  Dry run — no data was written. Remove --dry-run to execute.")
    else:
        print("\n  ✓ Migration complete.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
