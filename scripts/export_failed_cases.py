#!/usr/bin/env python3
"""Export failed cases from PostgreSQL to JSONL for offline analysis.

Usage: python scripts/export_failed_cases.py [output_path]
"""

from __future__ import annotations

import json, os, sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
_env_file = _project_root / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if not line.strip() or line.startswith("#") or "=" not in line: continue
        k, _, v = line.partition("="); k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and k not in os.environ: os.environ[k] = v
sys.path.insert(0, str(_project_root))

OUTPUT = sys.argv[1] if len(sys.argv) > 1 else str(_project_root / "data" / "eval" / "exported_failed_cases.jsonl")

import asyncio
from enterprise_agentic_rag.storage.database import DatabaseManager
from sqlalchemy import text


async def main():
    dbm = DatabaseManager()
    if not await dbm.check_connection():
        print("PostgreSQL not available — check data/eval/failed_cases.jsonl for JSONL fallback")
        return

    async with dbm.session() as sess:
        r = await sess.execute(text("SELECT * FROM failed_cases ORDER BY created_at DESC LIMIT 1000"))
        rows = r.fetchall()

    if not rows:
        print("No failed cases in PostgreSQL")
        return

    cols = list(rows[0]._mapping.keys())
    with open(OUTPUT, "w") as fh:
        for row in rows:
            obj = {c: str(row._mapping[c]) for c in cols}
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Exported {len(rows)} failed cases to {OUTPUT}")
    await dbm.close()

asyncio.run(main())
