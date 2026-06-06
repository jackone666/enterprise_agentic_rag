#!/usr/bin/env python3
"""Import eval cases from JSONL into PostgreSQL.

Usage: python scripts/import_eval_cases.py [path_to_jsonl]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
_env_file = _project_root / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if not line.strip() or line.startswith("#") or "=" not in line: continue
        k, _, v = line.partition("="); k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and k not in os.environ: os.environ[k] = v
sys.path.insert(0, str(_project_root))

import asyncio

from enterprise_agentic_rag.storage.database import DatabaseManager
from enterprise_agentic_rag.storage.repositories import insert_eval_case

JSONL_PATH = sys.argv[1] if len(sys.argv) > 1 else str(_project_root / "data" / "eval" / "regression_cases.jsonl")


async def main():
    dbm = DatabaseManager()
    if not await dbm.check_connection():
        print("PostgreSQL not available")
        return
    with open(JSONL_PATH) as fh:
        lines = [l.strip() for l in fh if l.strip()]

    async with dbm.session() as sess:
        for line in lines:
            obj = json.loads(line)
            await insert_eval_case(sess, query=obj.get("query", ""),
                expected_intent=obj.get("expected_intent", ""),
                expected_sources=obj.get("expected_sources", []),
                expected_answer_keywords=obj.get("expected_answer_keywords", []),
                difficulty=obj.get("difficulty", "medium"), source="import")
        await sess.commit()
    print(f"Imported {len(lines)} eval cases to PostgreSQL")
    await dbm.close()

asyncio.run(main())
