#!/usr/bin/env python3
"""Initialise PostgreSQL tables and insert demo data.

Usage:
    python scripts/init_db.py

Requires DATABASE_URL in environment (or reads .env).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Load .env BEFORE any enterprise_agentic_rag imports ---
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
    print(f"✓ Loaded {_env_file}")

# Add project root to path
sys.path.insert(0, str(_project_root))

import asyncio

from enterprise_agentic_rag.storage.database import DatabaseManager
from enterprise_agentic_rag.storage.repositories import get_user, upsert_user


DEMO_USERS = [
    {
        "user_id": "u001",
        "name": "张三",
        "role": "developer",
        "department": "ecosystem",
        "email": "zhangsan@company.com",
        "permissions": ["basic", "policy:read", "docs:read", "ticket:read", "system:read"],
        "preferred_language": "zh-CN",
    },
    {
        "user_id": "u002",
        "name": "李四",
        "role": "developer",
        "department": "产品研发部",
        "email": "lisi@company.com",
        "permissions": ["basic", "docs:read"],
        "preferred_language": "zh-CN",
    },
    {
        "user_id": "u003",
        "name": "王五",
        "role": "basic",
        "department": "市场部",
        "email": "wangwu@company.com",
        "permissions": ["basic", "docs:read"],
        "preferred_language": "zh-CN",
    },
]


async def main() -> None:
    print("=" * 60)
    print(" Enterprise Agentic RAG — Database Initialisation")
    print("=" * 60)

    dbm = DatabaseManager()

    # 1. Check connection
    print("\n→ Checking PostgreSQL connection ...")
    if not await dbm.check_connection():
        print("  ✗ PostgreSQL is not available.")
        print("    Start it with: docker compose up -d postgres")
        print("    Or run without DB (system will use in-memory fallback).")
        return

    print("  ✓ Connected")

    # 2. Create tables
    print("\n→ Creating tables ...")
    await dbm.init_tables()
    print("  ✓ Tables created")

    # 3. Insert demo users
    print("\n→ Inserting demo users ...")
    async with dbm.session() as sess:
        for u in DEMO_USERS:
            await upsert_user(
                sess,
                user_id=u["user_id"],
                name=u["name"],
                role=u["role"],
                department=u["department"],
                permissions=u["permissions"],
                email=u["email"],
            )
            print(f"  ✓ {u['user_id']} ({u['name']}) — {u['role']}")
        await sess.commit()

    # 4. Verify
    print("\n→ Verifying ...")
    async with dbm.session() as sess:
        for uid in ["u001", "u002", "u003"]:
            user = await get_user(sess, uid)
            if user:
                perms = user.get("permissions", [])
                print(f"  ✓ {user['user_id']}: {user['name']} [{user['role']}] "
                      f"perms={perms}")
            else:
                print(f"  ✗ {uid} not found!")

    # 5. Milvus collection
    print("\n→ Checking Milvus ...")
    try:
        from enterprise_agentic_rag.rag.milvus_store import MilvusStore
        ms = MilvusStore(vector_size=768)
        if ms.available:
            ms.ensure_collection()
            print("  ✓ Milvus collection ready")
        else:
            print("  ⚠ Milvus not available — vector search disabled")
    except Exception as e:
        print(f"  ⚠ Milvus error: {e}")

    # 6. MinIO bucket
    print("\n→ Checking MinIO ...")
    try:
        from enterprise_agentic_rag.rag.minio_store import MinIOStore
        ms = MinIOStore()
        if ms.available:
            ms.ensure_bucket()
            print("  ✓ MinIO bucket ready")
        else:
            print("  ⚠ MinIO not available — document storage disabled")
    except Exception as e:
        print(f"  ⚠ MinIO error: {e}")

    await dbm.close()
    print("\n✓ Initialisation complete.")


if __name__ == "__main__":
    asyncio.run(main())
