#!/usr/bin/env bash
# ============================================================
# stop_dev.sh — Stop all Docker Compose services (keep data)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "→ Stopping Docker Compose services (volumes preserved) ..."
docker compose down

echo "✓ All services stopped. Data preserved in Docker volumes."
echo "  Use ./scripts/start_dev.sh to restart."
echo "  Use ./scripts/reset_dev.sh to wipe all data."
