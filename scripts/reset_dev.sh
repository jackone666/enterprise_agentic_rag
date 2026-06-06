#!/usr/bin/env bash
# ============================================================
# reset_dev.sh — Tear down EVERYTHING and start fresh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "⚠️  This will DELETE ALL Docker volumes and data!"
read -rp "   Are you sure? Type 'yes' to continue: " confirm
if [ "$confirm" != "yes" ]; then
  echo "Cancelled."
  exit 0
fi

echo "→ Stopping services and removing volumes ..."
docker compose down -v

echo "→ Cleaning local data directories ..."
rm -rf data/db data/logs/events.jsonl data/eval/failed_cases.jsonl 2>/dev/null || true

echo "→ Re-creating .env ..."
cp .env.example .env
echo "  ✓ .env reset"

echo "→ Starting fresh stack ..."
docker compose up -d

echo ""
echo "→ Running healthcheck ..."
sleep 5
"$SCRIPT_DIR/healthcheck.sh"

echo ""
echo "✓ Reset complete! All data wiped, services are fresh."
