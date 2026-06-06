#!/usr/bin/env bash
# ============================================================
# start_dev.sh — Launch all Docker Compose services + healthcheck
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=============================================="
echo " Enterprise Agentic RAG — Starting Dev Stack"
echo "=============================================="

# Copy .env.example → .env if .env doesn't exist
if [ ! -f .env ]; then
  echo "→ Creating .env from .env.example ..."
  cp .env.example .env
  echo "  ✓ .env created — edit if needed"
else
  echo "→ .env already exists, using it"
fi

# Start services
echo "→ Starting Docker Compose services ..."
docker compose up -d

echo ""
echo "→ Waiting for all services to be healthy ..."
echo "  (this may take 30-60 seconds on first run)"
echo ""

# Run healthcheck
"$SCRIPT_DIR/healthcheck.sh"

echo ""
echo "=============================================="
echo " All services are up!"
echo ""
echo " PostgreSQL  : postgresql://rag_user:rag_password_dev@localhost:5432/enterprise_rag"
echo " Redis       : redis://localhost:6379"
echo " Milvus      : http://localhost:19530 (gRPC) / http://localhost:9091 (health)"
echo " Elasticsearch: http://localhost:9200"
echo " Neo4j       : http://localhost:7474 (Bolt: 7687)"
echo " MinIO       : http://localhost:9000 (API)  / Console http://localhost:9001"
echo " Prometheus  : http://localhost:9090"
echo " Grafana     : http://localhost:3000 (admin / admin_dev)"
echo " OTel Col.   : grpc://localhost:4317"
echo ""
echo " Now start the backend:"
echo "   .venv/bin/python -m uvicorn enterprise_agentic_rag.app.main:app --reload --port 8000"
echo "=============================================="
