#!/usr/bin/env bash
# ============================================================
# healthcheck.sh — Verify all Docker services are reachable
# Covers all 9 services: PG, Redis, Milvus, MinIO, ES, Neo4j,
# Prometheus, Grafana, OTel Collector
# ============================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass=0
fail=0

check() {
  local name="$1"
  local cmd="$2"
  local url="${3:-}"

  printf "  %-20s ... " "$name"
  if eval "$cmd" &>/dev/null; then
    echo -e "${GREEN}✓ OK${NC}"
    ((pass++))
  else
    echo -e "${RED}✗ FAIL${NC}"
    if [ -n "$url" ]; then
      echo -e "    ${YELLOW}→ $url${NC}"
    fi
    ((fail++))
  fi
}

echo ""
echo "=============================================="
echo " Enterprise Agentic RAG — Health Check"
echo "=============================================="
echo ""

# --- PostgreSQL ---
check "PostgreSQL" \
  "pg_isready -h localhost -p ${POSTGRES_PORT:-5432} -U ${POSTGRES_USER:-rag_user} -d ${POSTGRES_DB:-enterprise_rag} -q" \
  "psql -h localhost -p 5432 -U rag_user -d enterprise_rag"

# --- Redis ---
check "Redis" \
  "redis-cli -h localhost -p ${REDIS_PORT:-6379} ping" \
  "redis-cli -h localhost -p 6379 ping"

# --- Milvus ---
check "Milvus" \
  "curl -fsS http://localhost:9091/healthz" \
  "http://localhost:9091/healthz (internal mgmt port)"

# --- MinIO ---
check "MinIO" \
  "curl -sf http://localhost:${MINIO_PORT:-9000}/minio/health/live" \
  "http://localhost:9001 (Console)"

# --- Elasticsearch ---
check "Elasticsearch" \
  "curl -fsS http://localhost:${ES_PORT:-9200}/_cluster/health" \
  "http://localhost:9200"

# --- Neo4j ---
check "Neo4j" \
  "curl -fsS -u ${NEO4J_USER:-neo4j}:${NEO4J_PASSWORD:-password} http://localhost:7474" \
  "http://localhost:7474 (Bolt: 7687)"

# --- Prometheus ---
check "Prometheus" \
  "curl -sf http://localhost:${PROMETHEUS_PORT:-9090}/-/healthy" \
  "http://localhost:9090"

# --- Grafana ---
check "Grafana" \
  "curl -sf http://localhost:${GRAFANA_PORT:-3000}/api/health" \
  "http://localhost:3000 (admin / admin_dev)"

# --- OTel Collector ---
check "OTel Collector" \
  "wget --spider -q http://localhost:13133/" \
  "http://localhost:13133/ (internal health endpoint)"

echo ""
echo "----------------------------------------------"
echo -e "  Passed: ${GREEN}${pass}${NC}  Failed: ${RED}${fail}${NC}"
echo "----------------------------------------------"

if [ "$fail" -gt 0 ]; then
  echo ""
  echo "Some services are not healthy. Try:"
  echo "  docker compose ps        # check container status"
  echo "  docker compose logs -f    # tail logs"
  exit 1
else
  echo "✓ All services healthy!"
  exit 0
fi
