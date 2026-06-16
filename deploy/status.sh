#!/usr/bin/env bash
# ============================================================================
# SarabCraft — health/status check for all components
#     bash /workspace/SarabCraft/deploy/status.sh
# ============================================================================
ENV_FILE="${SARAB_ENV_FILE:-/workspace/sarab.env}"
[ -f "$ENV_FILE" ] && { set -a; source "$ENV_FILE"; set +a; }
PORT="${SARAB_PORT:-8888}"
LOG_DIR="${SARAB_LOG_DIR:-/workspace}"

ok()   { echo -e "  \033[1;32m[OK]\033[0m $*"; }
bad()  { echo -e "  \033[1;31m[!!]\033[0m $*"; }

echo "── Backing services ─────────────────────────────"
pg_isready -h "${POSTGRES_HOST:-127.0.0.1}" -p "${POSTGRES_PORT:-5432}" >/dev/null 2>&1 \
    && ok "PostgreSQL up (${POSTGRES_HOST:-127.0.0.1}:${POSTGRES_PORT:-5432})" || bad "PostgreSQL DOWN"
redis-cli ping >/dev/null 2>&1 && ok "Redis up" || bad "Redis DOWN"
curl -fsS http://127.0.0.1:9000/minio/health/live >/dev/null 2>&1 && ok "MinIO up" || bad "MinIO DOWN"

echo "── App processes ────────────────────────────────"
pgrep -f 'backend.main:app'    >/dev/null && ok "API process running"    || bad "API process NOT running"
pgrep -f 'backend.jobs.worker' >/dev/null && ok "Worker process running" || bad "Worker process NOT running"

echo "── API endpoints ────────────────────────────────"
C=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/api/models/catalog")
[ "$C" = "200" ] && ok "API serving (catalog=200)" || bad "API not ready (catalog=$C — still importing? wait ~60s)"
H=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/api/verification/image/heartbeat")
[ "$H" = "200" ] && ok "Verification heartbeat=200" || bad "Verification heartbeat=$H"

echo "── Recent errors (api.log) ──────────────────────"
grep -nE 'ERROR|Traceback' "$LOG_DIR/api.log" 2>/dev/null | tail -5 || echo "  (none)"
echo "Open: https://<POD_ID>-${PORT}.proxy.runpod.net"
