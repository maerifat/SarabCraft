#!/usr/bin/env bash
# ============================================================================
# SarabCraft — start / restart the API + worker (assumes services already up)
# ----------------------------------------------------------------------------
# Use this after a reboot, after pulling new code, or whenever the app stops.
# It (re)starts Postgres/Redis/MinIO if they are down, then relaunches the
# uvicorn API and the job worker, detached so they survive SSH disconnect.
#
# USAGE (on the pod):
#     bash /workspace/SarabCraft/deploy/run.sh
# ============================================================================
set -euo pipefail

ENV_FILE="${SARAB_ENV_FILE:-/workspace/sarab.env}"
set -a; source "$ENV_FILE"; set +a

REPO="${SARAB_REPO:-/workspace/SarabCraft}"
VENV="${SARAB_VENV:-/workspace/venv}"
PORT="${SARAB_PORT:-8888}"
LOG_DIR="${SARAB_LOG_DIR:-/workspace}"
PGDATA="${PGDATA:-/workspace/pgdata}"
MINIO_DATA="${MINIO_DATA:-/workspace/minio-data}"

log() { echo -e "\033[1;32m[run]\033[0m $*"; }

# ── Ensure backing services are alive (in case the pod rebooted) ───────────
if ! pg_isready -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" >/dev/null 2>&1; then
    log "Starting PostgreSQL..."
    PG_BIN="$(dirname "$(ls /usr/lib/postgresql/*/bin/postgres | head -1)")"
    su postgres -c "$PG_BIN/pg_ctl -D $PGDATA -l $LOG_DIR/pg.log -o '-p ${POSTGRES_PORT}' -w start" || true
fi
if ! redis-cli ping >/dev/null 2>&1; then
    log "Starting Redis..."
    redis-server --daemonize yes --appendonly yes || true
fi
if ! curl -fsS http://127.0.0.1:9000/minio/health/live >/dev/null 2>&1; then
    log "Starting MinIO..."
    MINIO_ROOT_USER="$MINIO_ROOT_USER" MINIO_ROOT_PASSWORD="$MINIO_ROOT_PASSWORD" \
        setsid minio server "$MINIO_DATA" --address ":9000" --console-address ":9001" \
        >> "$LOG_DIR/minio.log" 2>&1 < /dev/null & disown
fi

# ── Stop any existing API/worker ───────────────────────────────────────────
log "Stopping existing API/worker (if any)..."
pkill -f 'backend.main:app'     2>/dev/null || true
pkill -f 'backend.jobs.worker'  2>/dev/null || true
sleep 2

# ── Launch API (detached) ──────────────────────────────────────────────────
log "Launching API on port $PORT (logs: $LOG_DIR/api.log)..."
setsid bash -c "cd '$REPO'; set -a; source '$ENV_FILE'; set +a; \
    exec '$VENV/bin/uvicorn' backend.main:app --host 0.0.0.0 --port $PORT \
    >> '$LOG_DIR/api.log' 2>&1" < /dev/null > /dev/null 2>&1 & disown

# ── Launch worker (detached) ───────────────────────────────────────────────
log "Launching worker (logs: $LOG_DIR/worker.log)..."
setsid bash -c "cd '$REPO'; set -a; source '$ENV_FILE'; set +a; \
    exec '$VENV/bin/python' -m backend.jobs.worker \
    >> '$LOG_DIR/worker.log' 2>&1" < /dev/null > /dev/null 2>&1 & disown

log "Launched. The API imports models on first boot (~60-90s) before it serves."
log "Watch readiness:  bash $REPO/deploy/status.sh"
