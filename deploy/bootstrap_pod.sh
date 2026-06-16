#!/usr/bin/env bash
# ============================================================================
# SarabCraft — FROM-SCRATCH bootstrap for a fresh RunPod (RunPod PyTorch image)
# ----------------------------------------------------------------------------
# Brings a brand-new pod from zero to a running app: installs Postgres, Redis,
# MinIO, clones the repo, builds the venv (reusing the pod's pre-installed CUDA
# torch), builds the React frontend, initializes the artifact bucket, then
# starts the API + worker via deploy/run.sh.
#
# Designed for: runpod/pytorch:*-cu*-torch*-ubuntu* (root, /workspace volume).
# Idempotent: safe to re-run; it skips steps already done.
#
# USAGE (on the pod, over SSH):
#     export GIT_REPO=https://github.com/maerifat/SarabCraft.git
#     bash bootstrap_pod.sh
#
# After it finishes, the app is reachable on the pod's proxied port 8888:
#     https://<POD_ID>-8888.proxy.runpod.net
# ============================================================================
set -euo pipefail

# ── Config (override via env before running) ───────────────────────────────
GIT_REPO="${GIT_REPO:-https://github.com/maerifat/SarabCraft.git}"
REPO="${SARAB_REPO:-/workspace/SarabCraft}"
VENV="${SARAB_VENV:-/workspace/venv}"
ENV_FILE="${SARAB_ENV_FILE:-/workspace/sarab.env}"
PORT="${SARAB_PORT:-8888}"
PGDATA="${PGDATA:-/workspace/pgdata}"
MINIO_DATA="${MINIO_DATA:-/workspace/minio-data}"
STATE_DIR="${SARABCRAFT_STATE_DIR:-/var/lib/sarabcraft}"

log() { echo -e "\n\033[1;36m[bootstrap]\033[0m $*"; }

# ── 1. System packages: postgres, redis, build tools ───────────────────────
log "Installing system packages (postgresql, redis, ffmpeg, libsndfile, nodejs)..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    postgresql postgresql-client redis-server \
    ffmpeg libsndfile1 git curl ca-certificates

# Node 22 (for the frontend build) if not present
if ! command -v node >/dev/null 2>&1 || [ "$(node -v | sed 's/v\([0-9]*\).*/\1/')" -lt 18 ]; then
    log "Installing Node.js 22..."
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y nodejs
fi

# ── 2. MinIO binary (S3-compatible artifact store) ─────────────────────────
if [ ! -x /usr/local/bin/minio ]; then
    log "Downloading MinIO server + client..."
    curl -fsSL https://dl.min.io/server/minio/release/linux-amd64/minio -o /usr/local/bin/minio
    curl -fsSL https://dl.min.io/client/mc/release/linux-amd64/mc -o /usr/local/bin/mc
    chmod +x /usr/local/bin/minio /usr/local/bin/mc
fi

# ── 3. Clone (or update) the repo ──────────────────────────────────────────
if [ ! -d "$REPO/.git" ]; then
    log "Cloning $GIT_REPO -> $REPO ..."
    git clone "$GIT_REPO" "$REPO"
else
    log "Repo exists; pulling latest..."
    git -C "$REPO" pull --ff-only || log "(pull skipped — local changes present)"
fi

# ── 4. Env file ────────────────────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    log "Creating $ENV_FILE from template (EDIT secrets if needed)..."
    cp "$REPO/deploy/sarab.env.example" "$ENV_FILE"
fi
set -a; source "$ENV_FILE"; set +a

mkdir -p "$STATE_DIR" "$PGDATA" "$MINIO_DATA" "$SARAB_LOG_DIR"
chmod 700 "$STATE_DIR"

# ── 5. PostgreSQL: init cluster, start, create role+db ─────────────────────
log "Configuring PostgreSQL..."
PG_BIN="$(dirname "$(ls /usr/lib/postgresql/*/bin/postgres | head -1)")"
if [ ! -s "$PGDATA/PG_VERSION" ]; then
    chown -R postgres:postgres "$PGDATA"
    su postgres -c "$PG_BIN/initdb -D $PGDATA"
fi
# Start postgres (detached) if not already listening
if ! pg_isready -h 127.0.0.1 -p "${POSTGRES_PORT:-5432}" >/dev/null 2>&1; then
    su postgres -c "$PG_BIN/pg_ctl -D $PGDATA -l $SARAB_LOG_DIR/pg.log -o '-p ${POSTGRES_PORT:-5432}' -w start"
fi
# Create role + database (ignore 'already exists')
su postgres -c "psql -p ${POSTGRES_PORT:-5432} -tc \"SELECT 1 FROM pg_roles WHERE rolname='${POSTGRES_USER}'\"" \
    | grep -q 1 || su postgres -c "psql -p ${POSTGRES_PORT:-5432} -c \"CREATE ROLE ${POSTGRES_USER} LOGIN PASSWORD '${POSTGRES_PASSWORD}';\""
su postgres -c "psql -p ${POSTGRES_PORT:-5432} -tc \"SELECT 1 FROM pg_database WHERE datname='${POSTGRES_DB}'\"" \
    | grep -q 1 || su postgres -c "psql -p ${POSTGRES_PORT:-5432} -c \"CREATE DATABASE ${POSTGRES_DB} OWNER ${POSTGRES_USER};\""

# ── 6. Redis ────────────────────────────────────────────────────────────────
log "Starting Redis..."
if ! redis-cli ping >/dev/null 2>&1; then
    redis-server --daemonize yes --appendonly yes
fi

# ── 7. MinIO + bucket ───────────────────────────────────────────────────────
log "Starting MinIO + creating artifact bucket..."
if ! curl -fsS http://127.0.0.1:9000/minio/health/live >/dev/null 2>&1; then
    MINIO_ROOT_USER="$MINIO_ROOT_USER" MINIO_ROOT_PASSWORD="$MINIO_ROOT_PASSWORD" \
        setsid minio server "$MINIO_DATA" --address ":9000" --console-address ":9001" \
        >> "$SARAB_LOG_DIR/minio.log" 2>&1 < /dev/null & disown
    sleep 5
fi
mc alias set local http://127.0.0.1:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null 2>&1 || true
mc mb --ignore-existing "local/${ARTIFACT_BUCKET}" >/dev/null 2>&1 || true

# ── 8. Python venv (reuse the pod's pre-installed CUDA torch) ──────────────
log "Building Python venv (inherits system CUDA torch)..."
if [ ! -d "$VENV" ]; then
    python3 -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/pip" install --upgrade pip wheel >/dev/null
# Install app deps but DO NOT reinstall torch (the pod already has the CUDA build)
grep -ivE '^(torch|torchvision|torchaudio)([=<>!~ ]|$)' "$REPO/requirements.txt" > /tmp/reqs_no_torch.txt
"$VENV/bin/pip" install --no-cache-dir -r /tmp/reqs_no_torch.txt
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ── 9. Frontend build ───────────────────────────────────────────────────────
log "Building React frontend..."
if [ ! -d "$REPO/frontend/dist" ] || [ "${REBUILD_FRONTEND:-0}" = "1" ]; then
    ( cd "$REPO/frontend" && npm ci && npm run build )
fi

# ── 10. Free the proxied port (RunPod runs Jupyter on 8888 by default) ─────
log "Freeing port $PORT (stopping Jupyter if present)..."
pkill -f jupyter-lab 2>/dev/null || true
sleep 1

# ── 11. Start the app ───────────────────────────────────────────────────────
log "Starting API + worker..."
bash "$REPO/deploy/run.sh"

log "DONE. Open: https://<POD_ID>-${PORT}.proxy.runpod.net"
log "Check status anytime with:  bash $REPO/deploy/status.sh"
