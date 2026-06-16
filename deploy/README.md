# SarabCraft — Deployment Runbook

How to start the SarabCraft app from scratch on a **fresh RunPod**, restart it,
and run it **locally with Docker**. This is the single source of truth for the
demo/viva setup.

> **TL;DR (new pod):** SSH in → run `bootstrap_pod.sh` → open the proxied URL.
> **TL;DR (restart):** SSH in → `bash /workspace/SarabCraft/deploy/run.sh`.
> **TL;DR (check):** `bash /workspace/SarabCraft/deploy/status.sh`.

---

## What the app needs

SarabCraft is a FastAPI backend + React frontend + a job worker, backed by:

| Component  | Purpose                                   | Native port |
|------------|-------------------------------------------|-------------|
| PostgreSQL | jobs, history, results                    | 5432        |
| Redis      | job queue / locks                         | 6379        |
| MinIO      | S3-compatible artifact store (adv images) | 9000 / 9001 |
| API        | FastAPI + serves built React frontend     | 8888        |
| Worker     | runs attacks/benchmarks off the queue     | —           |

All configuration is read from a single env file (`/workspace/sarab.env`).
The exact variable names match `backend/jobs/core.py` and the verifiers — see
`deploy/sarab.env.example`.

---

## A. Fresh RunPod — from scratch

### 1. Create the pod
- Template: **RunPod PyTorch** (e.g. `runpod/pytorch:*-cu*-torch*-ubuntu*`).
  The GPU torch build is pre-installed; the bootstrap reuses it.
- GPU for the demo: **RTX 4090** (best value) or **RTX 5090** (fastest). Attack
  speed is compute-bound (iterations on a single image), so raw GPU speed — not
  VRAM — is what matters.
- Expose **HTTP port 8888** (RunPod proxies it as `https://<POD_ID>-8888.proxy.runpod.net`).

### 2. SSH in
RunPod gives you a direct TCP SSH line on the pod's *Connect* tab, e.g.:
```bash
ssh root@<POD_IP> -p <POD_PORT> -i ~/.ssh/<your_key>
```
> Use the **"SSH over exposed TCP"** line (supports SCP/SFTP), *not* the
> `ssh.runpod.io` proxy line. The IP and port change every time you start a
> new pod — copy them fresh from the dashboard.

### 3. Run the bootstrap
```bash
# copy the bootstrap script up (or just clone the repo and use deploy/bootstrap_pod.sh)
export GIT_REPO=https://github.com/maerifat/SarabCraft.git
bash bootstrap_pod.sh
```
This single script:
1. installs Postgres, Redis, MinIO, ffmpeg, Node 22;
2. clones the repo to `/workspace/SarabCraft`;
3. creates `/workspace/sarab.env` from the template (edit secrets if needed);
4. inits + starts Postgres, creates the role/db; starts Redis; starts MinIO and
   creates the artifact bucket;
5. builds a venv with `--system-site-packages` (so it reuses the pod's CUDA
   torch) and installs the app's Python deps **without** reinstalling torch;
6. builds the React frontend;
7. stops Jupyter (which squats on 8888) and starts the API + worker.

First boot of the API imports models and takes **~60–90s** before it serves.

### 4. Verify + open
```bash
bash /workspace/SarabCraft/deploy/status.sh
```
When `catalog=200`, open `https://<POD_ID>-8888.proxy.runpod.net`.

### 5. Add cloud credentials (optional, for AWS Rekognition demo)
Add them in the UI: **Settings → Models → AWS** (persists to `profiles.json` in
`/var/lib/sarabcraft`). Adding the first profile applies it to the running
process automatically — no restart needed.

> **AWS image format:** `AWS_REKOGNITION_IMAGE_FORMAT` controls how adversarial
> images are sent to Rekognition: `png` (lossless/pixel-perfect, may fail if the
> PNG is >5 MB), `jpeg` (compressed), or `auto` (default: try PNG, fall back to
> JPEG only if it exceeds AWS's 5 MB limit). Local models are unaffected — they
> always use the exact adversarial image.

---

## B. Restart / recover an existing pod

After a reboot, after pulling new code, or if the app stopped:
```bash
bash /workspace/SarabCraft/deploy/run.sh      # restarts services + API + worker
bash /workspace/SarabCraft/deploy/status.sh   # confirm healthy
```
`run.sh` is idempotent: it brings up any down backing service, kills old
API/worker processes, and relaunches them detached (survives SSH disconnect via
`setsid` + `disown`).

### Pull latest code, then restart
```bash
cd /workspace/SarabCraft && git pull --ff-only
REBUILD_FRONTEND=1 bash deploy/run.sh   # rebuild UI only if frontend changed
```

### Logs
```bash
tail -f /workspace/api.log       # API
tail -f /workspace/worker.log    # attacks/benchmarks
tail -f /workspace/minio.log     # MinIO
tail -f /workspace/pg.log        # Postgres
```

---

## C. Local run with Docker (laptop)

The repo ships a full Docker setup that runs everything in containers (CPU torch):
```bash
cd mlsec_app_new
cp .env.example .env        # or use the committed .env for local dev
docker compose up --build
```
- App: <http://localhost:7860>
- MinIO console: <http://localhost:9001>

The Docker path builds the frontend in a Node stage and installs CPU-only torch,
so it works on any laptop without a GPU (attacks just run slower).

---

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `ssh: connect ... Connection refused` | Pod stopped or IP/port changed. Copy fresh SSH details from the RunPod dashboard. |
| API URL shows nothing / 502 | API still importing models (~60–90s on first boot). Re-check `status.sh`. |
| Port 8888 in use | Jupyter is squatting it. `pkill -f jupyter-lab` then `run.sh`. |
| `pip` PEP 668 / "externally managed" | venv must be created with `--system-site-packages` (bootstrap does this). |
| AWS Rekognition shows "Verification failed" | Check `grep -nE 'ERROR\|Rekognition' /workspace/api.log` — the real AWS error is now logged. Common: image >5 MB (use `AWS_REKOGNITION_IMAGE_FORMAT=auto`) or expired session token. |
| Two workers running | A stale worker survived. `pkill -f backend.jobs.worker` then `run.sh`. |

---

## File map

```
deploy/
├── README.md            ← this runbook
├── sarab.env.example    ← env template (copy to /workspace/sarab.env)
├── bootstrap_pod.sh     ← fresh pod, zero → running app
├── run.sh               ← start/restart API + worker (idempotent)
└── status.sh            ← health check for all components
```
