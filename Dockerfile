# ── Stage 1: Build React frontend ──────────────────────────────────────
FROM node:22-slim AS frontend-build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime ───────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (heaviest layer — cached unless requirements change)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader -d /usr/local/nltk_data wordnet omw-1.4

# Shared library modules (at /app/ so sys.path hacks resolve correctly)
COPY config.py ./config.py
COPY attacks/ ./attacks/
COPY models/ ./models/
COPY utils/ ./utils/
COPY verification/ ./verification/
COPY lab/ ./lab/
COPY plugins/ ./plugins/

# Backend & frontend at /app/ — flat layout matching the repo structure
COPY backend/ ./backend/
COPY --from=frontend-build /build/dist/ ./frontend/dist/

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
