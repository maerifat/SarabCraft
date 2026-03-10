"""
SarabCraft — FastAPI Backend

REST API for image/audio adversarial attacks, transfer verification,
and configuration. Replaces Gradio with a clean API for the React frontend.
"""

import logging
import os
import sys
import uuid

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load credentials at startup
from utils.credentials import apply_credentials
apply_credentials()

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from backend.routes import attacks, verification, config, tts, audio_attacks, plugins, variables, models
from backend.routes import history, explainability
from backend.routes import batch
from backend.routes import benchmark
from backend.jobs import routes as job_routes
from backend.jobs.core import initialize_runtime, ping_artifact_storage, ping_database, ping_redis
from backend.models.registry import initialize_model_registry, list_source_models

# ── Structured Logging ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("mlsec")


# ── Request ID Middleware ─────────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app = FastAPI(
    title="SarabCraft",
    description="""
## Crafting illusions that machines believe.

**SarabCraft** is a multimodal adversarial AI research framework for crafting inputs that look,
sound, or read as intended to humans while deceiving machines into perceiving something else.
It enables researchers to generate, measure, and validate adversarial examples across diverse
AI models and systems.

### Capabilities
- **32+ Image Attack Algorithms** — FGSM, PGD, C&W, AutoAttack, SarabCraft R1, and more
- **8 Audio Attack Types** — Including ASR transcription, hidden commands, and psychoacoustic masking
- **Transfer Verification** — Test adversarial examples against cloud APIs (AWS, Azure, GCP)
- **Plugin System** — Extend with custom Python classifiers
- **Explainability** — GradCAM attention overlays
- **Perturbation Metrics** — L0, L1, L2, L∞, SSIM, PSNR

### Links
- [Black Hat Arsenal](https://www.blackhat.com/arsenal.html)
- [GitHub Repository](https://github.com/mlsec-lab/mlsec)
    """,
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    contact={"name": "SarabCraft", "url": "https://github.com/mlsec-lab/mlsec"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
)

app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# ── Global Exception Handler ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "Unhandled exception | request_id=%s method=%s path=%s error=%s",
        request_id, request.method, request.url.path, exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred", "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


@app.on_event("startup")
def startup_runtime():
    try:
        initialize_runtime()
        initialize_model_registry()
    except Exception:
        logger.exception("Failed to initialize persistence runtime during startup")

app.include_router(attacks.router, prefix="/api/attacks", tags=["Image Attacks"])
app.include_router(audio_attacks.router, prefix="/api/attacks", tags=["Audio Attacks"])
app.include_router(verification.router, prefix="/api/verification", tags=["Transfer Verification"])
app.include_router(config.router, prefix="/api/config", tags=["Credentials & Configuration"])
app.include_router(models.router, prefix="/api/models", tags=["Models Registry"])
app.include_router(tts.router, prefix="/api/tts", tags=["Text-to-Speech"])
app.include_router(plugins.router, prefix="/api/plugins", tags=["Plugin System"])
app.include_router(variables.router, prefix="/api/variables", tags=["Global Variables"])
app.include_router(history.router, prefix="/api/history", tags=["Attack History"])
app.include_router(explainability.router, prefix="/api/explainability", tags=["Explainability (GradCAM)"])
app.include_router(batch.router, prefix="/api/attacks", tags=["Batch & Robustness"])
app.include_router(benchmark.router, prefix="/api/attacks", tags=["Attack Benchmark"])
app.include_router(job_routes.router, prefix="/api/jobs", tags=["Jobs"])


@app.get("/api/health", tags=["System"])
def health():
    """Health check — GPU, persistence, artifact storage, plugin directory."""
    import torch
    from plugins._base import PLUGINS_DIR

    checks = {}

    checks["gpu"] = {
        "available": torch.cuda.is_available(),
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    checks["postgres"] = ping_database()
    checks["redis"] = ping_redis()
    checks["artifact_storage"] = ping_artifact_storage()
    checks["history_backend"] = "postgres"

    checks["plugin_directory"] = {
        "exists": PLUGINS_DIR.is_dir(),
        "path": str(PLUGINS_DIR),
    }

    overall = (
        checks["postgres"]
        and checks["redis"]
        and checks["artifact_storage"]
        and checks["plugin_directory"]["exists"]
    )
    return {"status": "ok" if overall else "degraded", "checks": checks}


@app.get("/api/system/info", tags=["System"])
def system_info():
    """System information — GPU availability, loaded models, etc."""
    import torch
    from config import device, ATTACK_REGISTRY
    return {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "image_models": len(list_source_models("image", task="image_classification")),
        "image_attacks": len(ATTACK_REGISTRY),
        "audio_models": len(list_source_models("audio", task="audio_classification")),
        "asr_models": len(list_source_models("audio", task="asr")),
    }


# Serve React build (production) — must be last so API routes match first
_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=_frontend_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def serve_spa(path: str):
        from fastapi.responses import FileResponse
        if path.startswith("api"):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        fp = Path(_frontend_dist) / path
        if path and fp.is_file():
            return FileResponse(fp)
        return FileResponse(_frontend_dist / "index.html")
