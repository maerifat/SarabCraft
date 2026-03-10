import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from backend.jobs.core import (
    append_job_event,
    create_job,
    enqueue_job,
    get_job,
    list_jobs,
    replace_job_request,
    request_job_cancel,
    resume_job,
    store_bytes_artifact,
)
from backend.jobs.handlers import get_job_definition
from backend.models.registry import enrich_job_fields
from utils.attack_names import normalize_attack_payload

router = APIRouter()
logger = logging.getLogger("mlsec.jobs.routes")

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_IMAGE_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_AUDIO_UPLOAD_BYTES = 50 * 1024 * 1024

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}


def _append_field(target: dict[str, Any], key: str, value: Any) -> None:
    if key not in target:
        target[key] = value
        return
    current = target[key]
    if isinstance(current, list):
        current.append(value)
    else:
        target[key] = [current, value]


def _file_policy(kind: str, fields: dict[str, Any]) -> dict[str, str]:
    if kind in {"image_attack", "image_robustness"}:
        return {"input_file": "image", "target_file": "image"}
    if kind == "batch_attack":
        return {"input_files": "image", "target_file": "image"}
    if kind == "benchmark":
        if str(fields.get("domain", "image")) == "audio":
            return {"input_file": "audio"}
        return {"input_file": "image", "target_file": "image"}
    if kind == "audio_classification":
        return {"audio_file": "audio"}
    if kind == "asr_hidden_command":
        return {"carrier_file": "audio"}
    if kind in {
        "asr_transcription",
        "asr_universal_muting",
        "asr_psychoacoustic",
        "asr_over_the_air",
        "asr_speech_jamming",
        "asr_ua3",
        "audio_robustness",
    }:
        return {"audio_file": "audio"}
    return {}


def _file_kind_matches(expected: str, filename: str | None, content_type: str | None) -> bool:
    ext = Path(filename or "").suffix.lower()
    mime = (content_type or "").lower()
    if expected == "image":
        return mime.startswith("image/") or ext in IMAGE_EXTENSIONS
    if expected == "audio":
        return mime.startswith("audio/") or ext in AUDIO_EXTENSIONS
    return False


def _max_size_for(expected: str) -> int:
    if expected == "image":
        return MAX_IMAGE_UPLOAD_BYTES
    if expected == "audio":
        return MAX_AUDIO_UPLOAD_BYTES
    return MAX_UPLOAD_BYTES


@router.get("")
def list_jobs_route(
    limit: int = Query(50, ge=1, le=200),
    status: str | None = Query(None),
):
    return {"jobs": list_jobs(limit=limit, status=status)}


@router.get("/{job_id}")
def get_job_route(
    job_id: str,
    after_event_id: int = Query(0, ge=0),
):
    job = get_job(job_id, include_events=True, after_event_id=after_event_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.post("/{job_id}/cancel")
def cancel_job_route(job_id: str):
    existing = get_job(job_id)
    if not existing:
        raise HTTPException(404, "Job not found")
    if existing.get("status") not in {"queued", "running"}:
        raise HTTPException(409, "Only queued or running jobs can be cancelled")
    job = request_job_cancel(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    logger.info("Job cancellation requested | job_id=%s status=%s", job_id, job.get("status"))
    return job


@router.post("/{job_id}/resume")
def resume_job_route(job_id: str):
    existing = get_job(job_id)
    if not existing:
        raise HTTPException(404, "Job not found")
    if not existing.get("resume_supported"):
        raise HTTPException(400, "This job type does not support resume")
    normalized_request = normalize_attack_payload(existing.get("request") or {})
    replace_job_request(job_id, normalized_request)
    job = resume_job(job_id)
    if not job:
        raise HTTPException(400, "Only failed or cancelled jobs can be resumed")
    logger.info("Job resumed | job_id=%s", job_id)
    return job


@router.post("/submit/{kind}")
async def submit_job(kind: str, request: Request):
    definition = get_job_definition(kind)
    if not definition:
        raise HTTPException(400, f"Unsupported job kind: {kind}")

    try:
        form = await request.form()
    except Exception as exc:
        logger.warning("Invalid multipart payload for kind=%s: %s", kind, exc)
        raise HTTPException(400, "Invalid multipart form payload") from exc

    fields: dict[str, Any] = {}
    uploads: list[tuple[str, Any]] = []
    for key, value in form.multi_items():
        if hasattr(value, "filename") and hasattr(value, "read"):
            uploads.append((key, value))
        else:
            _append_field(fields, key, value)

    domain = definition["domain"]
    if kind == "benchmark":
        domain = str(fields.get("domain", "image"))
    policy = _file_policy(kind, fields)

    draft = create_job(
        kind=kind,
        domain=domain,
        title=definition["title"],
        request_payload={"fields": {}, "files": {}},
        resume_supported=definition["resume_supported"],
        enqueue_immediately=False,
    )
    job_id = draft["job_id"]
    files: dict[str, Any] = {}

    try:
        for key, upload in uploads:
            data = await upload.read()
            expected_kind = policy.get(key)
            if not expected_kind:
                raise HTTPException(400, f"Unexpected upload field: {key}")
            if not _file_kind_matches(expected_kind, upload.filename, upload.content_type):
                raise HTTPException(400, f"Unsupported {expected_kind} upload for field: {key}")
            max_bytes = _max_size_for(expected_kind)
            if len(data) > max_bytes:
                raise HTTPException(400, f"{upload.filename or key} exceeds upload limit")
            artifact = store_bytes_artifact(
                job_id,
                f"input-{key}",
                data,
                filename=upload.filename,
                mime_type=upload.content_type or "application/octet-stream",
                metadata={"field": key},
            )
            file_ref = {
                "artifact_id": artifact["id"],
                "storage_key": artifact["storage_key"],
                "filename": upload.filename,
                "mime_type": upload.content_type,
                "size_bytes": artifact["size_bytes"],
            }
            if key in files:
                current = files[key]
                if isinstance(current, list):
                    current.append(file_ref)
                else:
                    files[key] = [current, file_ref]
            else:
                files[key] = file_ref

        missing = [field_name for field_name in policy if field_name not in files]
        if missing:
            raise HTTPException(400, f"Missing required upload field(s): {', '.join(missing)}")

        enriched_fields = enrich_job_fields(kind, fields)
        replace_job_request(job_id, {"fields": enriched_fields, "files": files})
        enqueue_job(job_id)
        append_job_event(job_id, "queued", {"message": "Job queued"})
        logger.info(
            "Job created | job_id=%s kind=%s domain=%s files=%s ip=%s",
            job_id,
            kind,
            domain,
            sorted(files.keys()),
            request.client.host if request.client else "unknown",
        )
        job = get_job(job_id)
        if not job:
            raise HTTPException(500, "Failed to persist job")
        return job
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to create job %s (%s)", job_id, kind)
        raise HTTPException(500, "Failed to create job") from exc
