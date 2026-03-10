import json
import logging
import os
import posixpath
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import boto3
import pg8000.dbapi
import redis
from botocore.config import Config as BotoConfig
from utils.attack_names import normalize_attack_payload

logger = logging.getLogger("mlsec.jobs")

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_CANCELLED = "cancelled"

QUEUE_NAME = "sarabcraft:jobs:queue"

JSON_COLUMNS = {
    "request_json",
    "result_json",
    "checkpoint_json",
    "payload_json",
    "metadata_json",
    "entry_json",
    "settings_json",
    "compatibility_json",
    "aliases_json",
    "item_ids_json",
}

SCHEMA_LOCK_KEY = 730001


@dataclass(frozen=True)
class JobsSettings:
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "sarabcraft")
    postgres_user: str = os.getenv("POSTGRES_USER", "sarabcraft")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "sarabcraft")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    artifact_bucket: str = os.getenv("ARTIFACT_BUCKET", "sarabcraft-artifacts")
    artifact_endpoint_url: Optional[str] = os.getenv("S3_ENDPOINT_URL") or None
    artifact_region: str = os.getenv("ARTIFACT_REGION", os.getenv("AWS_REGION", "us-east-1"))
    artifact_access_key_id: Optional[str] = os.getenv("ARTIFACT_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID") or None
    artifact_secret_access_key: Optional[str] = os.getenv("ARTIFACT_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY") or None
    artifact_force_path_style: bool = os.getenv("ARTIFACT_FORCE_PATH_STYLE", "true").lower() in {"1", "true", "yes", "on"}


def get_settings() -> JobsSettings:
    return JobsSettings()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def _maybe_json(value: Any) -> Any:
    if isinstance(value, (dict, list)) or value is None:
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _rows_to_dicts(cursor, rows) -> list[dict]:
    columns = [desc[0] for desc in cursor.description or []]
    out = []
    for row in rows:
        item = {}
        for key, value in zip(columns, row):
            item[key] = _maybe_json(value) if key in JSON_COLUMNS else value
        out.append(item)
    return out


def get_db_connection(app_name: str = "sarabcraft-api"):
    settings = get_settings()
    return pg8000.dbapi.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        timeout=10,
        application_name=app_name,
    )


@contextmanager
def db_cursor(app_name: str = "sarabcraft-api"):
    conn = get_db_connection(app_name=app_name)
    try:
        cursor = conn.cursor()
        yield conn, cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def fetch_one(query: str, params: Iterable[Any] | None = None, app_name: str = "sarabcraft-api") -> Optional[dict]:
    with db_cursor(app_name=app_name) as (_, cursor):
        cursor.execute(query, tuple(params or ()))
        rows = cursor.fetchall()
        parsed = _rows_to_dicts(cursor, rows)
        return parsed[0] if parsed else None


def fetch_all(query: str, params: Iterable[Any] | None = None, app_name: str = "sarabcraft-api") -> list[dict]:
    with db_cursor(app_name=app_name) as (_, cursor):
        cursor.execute(query, tuple(params or ()))
        return _rows_to_dicts(cursor, cursor.fetchall())


def execute(query: str, params: Iterable[Any] | None = None, app_name: str = "sarabcraft-api") -> None:
    with db_cursor(app_name=app_name) as (_, cursor):
        cursor.execute(query, tuple(params or ()))


def get_redis_client() -> redis.Redis:
    settings = get_settings()
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def get_s3_client():
    settings = get_settings()
    return boto3.client(
        "s3",
        region_name=settings.artifact_region,
        endpoint_url=settings.artifact_endpoint_url,
        aws_access_key_id=settings.artifact_access_key_id,
        aws_secret_access_key=settings.artifact_secret_access_key,
        config=BotoConfig(
            s3={"addressing_style": "path" if settings.artifact_force_path_style else "auto"},
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def ensure_artifact_bucket() -> None:
    settings = get_settings()
    client = get_s3_client()
    try:
        client.head_bucket(Bucket=settings.artifact_bucket)
        return
    except Exception:
        logger.info("Creating artifact bucket %s", settings.artifact_bucket)

    kwargs = {"Bucket": settings.artifact_bucket}
    if not settings.artifact_endpoint_url and settings.artifact_region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": settings.artifact_region}
    client.create_bucket(**kwargs)


def ping_database() -> bool:
    try:
        row = fetch_one("SELECT 1 AS ok", app_name="sarabcraft-health")
        return bool(row and row.get("ok") == 1)
    except Exception:
        return False


def ping_redis() -> bool:
    try:
        return bool(get_redis_client().ping())
    except Exception:
        return False


def ping_artifact_storage() -> bool:
    try:
        ensure_artifact_bucket()
        return True
    except Exception:
        return False


def init_schema() -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            domain TEXT NOT NULL,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            request_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            result_json JSONB,
            checkpoint_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            progress_current INTEGER NOT NULL DEFAULT 0,
            progress_total INTEGER NOT NULL DEFAULT 1,
            progress_message TEXT,
            cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
            resume_supported BOOLEAN NOT NULL DEFAULT FALSE,
            error_message TEXT,
            worker_id TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS jobs_kind_idx ON jobs (kind, created_at DESC)",
        """
        CREATE TABLE IF NOT EXISTS job_artifacts (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            filename TEXT,
            storage_key TEXT NOT NULL,
            mime_type TEXT,
            size_bytes BIGINT NOT NULL DEFAULT 0,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS job_artifacts_job_idx ON job_artifacts (job_id, created_at ASC)",
        """
        CREATE TABLE IF NOT EXISTS job_events (
            id BIGSERIAL PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            event_type TEXT NOT NULL,
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS job_events_job_idx ON job_events (job_id, id ASC)",
        """
        CREATE TABLE IF NOT EXISTS history_entries (
            id TEXT PRIMARY KEY,
            entry_timestamp DOUBLE PRECISION NOT NULL,
            domain TEXT,
            attack TEXT,
            model TEXT,
            success BOOLEAN,
            entry_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS history_entries_timestamp_idx ON history_entries (entry_timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS history_entries_domain_idx ON history_entries (domain, entry_timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS history_entries_attack_idx ON history_entries (attack, entry_timestamp DESC)",
    ]
    with db_cursor(app_name="sarabcraft-init") as (_, cursor):
        # Serialize DDL so API and worker startup cannot race table type creation.
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", (SCHEMA_LOCK_KEY,))
        for statement in statements:
            cursor.execute(statement)


def initialize_runtime() -> None:
    init_schema()
    ensure_artifact_bucket()


def _artifact_key(job_id: str, role: str, filename: Optional[str]) -> str:
    safe_name = Path(filename or f"{role}.bin").name
    return posixpath.join("jobs", job_id, role, f"{uuid.uuid4().hex}-{safe_name}")


def store_bytes_artifact(
    job_id: str,
    role: str,
    data: bytes,
    *,
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    settings = get_settings()
    storage_key = _artifact_key(job_id, role, filename)
    client = get_s3_client()
    extra = {}
    if mime_type:
        extra["ContentType"] = mime_type
    client.put_object(Bucket=settings.artifact_bucket, Key=storage_key, Body=data, **extra)

    artifact_id = uuid.uuid4().hex
    row = fetch_one(
        """
        INSERT INTO job_artifacts (
            id, job_id, role, filename, storage_key, mime_type, size_bytes, metadata_json
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, CAST(%s AS JSONB))
        RETURNING *
        """,
        (
            artifact_id,
            job_id,
            role,
            filename,
            storage_key,
            mime_type,
            len(data),
            _json_dumps(metadata or {}),
        ),
    )
    return row or {
        "id": artifact_id,
        "job_id": job_id,
        "role": role,
        "filename": filename,
        "storage_key": storage_key,
        "mime_type": mime_type,
        "size_bytes": len(data),
        "metadata_json": metadata or {},
    }


def load_artifact_bytes(storage_key: str) -> bytes:
    settings = get_settings()
    client = get_s3_client()
    obj = client.get_object(Bucket=settings.artifact_bucket, Key=storage_key)
    return obj["Body"].read()


def list_job_artifacts(job_id: str) -> list[dict]:
    return fetch_all(
        "SELECT * FROM job_artifacts WHERE job_id = %s ORDER BY created_at ASC",
        (job_id,),
    )


def create_job(
    *,
    kind: str,
    domain: str,
    title: str,
    request_payload: dict,
    resume_supported: bool = False,
    enqueue_immediately: bool = True,
) -> dict:
    job_id = uuid.uuid4().hex
    row = fetch_one(
        """
        INSERT INTO jobs (
            id, kind, domain, title, status, request_json, resume_supported
        )
        VALUES (%s, %s, %s, %s, %s, CAST(%s AS JSONB), %s)
        RETURNING *
        """,
        (
            job_id,
            kind,
            domain,
            title,
            JOB_STATUS_QUEUED,
            _json_dumps(request_payload),
            resume_supported,
        ),
    )
    if enqueue_immediately:
        enqueue_job(job_id)
        append_job_event(job_id, "queued", {"message": "Job queued"})
    return hydrate_job(row or {"id": job_id, "kind": kind, "domain": domain, "title": title, "status": JOB_STATUS_QUEUED})


def replace_job_request(job_id: str, request_payload: dict) -> Optional[dict]:
    row = fetch_one(
        """
        UPDATE jobs
        SET request_json = CAST(%s AS JSONB),
            updated_at = NOW()
        WHERE id = %s
        RETURNING *
        """,
        (_json_dumps(request_payload), job_id),
    )
    return hydrate_job(row) if row else None


def hydrate_job(row: Optional[dict], *, include_events: bool = False, after_event_id: int = 0) -> Optional[dict]:
    if row is None:
        return None
    job = {
        "job_id": row["id"],
        "kind": row["kind"],
        "domain": row["domain"],
        "title": row["title"],
        "status": row["status"],
        "progress": {
            "current": row.get("progress_current", 0),
            "total": row.get("progress_total", 1),
            "message": row.get("progress_message"),
            "percent": round(
                ((row.get("progress_current", 0) / row.get("progress_total", 1)) * 100)
                if row.get("progress_total", 1)
                else 0,
                2,
            ),
        },
        "request": normalize_attack_payload(row.get("request_json") or {}),
        "result": normalize_attack_payload(row.get("result_json")),
        "checkpoint": normalize_attack_payload(row.get("checkpoint_json") or {}),
        "cancel_requested": bool(row.get("cancel_requested")),
        "resume_supported": bool(row.get("resume_supported")),
        "error_message": row.get("error_message"),
        "worker_id": row.get("worker_id"),
        "attempts": row.get("attempts", 0),
        "created_at": row.get("created_at"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "updated_at": row.get("updated_at"),
        "artifacts": list_job_artifacts(row["id"]),
    }
    if include_events:
        job["events"] = normalize_attack_payload(get_job_events(row["id"], after_event_id=after_event_id))
    return job


def get_job(job_id: str, *, include_events: bool = False, after_event_id: int = 0) -> Optional[dict]:
    row = fetch_one("SELECT * FROM jobs WHERE id = %s", (job_id,))
    return hydrate_job(row, include_events=include_events, after_event_id=after_event_id)


def list_jobs(limit: int = 50, status: Optional[str] = None) -> list[dict]:
    params: list[Any] = []
    where = ""
    if status:
        where = "WHERE status = %s"
        params.append(status)
    params.append(limit)
    rows = fetch_all(
        f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT %s",
        tuple(params),
    )
    return [hydrate_job(row, include_events=False) for row in rows]


def get_job_events(job_id: str, *, after_event_id: int = 0, limit: int = 500) -> list[dict]:
    return fetch_all(
        """
        SELECT id, job_id, event_type, payload_json, created_at
        FROM job_events
        WHERE job_id = %s AND id > %s
        ORDER BY id ASC
        LIMIT %s
        """,
        (job_id, after_event_id, limit),
    )


def append_job_event(job_id: str, event_type: str, payload: Optional[dict] = None) -> dict:
    return fetch_one(
        """
        INSERT INTO job_events (job_id, event_type, payload_json)
        VALUES (%s, %s, CAST(%s AS JSONB))
        RETURNING id, job_id, event_type, payload_json, created_at
        """,
        (job_id, event_type, _json_dumps(payload or {})),
    ) or {}


def enqueue_job(job_id: str) -> None:
    get_redis_client().lpush(QUEUE_NAME, job_id)


def dequeue_job(timeout: int = 5) -> Optional[str]:
    item = get_redis_client().brpop(QUEUE_NAME, timeout=timeout)
    if not item:
        return None
    _, job_id = item
    return job_id


def claim_job(job_id: str, worker_id: str) -> Optional[dict]:
    row = fetch_one(
        """
        UPDATE jobs
        SET status = %s,
            worker_id = %s,
            started_at = COALESCE(started_at, NOW()),
            attempts = attempts + 1,
            updated_at = NOW()
        WHERE id = %s
          AND status = %s
        RETURNING *
        """,
        (JOB_STATUS_RUNNING, worker_id, job_id, JOB_STATUS_QUEUED),
        app_name="sarabcraft-worker",
    )
    return hydrate_job(row) if row else None


def update_job_progress(
    job_id: str,
    *,
    current: Optional[int] = None,
    total: Optional[int] = None,
    message: Optional[str] = None,
    result: Optional[dict] = None,
    checkpoint: Optional[dict] = None,
) -> Optional[dict]:
    current_row = fetch_one("SELECT * FROM jobs WHERE id = %s", (job_id,), app_name="sarabcraft-worker")
    if current_row is None:
        return None

    next_result = current_row.get("result_json")
    if result is not None:
        next_result = result

    next_checkpoint = current_row.get("checkpoint_json") or {}
    if checkpoint is not None:
        next_checkpoint = checkpoint

    row = fetch_one(
        """
        UPDATE jobs
        SET progress_current = %s,
            progress_total = %s,
            progress_message = %s,
            result_json = CAST(%s AS JSONB),
            checkpoint_json = CAST(%s AS JSONB),
            updated_at = NOW()
        WHERE id = %s
        RETURNING *
        """,
        (
            current if current is not None else current_row.get("progress_current", 0),
            total if total is not None else current_row.get("progress_total", 1),
            message if message is not None else current_row.get("progress_message"),
            _json_dumps(next_result) if next_result is not None else "null",
            _json_dumps(next_checkpoint),
            job_id,
        ),
        app_name="sarabcraft-worker",
    )
    return hydrate_job(row)


def complete_job(job_id: str, result: Optional[dict] = None) -> Optional[dict]:
    current_row = fetch_one("SELECT * FROM jobs WHERE id = %s", (job_id,), app_name="sarabcraft-worker")
    if current_row is None:
        return None
    progress_total = current_row.get("progress_total", 1) or 1
    row = fetch_one(
        """
        UPDATE jobs
        SET status = %s,
            result_json = CAST(%s AS JSONB),
            progress_current = %s,
            progress_total = %s,
            progress_message = %s,
            finished_at = NOW(),
            updated_at = NOW()
        WHERE id = %s
        RETURNING *
        """,
        (
            JOB_STATUS_COMPLETED,
            _json_dumps(result if result is not None else current_row.get("result_json")),
            progress_total,
            progress_total,
            "Completed",
            job_id,
        ),
        app_name="sarabcraft-worker",
    )
    append_job_event(job_id, "done", {})
    return hydrate_job(row)


def fail_job(job_id: str, message: str) -> Optional[dict]:
    row = fetch_one(
        """
        UPDATE jobs
        SET status = %s,
            error_message = %s,
            progress_message = %s,
            finished_at = NOW(),
            updated_at = NOW()
        WHERE id = %s
        RETURNING *
        """,
        (JOB_STATUS_FAILED, message, "Failed", job_id),
        app_name="sarabcraft-worker",
    )
    return hydrate_job(row)


def cancel_job(job_id: str, *, result: Optional[dict] = None) -> Optional[dict]:
    current_row = fetch_one("SELECT * FROM jobs WHERE id = %s", (job_id,), app_name="sarabcraft-worker")
    if current_row is None:
        return None
    row = fetch_one(
        """
        UPDATE jobs
        SET status = %s,
            cancel_requested = FALSE,
            result_json = CAST(%s AS JSONB),
            progress_message = %s,
            finished_at = NOW(),
            updated_at = NOW()
        WHERE id = %s
        RETURNING *
        """,
        (
            JOB_STATUS_CANCELLED,
            _json_dumps(result if result is not None else current_row.get("result_json")),
            "Cancelled",
            job_id,
        ),
        app_name="sarabcraft-worker",
    )
    append_job_event(job_id, "done", {"cancelled": True})
    return hydrate_job(row)


def request_job_cancel(job_id: str) -> Optional[dict]:
    existing = fetch_one("SELECT * FROM jobs WHERE id = %s", (job_id,))
    if not existing:
        return None
    if existing["status"] == JOB_STATUS_QUEUED:
        row = fetch_one(
            """
            UPDATE jobs
            SET status = %s,
                cancel_requested = FALSE,
                progress_message = %s,
                finished_at = NOW(),
                updated_at = NOW()
            WHERE id = %s
            RETURNING *
            """,
            (JOB_STATUS_CANCELLED, "Cancelled", job_id),
        )
        append_job_event(job_id, "cancel_requested", {"message": "Cancellation requested"})
        append_job_event(job_id, "done", {"cancelled": True})
        return hydrate_job(row)

    row = fetch_one(
        """
        UPDATE jobs
        SET cancel_requested = TRUE,
            updated_at = NOW()
        WHERE id = %s
          AND status = %s
        RETURNING *
        """,
        (job_id, JOB_STATUS_RUNNING),
    )
    if row:
        append_job_event(job_id, "cancel_requested", {"message": "Cancellation requested"})
    return hydrate_job(row) if row else hydrate_job(existing)


def clear_job_error(job_id: str) -> Optional[dict]:
    row = fetch_one(
        """
        UPDATE jobs
        SET error_message = NULL,
            updated_at = NOW()
        WHERE id = %s
        RETURNING *
        """,
        (job_id,),
    )
    return hydrate_job(row)


def resume_job(job_id: str) -> Optional[dict]:
    row = fetch_one(
        """
        UPDATE jobs
        SET status = %s,
            cancel_requested = FALSE,
            error_message = NULL,
            worker_id = NULL,
            finished_at = NULL,
            progress_message = %s,
            updated_at = NOW()
        WHERE id = %s
          AND status IN (%s, %s)
        RETURNING *
        """,
        (
            JOB_STATUS_QUEUED,
            "Resumed",
            job_id,
            JOB_STATUS_FAILED,
            JOB_STATUS_CANCELLED,
        ),
    )
    if row:
        enqueue_job(job_id)
        append_job_event(job_id, "resumed", {"message": "Job re-queued"})
    return hydrate_job(row)


def is_cancel_requested(job_id: str) -> bool:
    row = fetch_one("SELECT cancel_requested FROM jobs WHERE id = %s", (job_id,), app_name="sarabcraft-worker")
    return bool(row and row.get("cancel_requested"))


def insert_history_entry(entry: dict) -> dict:
    row = fetch_one(
        """
        INSERT INTO history_entries (
            id, entry_timestamp, domain, attack, model, success, entry_json
        )
        VALUES (%s, %s, %s, %s, %s, %s, CAST(%s AS JSONB))
        ON CONFLICT (id) DO UPDATE
        SET entry_timestamp = EXCLUDED.entry_timestamp,
            domain = EXCLUDED.domain,
            attack = EXCLUDED.attack,
            model = EXCLUDED.model,
            success = EXCLUDED.success,
            entry_json = EXCLUDED.entry_json
        RETURNING *
        """,
        (
            entry["id"],
            entry["timestamp"],
            entry.get("domain"),
            entry.get("attack"),
            entry.get("model"),
            entry.get("success"),
            _json_dumps(entry),
        ),
    )
    return row or {"id": entry["id"], "entry_json": entry}


def list_history_rows(
    *,
    limit: int = 50,
    offset: int = 0,
    domain: Optional[str] = None,
    attack_type: Optional[str] = None,
) -> tuple[int, list[dict]]:
    filters = []
    params: list[Any] = []
    if domain:
        filters.append("domain = %s")
        params.append(domain)
    if attack_type:
        filters.append("attack = %s")
        params.append(attack_type)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""

    total_row = fetch_one(
        f"SELECT COUNT(*) AS total FROM history_entries {where}",
        tuple(params),
    )
    total = int(total_row["total"]) if total_row else 0

    rows = fetch_all(
        f"""
        SELECT entry_json
        FROM history_entries
        {where}
        ORDER BY entry_timestamp DESC
        LIMIT %s OFFSET %s
        """,
        tuple([*params, limit, offset]),
    )
    return total, [row["entry_json"] for row in rows]


def list_all_history_rows() -> list[dict]:
    rows = fetch_all(
        "SELECT entry_json FROM history_entries ORDER BY entry_timestamp DESC"
    )
    return [row["entry_json"] for row in rows]


def get_history_entry(entry_id: str) -> Optional[dict]:
    row = fetch_one("SELECT entry_json FROM history_entries WHERE id = %s", (entry_id,))
    return row["entry_json"] if row else None


def delete_history_entry(entry_id: str) -> bool:
    row = fetch_one(
        "DELETE FROM history_entries WHERE id = %s RETURNING id",
        (entry_id,),
    )
    return bool(row)


def clear_history_entries() -> None:
    execute("DELETE FROM history_entries")
