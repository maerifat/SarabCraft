import hashlib
import json
import logging
import re
import uuid
from typing import Any, Iterable

from backend.jobs.core import db_cursor, execute, fetch_all, fetch_one
from backend.models.hf_presets import HF_PRESET_MODELS
from config import AVAILABLE_ASR_MODELS, AVAILABLE_AUDIO_MODELS, AVAILABLE_MODELS

logger = logging.getLogger("mlsec.models")

KIND_SOURCE = "source_model"
KIND_TARGET = "verification_target"
BACKEND_LOCAL_IMAGE = "hf_local_image"
BACKEND_LOCAL_AUDIO = "hf_local_audio"
BACKEND_LOCAL_ASR = "hf_local_asr"
BACKEND_HF_API = "hf_api"
BACKEND_SERVICE = "verifier_service"

TASK_IMAGE_CLASSIFICATION = "image_classification"
TASK_AUDIO_CLASSIFICATION = "audio_classification"
TASK_ASR = "asr"
TASK_IMAGE_VERIFICATION = "image_verification"
TASK_AUDIO_VERIFICATION = "audio_verification"
SERVICE_LOCAL_MODELS = "Local Models"
SERVICE_HF_API = "HuggingFace API"
MODEL_REGISTRY_LOCK_KEY = 730002

IMAGE_SERVICE_TARGETS = [
    ("target-image-aws-rekognition", "AWS Rekognition", "aws", "Cloud Image"),
    ("target-image-azure-vision", "Azure Vision", "azure", "Cloud Image"),
    ("target-image-gcp-vision", "GCP Vision", "gcp", "Cloud Image"),
]

AUDIO_SERVICE_TARGETS = [
    ("target-audio-aws-transcribe", "AWS Transcribe", "aws", "Cloud Audio"),
    ("target-audio-elevenlabs-stt", "ElevenLabs STT", "elevenlabs", "Cloud Audio"),
]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "entry"


def _seed_id(prefix: str, value: str) -> str:
    slug = _slugify(value)
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{slug[:48]}-{digest}"


def _parse_family(display_name: str) -> str:
    match = re.match(r"^\[([^\]]+)\]", display_name or "")
    return match.group(1).strip() if match else "Other"


def _parse_provider(display_name: str, fallback: str = "") -> str:
    matches = re.findall(r"\(([^)]+)\)", display_name or "")
    if not matches:
        return fallback
    return matches[-1].strip()


def _dedupe(values: Iterable[Any]) -> list[Any]:
    seen = set()
    out = []
    for value in values:
        if value in (None, ""):
            continue
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
            continue
        marker = json.dumps(value, sort_keys=True, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(value)
    return out


def _json_param(value: Any) -> str:
    return json.dumps(value, default=str)


def _entry_from_row(row: dict | None) -> dict | None:
    if not row:
        return None
    return {
        "id": row["id"],
        "kind": row["kind"],
        "domain": row["domain"],
        "task": row["task"],
        "backend": row["backend"],
        "display_name": row["display_name"],
        "description": row.get("description") or "",
        "provider": row.get("provider") or "",
        "family": row.get("family") or "",
        "model_ref": row.get("model_ref"),
        "settings": row.get("settings_json") or {},
        "compatibility": row.get("compatibility_json") or [],
        "aliases": row.get("aliases_json") or [],
        "item_ids": row.get("item_ids_json") or [],
        "credential_profile_id": row.get("credential_profile_id"),
        "enabled": bool(row.get("enabled")),
        "builtin": bool(row.get("builtin")),
        "sort_order": int(row.get("sort_order") or 0),
        "archived_at": row.get("archived_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _query_entries(
    *,
    kind: str | None = None,
    domain: str | None = None,
    task: str | None = None,
    include_archived: bool = False,
    enabled_only: bool = False,
    compatibility: str | None = None,
    include_mixed: bool = False,
) -> list[dict]:
    query = ["SELECT * FROM model_registry WHERE 1=1"]
    params: list[Any] = []
    if kind:
        query.append("AND kind = %s")
        params.append(kind)
    if domain:
        if include_mixed:
            query.append("AND (domain = %s OR domain = 'mixed')")
        else:
            query.append("AND domain = %s")
        params.append(domain)
    if task:
        query.append("AND task = %s")
        params.append(task)
    if not include_archived:
        query.append("AND archived_at IS NULL")
    if enabled_only:
        query.append("AND enabled = TRUE")
    if compatibility:
        query.append("AND compatibility_json @> %s::jsonb")
        params.append(_json_param([compatibility]))
    query.append("ORDER BY builtin DESC, sort_order ASC, display_name ASC")
    return [_entry_from_row(row) for row in fetch_all(" ".join(query), params, app_name="sarabcraft-models")]  # type: ignore[arg-type]


def list_entries(
    *,
    kind: str | None = None,
    domain: str | None = None,
    task: str | None = None,
    include_archived: bool = False,
    enabled_only: bool = False,
    compatibility: str | None = None,
    include_mixed: bool = False,
) -> list[dict]:
    return _query_entries(
        kind=kind,
        domain=domain,
        task=task,
        include_archived=include_archived,
        enabled_only=enabled_only,
        compatibility=compatibility,
        include_mixed=include_mixed,
    )


def get_entry(entry_id: str, *, include_archived: bool = True) -> dict | None:
    row = fetch_one(
        """
        SELECT * FROM model_registry
        WHERE id = %s
          AND (%s OR archived_at IS NULL)
        """,
        [entry_id, include_archived],
        app_name="sarabcraft-models",
    )
    return _entry_from_row(row)


def _normalize_kind(payload: dict[str, Any], existing: dict | None = None) -> str:
    kind = str(payload.get("kind") or (existing or {}).get("kind") or KIND_SOURCE).strip()
    if kind not in {KIND_SOURCE, KIND_TARGET}:
        raise ValueError(f"Unsupported model kind: {kind}")
    return kind


def _normalize_domain(payload: dict[str, Any], existing: dict | None = None) -> str:
    domain = str(payload.get("domain") or (existing or {}).get("domain") or "image").strip()
    if domain not in {"image", "audio", "mixed"}:
        raise ValueError(f"Unsupported domain: {domain}")
    return domain


def _normalize_task(kind: str, payload: dict[str, Any], existing: dict | None = None) -> str:
    if kind == KIND_SOURCE:
        default = TASK_IMAGE_CLASSIFICATION
        if str(payload.get("domain") or (existing or {}).get("domain") or "image") == "audio":
            default = TASK_AUDIO_CLASSIFICATION
    else:
        default = TASK_IMAGE_VERIFICATION
        if str(payload.get("domain") or (existing or {}).get("domain") or "image") == "audio":
            default = TASK_AUDIO_VERIFICATION
    task = str(payload.get("task") or (existing or {}).get("task") or default).strip()
    allowed = {
        KIND_SOURCE: {TASK_IMAGE_CLASSIFICATION, TASK_AUDIO_CLASSIFICATION, TASK_ASR},
        KIND_TARGET: {TASK_IMAGE_VERIFICATION, TASK_AUDIO_VERIFICATION},
    }[kind]
    if task not in allowed:
        raise ValueError(f"Unsupported task '{task}' for kind '{kind}'")
    return task


def _default_backend(kind: str, task: str) -> str:
    if kind == KIND_TARGET:
        return BACKEND_HF_API if task == TASK_IMAGE_VERIFICATION else BACKEND_SERVICE
    if task == TASK_AUDIO_CLASSIFICATION:
        return BACKEND_LOCAL_AUDIO
    if task == TASK_ASR:
        return BACKEND_LOCAL_ASR
    return BACKEND_LOCAL_IMAGE


def _normalize_backend(kind: str, task: str, payload: dict[str, Any], existing: dict | None = None) -> str:
    backend = str(payload.get("backend") or (existing or {}).get("backend") or _default_backend(kind, task)).strip()
    allowed = {
        KIND_SOURCE: {BACKEND_LOCAL_IMAGE, BACKEND_LOCAL_AUDIO, BACKEND_LOCAL_ASR},
        KIND_TARGET: {BACKEND_HF_API, BACKEND_SERVICE},
    }[kind]
    if backend not in allowed:
        raise ValueError(f"Unsupported backend '{backend}' for kind '{kind}'")
    return backend


def _ensure_dict(value: Any, *, default: dict | None = None) -> dict:
    if value is None:
        return dict(default or {})
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Expected JSON object") from exc
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Expected JSON object")


def _ensure_list(value: Any, *, default: list | None = None) -> list:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("Expected JSON array") from exc
            if isinstance(parsed, list):
                return parsed
        return [part.strip() for part in text.split(",") if part.strip()]
    return list(default or [])


def _normalize_payload(payload: dict[str, Any], *, existing: dict | None = None, preserve_builtin: bool = True) -> dict[str, Any]:
    kind = _normalize_kind(payload, existing)
    domain = _normalize_domain(payload, existing)
    task = _normalize_task(kind, payload, existing)
    backend = _normalize_backend(kind, task, payload, existing)
    display_name = str(payload.get("display_name") or (existing or {}).get("display_name") or "").strip()
    if not display_name:
        raise ValueError("display_name is required")
    description = str(payload.get("description") or (existing or {}).get("description") or "").strip()
    provider = str(payload.get("provider") or (existing or {}).get("provider") or "").strip()
    family = str(payload.get("family") or (existing or {}).get("family") or "").strip()
    model_ref = payload.get("model_ref", (existing or {}).get("model_ref"))
    if isinstance(model_ref, str):
        model_ref = model_ref.strip() or None
    settings = _ensure_dict(payload.get("settings"), default=(existing or {}).get("settings") or {})
    item_ids = _dedupe(_ensure_list(payload.get("item_ids"), default=(existing or {}).get("item_ids") or []))
    compatibility = _dedupe(_ensure_list(payload.get("compatibility"), default=(existing or {}).get("compatibility") or []))
    aliases = _dedupe(
        [
            *_ensure_list(payload.get("aliases"), default=(existing or {}).get("aliases") or []),
            display_name,
            model_ref,
        ]
    )
    credential_profile_id = payload.get("credential_profile_id", (existing or {}).get("credential_profile_id"))
    if isinstance(credential_profile_id, str):
        credential_profile_id = credential_profile_id.strip() or None
    enabled = bool(payload.get("enabled", (existing or {}).get("enabled", True)))
    builtin = bool((existing or {}).get("builtin")) if preserve_builtin else bool(payload.get("builtin"))
    sort_order = int(payload.get("sort_order", (existing or {}).get("sort_order", 0)) or 0)

    if kind == KIND_SOURCE and not model_ref:
        raise ValueError("model_ref is required for source models")
    if kind == KIND_TARGET and backend == BACKEND_HF_API and not model_ref:
        raise ValueError("model_ref is required for HuggingFace API targets")
    if kind == KIND_TARGET and backend == BACKEND_SERVICE:
        service_name = str(settings.get("service_name") or model_ref or "").strip()
        if not service_name:
            raise ValueError("settings.service_name is required for service targets")
        settings["service_name"] = service_name
        model_ref = model_ref or service_name
        if not provider:
            provider = str(settings.get("provider") or provider or "").strip()

    if task == TASK_IMAGE_CLASSIFICATION:
        compatibility = _dedupe(
            compatibility
            or [
                "classify",
                "attack_source",
                "robustness_source",
                "benchmark_source",
                "verification_target_local",
                "benchmark_target_local",
            ]
        )
        settings.setdefault("trust_remote_code", False)
    elif task == TASK_AUDIO_CLASSIFICATION:
        compatibility = _dedupe(compatibility or ["attack_source"])
        settings.setdefault("trust_remote_code", False)
    elif task == TASK_ASR:
        compatibility = _dedupe(compatibility or ["attack_source", "robustness_source", "benchmark_source"])
        settings.setdefault("trust_remote_code", False)
    elif task == TASK_IMAGE_VERIFICATION:
        compatibility = _dedupe(compatibility or ["verification_target", "benchmark_target"])
    elif task == TASK_AUDIO_VERIFICATION:
        compatibility = _dedupe(compatibility or ["verification_target", "benchmark_target"])

    if not family:
        family = _parse_family(display_name)
    if not provider:
        provider = _parse_provider(display_name, fallback=provider)

    return {
        "kind": kind,
        "domain": domain,
        "task": task,
        "backend": backend,
        "display_name": display_name,
        "description": description,
        "provider": provider,
        "family": family,
        "model_ref": model_ref,
        "settings": settings,
        "compatibility": compatibility,
        "aliases": aliases,
        "item_ids": item_ids,
        "credential_profile_id": credential_profile_id,
        "enabled": enabled,
        "builtin": builtin,
        "sort_order": sort_order,
    }


def init_registry_schema() -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS model_registry (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            domain TEXT NOT NULL,
            task TEXT NOT NULL,
            backend TEXT NOT NULL,
            display_name TEXT NOT NULL,
            description TEXT,
            provider TEXT,
            family TEXT,
            model_ref TEXT,
            settings_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            compatibility_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            aliases_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            item_ids_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            credential_profile_id TEXT,
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            builtin BOOLEAN NOT NULL DEFAULT FALSE,
            sort_order INTEGER NOT NULL DEFAULT 0,
            archived_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS model_registry_kind_idx ON model_registry (kind, domain, task, sort_order ASC, display_name ASC)",
        "CREATE INDEX IF NOT EXISTS model_registry_enabled_idx ON model_registry (enabled, archived_at)",
        "CREATE INDEX IF NOT EXISTS model_registry_model_ref_idx ON model_registry (model_ref)",
    ]
    with db_cursor(app_name="sarabcraft-models-init") as (_, cursor):
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", (MODEL_REGISTRY_LOCK_KEY,))
        for statement in statements:
            cursor.execute(statement)


def _seed_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    image_seed_order = 0
    for display_name, model_ref in AVAILABLE_MODELS.items():
        entries.append(
            {
                "id": _seed_id("source-image", model_ref),
                "kind": KIND_SOURCE,
                "domain": "image",
                "task": TASK_IMAGE_CLASSIFICATION,
                "backend": BACKEND_LOCAL_IMAGE,
                "display_name": display_name,
                "description": "Local Hugging Face image classifier",
                "provider": _parse_provider(display_name, fallback="Hugging Face"),
                "family": _parse_family(display_name),
                "model_ref": model_ref,
                "settings": {"trust_remote_code": False},
                "compatibility": [
                    "classify",
                    "attack_source",
                    "robustness_source",
                    "benchmark_source",
                    "verification_target_local",
                    "benchmark_target_local",
                ],
                "aliases": [display_name, model_ref],
                "item_ids": [],
                "credential_profile_id": None,
                "enabled": True,
                "builtin": True,
                "sort_order": image_seed_order,
            }
        )
        image_seed_order += 10

    audio_seed_order = 0
    for display_name, model_ref in AVAILABLE_AUDIO_MODELS.items():
        entries.append(
            {
                "id": _seed_id("source-audio", model_ref),
                "kind": KIND_SOURCE,
                "domain": "audio",
                "task": TASK_AUDIO_CLASSIFICATION,
                "backend": BACKEND_LOCAL_AUDIO,
                "display_name": display_name,
                "description": "Local Hugging Face audio classifier",
                "provider": _parse_provider(display_name, fallback="Hugging Face"),
                "family": _parse_family(display_name),
                "model_ref": model_ref,
                "settings": {"trust_remote_code": False},
                "compatibility": ["attack_source"],
                "aliases": [display_name, model_ref],
                "item_ids": [],
                "credential_profile_id": None,
                "enabled": True,
                "builtin": True,
                "sort_order": audio_seed_order,
            }
        )
        audio_seed_order += 10

    asr_seed_order = 0
    for display_name, model_ref in AVAILABLE_ASR_MODELS.items():
        entries.append(
            {
                "id": _seed_id("source-asr", model_ref),
                "kind": KIND_SOURCE,
                "domain": "audio",
                "task": TASK_ASR,
                "backend": BACKEND_LOCAL_ASR,
                "display_name": display_name,
                "description": "Local automatic speech recognition model",
                "provider": _parse_provider(display_name, fallback="Hugging Face"),
                "family": _parse_family(display_name),
                "model_ref": model_ref,
                "settings": {"trust_remote_code": False},
                "compatibility": ["attack_source", "robustness_source", "benchmark_source"],
                "aliases": [display_name, model_ref],
                "item_ids": [],
                "credential_profile_id": None,
                "enabled": True,
                "builtin": True,
                "sort_order": asr_seed_order,
            }
        )
        asr_seed_order += 10

    hf_seed_order = 0
    for model_ref, label in HF_PRESET_MODELS:
        entries.append(
            {
                "id": _seed_id("target-hf", model_ref),
                "kind": KIND_TARGET,
                "domain": "image",
                "task": TASK_IMAGE_VERIFICATION,
                "backend": BACKEND_HF_API,
                "display_name": label,
                "description": "HuggingFace Inference API verification target",
                "provider": _parse_provider(label, fallback="Hugging Face"),
                "family": "HuggingFace API",
                "model_ref": model_ref,
                "settings": {"service_name": SERVICE_HF_API},
                "compatibility": ["verification_target", "benchmark_target"],
                "aliases": [label, model_ref, SERVICE_HF_API],
                "item_ids": [],
                "credential_profile_id": None,
                "enabled": True,
                "builtin": True,
                "sort_order": hf_seed_order,
            }
        )
        hf_seed_order += 10

    for idx, (entry_id, display_name, provider, family) in enumerate(IMAGE_SERVICE_TARGETS):
        entries.append(
            {
                "id": entry_id,
                "kind": KIND_TARGET,
                "domain": "image",
                "task": TASK_IMAGE_VERIFICATION,
                "backend": BACKEND_SERVICE,
                "display_name": display_name,
                "description": "Cloud image verification target",
                "provider": provider,
                "family": family,
                "model_ref": display_name,
                "settings": {"service_name": display_name, "provider": provider},
                "compatibility": ["verification_target", "benchmark_target"],
                "aliases": [display_name],
                "item_ids": [],
                "credential_profile_id": None,
                "enabled": True,
                "builtin": True,
                "sort_order": idx * 10,
            }
        )

    for idx, (entry_id, display_name, provider, family) in enumerate(AUDIO_SERVICE_TARGETS):
        entries.append(
            {
                "id": entry_id,
                "kind": KIND_TARGET,
                "domain": "audio",
                "task": TASK_AUDIO_VERIFICATION,
                "backend": BACKEND_SERVICE,
                "display_name": display_name,
                "description": "Cloud audio verification target",
                "provider": provider,
                "family": family,
                "model_ref": display_name,
                "settings": {"service_name": display_name, "provider": provider},
                "compatibility": ["verification_target", "benchmark_target"],
                "aliases": [display_name],
                "item_ids": [],
                "credential_profile_id": None,
                "enabled": True,
                "builtin": True,
                "sort_order": idx * 10,
            }
        )

    return entries


def seed_builtin_entries() -> None:
    with db_cursor(app_name="sarabcraft-models-seed") as (_, cursor):
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", (MODEL_REGISTRY_LOCK_KEY,))
        for entry in _seed_entries():
            cursor.execute(
                """
                INSERT INTO model_registry (
                    id, kind, domain, task, backend, display_name, description, provider, family,
                    model_ref, settings_json, compatibility_json, aliases_json, item_ids_json,
                    credential_profile_id, enabled, builtin, sort_order
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb,
                    %s, %s, %s, %s
                )
                ON CONFLICT (id) DO NOTHING
                """,
                [
                    entry["id"],
                    entry["kind"],
                    entry["domain"],
                    entry["task"],
                    entry["backend"],
                    entry["display_name"],
                    entry["description"],
                    entry["provider"],
                    entry["family"],
                    entry["model_ref"],
                    _json_param(entry["settings"]),
                    _json_param(entry["compatibility"]),
                    _json_param(entry["aliases"]),
                    _json_param(entry["item_ids"]),
                    entry["credential_profile_id"],
                    entry["enabled"],
                    entry["builtin"],
                    entry["sort_order"],
                ],
            )


def initialize_model_registry() -> None:
    init_registry_schema()
    seed_builtin_entries()


def list_source_models(
    domain: str,
    *,
    task: str | None = None,
    enabled_only: bool = True,
    include_archived: bool = False,
) -> list[dict]:
    return list_entries(
        kind=KIND_SOURCE,
        domain=domain,
        task=task,
        enabled_only=enabled_only,
        include_archived=include_archived,
    )


def list_local_verification_models(*, enabled_only: bool = True, include_archived: bool = False) -> list[dict]:
    return list_entries(
        kind=KIND_SOURCE,
        domain="image",
        enabled_only=enabled_only,
        include_archived=include_archived,
        compatibility="verification_target_local",
    )


def list_verification_targets(
    domain: str,
    *,
    enabled_only: bool = True,
    include_archived: bool = False,
) -> list[dict]:
    task = TASK_IMAGE_VERIFICATION if domain == "image" else TASK_AUDIO_VERIFICATION
    return list_entries(
        kind=KIND_TARGET,
        domain=domain,
        task=task,
        enabled_only=enabled_only,
        include_archived=include_archived,
    )


def _response_item(entry: dict) -> dict:
    return {
        **entry,
        "label": entry["display_name"],
        "value": entry["id"],
        "group": entry["family"] or entry["task"],
        "archived": bool(entry.get("archived_at")),
    }


def build_source_models_response(domain: str, task: str) -> dict:
    models = list_source_models(domain, task=task)
    items = [_response_item(item) for item in models]
    return {
        "models": [item["display_name"] for item in models],
        "items": items,
        "default_id": items[0]["id"] if items else None,
    }


def build_verification_targets_response(domain: str) -> dict:
    remote_targets = [_response_item(item) for item in list_verification_targets(domain)]
    local_targets = [_response_item(item) for item in list_local_verification_models()] if domain == "image" else []
    return {
        "targets": remote_targets,
        "local_targets": local_targets,
    }


def build_catalog_response() -> dict:
    image_sources = [_response_item(item) for item in list_source_models("image", task=TASK_IMAGE_CLASSIFICATION, enabled_only=False, include_archived=True)]
    audio_sources = [_response_item(item) for item in list_source_models("audio", task=TASK_AUDIO_CLASSIFICATION, enabled_only=False, include_archived=True)]
    asr_sources = [_response_item(item) for item in list_source_models("audio", task=TASK_ASR, enabled_only=False, include_archived=True)]
    image_targets = [_response_item(item) for item in list_verification_targets("image", enabled_only=False, include_archived=True)]
    audio_targets = [_response_item(item) for item in list_verification_targets("audio", enabled_only=False, include_archived=True)]
    return {
        "summary": {
            "source_models": len(image_sources) + len(audio_sources) + len(asr_sources),
            "verification_targets": len(image_targets) + len(audio_targets),
        },
        "source_models": {
            "image": image_sources,
            "audio_classification": audio_sources,
            "asr": asr_sources,
        },
        "verification_targets": {
            "image": image_targets,
            "audio": audio_targets,
        },
    }


def _select_best_match(entries: list[dict], *, allow_archived: bool = False) -> dict | None:
    if not entries:
        return None
    preferred = sorted(
        entries,
        key=lambda item: (
            bool(item.get("archived_at")) and not allow_archived,
            not item.get("enabled", True),
            not item.get("builtin", False),
            item.get("sort_order", 0),
            item.get("display_name", ""),
        ),
    )
    return preferred[0]


def resolve_entry(
    value: str | None,
    *,
    kind: str | None = None,
    domain: str | None = None,
    task: str | None = None,
    allow_archived: bool = False,
    include_mixed: bool = False,
) -> dict | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    direct = get_entry(text, include_archived=allow_archived)
    if direct:
        if kind and direct["kind"] != kind:
            direct = None
        if direct and domain and direct["domain"] not in {domain, "mixed"}:
            direct = None
        if direct and task and direct["task"] != task:
            direct = None
        if direct:
            return direct

    query = ["SELECT * FROM model_registry WHERE archived_at IS NULL"]
    params: list[Any] = []
    if allow_archived:
        query = ["SELECT * FROM model_registry WHERE 1=1"]
    if kind:
        query.append("AND kind = %s")
        params.append(kind)
    if domain:
        if include_mixed:
            query.append("AND (domain = %s OR domain = 'mixed')")
        else:
            query.append("AND domain = %s")
        params.append(domain)
    if task:
        query.append("AND task = %s")
        params.append(task)
    query.append("AND (display_name = %s OR model_ref = %s OR aliases_json @> %s::jsonb)")
    params.extend([text, text, _json_param([text])])
    query.append("ORDER BY builtin DESC, enabled DESC, sort_order ASC, display_name ASC")
    matches = [_entry_from_row(row) for row in fetch_all(" ".join(query), params, app_name="sarabcraft-models-resolve")]  # type: ignore[arg-type]
    return _select_best_match(matches, allow_archived=allow_archived)


def _ephemeral_source(value: str, domain: str, task: str) -> dict:
    family = "Custom"
    backend = BACKEND_LOCAL_IMAGE
    if task == TASK_AUDIO_CLASSIFICATION:
        backend = BACKEND_LOCAL_AUDIO
        family = "Audio"
    elif task == TASK_ASR:
        backend = BACKEND_LOCAL_ASR
        family = "ASR"
    compatibility_map = {
        TASK_IMAGE_CLASSIFICATION: [
            "classify",
            "attack_source",
            "robustness_source",
            "benchmark_source",
            "verification_target_local",
            "benchmark_target_local",
        ],
        TASK_AUDIO_CLASSIFICATION: ["attack_source"],
        TASK_ASR: ["attack_source", "robustness_source", "benchmark_source"],
    }
    return {
        "id": None,
        "kind": KIND_SOURCE,
        "domain": domain,
        "task": task,
        "backend": backend,
        "display_name": value,
        "description": "Ephemeral model reference",
        "provider": "Hugging Face",
        "family": family,
        "model_ref": value,
        "settings": {"trust_remote_code": False},
        "compatibility": compatibility_map.get(task, []),
        "aliases": [value],
        "item_ids": [],
        "credential_profile_id": None,
        "enabled": True,
        "builtin": False,
        "sort_order": 0,
        "archived_at": None,
        "created_at": None,
        "updated_at": None,
    }


def _ephemeral_target(value: str, domain: str) -> dict:
    service_map = {
        SERVICE_HF_API: {
            "id": None,
            "kind": KIND_TARGET,
            "domain": domain,
            "task": TASK_IMAGE_VERIFICATION,
            "backend": BACKEND_HF_API,
            "display_name": value,
            "description": "Ephemeral HuggingFace API target",
            "provider": "Hugging Face",
            "family": "HuggingFace API",
            "model_ref": value,
            "settings": {"service_name": SERVICE_HF_API},
            "compatibility": ["verification_target", "benchmark_target"],
            "aliases": [value, SERVICE_HF_API],
            "item_ids": [],
            "credential_profile_id": None,
            "enabled": True,
            "builtin": False,
            "sort_order": 0,
            "archived_at": None,
            "created_at": None,
            "updated_at": None,
        }
    }
    if value in service_map:
        return service_map[value]
    return {
        "id": None,
        "kind": KIND_TARGET,
        "domain": domain,
        "task": TASK_IMAGE_VERIFICATION if domain == "image" else TASK_AUDIO_VERIFICATION,
        "backend": BACKEND_SERVICE,
        "display_name": value,
        "description": "Ephemeral verification target",
        "provider": "",
        "family": "Cloud Service",
        "model_ref": value,
        "settings": {"service_name": value},
        "compatibility": ["verification_target", "benchmark_target"],
        "aliases": [value],
        "item_ids": [],
        "credential_profile_id": None,
        "enabled": True,
        "builtin": False,
        "sort_order": 0,
        "archived_at": None,
        "created_at": None,
        "updated_at": None,
    }


def resolve_source_model(value: str | None, *, domain: str, task: str, allow_archived: bool = False) -> dict | None:
    entry = resolve_entry(value, kind=KIND_SOURCE, domain=domain, task=task, allow_archived=allow_archived)
    if entry:
        return entry
    text = str(value or "").strip()
    if "/" in text:
        return _ephemeral_source(text, domain, task)
    return None


def resolve_verification_target(value: str | None, *, domain: str, allow_archived: bool = False) -> dict | None:
    entry = resolve_entry(
        value,
        kind=KIND_TARGET,
        domain=domain,
        task=TASK_IMAGE_VERIFICATION if domain == "image" else TASK_AUDIO_VERIFICATION,
        allow_archived=allow_archived,
    )
    if entry:
        return entry
    text = str(value or "").strip()
    if not text:
        return None
    if "/" in text and domain == "image":
        return {
            **_ephemeral_target(text, domain),
            "backend": BACKEND_HF_API,
            "family": "HuggingFace API",
            "provider": "Hugging Face",
            "settings": {"service_name": SERVICE_HF_API},
        }
    return _ephemeral_target(text, domain)


def snapshot_entry(entry: dict | None, *, include_sensitive: bool = False) -> dict | None:
    if not entry:
        return None
    snapshot = {
        "id": entry.get("id"),
        "kind": entry.get("kind"),
        "domain": entry.get("domain"),
        "task": entry.get("task"),
        "backend": entry.get("backend"),
        "display_name": entry.get("display_name"),
        "description": entry.get("description"),
        "provider": entry.get("provider"),
        "family": entry.get("family"),
        "model_ref": entry.get("model_ref"),
        "compatibility": entry.get("compatibility") or [],
        "credential_profile_id": entry.get("credential_profile_id"),
        "builtin": bool(entry.get("builtin")),
        "enabled": bool(entry.get("enabled", True)),
    }
    if include_sensitive:
        snapshot["settings"] = entry.get("settings") or {}
    else:
        snapshot["settings"] = {
            key: value
            for key, value in (entry.get("settings") or {}).items()
            if key not in {"api_key", "secret", "token"}
        }
    if entry.get("item_ids"):
        snapshot["item_ids"] = entry.get("item_ids") or []
    return snapshot


def snapshot_source_model(value: str | None, *, domain: str, task: str, allow_archived: bool = True) -> dict | None:
    return snapshot_entry(resolve_source_model(value, domain=domain, task=task, allow_archived=allow_archived))


def snapshot_verification_target(value: str | None, *, domain: str, allow_archived: bool = True) -> dict | None:
    return snapshot_entry(resolve_verification_target(value, domain=domain, allow_archived=allow_archived))


def _snapshot_list(values: Iterable[Any], *, domain: str, task: str | None = None, kind: str) -> list[dict]:
    snapshots = []
    for value in values:
        if kind == KIND_SOURCE:
            snap = snapshot_source_model(str(value), domain=domain, task=task or TASK_IMAGE_CLASSIFICATION)
        else:
            snap = snapshot_verification_target(str(value), domain=domain)
        if snap:
            snapshots.append(snap)
    return snapshots


def build_image_transfer_target_snapshot(payload: dict[str, Any]) -> dict:
    local_values = _ensure_list(payload.get("local_model_ids"))
    remote_values = _ensure_list(payload.get("remote_target_ids"))
    return {
        "local_models": _snapshot_list(local_values, domain="image", task=TASK_IMAGE_CLASSIFICATION, kind=KIND_SOURCE),
        "remote_targets": _snapshot_list(remote_values, domain="image", kind=KIND_TARGET),
        "preprocess_mode": str(payload.get("preprocess_mode") or "exact"),
    }


def build_audio_transfer_target_snapshot(payload: dict[str, Any]) -> dict:
    remote_values = _ensure_list(payload.get("remote_target_ids"))
    return {
        "remote_targets": _snapshot_list(remote_values, domain="audio", kind=KIND_TARGET),
        "language": str(payload.get("language") or "en-US"),
    }


def enrich_job_fields(kind: str, fields: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(fields)
    if kind in {"image_attack", "batch_attack"}:
        snap = snapshot_source_model(enriched.get("model"), domain="image", task=TASK_IMAGE_CLASSIFICATION)
        if snap:
            enriched["model_snapshot_json"] = json.dumps(snap)
        if kind == "image_attack":
            ensemble_values = _ensure_list(enriched.get("ensemble_models"))
            if ensemble_values:
                ensemble_snaps = _snapshot_list(ensemble_values, domain="image", task=TASK_IMAGE_CLASSIFICATION, kind=KIND_SOURCE)
                enriched["ensemble_model_snapshots_json"] = json.dumps(ensemble_snaps)
    elif kind == "image_robustness":
        model_values = _ensure_list(enriched.get("models_json"))
        if model_values:
            enriched["models_snapshot_json"] = json.dumps(
                _snapshot_list(model_values, domain="image", task=TASK_IMAGE_CLASSIFICATION, kind=KIND_SOURCE)
            )
    elif kind == "audio_robustness":
        model_values = _ensure_list(enriched.get("models_json"))
        if model_values:
            enriched["models_snapshot_json"] = json.dumps(
                _snapshot_list(model_values, domain="audio", task=TASK_ASR, kind=KIND_SOURCE)
            )
    elif kind == "audio_classification":
        snap = snapshot_source_model(enriched.get("model"), domain="audio", task=TASK_AUDIO_CLASSIFICATION)
        if snap:
            enriched["model_snapshot_json"] = json.dumps(snap)
    elif kind in {
        "asr_transcription",
        "asr_hidden_command",
        "asr_universal_muting",
        "asr_psychoacoustic",
        "asr_over_the_air",
        "asr_speech_jamming",
        "asr_ua3",
    }:
        snap = snapshot_source_model(enriched.get("model"), domain="audio", task=TASK_ASR)
        if snap:
            enriched["model_snapshot_json"] = json.dumps(snap)
    elif kind == "benchmark":
        domain = str(enriched.get("domain") or "image")
        if domain == "audio":
            snap = snapshot_source_model(enriched.get("source_model"), domain="audio", task=TASK_ASR)
            if snap:
                enriched["source_model_snapshot_json"] = json.dumps(snap)
            transfer_targets = _ensure_dict(enriched.get("transfer_targets_json"), default={})
            enriched["transfer_targets_snapshot_json"] = json.dumps(build_audio_transfer_target_snapshot(transfer_targets))
        else:
            snap = snapshot_source_model(enriched.get("source_model"), domain="image", task=TASK_IMAGE_CLASSIFICATION)
            if snap:
                enriched["source_model_snapshot_json"] = json.dumps(snap)
            transfer_targets = _ensure_dict(enriched.get("transfer_targets_json"), default={})
            enriched["transfer_targets_snapshot_json"] = json.dumps(build_image_transfer_target_snapshot(transfer_targets))
    return enriched


def snapshot_model_ref(snapshot: dict | None, fallback: str | None = None) -> str | None:
    if snapshot and snapshot.get("model_ref"):
        return str(snapshot["model_ref"])
    return fallback


def snapshot_display_name(snapshot: dict | None, fallback: str | None = None) -> str | None:
    if snapshot and snapshot.get("display_name"):
        return str(snapshot["display_name"])
    return fallback


def create_entry(payload: dict[str, Any]) -> dict:
    normalized = _normalize_payload(payload, preserve_builtin=False)
    entry_id = str(payload.get("id") or "").strip() or f"{_slugify(normalized['kind'])}-{uuid.uuid4().hex[:10]}"
    execute(
        """
        INSERT INTO model_registry (
            id, kind, domain, task, backend, display_name, description, provider, family,
            model_ref, settings_json, compatibility_json, aliases_json, item_ids_json,
            credential_profile_id, enabled, builtin, sort_order
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb,
            %s, %s, %s, %s
        )
        """,
        [
            entry_id,
            normalized["kind"],
            normalized["domain"],
            normalized["task"],
            normalized["backend"],
            normalized["display_name"],
            normalized["description"],
            normalized["provider"],
            normalized["family"],
            normalized["model_ref"],
            _json_param(normalized["settings"]),
            _json_param(normalized["compatibility"]),
            _json_param(normalized["aliases"]),
            _json_param(normalized["item_ids"]),
            normalized["credential_profile_id"],
            normalized["enabled"],
            False,
            normalized["sort_order"],
        ],
        app_name="sarabcraft-models-create",
    )
    return get_entry(entry_id) or {"id": entry_id, **normalized}


def update_entry(entry_id: str, payload: dict[str, Any]) -> dict:
    existing = get_entry(entry_id, include_archived=True)
    if not existing:
        raise ValueError("Model entry not found")
    normalized = _normalize_payload(payload, existing=existing)
    execute(
        """
        UPDATE model_registry
        SET kind = %s,
            domain = %s,
            task = %s,
            backend = %s,
            display_name = %s,
            description = %s,
            provider = %s,
            family = %s,
            model_ref = %s,
            settings_json = %s::jsonb,
            compatibility_json = %s::jsonb,
            aliases_json = %s::jsonb,
            item_ids_json = %s::jsonb,
            credential_profile_id = %s,
            enabled = %s,
            sort_order = %s,
            updated_at = NOW()
        WHERE id = %s
        """,
        [
            normalized["kind"],
            normalized["domain"],
            normalized["task"],
            normalized["backend"],
            normalized["display_name"],
            normalized["description"],
            normalized["provider"],
            normalized["family"],
            normalized["model_ref"],
            _json_param(normalized["settings"]),
            _json_param(normalized["compatibility"]),
            _json_param(normalized["aliases"]),
            _json_param(normalized["item_ids"]),
            normalized["credential_profile_id"],
            normalized["enabled"],
            normalized["sort_order"],
            entry_id,
        ],
        app_name="sarabcraft-models-update",
    )
    return get_entry(entry_id, include_archived=True) or {**existing, **normalized}


def duplicate_entry(entry_id: str) -> dict:
    existing = get_entry(entry_id, include_archived=True)
    if not existing:
        raise ValueError("Model entry not found")
    payload = {
        **existing,
        "display_name": f"{existing['display_name']} Copy",
        "builtin": False,
        "enabled": existing.get("enabled", True),
        "aliases": [],
    }
    for key in ("id", "created_at", "updated_at", "archived_at"):
        payload.pop(key, None)
    return create_entry(payload)


def toggle_entry(entry_id: str, enabled: bool) -> dict:
    existing = get_entry(entry_id, include_archived=True)
    if not existing:
        raise ValueError("Model entry not found")
    execute(
        "UPDATE model_registry SET enabled = %s, updated_at = NOW() WHERE id = %s",
        [enabled, entry_id],
        app_name="sarabcraft-models-toggle",
    )
    return get_entry(entry_id, include_archived=True) or {**existing, "enabled": enabled}


def _reference_counts(entry_id: str) -> dict[str, int]:
    jobs = fetch_one(
        "SELECT COUNT(*) AS count FROM jobs WHERE request_json::text LIKE %s",
        [f"%{entry_id}%"],
        app_name="sarabcraft-models-refs",
    )
    history = fetch_one(
        "SELECT COUNT(*) AS count FROM history_entries WHERE entry_json::text LIKE %s",
        [f"%{entry_id}%"],
        app_name="sarabcraft-models-refs",
    )
    return {
        "jobs": int((jobs or {}).get("count") or 0),
        "history": int((history or {}).get("count") or 0),
    }


def delete_entry(entry_id: str) -> dict:
    existing = get_entry(entry_id, include_archived=True)
    if not existing:
        raise ValueError("Model entry not found")
    refs = _reference_counts(entry_id)
    if any(refs.values()):
        execute(
            """
            UPDATE model_registry
            SET enabled = FALSE,
                archived_at = COALESCE(archived_at, NOW()),
                updated_at = NOW()
            WHERE id = %s
            """,
            [entry_id],
            app_name="sarabcraft-models-archive",
        )
        archived = get_entry(entry_id, include_archived=True) or {**existing, "enabled": False}
        return {"action": "archived", "references": refs, "entry": archived}
    execute("DELETE FROM model_registry WHERE id = %s", [entry_id], app_name="sarabcraft-models-delete")
    return {"action": "deleted", "references": refs, "entry": existing}


def test_entry(entry_id: str) -> dict:
    entry = get_entry(entry_id, include_archived=True)
    if not entry:
        raise ValueError("Model entry not found")
    if entry["backend"] == BACKEND_LOCAL_IMAGE:
        from models.loader import load_model

        load_model(entry["model_ref"], progress=None)
        return {"ok": True, "message": f"Loaded image model {entry['display_name']}"}
    if entry["backend"] == BACKEND_LOCAL_AUDIO:
        from models.audio_loader import load_audio_model

        load_audio_model(entry["model_ref"], progress=None)
        return {"ok": True, "message": f"Loaded audio model {entry['display_name']}"}
    if entry["backend"] == BACKEND_LOCAL_ASR:
        from models.asr_loader import load_asr_model

        load_asr_model(entry["model_ref"], progress=None)
        return {"ok": True, "message": f"Loaded ASR model {entry['display_name']}"}
    if entry["backend"] == BACKEND_HF_API:
        from verification.huggingface_api import _resolve_pipeline_tag

        task = _resolve_pipeline_tag(entry["model_ref"])
        return {
            "ok": True,
            "message": f"HuggingFace target ready ({task or 'unknown pipeline'})",
            "details": {"pipeline_tag": task or "unknown"},
        }
    if entry["backend"] == BACKEND_SERVICE:
        service_name = str((entry.get("settings") or {}).get("service_name") or entry["display_name"])
        verifiers = []
        if entry["domain"] == "image":
            from verification.registry import get_all_verifiers

            verifiers = get_all_verifiers()
        else:
            from verification.audio_registry import get_all_audio_verifiers

            verifiers = get_all_audio_verifiers()
        verifier = next((item for item in verifiers if item.name == service_name), None)
        if verifier is None:
            raise ValueError(f"Verifier '{service_name}' not found")
        heartbeat = verifier.heartbeat()
        details = verifier.detailed_status() if hasattr(verifier, "detailed_status") else {}
        ok = bool(heartbeat.get("ok", verifier.is_available()))
        message = heartbeat.get("message") or details.get("reason") or verifier.status_message()
        return {"ok": ok, "message": message, "details": details}
    raise ValueError(f"Unsupported backend: {entry['backend']}")
