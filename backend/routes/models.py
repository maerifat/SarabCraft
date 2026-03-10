import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from backend.models.registry import (
    build_catalog_response,
    build_source_models_response,
    build_verification_targets_response,
    create_entry,
    delete_entry,
    duplicate_entry,
    get_entry,
    test_entry,
    toggle_entry,
    update_entry,
)

router = APIRouter()
logger = logging.getLogger("mlsec.models.routes")


@router.get("/catalog")
def get_model_catalog():
    return build_catalog_response()


@router.get("/sources")
def get_source_models(
    domain: str = Query("image", pattern="^(image|audio)$"),
    task: str = Query("image_classification"),
):
    return build_source_models_response(domain, task)


@router.get("/verification")
def get_verification_targets(
    domain: str = Query("image", pattern="^(image|audio)$"),
):
    return build_verification_targets_response(domain)


@router.post("")
def create_model_entry(payload: dict[str, Any]):
    try:
        return create_entry(payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.put("/{entry_id}")
def update_model_entry(entry_id: str, payload: dict[str, Any]):
    try:
        return update_entry(entry_id, payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/{entry_id}/toggle")
def toggle_model_entry(entry_id: str, payload: dict[str, Any]):
    enabled = bool(payload.get("enabled", True))
    try:
        return toggle_entry(entry_id, enabled)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/{entry_id}/duplicate")
def duplicate_model_entry(entry_id: str):
    try:
        return duplicate_entry(entry_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/{entry_id}/test")
def test_model(entry_id: str):
    try:
        return test_entry(entry_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive route guard
        logger.exception("Model validation failed | entry_id=%s", entry_id)
        raise HTTPException(500, "Model validation failed") from exc


@router.delete("/{entry_id}")
def delete_model_entry(entry_id: str):
    try:
        return delete_entry(entry_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{entry_id}")
def get_model_entry(entry_id: str):
    entry = get_entry(entry_id, include_archived=True)
    if not entry:
        raise HTTPException(404, "Model entry not found")
    return entry
