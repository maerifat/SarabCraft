"""
Attack history persistence — stores every attack result for replay, comparison,
and dashboard analytics.  Uses a lightweight JSON-lines file (~/.mlsec/history.jsonl).
"""

import csv
import io
import json
import logging
import os
import threading
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from backend.jobs.core import (
    clear_history_entries,
    delete_history_entry as db_delete_history_entry,
    get_history_entry as db_get_history_entry,
    insert_history_entry,
    list_all_history_rows,
    list_history_rows,
)
from utils.attack_names import normalize_attack_payload

router = APIRouter()
logger = logging.getLogger(__name__)

HISTORY_DIR = os.path.join(os.path.expanduser("~"), ".mlsec")
HISTORY_FILE = os.path.join(HISTORY_DIR, "attack_history.jsonl")

os.makedirs(HISTORY_DIR, exist_ok=True)

_history_lock = threading.Lock()


def save_entry(entry: dict) -> dict:
    entry = normalize_attack_payload(dict(entry))
    entry["id"] = str(uuid.uuid4())
    entry["timestamp"] = time.time()
    try:
        insert_history_entry(entry)
    except Exception as exc:
        logger.warning("DB history insert failed, using JSONL fallback: %s", exc)
        with _history_lock:
            with open(HISTORY_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
    return entry


def _load_all_file() -> list[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    entries = []
    with open(HISTORY_FILE) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    entries.append(normalize_attack_payload(json.loads(line)))
                except json.JSONDecodeError as exc:
                    logger.warning("Corrupt history line %d: %s — %s", line_num, exc, line[:120])
    return entries


def _load_all() -> list[dict]:
    try:
        return normalize_attack_payload(list_all_history_rows())
    except Exception as exc:
        logger.warning("DB history read failed, using JSONL fallback: %s", exc)
        return _load_all_file()


# Static routes MUST be declared before the /{entry_id} wildcard

@router.get("/list")
def list_history(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    attack_type: Optional[str] = Query(None),
    domain: Optional[str] = Query(None),
):
    try:
        total, entries = list_history_rows(limit=limit, offset=offset, domain=domain, attack_type=attack_type)
    except Exception as exc:
        logger.warning("DB history listing failed, using JSONL fallback: %s", exc)
        entries = _load_all_file()
        if domain:
            entries = [e for e in entries if e.get("domain") == domain]
        if attack_type:
            entries = [e for e in entries if e.get("attack") == attack_type]
        entries.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        total = len(entries)
        entries = entries[offset : offset + limit]
    entries = normalize_attack_payload(entries)
    for e in entries:
        e.pop("adversarial_b64", None)
        e.pop("perturbation_b64", None)
        e.pop("audio_b64", None)
        e.pop("adversarial_wav_b64", None)
    return {"total": total, "entries": entries}


@router.get("/stats/summary")
def history_stats():
    entries = _load_all()
    if not entries:
        return {"total": 0, "success_rate": 0, "attacks": {}, "models": {}, "domains": {}}

    total = len(entries)
    successes = sum(1 for e in entries if e.get("success"))

    attacks = {}
    models = {}
    domains = {}
    for e in entries:
        a = e.get("attack", "unknown")
        m = e.get("model", "unknown")
        d = e.get("domain", "image")
        attacks[a] = attacks.get(a, {"total": 0, "success": 0})
        attacks[a]["total"] += 1
        if e.get("success"):
            attacks[a]["success"] += 1
        models[m] = models.get(m, {"total": 0, "success": 0})
        models[m]["total"] += 1
        if e.get("success"):
            models[m]["success"] += 1
        domains[d] = domains.get(d, 0) + 1

    return {
        "total": total,
        "success_rate": successes / total if total else 0,
        "attacks": attacks,
        "models": models,
        "domains": domains,
    }


@router.get("/stats/heatmap")
def transferability_heatmap():
    """Build attack x model success-rate matrix for dashboard heatmap."""
    entries = [e for e in _load_all() if e.get("domain") == "image"]
    if not entries:
        return {"attacks": [], "models": [], "matrix": []}

    attacks_set = sorted({e.get("attack", "") for e in entries})
    models_set = sorted({e.get("model", "") for e in entries})

    grid = {}
    for e in entries:
        a = e.get("attack", "")
        m = e.get("model", "")
        key = (a, m)
        if key not in grid:
            grid[key] = {"total": 0, "success": 0}
        grid[key]["total"] += 1
        if e.get("success"):
            grid[key]["success"] += 1

    matrix = []
    for a in attacks_set:
        row = []
        for m in models_set:
            cell = grid.get((a, m))
            row.append(cell["success"] / cell["total"] if cell and cell["total"] else None)
        matrix.append(row)

    return {"attacks": attacks_set, "models": models_set, "matrix": matrix}


@router.get("/export")
def export_history(format: str = Query("json")):
    """Export full history as JSON or CSV."""
    entries = _load_all()
    entries = normalize_attack_payload(entries)
    for e in entries:
        e.pop("adversarial_b64", None)
        e.pop("perturbation_b64", None)
        e.pop("audio_b64", None)
        e.pop("adversarial_wav_b64", None)

    if format == "csv":
        output = io.StringIO()
        if entries:
            fieldnames = [
                "id", "timestamp", "domain", "attack", "model",
                "epsilon", "iterations", "success", "original_class",
                "adversarial_class", "target_class",
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for e in entries:
                writer.writerow(e)
        from fastapi.responses import Response
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=mlsec_history.csv"},
        )

    return entries


@router.delete("/")
def clear_history():
    try:
        clear_history_entries()
    except Exception as exc:
        logger.warning("DB history clear failed, using JSONL fallback: %s", exc)
        with _history_lock:
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
    return {"ok": True}


# Wildcard routes AFTER all static routes

@router.get("/{entry_id}")
def get_entry(entry_id: str):
    try:
        entry = db_get_history_entry(entry_id)
        if entry:
            return normalize_attack_payload(entry)
    except Exception as exc:
        logger.warning("DB history lookup failed, using JSONL fallback: %s", exc)
    for e in _load_all_file():
        if e.get("id") == entry_id:
            return normalize_attack_payload(e)
    raise HTTPException(404, "Entry not found")


@router.delete("/{entry_id}")
def delete_entry(entry_id: str):
    try:
        deleted = db_delete_history_entry(entry_id)
        if deleted:
            return {"ok": True}
    except Exception as exc:
        logger.warning("DB history delete failed, using JSONL fallback: %s", exc)

    with _history_lock:
        entries = _load_all_file()
        remaining = [e for e in entries if e.get("id") != entry_id]
        if len(remaining) == len(entries):
            raise HTTPException(404, "Entry not found")
        with open(HISTORY_FILE, "w") as f:
            for e in remaining:
                f.write(json.dumps(e) + "\n")
    return {"ok": True}
