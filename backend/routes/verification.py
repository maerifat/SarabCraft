"""
Image and audio transfer verification API.
"""

import base64
import io
import os
import concurrent.futures
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.models.registry import (
    resolve_verification_target,
)
from verification.registry import run_verification, get_all_verifiers
from verification.registry import _ensure_loaded as _ensure_image_loaded
from verification.audio_registry import run_audio_verification, get_all_audio_verifiers
from verification.audio_registry import _ensure_audio_loaded
from plugins._base import get_enabled_plugins, run_local_plugin

router = APIRouter()


def _service_status_payload(verifier):
    try:
        details = verifier.detailed_status()
        return {
            **details,
            "name": verifier.name,
            "service_type": verifier.service_type,
        }
    except Exception as exc:
        return {
            "name": verifier.name,
            "service_type": verifier.service_type,
            "level": "unavailable",
            "reason": str(exc),
            "status": "not_configured",
        }


def _service_heartbeat_payload(verifier):
    try:
        hb = verifier.heartbeat()
        ok = bool(hb.get("ok"))
        message = hb.get("message") or hb.get("reason") or verifier.status_message()
        return {
            "name": verifier.name,
            "service_type": verifier.service_type,
            "ok": ok,
            "message": message,
            "level": hb.get("level") or ("ready" if ok else "unavailable"),
            "reason": hb.get("reason") or message,
            "status": hb.get("status") or ("ok" if ok else "not_configured"),
        }
    except Exception as exc:
        return {
            "name": verifier.name,
            "service_type": verifier.service_type,
            "ok": False,
            "message": str(exc),
            "level": "unavailable",
            "reason": str(exc),
            "status": "not_configured",
        }


class ImageVerificationRequest(BaseModel):
    adversarial_b64: str
    original_b64: Optional[str] = None
    target_label: Optional[str] = None
    original_label: Optional[str] = None
    remote_target_ids: Optional[list[str]] = None
    local_model_ids: Optional[list[str]] = None
    preprocess_mode: str = "exact"
    plugin_ids: Optional[list[str]] = None


class AudioVerificationRequest(BaseModel):
    adversarial_b64: str
    sample_rate: int = 16000
    original_b64: Optional[str] = None
    target_text: Optional[str] = None
    original_text: Optional[str] = None
    remote_target_ids: Optional[list[str]] = None
    language: str = "en-US"
    plugin_ids: Optional[list[str]] = None


@router.get("/image/status")
def image_service_status():
    """List image verification services and their status."""
    _ensure_image_loaded()
    verifiers = get_all_verifiers()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        services = list(pool.map(_service_status_payload, verifiers))
    return {"services": services}


@router.get("/image/heartbeat")
def image_heartbeat():
    """Live connectivity test for each image verification service."""
    _ensure_image_loaded()
    verifiers = get_all_verifiers()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_service_heartbeat_payload, verifiers))
    return {"services": results}


@router.post("/image/run")
async def run_image_verification(req: ImageVerificationRequest):
    """Run image transfer verification."""
    try:
        adv_bytes = base64.b64decode(req.adversarial_b64)
        if len(adv_bytes) > 50 * 1024 * 1024:
            raise HTTPException(400, f"Image exceeds 50 MB limit ({len(adv_bytes)} bytes)")
        adv_img = Image.open(io.BytesIO(adv_bytes)).convert("RGB")
        orig_img = None
        if req.original_b64:
            orig_bytes = base64.b64decode(req.original_b64)
            orig_img = Image.open(io.BytesIO(orig_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    local_names = list(req.local_model_ids or [])
    remote_targets = []
    for target_id in req.remote_target_ids or []:
        target = resolve_verification_target(target_id, domain="image")
        if target:
            remote_targets.append(target)

    try:
        results = run_verification(
            adv_img, orig_img, req.target_label, req.original_label,
            local_model_names=local_names,
            remote_targets=remote_targets or None,
            local_exact_preprocess=(req.preprocess_mode == "exact"),
        )

        def to_dict(r):
            d = {
                "verifier_name": r.verifier_name,
                "service_type": r.service_type,
                "matched_target": r.matched_target,
                "original_label_gone": r.original_label_gone,
                "confidence_drop": r.confidence_drop,
                "elapsed_ms": r.elapsed_ms,
                "error": r.error,
            }
            d["predictions"] = [{"label": p.label, "confidence": p.confidence} for p in r.predictions[:5]]
            return d

        result_dicts = [to_dict(r) for r in results]

        if req.plugin_ids:
            plugin_results = _run_plugins_for_image(req.plugin_ids, adv_img, orig_img, req.target_label)
            result_dicts.extend(plugin_results)

        return {"results": result_dicts}
    except Exception as e:
        raise HTTPException(500, f"Verification failed: {str(e)}")


def _run_plugins_for_image(plugin_ids, adv_img, orig_img, target_label):
    """Run selected plugins and return results in the same format as verifiers."""
    results = []
    all_plugins = get_enabled_plugins("image")
    for pid in plugin_ids:
        raw = run_local_plugin(pid, adversarial_image=adv_img, original_image=orig_img)
        preds = raw.get("predictions", [])
        top_label = preds[0]["label"] if preds else None
        matched = bool(target_label and top_label and target_label.lower() in top_label.lower())
        name = next((p["name"] for p in all_plugins if p["id"] == pid), pid)

        results.append({
            "verifier_name": name,
            "service_type": "plugin",
            "matched_target": matched,
            "original_label_gone": False,
            "confidence_drop": 0,
            "elapsed_ms": raw.get("elapsed_ms", 0),
            "error": raw.get("error"),
            "predictions": [{"label": p["label"], "confidence": p["confidence"]} for p in preds[:5]],
        })
    return results


@router.get("/audio/status")
def audio_service_status():
    """List audio verification services and their status."""
    _ensure_audio_loaded()
    verifiers = get_all_audio_verifiers()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        services = list(pool.map(_service_status_payload, verifiers))
    return {"services": services}


@router.get("/audio/heartbeat")
def audio_heartbeat():
    """Live connectivity test for each audio verification service."""
    _ensure_audio_loaded()
    verifiers = get_all_audio_verifiers()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_service_heartbeat_payload, verifiers))
    return {"services": results}


@router.post("/audio/run")
async def run_audio_verification_api(req: AudioVerificationRequest):
    """Run audio transfer verification."""
    try:
        adv_bytes = base64.b64decode(req.adversarial_b64)
        if len(adv_bytes) == 0:
            raise ValueError("Empty audio")
        if len(adv_bytes) > 50 * 1024 * 1024:
            raise ValueError(f"Audio exceeds 50 MB limit ({len(adv_bytes)} bytes)")

        import soundfile as sf
        if adv_bytes[:4] == b"RIFF":
            adv_np, sr = sf.read(io.BytesIO(adv_bytes))
            sample_rate = int(sr)
        else:
            adv_np = np.frombuffer(adv_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sample_rate = req.sample_rate

        orig_np = None
        if req.original_b64:
            orig_bytes = base64.b64decode(req.original_b64)
            if orig_bytes[:4] == b"RIFF":
                orig_np, _ = sf.read(io.BytesIO(orig_bytes))
            else:
                orig_np = np.frombuffer(orig_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        raise HTTPException(400, f"Invalid audio: {e}")

    try:
        remote_targets = []
        for target_id in req.remote_target_ids or []:
            target = resolve_verification_target(target_id, domain="audio")
            if target:
                remote_targets.append(target)
        results = run_audio_verification(
            adversarial_audio=adv_np,
            original_audio=orig_np,
            sample_rate=sample_rate,
            target_text=req.target_text,
            original_text=req.original_text,
            remote_targets=remote_targets or None,
            language=req.language,
        )
        audio_result_dicts = [
                {
                    "verifier_name": r.verifier_name,
                    "service_type": r.service_type,
                    "transcription": r.transcription,
                    "original_transcription": r.original_transcription,
                    "target_text": r.target_text,
                    "exact_match": r.exact_match,
                    "contains_target": r.contains_target,
                    "wer": r.wer,
                    "elapsed_ms": r.elapsed_ms,
                    "error": r.error,
                }
                for r in results
            ]

        if req.plugin_ids:
            audio_result_dicts.extend(
                _run_plugins_for_audio(req.plugin_ids, adv_np, orig_np, sample_rate, req.target_text)
            )

        return {"results": audio_result_dicts}
    except Exception as e:
        raise HTTPException(500, f"Audio verification failed: {str(e)}")


def _run_plugins_for_audio(plugin_ids, adv_np, orig_np, sample_rate, target_text):
    """Run selected audio plugins and return results in audio verifier format."""
    results = []
    all_plugins = get_enabled_plugins("audio")
    for pid in plugin_ids:
        raw = run_local_plugin(pid, adversarial_audio=adv_np, original_audio=orig_np,
                               sample_rate=sample_rate)
        preds = raw.get("predictions", [])
        top_label = preds[0]["label"] if preds else None
        exact = bool(target_text and top_label and target_text.lower().strip() == top_label.lower().strip())
        contains = bool(target_text and top_label and target_text.lower().strip() in top_label.lower().strip())
        name = next((p["name"] for p in all_plugins if p["id"] == pid), pid)

        results.append({
            "verifier_name": name,
            "service_type": "plugin",
            "transcription": top_label,
            "original_transcription": None,
            "target_text": target_text,
            "exact_match": exact,
            "contains_target": contains,
            "wer": None,
            "elapsed_ms": raw.get("elapsed_ms", 0),
            "error": raw.get("error"),
        })
    return results
