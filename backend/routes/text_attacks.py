"""
Text attack API endpoints.

Mirrors backend/routes/attacks.py for image and audio_attacks.py for audio.
"""

import os
import sys
from typing import Optional

from fastapi import APIRouter, Form, HTTPException

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from attacks.text.config import TEXT_ATTACK_REGISTRY, AVAILABLE_TEXT_MODELS, DEFAULT_TEXT_MODEL
from models.text_loader import load_text_model, get_predictions
from attacks.text.result_builder import run_and_build_result

router = APIRouter()


# ── List endpoints ───────────────────────────────────────────────────────────

@router.get("/text/models")
def list_text_models():
    """List available text classification models."""
    return {
        "models": [
            {"display_name": name, "model_id": mid}
            for name, mid in AVAILABLE_TEXT_MODELS.items()
        ],
        "default": DEFAULT_TEXT_MODEL,
    }


@router.get("/text/methods")
def list_text_methods():
    """List available text attack methods with metadata."""
    return {
        "attacks": {
            name: {
                "category": info["cat"],
                "threat_model": info["threat"],
                "description": info["desc"],
                "paper": info.get("paper", ""),
                "authors": info.get("authors", ""),
                "year": info.get("year"),
                "arxiv": info.get("arxiv", ""),
                "params": info.get("params", {}),
            }
            for name, info in TEXT_ATTACK_REGISTRY.items()
        }
    }


# ── Classify endpoint ───────────────────────────────────────────────────────

@router.post("/text/classify")
def classify_text(
    text: str = Form(...),
    model: str = Form(DEFAULT_TEXT_MODEL),
    top_k: int = Form(5),
):
    """Classify text using a text classification model."""
    try:
        m, tok = load_text_model(model)
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")

    preds = get_predictions(m, tok, text, top_k=top_k)
    return {"predictions": preds, "model": model}


# ── Run attack endpoint ─────────────────────────────────────────────────────

@router.post("/text/run")
def run_text_attack_endpoint(
    text: str = Form(...),
    model: str = Form(DEFAULT_TEXT_MODEL),
    attack: str = Form(...),
    target_label: Optional[str] = Form(None),
    params: Optional[str] = Form("{}"),
):
    """Run a text adversarial attack."""
    import json
    from dataclasses import asdict

    if attack not in TEXT_ATTACK_REGISTRY:
        raise HTTPException(400, f"Unknown attack: {attack}")

    try:
        attack_params = json.loads(params) if params else {}
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid params JSON")

    try:
        m, tok = load_text_model(model)
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")

    try:
        result = run_and_build_result(
            attack_name=attack,
            model=m,
            tokenizer=tok,
            text=text,
            target_label=target_label if target_label else None,
            params=attack_params,
        )
        return {"result": asdict(result)}
    except Exception as e:
        raise HTTPException(500, f"Attack failed: {e}")
