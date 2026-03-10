"""
Image attack API endpoints.
"""

import base64
import io
import os
from typing import Optional

import numpy as np
import torch
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import AVAILABLE_ATTACKS, ATTACK_REGISTRY
from backend.models.registry import (
    TASK_IMAGE_CLASSIFICATION,
    build_source_models_response,
    resolve_source_model,
    snapshot_display_name,
    snapshot_entry,
)
from models.loader import load_model
from utils.image import preprocess_image, tensor_to_pil, get_predictions
from utils.metrics import compute_metrics
from utils.attack_names import SARABCRAFT_R1_NAME
from attacks.router import run_attack_method
from backend.routes.history import save_entry

router = APIRouter()

MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")


def _resolve_model(model: str) -> str:
    entry = resolve_source_model(model, domain="image", task=TASK_IMAGE_CLASSIFICATION)
    if not entry or not entry.get("model_ref"):
        raise HTTPException(400, f"Unknown model key: {model}")
    return str(entry["model_ref"])


def _resolve_model_entry(model: str) -> dict:
    entry = resolve_source_model(model, domain="image", task=TASK_IMAGE_CLASSIFICATION)
    if not entry or not entry.get("model_ref"):
        raise HTTPException(400, f"Unknown model key: {model}")
    return entry


def _build_attack_params(**kwargs) -> dict:
    """Collect all attack parameters into a single dict."""
    return {k: v for k, v in kwargs.items() if v is not None}


def _load_ensemble_models(
    ensemble_str: Optional[str],
    primary_model: str,
    ensemble_model_snapshots: Optional[list[dict]] = None,
):
    if not ensemble_str and not ensemble_model_snapshots:
        return []
    models = []
    model_refs = []
    if ensemble_model_snapshots:
        for snapshot in ensemble_model_snapshots:
            ref = (snapshot or {}).get("model_ref")
            if ref:
                model_refs.append(str(ref))
    elif ensemble_str:
        for candidate in ensemble_str.split(","):
            candidate = candidate.strip()
            if not candidate:
                continue
            entry = resolve_source_model(candidate, domain="image", task=TASK_IMAGE_CLASSIFICATION)
            ref = (entry or {}).get("model_ref")
            if ref:
                model_refs.append(str(ref))

    for model_ref in model_refs:
        if model_ref != primary_model:
            mdl, _ = load_model(model_ref, progress=None)
            models.append(mdl)
    return models


def _encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _build_perturbation_image(adv_tensor, input_tensor) -> str:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(adv_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(adv_tensor.device)
    adv_px = (adv_tensor * std + mean).clamp(0, 1)
    inp_px = (input_tensor * std + mean).clamp(0, 1)
    pert = (adv_px - inp_px).abs().squeeze(0).cpu()
    pert = (pert * 10 + 0.5).clamp(0, 1)
    pert = pert.permute(1, 2, 0).numpy()
    pert_img = Image.fromarray((pert * 255).astype(np.uint8))
    return _encode_image(pert_img)


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/models")
def list_models():
    return build_source_models_response("image", TASK_IMAGE_CLASSIFICATION)


@router.get("/methods")
def list_attacks():
    return {"attacks": list(AVAILABLE_ATTACKS.keys()), "registry": ATTACK_REGISTRY}


@router.post("/image/classify")
async def classify_image(image_file: UploadFile = File(...), model: str = Form("microsoft/resnet-50")):
    data = await image_file.read()
    if len(data) > MAX_IMAGE_SIZE:
        raise HTTPException(400, f"Image exceeds 20 MB limit ({len(data)} bytes)")
    img = _parse_image(data)
    model_entry = _resolve_model_entry(model)
    model_name = str(model_entry["model_ref"])
    try:
        mdl, _ = load_model(model_name, progress=None)
        tensor = preprocess_image(img, model_name)
        preds, top_class, _ = get_predictions(mdl, tensor)
        return {"class": top_class, "confidence": preds.get(top_class, 0) * 100, "predictions": preds}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/image/run")
async def run_image_attack(
    input_file: UploadFile = File(...),
    target_file: UploadFile = File(...),
    model: str = Form("microsoft/resnet-50"),
    attack: str = Form("PGD"),
    epsilon: float = Form(16),
    iterations: int = Form(40),
    alpha: float = Form(1.0),
    momentum_decay: float = Form(1.0),
    random_start: bool = Form(True),
    overshoot: float = Form(0.02),
    cw_confidence: float = Form(0.0),
    cw_lr: float = Form(0.01),
    cw_c: float = Form(1.0),
    p_di: float = Form(0.7),
    kernel_size: int = Form(5),
    n_scale: int = Form(5),
    n_var: int = Form(20),
    beta_var: float = Form(1.5),
    cfm_mix_prob: float = Form(0.1),
    cfm_mix_upper: float = Form(0.75),
    r1_multi_image: bool = Form(False),
    r1_multi_image_strategy: str = Form("tile_shuffle"),
    r1_multi_image_count: int = Form(10),
    apgd_loss: str = Form("dlr"),
    n_restarts: int = Form(1),
    rho: float = Form(0.75),
    jitter_ratio: float = Form(0.1),
    amplification: float = Form(10.0),
    pi_prob: float = Form(0.7),
    pi_kern_size: int = Form(3),
    n_spectrum: int = Form(20),
    rho_spectrum: float = Form(0.5),
    n_mix: int = Form(5),
    mix_ratio: float = Form(0.2),
    n_block: int = Form(3),
    rotation_range: float = Form(10),
    theta: float = Form(1.0),
    gamma: float = Form(0.1),
    ead_beta: float = Form(0.001),
    ead_rule: str = Form("EN"),
    sf_lambda: float = Form(1.0),
    fab_alpha_max: float = Form(0.1),
    fab_eta: float = Form(1.05),
    fab_beta: float = Form(0.9),
    n_queries: int = Form(5000),
    p_init: float = Form(0.8),
    spsa_delta: float = Form(0.01),
    spsa_lr: float = Form(0.01),
    nb_sample: int = Form(128),
    pixels: int = Form(5),
    popsize: int = Form(400),
    boundary_delta: float = Form(0.01),
    boundary_step: float = Form(0.01),
    hsj_init_evals: int = Form(100),
    hsj_max_evals: int = Form(1000),
    hsj_gamma: float = Form(1.0),
    patch_ratio: float = Form(0.1),
    patch_lr: float = Form(0.01),
    aa_version: str = Form("standard"),
    nb_di_prob: float = Form(0.5),
    nb_di_resize_rate: float = Form(0.9),
    nb_ti_len: int = Form(3),
    nb_npatch: int = Form(128),
    nb_grid_scale: int = Form(16),
    nb_enable_un: bool = Form(True),
    nb_enable_pi: bool = Form(True),
    nb_enable_di: bool = Form(True),
    nb_enable_ti: bool = Form(True),
    nb_enable_ni: bool = Form(True),
    ensemble_models: Optional[str] = Form(None),
    ensemble_mode: str = Form("Simultaneous"),
):
    if attack not in ATTACK_REGISTRY:
        raise HTTPException(400, f"Unknown attack: {attack}. Available: {list(ATTACK_REGISTRY.keys())}")
    if not (0 <= epsilon <= 255):
        raise HTTPException(400, f"epsilon must be between 0 and 255, got {epsilon}")
    if not (1 <= iterations <= 10000):
        raise HTTPException(400, f"iterations must be between 1 and 10000, got {iterations}")

    input_data = await input_file.read()
    if len(input_data) > MAX_IMAGE_SIZE:
        raise HTTPException(400, f"Input image exceeds 20 MB limit ({len(input_data)} bytes)")
    target_data = await target_file.read()
    if len(target_data) > MAX_IMAGE_SIZE:
        raise HTTPException(400, f"Target image exceeds 20 MB limit ({len(target_data)} bytes)")

    input_img = _parse_image(input_data)
    target_img = _parse_image(target_data)
    model_entry = _resolve_model_entry(model)
    model_name = str(model_entry["model_ref"])
    model_snapshot = snapshot_entry(model_entry)

    try:
        mdl, _ = load_model(model_name, progress=None)
        input_tensor = preprocess_image(input_img, model_name)
        target_tensor = preprocess_image(target_img, model_name)
        orig_preds, orig_class, _ = get_predictions(mdl, input_tensor)
        _, target_class, target_idx = get_predictions(mdl, target_tensor)

        attack_params = _build_attack_params(
            alpha=alpha, momentum_decay=momentum_decay, random_start=random_start,
            overshoot=overshoot, cw_confidence=cw_confidence, cw_lr=cw_lr, cw_c=cw_c,
            p_di=p_di, kernel_size=kernel_size, n_scale=n_scale, n_var=n_var,
            beta_var=beta_var, cfm_mix_prob=cfm_mix_prob, cfm_mix_upper=cfm_mix_upper,
            r1_multi_image=r1_multi_image if attack == SARABCRAFT_R1_NAME else None,
            r1_multi_image_strategy=r1_multi_image_strategy if attack == SARABCRAFT_R1_NAME and r1_multi_image else None,
            r1_multi_image_count=r1_multi_image_count if attack == SARABCRAFT_R1_NAME and r1_multi_image else None,
            ensemble_mode=ensemble_mode.lower() if ensemble_mode else "simultaneous",
            apgd_loss=apgd_loss, n_restarts=n_restarts, rho=rho,
            jitter_ratio=jitter_ratio, amplification=amplification,
            pi_prob=pi_prob, pi_kern_size=pi_kern_size,
            n_spectrum=n_spectrum, rho_spectrum=rho_spectrum,
            n_mix=n_mix, mix_ratio=mix_ratio,
            n_block=n_block, rotation_range=rotation_range,
            theta=theta, gamma=gamma, ead_beta=ead_beta, ead_rule=ead_rule,
            sf_lambda=sf_lambda, fab_alpha_max=fab_alpha_max, fab_eta=fab_eta,
            fab_beta=fab_beta, n_queries=n_queries, p_init=p_init,
            spsa_delta=spsa_delta, spsa_lr=spsa_lr, nb_sample=nb_sample,
            pixels=pixels, popsize=popsize,
            boundary_delta=boundary_delta, boundary_step=boundary_step,
            hsj_init_evals=hsj_init_evals, hsj_max_evals=hsj_max_evals,
            hsj_gamma=hsj_gamma, patch_ratio=patch_ratio, patch_lr=patch_lr,
            aa_version=aa_version,
            nb_di_prob=nb_di_prob, nb_di_resize_rate=nb_di_resize_rate,
            nb_ti_len=nb_ti_len, nb_npatch=nb_npatch, nb_grid_scale=nb_grid_scale,
            nb_enable_un=nb_enable_un, nb_enable_pi=nb_enable_pi,
            nb_enable_di=nb_enable_di, nb_enable_ti=nb_enable_ti,
            nb_enable_ni=nb_enable_ni,
        )

        ensemble_entries = []
        if ensemble_models:
            for candidate in ensemble_models.split(","):
                candidate = candidate.strip()
                if not candidate:
                    continue
                resolved = resolve_source_model(candidate, domain="image", task=TASK_IMAGE_CLASSIFICATION)
                if resolved:
                    ensemble_entries.append(resolved)
        ens_mdls = _load_ensemble_models(
            ensemble_models,
            model_name,
            ensemble_model_snapshots=[snapshot_entry(item) for item in ensemble_entries],
        )

        adv_tensor = run_attack_method(
            attack, mdl, input_tensor, target_idx, epsilon / 255.0, iterations,
            attack_params, ensemble_models=ens_mdls if ens_mdls else None,
        )
        adv_preds, adv_class, _ = get_predictions(mdl, adv_tensor)

        success = adv_class == target_class
        metrics = compute_metrics(input_tensor, adv_tensor)

        result = {
            "adversarial_b64": _encode_image(tensor_to_pil(adv_tensor)),
            "perturbation_b64": _build_perturbation_image(adv_tensor, input_tensor),
            "original_class": orig_class,
            "adversarial_class": adv_class,
            "target_class": target_class,
            "original_preds": orig_preds,
            "adversarial_preds": adv_preds,
            "status": f"SUCCESS: classified as {adv_class}" if success else f"Partial: {adv_class} (target: {target_class})",
            "success": success,
            "metrics": metrics,
        }

        try:
            save_entry({
                "domain": "image",
                "attack": attack,
                "model": snapshot_display_name(model_snapshot, model_name),
                "model_id": model_snapshot.get("id") if model_snapshot else None,
                "model_ref": model_name,
                "model_snapshot": model_snapshot,
                "epsilon": epsilon / 255.0,
                "iterations": iterations,
                "success": success,
                "original_class": orig_class,
                "adversarial_class": adv_class,
                "target_class": target_class,
                "metrics": metrics,
                "adversarial_b64": result["adversarial_b64"],
                "perturbation_b64": result["perturbation_b64"],
                "ensemble_count": len(ens_mdls) if ens_mdls else 0,
                "ensemble_models": [snapshot_entry(item) for item in ensemble_entries],
            })
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Failed to save history entry: %s", exc)

        return result
    except Exception as e:
        raise HTTPException(500, f"Attack failed: {str(e)}")
