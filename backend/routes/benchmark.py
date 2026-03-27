"""
Attack Benchmark — SSE streaming endpoint.

Runs multiple attacks (with optional parameter sweeps) against the same input,
then auto-tests each adversarial output against a configurable set of transfer
targets. Streams results back as Server-Sent Events so the frontend can display
a live leaderboard.
"""

import asyncio
import base64
import io
import json
import logging
import threading
import time
from itertools import product
from typing import Optional

import numpy as np
import torch
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from config import ATTACK_REGISTRY, device
from backend.audio_attack_catalog import (
    audio_attack_is_untargeted,
    audio_attack_key,
    audio_attack_requires_target_text,
    audio_attack_target_text,
    available_audio_attack_names,
    canonicalize_audio_attack_name,
    evaluate_audio_source_success,
    evaluate_audio_transfer_success,
)
from backend.models.registry import (
    TASK_ASR,
    TASK_IMAGE_CLASSIFICATION,
    resolve_source_model,
    resolve_verification_target,
    snapshot_display_name,
    snapshot_entry,
)

logger = logging.getLogger("mlsec.benchmark")

router = APIRouter()

# ---------------------------------------------------------------------------
# Preset profiles
# ---------------------------------------------------------------------------

IMAGE_PRESETS = {
    "conservative": {"epsilon": 4, "iterations": 20, "alpha": 1.0},
    "balanced":     {"epsilon": 16, "iterations": 50, "alpha": 1.0},
    "aggressive":   {"epsilon": 32, "iterations": 100, "alpha": 2.0},
}

AUDIO_PRESETS = {
    "conservative": {"epsilon": 0.02, "iterations": 100, "lr": 0.001},
    "balanced":     {"epsilon": 0.05, "iterations": 300, "lr": 0.005},
    "aggressive":   {"epsilon": 0.10, "iterations": 500, "lr": 0.01},
}

MAX_COMBOS = 200
MAX_TRANSFER_TARGETS = 20

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


def _parse_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _tensor_to_pil(t: torch.Tensor, model_name: str) -> Image.Image:
    from utils.image import tensor_to_pil
    return tensor_to_pil(t)


def _resolve_image_transfer_targets(payload: dict | None) -> dict:
    payload = payload or {}
    local_models = []
    for value in payload.get("local_model_ids", []) or []:
        snap = snapshot_entry(resolve_source_model(value, domain="image", task=TASK_IMAGE_CLASSIFICATION))
        if snap:
            local_models.append(snap)

    remote_targets = []
    for value in payload.get("remote_target_ids", []) or []:
        snap = snapshot_entry(resolve_verification_target(value, domain="image"))
        if snap:
            remote_targets.append(snap)

    return {
        "local_models": local_models,
        "remote_targets": remote_targets,
        "preprocess_mode": str(payload.get("preprocess_mode") or "exact"),
    }


def _resolve_audio_transfer_targets(payload: dict | None) -> dict:
    payload = payload or {}
    remote_targets = []
    for value in payload.get("remote_target_ids", []) or []:
        snap = snapshot_entry(resolve_verification_target(value, domain="audio"))
        if snap:
            remote_targets.append(snap)
    return {
        "remote_targets": remote_targets,
        "language": str(payload.get("language") or "en-US"),
    }


def _build_image_param_combos(attacks, param_mode, preset_name, sweep_cfg):
    """Return list of (attack_name, params_dict) tuples."""
    combos = []
    if param_mode == "sweep" and sweep_cfg:
        eps_list = sweep_cfg.get("epsilon", [16])
        iter_list = sweep_cfg.get("iterations", [50])
        for atk in attacks:
            for eps, iters in product(eps_list, iter_list):
                combos.append((atk, {"epsilon": eps, "iterations": iters, "alpha": 1.0}))
    else:
        params = IMAGE_PRESETS.get(preset_name, IMAGE_PRESETS["balanced"])
        for atk in attacks:
            combos.append((atk, dict(params)))
    return combos


def _build_audio_param_combos(attacks, param_mode, preset_name, sweep_cfg):
    combos = []
    if param_mode == "sweep" and sweep_cfg:
        eps_list = sweep_cfg.get("epsilon", [0.05])
        iter_list = sweep_cfg.get("iterations", [300])
        for atk in attacks:
            for eps, iters in product(eps_list, iter_list):
                combos.append((atk, {"epsilon": eps, "iterations": iters, "lr": 0.005}))
    else:
        params = AUDIO_PRESETS.get(preset_name, AUDIO_PRESETS["balanced"])
        for atk in attacks:
            combos.append((atk, dict(params)))
    return combos


# ---------------------------------------------------------------------------
# Image benchmark core
# ---------------------------------------------------------------------------

def _run_image_benchmark(
    input_img, target_img, source_model, combos,
    transfer_targets, preprocess_mode, cancel_event=None, should_cancel=None,
):
    from models.loader import load_model
    from utils.image import preprocess_image, get_predictions, tensor_to_pil
    from attacks.image.router import AttackCancelledError, run_attack_method
    from utils.metrics import compute_metrics

    source_entry = resolve_source_model(source_model, domain="image", task=TASK_IMAGE_CLASSIFICATION)
    if not source_entry or not source_entry.get("model_ref"):
        raise ValueError(f"Unknown source model: {source_model}")
    source_model_ref = str(source_entry["model_ref"])
    source_snapshot = snapshot_entry(source_entry)

    mdl, _ = load_model(source_model_ref, progress=None)
    input_tensor = preprocess_image(input_img, source_model_ref)
    target_tensor = preprocess_image(target_img, source_model_ref)
    _, _, target_idx = get_predictions(mdl, target_tensor)
    orig_preds, orig_class, _ = get_predictions(mdl, input_tensor)
    resolved_targets = _resolve_image_transfer_targets(transfer_targets)

    for idx, (atk_name, params) in enumerate(combos):
        if cancel_event and cancel_event.is_set():
            logger.info("Benchmark cancelled by client at combo %d/%d", idx, len(combos))
            return
        if should_cancel and should_cancel():
            logger.info("Benchmark cancelled by job request at combo %d/%d", idx, len(combos))
            return

        row = {
            "index": idx,
            "attack": atk_name,
            "params": params,
            "domain": "image",
            "source_model": snapshot_display_name(source_snapshot, source_model_ref),
            "source_model_ref": source_model_ref,
            "source_model_snapshot": source_snapshot,
        }
        t0 = time.time()
        try:
            eps_norm = params["epsilon"] / 255.0
            iters = params.get("iterations", 50)
            extra = {k: v for k, v in params.items() if k not in ("epsilon", "iterations")}

            adv_tensor = run_attack_method(
                atk_name, mdl, input_tensor, target_idx,
                eps_norm, iters, extra, should_cancel=should_cancel,
            )

            adv_preds, adv_class, _ = get_predictions(mdl, adv_tensor)
            source_success = (adv_class == get_predictions(mdl, target_tensor)[1])

            metrics = compute_metrics(input_tensor, adv_tensor)
            adv_pil = _tensor_to_pil(adv_tensor, source_model_ref)
            adv_b64 = _image_to_b64(adv_pil)

            row["source_success"] = source_success
            row["source_class"] = adv_class
            row["original_class"] = orig_class
            row["distortion"] = {
                "l2": round(metrics.get("l2", 0), 4),
                "ssim": round(metrics.get("ssim", 0), 4),
                "psnr": round(metrics.get("psnr", 0), 1),
            }
            row["adversarial_b64"] = adv_b64

            # Transfer verification
            transfer_results = []
            if resolved_targets["local_models"] or resolved_targets["remote_targets"]:
                from verification.registry import run_verification
                orig_pil = _tensor_to_pil(input_tensor, source_model_ref)
                _, target_class_label, _ = get_predictions(mdl, target_tensor)
                exact = resolved_targets["preprocess_mode"] == "exact"

                logger.info(
                    "Transfer verify: target_label=%r, original_class=%r, remote_targets=%r, local_models=%r, exact=%r",
                    target_class_label,
                    orig_class,
                    [target.get("display_name") for target in resolved_targets["remote_targets"]],
                    [target.get("display_name") for target in resolved_targets["local_models"]],
                    exact,
                )

                vr = run_verification(
                    adversarial_image=adv_pil,
                    original_image=orig_pil,
                    target_label=target_class_label,
                    original_label=orig_class,
                    local_model_names=[
                        target.get("id") or target.get("model_ref")
                        for target in resolved_targets["local_models"]
                    ] or None,
                    remote_targets=resolved_targets["remote_targets"] or None,
                    local_exact_preprocess=exact,
                )
                for vres in vr:
                    logger.info(
                        "  -> %s: matched_target=%r, top_preds=%s, error=%s",
                        vres.verifier_name,
                        vres.matched_target,
                        [(p.label, round(p.confidence, 3)) for p in (vres.predictions or [])[:3]],
                        vres.error,
                    )
                    transfer_results.append({
                        "target": vres.verifier_name,
                        "service_type": vres.service_type,
                        "success": vres.matched_target,
                        "original_gone": vres.original_label_gone,
                        "confidence_drop": round(vres.confidence_drop, 3),
                        "predictions": [
                            {"label": p.label, "confidence": round(p.confidence, 4)}
                            for p in (vres.predictions or [])[:3]
                        ],
                        "error": vres.error,
                    })

            n_tested = len([t for t in transfer_results if not t.get("error")])
            n_success = len([t for t in transfer_results if t.get("success")])
            row["transfer_results"] = transfer_results
            row["transfer_rate"] = round(n_success / n_tested, 3) if n_tested else 0
            row["transfer_tested"] = n_tested
            row["transfer_success"] = n_success

        except AttackCancelledError:
            logger.info("Benchmark cancelled during combo %d/%d", idx, len(combos))
            return
        except Exception as e:
            logger.warning("Benchmark combo failed: %s %s — %s", atk_name, params, e)
            row["error"] = str(e)

        row["elapsed_ms"] = round((time.time() - t0) * 1000)
        yield row


# ---------------------------------------------------------------------------
# Audio benchmark core
# ---------------------------------------------------------------------------

def _run_audio_benchmark(
    audio_bytes, source_model, target_text, combos,
    transfer_targets, cancel_event=None,
):
    from backend.routes import batch as batch_routes

    wav_tensor, sr, waveform = batch_routes._load_audio_bytes(audio_bytes)
    source_entry = resolve_source_model(source_model, domain="audio", task=TASK_ASR)
    if not source_entry or not source_entry.get("model_ref"):
        raise ValueError(f"Unknown source model: {source_model}")
    source_model_ref = str(source_entry["model_ref"])
    source_snapshot = snapshot_entry(source_entry)
    resolved_targets = _resolve_audio_transfer_targets(transfer_targets)

    for idx, (atk_name, params) in enumerate(combos):
        if cancel_event and cancel_event.is_set():
            logger.info("Audio benchmark cancelled by client at combo %d/%d", idx, len(combos))
            return

        row = {
            "index": idx,
            "attack": atk_name,
            "params": params,
            "domain": "audio",
            "source_model": snapshot_display_name(source_snapshot, source_model_ref),
            "source_model_ref": source_model_ref,
            "source_model_snapshot": source_snapshot,
        }
        t0 = time.time()
        try:
            canonical_attack = canonicalize_audio_attack_name(atk_name)
            atk_key = audio_attack_key(canonical_attack)
            if atk_key is None:
                raise ValueError(f"Unknown audio attack: {atk_name}")
            eps = params.get("epsilon", 0.05)
            iters = params.get("iterations", 300)
            lr = params.get("lr", 0.005)
            effective_target_text = audio_attack_target_text(canonical_attack, target_text)

            adv_wav, result_text, orig_text = _run_single_audio_attack(
                atk_key, wav_tensor, sr, source_model_ref, effective_target_text, eps, iters, lr,
            )

            from utils.asr_utils import compute_snr
            snr_val = compute_snr(wav_tensor, adv_wav) if adv_wav is not None else 0
            source_success = evaluate_audio_source_success(
                canonical_attack,
                result_text,
                target_text=effective_target_text,
                original_text=orig_text,
            )

            # Encode adversarial audio
            adv_np = adv_wav.squeeze().cpu().numpy() if adv_wav is not None else waveform
            adv_b64 = batch_routes._encode_wav(adv_np, sr)

            row["source_success"] = source_success
            row["result_text"] = result_text or ""
            row["original_text"] = orig_text or ""
            row["target_text"] = effective_target_text
            row["evaluation_mode"] = "untargeted" if audio_attack_is_untargeted(canonical_attack) else "targeted"
            row["distortion"] = {"snr_db": round(snr_val, 1) if snr_val else 0}
            row["adversarial_b64"] = adv_b64

            # Audio transfer verification
            transfer_results = []
            if resolved_targets["remote_targets"]:
                remote_targets = resolved_targets["remote_targets"]
                if remote_targets:
                    from verification.audio_registry import run_audio_verification
                    vr = run_audio_verification(
                        adversarial_audio=adv_np,
                        original_audio=waveform,
                        sample_rate=sr,
                        target_text=effective_target_text,
                        original_text=orig_text,
                        remote_targets=remote_targets,
                        language=resolved_targets["language"],
                    )
                    for vres in vr:
                        transfer_success = evaluate_audio_transfer_success(
                            canonical_attack,
                            vres.transcription,
                            target_text=effective_target_text,
                            original_text=orig_text,
                            original_transcription=vres.original_transcription,
                        )
                        transfer_results.append({
                            "target": vres.verifier_name,
                            "service_type": vres.service_type,
                            "success": transfer_success,
                            "transcription": vres.transcription or "",
                            "original_transcription": vres.original_transcription or "",
                            "target_text": vres.target_text or effective_target_text,
                            "wer": round(vres.wer, 3) if vres.wer is not None else None,
                            "error": vres.error,
                        })

            n_tested = len([t for t in transfer_results if not t.get("error")])
            n_success = len([t for t in transfer_results if t.get("success")])
            row["transfer_results"] = transfer_results
            row["transfer_rate"] = round(n_success / n_tested, 3) if n_tested else 0
            row["transfer_tested"] = n_tested
            row["transfer_success"] = n_success

        except Exception as e:
            logger.warning("Audio benchmark combo failed: %s %s — %s", atk_name, params, e)
            row["error"] = str(e)

        row["elapsed_ms"] = round((time.time() - t0) * 1000)
        yield row


def _run_single_audio_attack(atk_key, wav_tensor, sr, model_key, target_text, eps, iters, lr):
    """Dispatch to the correct ASR attack, return (adv_tensor, result_text, orig_text)."""
    from models.asr_loader import load_asr_model

    model_name = resolve_source_model(model_key, domain="audio", task=TASK_ASR)
    if not model_name or not model_name.get("model_ref"):
        raise ValueError(f"Unknown ASR model: {model_key}")
    model_name = str(model_name["model_ref"])
    wrapper, _ = load_asr_model(model_name, progress=None)
    orig_text = wrapper.transcribe(wav_tensor)

    if atk_key == "transcription":
        from attacks.audio.transcription_attack import targeted_transcription_attack
        adv = targeted_transcription_attack(wrapper, wav_tensor, target_text, epsilon=eps, iterations=iters, lr=lr)
    elif atk_key == "hidden_command":
        from attacks.audio.hidden_command import hidden_command_attack
        adv = hidden_command_attack(wrapper, wav_tensor, target_text, epsilon=eps, iterations=iters, lr=lr)
    elif atk_key == "psychoacoustic":
        from attacks.audio.psychoacoustic_attack import psychoacoustic_transcription_attack
        adv = psychoacoustic_transcription_attack(wrapper, wav_tensor, target_text, iterations=iters, lr=lr)
    elif atk_key == "ota":
        from attacks.audio.over_the_air_attack import over_the_air_attack
        adv = over_the_air_attack(wrapper, wav_tensor, target_text, epsilon=eps, iterations=iters, lr=lr)
    elif atk_key == "jamming":
        from attacks.audio.speech_jamming import speech_jamming_untargeted
        adv = speech_jamming_untargeted(wrapper, wav_tensor, epsilon=eps, iterations=iters, lr=lr)
    else:
        raise ValueError(f"Unknown audio attack key: {atk_key}")

    result_text = wrapper.transcribe(adv) if adv is not None else ""
    return adv, result_text, orig_text


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------

@router.post("/benchmark/run")
async def run_benchmark(
    request: Request,
    input_file: UploadFile = File(...),
    domain: str = Form("image"),
    target_file: Optional[UploadFile] = File(None),
    target_text: str = Form(""),
    source_model: str = Form("microsoft/resnet-50"),
    attacks_json: str = Form("[]"),
    param_mode: str = Form("preset"),
    param_preset: str = Form("balanced"),
    param_sweep_json: str = Form("{}"),
    transfer_targets_json: str = Form("{}"),
    preprocess_mode: str = Form("exact"),
):
    """Run attack benchmark with SSE streaming results."""
    try:
        attacks = json.loads(attacks_json)
    except json.JSONDecodeError:
        raise HTTPException(400, "attacks_json must be valid JSON array")
    if not attacks:
        raise HTTPException(400, "Select at least one attack")

    try:
        sweep_cfg = json.loads(param_sweep_json)
    except json.JSONDecodeError:
        sweep_cfg = {}

    try:
        transfer_targets = json.loads(transfer_targets_json)
    except json.JSONDecodeError:
        transfer_targets = {}

    input_data = await input_file.read()

    if domain == "image":
        if not target_file:
            raise HTTPException(400, "Target image required for image benchmark")
        target_data = await target_file.read()
        input_img = _parse_image(input_data)
        target_img = _parse_image(target_data)

        valid_attacks = [a for a in attacks if a in ATTACK_REGISTRY]
        if not valid_attacks:
            raise HTTPException(400, f"No valid attacks. Available: {list(ATTACK_REGISTRY.keys())[:10]}...")

        combos = _build_image_param_combos(valid_attacks, param_mode, param_preset, sweep_cfg)
        if len(combos) > MAX_COMBOS:
            raise HTTPException(400, f"Too many combinations ({len(combos)}). Max is {MAX_COMBOS}.")

        cancel = threading.Event()

        async def stream():
            yield _sse("init", {
                "total": len(combos),
                "domain": "image",
                "attacks": valid_attacks,
                "source_model": source_model,
            })
            results = []
            try:
                for row in _run_image_benchmark(
                    input_img, target_img, source_model, combos,
                    transfer_targets, preprocess_mode, cancel_event=cancel,
                ):
                    if await request.is_disconnected():
                        logger.info("Client disconnected, cancelling benchmark")
                        cancel.set()
                        return
                    results.append(row)
                    yield _sse("result", row)
                    await asyncio.sleep(0)
            except Exception as e:
                logger.exception("Error during image benchmark stream")
                yield _sse("error", {"message": str(e)})

            if not cancel.is_set():
                yield _sse("summary", _build_summary(results))
                yield _sse("done", {})

        return StreamingResponse(stream(), media_type="text/event-stream")

    elif domain == "audio":
        normalized_attacks = [canonicalize_audio_attack_name(a) for a in attacks]
        audio_atk_names = available_audio_attack_names()
        valid_attacks = [a for a in normalized_attacks if a in audio_atk_names]
        if not valid_attacks:
            raise HTTPException(400, f"No valid audio attacks. Available: {audio_atk_names}")
        if any(audio_attack_requires_target_text(a) for a in valid_attacks) and not target_text:
            raise HTTPException(400, "target_text required for targeted audio benchmark attacks")

        combos = _build_audio_param_combos(valid_attacks, param_mode, param_preset, sweep_cfg)
        if len(combos) > MAX_COMBOS:
            raise HTTPException(400, f"Too many combinations ({len(combos)}). Max is {MAX_COMBOS}.")

        cancel = threading.Event()

        async def stream():
            yield _sse("init", {
                "total": len(combos),
                "domain": "audio",
                "attacks": valid_attacks,
                "source_model": source_model,
            })
            results = []
            try:
                for row in _run_audio_benchmark(
                    input_data, source_model, target_text, combos,
                    transfer_targets, cancel_event=cancel,
                ):
                    if await request.is_disconnected():
                        logger.info("Client disconnected, cancelling audio benchmark")
                        cancel.set()
                        return
                    results.append(row)
                    yield _sse("result", row)
                    await asyncio.sleep(0)
            except Exception as e:
                logger.exception("Error during audio benchmark stream")
                yield _sse("error", {"message": str(e)})

            if not cancel.is_set():
                yield _sse("summary", _build_summary(results))
                yield _sse("done", {})

        return StreamingResponse(stream(), media_type="text/event-stream")

    else:
        raise HTTPException(400, f"Unknown domain: {domain}. Use 'image' or 'audio'.")


def _build_summary(results):
    successful = [r for r in results if not r.get("error")]
    if not successful:
        return {"total_combos": len(results), "best_attack": None, "best_transfer_rate": 0}

    def _best_score(row):
        distortion = row.get("distortion", {})
        if row.get("domain") == "audio":
            return (row.get("transfer_rate", 0), distortion.get("snr_db", 0))
        return (row.get("transfer_rate", 0), -(distortion.get("l2", 999) or 999))

    best = max(successful, key=_best_score)
    pareto = []
    for r in successful:
        dist = r.get("distortion", {}).get("l2", 0) or r.get("distortion", {}).get("snr_db", 0)
        pareto.append({
            "attack": r["attack"],
            "params": r["params"],
            "distortion": dist,
            "transfer_rate": r.get("transfer_rate", 0),
        })

    return {
        "total_combos": len(results),
        "successful_combos": len(successful),
        "failed_combos": len(results) - len(successful),
        "best_attack": best.get("attack"),
        "best_params": best.get("params"),
        "best_transfer_rate": best.get("transfer_rate", 0),
        "pareto": pareto,
    }


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------

@router.get("/benchmark/presets")
def get_benchmark_presets():
    """Return available preset profiles for both domains."""
    return {
        "image": IMAGE_PRESETS,
        "audio": AUDIO_PRESETS,
    }


@router.get("/benchmark/attacks")
def get_benchmark_attacks():
    """Return grouped attack lists for the benchmark UI."""
    image_groups = {}
    for name, meta in ATTACK_REGISTRY.items():
        cat = meta.get("cat", "Other")
        image_groups.setdefault(cat, []).append({
            "name": name,
            "year": meta.get("year"),
            "authors": meta.get("authors", ""),
        })

    audio_attacks = [
        {"name": "Targeted Transcription", "desc": "C&W/PGD optimized transcription"},
        {"name": "Hidden Command", "desc": "Embed hidden voice command in carrier audio"},
        {"name": "Psychoacoustic", "desc": "Masking-aware imperceptible perturbation"},
        {"name": "Over-the-Air Robust", "desc": "Robust to room impulse + noise"},
        {"name": "Speech Jamming", "desc": "Untargeted ASR disruption"},
    ]

    return {"image": image_groups, "audio": audio_attacks}
