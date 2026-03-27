"""
Batch attack and model robustness comparison endpoints.
Runs attacks across multiple images or multiple models for aggregate analysis.
"""

import base64
import io
import json
import logging
import struct
import time
from typing import Optional

import numpy as np
import torch
from fastapi import APIRouter, File, Form, UploadFile, HTTPException

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PIL import Image
from config import ATTACK_REGISTRY
from backend.audio_attack_catalog import (
    audio_attack_key,
    audio_attack_target_text,
    available_audio_attack_names,
    canonicalize_audio_attack_name,
    evaluate_audio_source_success,
)
from backend.models.registry import (
    TASK_ASR,
    TASK_IMAGE_CLASSIFICATION,
    list_source_models,
    resolve_source_model,
    snapshot_display_name,
    snapshot_entry,
)
from models.loader import load_model
from utils.image import preprocess_image, tensor_to_pil, get_predictions
from utils.metrics import compute_metrics
from attacks.image.router import run_attack_method
from backend.routes.history import save_entry

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_BATCH_IMAGES = 50
MAX_ROBUSTNESS_MODELS = 20

AUDIO_ATK_MAP = {name: audio_attack_key(name) for name in available_audio_attack_names()}


def _parse_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")


def _encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@router.post("/batch/run")
async def run_batch_attack(
    input_files: list[UploadFile] = File(...),
    target_file: UploadFile = File(...),
    model: str = Form("microsoft/resnet-50"),
    attack: str = Form("PGD"),
    epsilon: float = Form(16),
    iterations: int = Form(40),
):
    """Run the same attack on multiple input images. Returns aggregate statistics."""
    if len(input_files) > MAX_BATCH_IMAGES:
        raise HTTPException(400, f"Too many images: {len(input_files)} exceeds limit of {MAX_BATCH_IMAGES}")
    if attack not in ATTACK_REGISTRY:
        raise HTTPException(400, f"Unknown attack: {attack}. Available: {list(ATTACK_REGISTRY.keys())}")

    target_img = _parse_image(await target_file.read())
    model_entry = resolve_source_model(model, domain="image", task=TASK_IMAGE_CLASSIFICATION)
    if not model_entry or not model_entry.get("model_ref"):
        raise HTTPException(400, f"Unknown model key: {model}")
    model_name = str(model_entry["model_ref"])
    model_snapshot = snapshot_entry(model_entry)

    mdl, _ = load_model(model_name, progress=None)
    target_tensor = preprocess_image(target_img, model_name)
    _, target_class, target_idx = get_predictions(mdl, target_tensor)

    results = []
    total = len(input_files)

    for i, f in enumerate(input_files):
        entry = {"filename": f.filename, "index": i}
        try:
            data = await f.read()
            img = _parse_image(data)
            input_tensor = preprocess_image(img, model_name)
            orig_preds, orig_class, _ = get_predictions(mdl, input_tensor)

            adv_tensor = run_attack_method(
                attack, mdl, input_tensor, target_idx,
                epsilon / 255.0, iterations, {}
            )
            adv_preds, adv_class, _ = get_predictions(mdl, adv_tensor)
            metrics = compute_metrics(input_tensor, adv_tensor)
            success = adv_class == target_class

            entry.update({
                "success": success,
                "original_class": orig_class,
                "adversarial_class": adv_class,
                "metrics": metrics,
            })

            try:
                save_entry({
                    "domain": "image", "attack": attack, "model": snapshot_display_name(model_snapshot, model_name),
                    "model_id": model_snapshot.get("id") if model_snapshot else None,
                    "model_ref": model_name,
                    "model_snapshot": model_snapshot,
                    "epsilon": epsilon / 255.0, "iterations": iterations,
                    "success": success, "original_class": orig_class,
                    "adversarial_class": adv_class, "target_class": target_class,
                    "metrics": metrics, "batch": True,
                })
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Failed to save batch history entry: %s", exc)

        except Exception as e:
            entry["error"] = str(e)
            entry["success"] = False

        results.append(entry)

    successes = sum(1 for r in results if r.get("success"))
    metric_results = [r["metrics"] for r in results if r.get("metrics")]
    avg_l2 = float(np.mean([m["l2"] for m in metric_results])) if metric_results else 0.0
    avg_ssim = float(np.mean([m["ssim"] for m in metric_results])) if metric_results else 0.0

    return {
        "total": total,
        "successes": successes,
        "success_rate": successes / total if total else 0,
        "target_class": target_class,
        "attack": attack,
        "model": model_name,
        "avg_l2": round(float(avg_l2), 6),
        "avg_ssim": round(float(avg_ssim), 4),
        "results": results,
    }


@router.post("/robustness/run")
async def run_robustness_comparison(
    input_file: UploadFile = File(...),
    target_file: UploadFile = File(...),
    attack: str = Form("PGD"),
    epsilon: float = Form(16),
    iterations: int = Form(40),
    models_json: str = Form("[]"),
    attack_params_json: str = Form("{}"),
):
    """Stream per-model robustness results via SSE."""
    from fastapi.responses import StreamingResponse

    if attack not in ATTACK_REGISTRY:
        raise HTTPException(400, f"Unknown attack: {attack}. Available: {list(ATTACK_REGISTRY.keys())}")

    input_data = await input_file.read()
    target_data = await target_file.read()
    input_img = _parse_image(input_data)
    target_img = _parse_image(target_data)

    try:
        model_keys = json.loads(models_json)
    except json.JSONDecodeError:
        raise HTTPException(400, "models_json must be valid JSON")

    if not isinstance(model_keys, list):
        raise HTTPException(400, "models_json must be a JSON array")
    if len(model_keys) > MAX_ROBUSTNESS_MODELS:
        raise HTTPException(400, f"Too many models: {len(model_keys)} exceeds limit of {MAX_ROBUSTNESS_MODELS}")

    if not model_keys:
        model_keys = [item["id"] for item in list_source_models("image", task=TASK_IMAGE_CLASSIFICATION)[:10]]

    try:
        extra_params = json.loads(attack_params_json)
    except json.JSONDecodeError:
        extra_params = {}
    attack_extra = {k: v for k, v in extra_params.items() if k not in ("epsilon", "iterations")}

    def _sse_event(event_type, data):
        return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"

    def stream():
        total = len(model_keys)
        yield _sse_event("init", {"attack": attack, "epsilon": epsilon, "iterations": iterations, "total_models": total})

        results = []
        for idx, mk in enumerate(model_keys):
            model_entry = resolve_source_model(mk, domain="image", task=TASK_IMAGE_CLASSIFICATION)
            if not model_entry or not model_entry.get("model_ref"):
                entry["error"] = f"Unknown model key: {mk}"
                entry["success"] = False
                results.append(entry)
                yield _sse_event("result", {**entry, "index": idx})
                continue
            model_name = str(model_entry["model_ref"])
            model_snapshot = snapshot_entry(model_entry)
            short_name = model_name.split("/")[-1] if "/" in model_name else model_name
            yield _sse_event("progress", {"index": idx, "total": total, "model": short_name, "status": "running"})

            entry = {
                "model": snapshot_display_name(model_snapshot, model_name),
                "model_key": mk,
                "model_ref": model_name,
                "model_snapshot": model_snapshot,
            }
            try:
                t0 = time.time()
                mdl, _ = load_model(model_name, progress=None)

                input_tensor = preprocess_image(input_img, model_name)
                target_tensor = preprocess_image(target_img, model_name)
                orig_preds, orig_class, _ = get_predictions(mdl, input_tensor)
                _, target_class, target_idx = get_predictions(mdl, target_tensor)

                adv_tensor = run_attack_method(
                    attack, mdl, input_tensor, target_idx,
                    epsilon / 255.0, iterations, attack_extra
                )
                adv_preds, adv_class, _ = get_predictions(mdl, adv_tensor)
                metrics = compute_metrics(input_tensor, adv_tensor)
                success = adv_class == target_class
                elapsed = round((time.time() - t0) * 1000)

                entry.update({
                    "success": success,
                    "original_class": orig_class,
                    "adversarial_class": adv_class,
                    "target_class": target_class,
                    "metrics": metrics,
                    "elapsed_ms": elapsed,
                    "top_adv_conf": max(adv_preds.values()) if adv_preds else 0,
                })

                try:
                    save_entry({
                        "domain": "image", "attack": attack, "model": snapshot_display_name(model_snapshot, model_name),
                        "model_id": model_snapshot.get("id") if model_snapshot else None,
                        "model_ref": model_name,
                        "model_snapshot": model_snapshot,
                        "epsilon": epsilon / 255.0, "iterations": iterations,
                        "success": success, "original_class": orig_class,
                        "adversarial_class": adv_class, "target_class": target_class,
                        "metrics": metrics, "robustness_test": True,
                    })
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning("Failed to save robustness history entry: %s", exc)

            except Exception as e:
                entry["error"] = str(e)
                entry["success"] = False

            results.append(entry)
            yield _sse_event("result", {**entry, "index": idx})

        successes = sum(1 for r in results if r.get("success"))
        yield _sse_event("summary", {
            "attack": attack,
            "epsilon": epsilon,
            "iterations": iterations,
            "total_models": len(results),
            "successful_transfers": successes,
            "transfer_rate": successes / len(results) if results else 0,
            "results": results,
        })
        yield _sse_event("done", {})

    return StreamingResponse(stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Audio Model Robustness — test one ASR attack across multiple models
# ---------------------------------------------------------------------------

MAX_AUDIO_ROBUSTNESS_MODELS = 10


def _load_audio_bytes(audio_bytes: bytes) -> tuple:
    """Read audio bytes into (torch.Tensor[1, samples], sample_rate)."""
    import tempfile, shutil, subprocess
    from models.asr_loader import WHISPER_SAMPLE_RATE
    from utils.asr_utils import load_audio_for_asr

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)
        try:
            import soundfile as sf
            sf.info(tmp_path)
        except Exception:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                wav_path = tmp_path + "_conv.wav"
                subprocess.run(
                    [ffmpeg, "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                    capture_output=True, timeout=30, check=True,
                )
                os.replace(wav_path, tmp_path)
        wav_tensor, sr = load_audio_for_asr(tmp_path, target_sr=WHISPER_SAMPLE_RATE)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    raw_waveform = wav_tensor.detach().cpu().squeeze().numpy()
    return wav_tensor, sr, raw_waveform


def _encode_wav(adv_np: np.ndarray, sr: int) -> str:
    pcm = (adv_np * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(pcm)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(pcm)))
    buf.write(pcm)
    return base64.b64encode(buf.getvalue()).decode()


def _run_audio_attack_for_model(atk_key, wav_tensor, sr, model_key, target_text, params):
    """Run a single audio attack against a specific ASR model."""
    from models.asr_loader import load_asr_model

    model_entry = resolve_source_model(model_key, domain="audio", task=TASK_ASR)
    if not model_entry or not model_entry.get("model_ref"):
        raise ValueError(f"Unknown ASR model: {model_key}")
    model_name = str(model_entry["model_ref"])
    wrapper, _ = load_asr_model(model_name, progress=None)
    orig_text = wrapper.transcribe(wav_tensor)

    eps = params.get("epsilon", 0.05)
    iters = params.get("iterations", 300)
    lr = params.get("lr", 0.005)

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


@router.post("/robustness/audio/run")
async def run_audio_robustness(
    audio_file: UploadFile = File(...),
    target_text: str = Form(""),
    attack: str = Form("Targeted Transcription"),
    models_json: str = Form("[]"),
    attack_params_json: str = Form("{}"),
):
    """Stream per-ASR-model robustness results via SSE."""
    from fastapi.responses import StreamingResponse

    attack_name = canonicalize_audio_attack_name(attack)
    atk_key = AUDIO_ATK_MAP.get(attack_name)
    if atk_key is None:
        raise HTTPException(400, f"Unknown audio attack: {attack}. Available: {list(AUDIO_ATK_MAP.keys())}")
    effective_target_text = audio_attack_target_text(attack_name, target_text)

    audio_data = await audio_file.read()
    try:
        model_keys = json.loads(models_json)
    except json.JSONDecodeError:
        raise HTTPException(400, "models_json must be valid JSON")

    if not isinstance(model_keys, list):
        raise HTTPException(400, "models_json must be a JSON array")
    if len(model_keys) > MAX_AUDIO_ROBUSTNESS_MODELS:
        raise HTTPException(400, f"Too many models: {len(model_keys)} exceeds limit of {MAX_AUDIO_ROBUSTNESS_MODELS}")
    if not model_keys:
        model_keys = [item["id"] for item in list_source_models("audio", task=TASK_ASR)[:4]]

    try:
        extra_params = json.loads(attack_params_json)
    except json.JSONDecodeError:
        extra_params = {}

    def _sse(event_type, data):
        return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"

    def stream():
        from utils.asr_utils import compute_snr

        wav_tensor, sr, raw_np = _load_audio_bytes(audio_data)
        total = len(model_keys)
        yield _sse("init", {
            "attack": attack_name, "total_models": total, "domain": "audio",
            "params": extra_params,
        })

        results = []
        for idx, mk in enumerate(model_keys):
            model_entry = resolve_source_model(mk, domain="audio", task=TASK_ASR)
            if not model_entry or not model_entry.get("model_ref"):
                entry = {"model": mk, "model_key": mk, "error": f"Unknown ASR model: {mk}", "success": False}
                results.append(entry)
                yield _sse("result", {**entry, "index": idx})
                continue
            model_snapshot = snapshot_entry(model_entry)
            model_name = str(model_entry["model_ref"])
            short_name = snapshot_display_name(model_snapshot, model_name) or model_name
            yield _sse("progress", {"index": idx, "total": total, "model": short_name, "status": "running"})

            entry = {
                "model": snapshot_display_name(model_snapshot, model_name),
                "model_key": mk,
                "model_ref": model_name,
                "model_snapshot": model_snapshot,
            }
            try:
                t0 = time.time()
                adv_wav, result_text, orig_text = _run_audio_attack_for_model(
                    atk_key, wav_tensor, sr, mk, effective_target_text, extra_params
                )

                snr_val = compute_snr(wav_tensor, adv_wav) if adv_wav is not None else 0
                success = evaluate_audio_source_success(
                    attack_name,
                    result_text,
                    target_text=effective_target_text,
                    original_text=orig_text,
                )

                adv_np = adv_wav.squeeze().cpu().numpy() if adv_wav is not None else raw_np
                adv_b64 = _encode_wav(adv_np, sr)

                elapsed = round((time.time() - t0) * 1000)
                entry.update({
                    "success": success,
                    "original_text": orig_text or "",
                    "result_text": result_text or "",
                    "target_text": effective_target_text,
                    "evaluation_mode": "untargeted" if atk_key == "jamming" else "targeted",
                    "snr_db": round(snr_val, 1) if snr_val else 0,
                    "adversarial_b64": adv_b64,
                    "elapsed_ms": elapsed,
                })

                try:
                    save_entry({
                        "domain": "audio", "attack": attack_name, "model": snapshot_display_name(model_snapshot, model_name),
                        "model_id": model_snapshot.get("id") if model_snapshot else None,
                        "model_ref": model_name,
                        "model_snapshot": model_snapshot,
                        "success": success, "original_text": orig_text,
                        "result_text": result_text, "target_text": effective_target_text,
                        "robustness_test": True,
                    })
                except Exception as exc:
                    logger.warning("Failed to save audio robustness entry: %s", exc)

            except Exception as e:
                entry["error"] = str(e)
                entry["success"] = False
                entry["elapsed_ms"] = round((time.time() - t0) * 1000)

            results.append(entry)
            yield _sse("result", {**entry, "index": idx})

        successes = sum(1 for r in results if r.get("success"))
        yield _sse("summary", {
            "attack": attack_name,
            "domain": "audio",
            "total_models": len(results),
            "successful_attacks": successes,
            "attack_success_rate": successes / len(results) if results else 0,
            "results": results,
        })
        yield _sse("done", {})

    return StreamingResponse(stream(), media_type="text/event-stream")
