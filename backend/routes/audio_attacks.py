"""
Audio attack API endpoints.
"""

import base64
import io
import os
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import APIRouter, File, Form, UploadFile, HTTPException

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (
    AVAILABLE_AUDIO_ATTACKS,
    device,
)
from backend.models.registry import (
    TASK_ASR,
    TASK_AUDIO_CLASSIFICATION,
    build_source_models_response,
    resolve_source_model,
)
from models.audio_loader import load_audio_model
from models.asr_loader import load_asr_model, WHISPER_SAMPLE_RATE
from utils.audio import load_audio, get_audio_predictions, get_audio_label_choices, parse_target_label, waveform_to_numpy
from utils.asr_utils import (
    load_audio_for_asr, compute_wer, compute_snr,
    waveform_to_numpy as asr_waveform_to_numpy,
)
from attacks.audio.router import run_audio_attack_method
from attacks.audio.transcription_attack import targeted_transcription_attack, targeted_transcription_pgd
from attacks.audio.hidden_command import hidden_command_attack, hidden_command_pgd
from attacks.audio.universal_muting import universal_muting_attack, apply_universal_segment, generate_training_waveforms
from attacks.audio.psychoacoustic_attack import psychoacoustic_transcription_attack
from attacks.audio.over_the_air_attack import over_the_air_attack
from attacks.audio.speech_jamming import speech_jamming_untargeted, speech_jamming_band_noise

router = APIRouter()

MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_OPTIMIZERS = {"C&W (Adam)", "PGD (Sign-based)"}
ALLOWED_MUTING_MODES = {"Mute (Silence)", "Targeted Override"}
ALLOWED_JAMMING_METHODS = {"Untargeted (Max CE)", "Band-Limited Noise"}


# ── Shared helpers ───────────────────────────────────────────────────────────

def _save_temp(content: bytes, tag: str) -> str:
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".wav", prefix=f"_{tag}_")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(content)
    return path


def _cleanup(path: str):
    if path and os.path.exists(path):
        os.remove(path)


def _convert_to_wav_if_needed(path: str) -> str:
    """If the file isn't readable by soundfile (e.g. webm, mp3), convert via ffmpeg."""
    import shutil, tempfile, subprocess
    try:
        sf.info(path)
        return path
    except Exception:
        pass
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(400, "Unsupported audio format and ffmpeg not available for conversion")
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="_conv_")
    os.close(fd)
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, timeout=30, check=True,
        )
        _cleanup(path)
        return wav_path
    except Exception as e:
        _cleanup(wav_path)
        raise HTTPException(400, f"Audio conversion failed: {e}")


def _load_asr(model_key: str):
    model_entry = resolve_source_model(model_key, domain="audio", task=TASK_ASR)
    if not model_entry or not model_entry.get("model_ref"):
        raise HTTPException(400, "Invalid ASR model")
    model_name = str(model_entry["model_ref"])
    return load_asr_model(model_name, progress=None)


def _load_asr_audio(content: bytes, tag: str):
    path = _save_temp(content, tag)
    path = _convert_to_wav_if_needed(path)
    try:
        waveform, sr = load_audio_for_asr(path, target_sr=WHISPER_SAMPLE_RATE)
        return waveform, sr, path
    except Exception:
        _cleanup(path)
        raise


def _encode_wav(waveform_tensor, sr: int) -> str:
    wav_np = asr_waveform_to_numpy(waveform_tensor)
    buf = io.BytesIO()
    sf.write(buf, wav_np, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def _asr_response(adv, waveform, wrapper, sr, **extra):
    """Build standard ASR attack response."""
    adv_text = wrapper.transcribe(adv)
    orig_text = extra.pop("original_text", None)
    if orig_text is None:
        try:
            orig_text = wrapper.transcribe(waveform)
        except Exception:
            orig_text = None
    resp = {
        "adversarial_wav_b64": _encode_wav(adv, sr),
        "sample_rate": sr,
        "adversarial_text": adv_text,
    }
    if orig_text is not None:
        resp["original_text"] = orig_text
    try:
        resp["snr_db"] = compute_snr(waveform, adv)
    except Exception:
        pass
    resp.update(extra)
    return resp


# ── Model listing endpoints ─────────────────────────────────────────────────

@router.get("/audio/models")
def list_audio_models():
    return build_source_models_response("audio", TASK_AUDIO_CLASSIFICATION)


@router.get("/audio/methods")
def list_audio_attacks():
    return {"attacks": list(AVAILABLE_AUDIO_ATTACKS.keys())}


@router.get("/asr/models")
def list_asr_models():
    return build_source_models_response("audio", TASK_ASR)


@router.get("/audio/labels/{model_key}")
def get_audio_labels(model_key: str):
    model_entry = resolve_source_model(model_key, domain="audio", task=TASK_AUDIO_CLASSIFICATION)
    if not model_entry or not model_entry.get("model_ref"):
        raise HTTPException(404, "Model not found")
    model_name = str(model_entry["model_ref"])
    try:
        wrapper, _, _ = load_audio_model(model_name)
        return {"labels": get_audio_label_choices(wrapper)}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Audio classification attack ──────────────────────────────────────────────

@router.post("/audio/classification/run")
async def run_audio_classification_attack(
    audio_file: UploadFile = File(...),
    model: str = Form(...),
    attack: str = Form("PGD"),
    target_label: str = Form(...),
    epsilon: float = Form(0.01),
    iterations: int = Form(40),
    alpha: float = Form(1.0),
    momentum_decay: float = Form(1.0),
    overshoot: float = Form(0.02),
    cw_confidence: float = Form(0.0),
    cw_lr: float = Form(0.01),
    cw_c: float = Form(1.0),
):
    if attack not in AVAILABLE_AUDIO_ATTACKS:
        raise HTTPException(400, f"Unknown audio attack: {attack}. Available: {list(AVAILABLE_AUDIO_ATTACKS.keys())}")
    model_entry = resolve_source_model(model, domain="audio", task=TASK_AUDIO_CLASSIFICATION)
    if not model_entry or not model_entry.get("model_ref"):
        raise HTTPException(400, "Invalid model")
    model_name = str(model_entry["model_ref"])
    try:
        content = await audio_file.read()
        if len(content) > MAX_AUDIO_SIZE:
            raise HTTPException(400, f"Audio file exceeds 50 MB limit ({len(content)} bytes)")
        wrapper, _, expected_sr = load_audio_model(model_name, progress=None)
        path = _save_temp(content, "ac")
        path = _convert_to_wav_if_needed(path)
        try:
            waveform, sr = load_audio(path, target_sr=expected_sr)
        finally:
            _cleanup(path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Invalid audio: {e}")

    try:
        target_idx = parse_target_label(target_label)
        attack_params = {
            "alpha": alpha, "momentum_decay": momentum_decay, "random_start": True,
            "overshoot": overshoot, "cw_confidence": cw_confidence,
            "cw_lr": cw_lr, "cw_c": cw_c,
        }
        adv = run_audio_attack_method(attack, wrapper, waveform, target_idx, epsilon, iterations, attack_params)
        adv_preds, adv_class, _ = get_audio_predictions(wrapper, adv)
        orig_preds, orig_class, _ = get_audio_predictions(wrapper, waveform)
        adv_np = waveform_to_numpy(adv)
        buf = io.BytesIO()
        sf.write(buf, adv_np, sr, format="WAV")
        return {
            "adversarial_wav_b64": base64.b64encode(buf.getvalue()).decode(),
            "sample_rate": expected_sr,
            "original_class": orig_class,
            "adversarial_class": adv_class,
            "target_idx": target_idx,
            "success": adv_class == wrapper.config.id2label.get(target_idx, f"class_{target_idx}"),
        }
    except Exception as e:
        raise HTTPException(500, f"Attack failed: {str(e)}")


# ── ASR: Targeted Transcription ──────────────────────────────────────────────

@router.post("/asr/transcription/run")
async def run_transcription_attack(
    audio_file: UploadFile = File(...),
    model: str = Form(...),
    target_text: str = Form(...),
    epsilon: float = Form(0.05),
    iterations: int = Form(300),
    lr: float = Form(0.005),
    optimizer: str = Form("C&W (Adam)"),
):
    if optimizer not in ALLOWED_OPTIMIZERS:
        raise HTTPException(400, f"Invalid optimizer: {optimizer}. Allowed: {sorted(ALLOWED_OPTIMIZERS)}")
    content = await audio_file.read()
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(400, f"Audio file exceeds 50 MB limit ({len(content)} bytes)")
    wrapper, _ = _load_asr(model)
    waveform, sr, path = _load_asr_audio(content, "trans")
    try:
        attack_fn = targeted_transcription_pgd if optimizer == "PGD (Sign-based)" else targeted_transcription_attack
        adv = attack_fn(wrapper, waveform, target_text.strip(), epsilon=epsilon, iterations=int(iterations), **({"alpha": lr} if optimizer == "PGD (Sign-based)" else {"lr": lr}))
        resp = _asr_response(adv, waveform, wrapper, sr, target_text=target_text.strip())
        resp["wer"] = compute_wer(target_text.strip(), resp.get("adversarial_text", ""))
        return resp
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Transcription attack failed: {str(e)}")
    finally:
        _cleanup(path)


# ── ASR: Hidden Command ──────────────────────────────────────────────────────

@router.post("/asr/hidden-command/run")
async def run_hidden_command_attack(
    carrier_file: UploadFile = File(...),
    model: str = Form(...),
    command_text: str = Form(...),
    epsilon: float = Form(0.1),
    iterations: int = Form(500),
    lr: float = Form(0.005),
    optimizer: str = Form("C&W (Adam)"),
):
    if optimizer not in ALLOWED_OPTIMIZERS:
        raise HTTPException(400, f"Invalid optimizer: {optimizer}. Allowed: {sorted(ALLOWED_OPTIMIZERS)}")
    content = await carrier_file.read()
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(400, f"Audio file exceeds 50 MB limit ({len(content)} bytes)")
    wrapper, _ = _load_asr(model)
    waveform, sr, path = _load_asr_audio(content, "hc")
    try:
        attack_fn = hidden_command_pgd if optimizer == "PGD (Sign-based)" else hidden_command_attack
        adv = attack_fn(wrapper, waveform, command_text.strip(), epsilon=epsilon, iterations=int(iterations), **({"alpha": lr} if optimizer == "PGD (Sign-based)" else {"lr": lr}))
        return _asr_response(adv, waveform, wrapper, sr, command_text=command_text.strip())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Hidden command attack failed: {str(e)}")
    finally:
        _cleanup(path)


# ── ASR: Universal Muting ────────────────────────────────────────────────────

@router.post("/asr/universal-muting/run")
async def run_universal_muting(
    audio_file: UploadFile = File(...),
    model: str = Form(...),
    mode: str = Form("Mute (Silence)"),
    target_text: str = Form(""),
    segment_duration: float = Form(0.64),
    iterations: int = Form(300),
    lr: float = Form(0.01),
):
    if mode not in ALLOWED_MUTING_MODES:
        raise HTTPException(400, f"Invalid mode: {mode}. Allowed: {sorted(ALLOWED_MUTING_MODES)}")
    if mode == "Targeted Override" and not target_text.strip():
        raise HTTPException(400, "target_text is required for Targeted Override mode")
    content = await audio_file.read()
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(400, f"Audio file exceeds 50 MB limit ({len(content)} bytes)")
    wrapper, _ = _load_asr(model)
    waveform, sr, path = _load_asr_audio(content, "um")
    try:
        training_waveforms = generate_training_waveforms(wrapper, waveform, n_augments=3)
        m = "target" if mode == "Targeted Override" else "mute"
        tgt = target_text.strip() if m == "target" else None
        segment = universal_muting_attack(wrapper, training_waveforms, segment_duration=segment_duration, iterations=iterations, lr=lr, mode=m, target_text=tgt)
        adv = apply_universal_segment(segment, waveform)
        return _asr_response(adv, waveform, wrapper, sr)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Universal muting attack failed: {str(e)}")
    finally:
        _cleanup(path)


# ── ASR: Psychoacoustic ──────────────────────────────────────────────────────

@router.post("/asr/psychoacoustic/run")
async def run_psychoacoustic_attack(
    audio_file: UploadFile = File(...),
    model: str = Form(...),
    target_text: str = Form(...),
    iterations: int = Form(500),
    lr: float = Form(0.005),
    masking_weight: float = Form(1.0),
):
    content = await audio_file.read()
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(400, f"Audio file exceeds 50 MB limit ({len(content)} bytes)")
    wrapper, _ = _load_asr(model)
    waveform, sr, path = _load_asr_audio(content, "psy")
    try:
        adv = psychoacoustic_transcription_attack(wrapper, waveform, target_text.strip(), iterations=iterations, lr=lr, masking_weight=masking_weight)
        return _asr_response(adv, waveform, wrapper, sr)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Psychoacoustic attack failed: {str(e)}")
    finally:
        _cleanup(path)


# ── ASR: Over-the-Air ────────────────────────────────────────────────────────

@router.post("/asr/over-the-air/run")
async def run_over_the_air_attack(
    audio_file: UploadFile = File(...),
    model: str = Form(...),
    target_text: str = Form(...),
    epsilon: float = Form(0.08),
    iterations: int = Form(500),
    lr: float = Form(0.005),
    n_rooms: int = Form(5),
    noise_snr_db: float = Form(20.0),
):
    content = await audio_file.read()
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(400, f"Audio file exceeds 50 MB limit ({len(content)} bytes)")
    wrapper, _ = _load_asr(model)
    waveform, sr, path = _load_asr_audio(content, "ota")
    try:
        adv = over_the_air_attack(wrapper, waveform, target_text.strip(), epsilon=epsilon, iterations=iterations, lr=lr, n_rooms=n_rooms, noise_snr_db=noise_snr_db)
        return _asr_response(adv, waveform, wrapper, sr)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Over-the-air attack failed: {str(e)}")
    finally:
        _cleanup(path)


# ── ASR: Speech Jamming ──────────────────────────────────────────────────────

@router.post("/asr/speech-jamming/run")
async def run_speech_jamming_attack(
    audio_file: UploadFile = File(...),
    model: str = Form(...),
    method: str = Form("Untargeted (Max CE)"),
    epsilon: float = Form(0.05),
    iterations: int = Form(300),
    lr: float = Form(0.005),
    band_low_hz: int = Form(300),
    band_high_hz: int = Form(4000),
):
    if method not in ALLOWED_JAMMING_METHODS:
        raise HTTPException(400, f"Invalid method: {method}. Allowed: {sorted(ALLOWED_JAMMING_METHODS)}")
    if method == "Band-Limited Noise" and band_high_hz <= band_low_hz:
        raise HTTPException(400, "band_high_hz must be greater than band_low_hz")
    content = await audio_file.read()
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(400, f"Audio file exceeds 50 MB limit ({len(content)} bytes)")
    wrapper, _ = _load_asr(model)
    waveform, sr, path = _load_asr_audio(content, "jam")
    try:
        if method == "Band-Limited Noise":
            adv = speech_jamming_band_noise(wrapper, waveform, epsilon=epsilon, iterations=iterations, lr=lr, band_low_hz=band_low_hz, band_high_hz=band_high_hz)
        else:
            adv = speech_jamming_untargeted(wrapper, waveform, epsilon=epsilon, iterations=iterations, lr=lr)
        return _asr_response(adv, waveform, wrapper, sr)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Speech jamming attack failed: {str(e)}")
    finally:
        _cleanup(path)


# ── ASR: UA3 (Universal Audio Adversarial Attack) ────────────────────────────

_ua3_model_cache = {}

UA3_MODEL_MAP = {
    "Whisper (base.en)": ["openai/whisper-base.en"],
    "Wav2Vec2": ["facebook/wav2vec2-base-960h"],
    "HuBERT": ["facebook/hubert-large-ls960-ft"],
    "Whisper + Wav2Vec2": ["openai/whisper-base.en", "facebook/wav2vec2-base-960h"],
    "Whisper + Wav2Vec2 + HuBERT": [
        "openai/whisper-base.en", "facebook/wav2vec2-base-960h", "facebook/hubert-large-ls960-ft",
    ],
}


def _get_ua3_models(model_names, dev):
    key = tuple(sorted(model_names))
    if key in _ua3_model_cache:
        return _ua3_model_cache[key]
    from lab.audio_benchmark.universal_attack import WhisperAttackModel, CTCAttackModel
    models = []
    for name in model_names:
        if "whisper" in name.lower():
            wrapper, processor = load_asr_model(name)
            models.append(WhisperAttackModel(wrapper, processor, dev))
        else:
            models.append(CTCAttackModel(name, dev))
    _ua3_model_cache[key] = models
    return models


@router.get("/asr/ua3/models")
def list_ua3_models():
    return {"models": list(UA3_MODEL_MAP.keys())}


@router.post("/asr/ua3/run")
async def run_ua3_attack(
    audio_file: UploadFile = File(...),
    target_text: str = Form(...),
    model_selection: str = Form("Whisper + Wav2Vec2"),
    iterations: int = Form(1500),
    epsilon: float = Form(0.08),
    lr: float = Form(0.005),
):
    selected = UA3_MODEL_MAP.get(model_selection)
    if not selected:
        raise HTTPException(400, f"Invalid model selection. Choose from: {list(UA3_MODEL_MAP.keys())}")

    content = await audio_file.read()
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(400, f"Audio file exceeds 50 MB limit ({len(content)} bytes)")
    waveform, sr, path = _load_asr_audio(content, "ua3")
    try:
        models = _get_ua3_models(selected, device)
        from lab.audio_benchmark.universal_attack import ua3_attack, _normalize_text
        adv_waveform = ua3_attack(models, waveform, target_text.strip(), iterations=iterations, lr=lr, linf_budget=epsilon, use_augment=True)

        target_norm = _normalize_text(target_text)
        per_model = []
        all_match = True
        with torch.no_grad():
            for m in models:
                text = m.transcribe(adv_waveform)
                matched = _normalize_text(text) == target_norm
                if not matched:
                    all_match = False
                per_model.append({"name": m.name, "text": text, "matched": matched})

            delta = adv_waveform - waveform
            snr_val = 10 * torch.log10(waveform.pow(2).mean() / delta.pow(2).mean().clamp(min=1e-10)).item()
            linf_val = delta.abs().max().item()

        adv_np = adv_waveform.squeeze().cpu().numpy()
        buf = io.BytesIO()
        sf.write(buf, adv_np, sr, format="WAV")

        return {
            "adversarial_wav_b64": base64.b64encode(buf.getvalue()).decode(),
            "sample_rate": sr,
            "target_text": target_text.strip(),
            "per_model_results": per_model,
            "all_match": all_match,
            "snr_db": snr_val,
            "linf": linf_val,
            "num_models": len(models),
        }
    except Exception as e:
        raise HTTPException(500, f"UA3 attack failed: {str(e)}")
    finally:
        _cleanup(path)


# ── Quick transcribe (preview) ───────────────────────────────────────────────

@router.post("/asr/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...), model: str = Form(...)):
    content = await audio_file.read()
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(400, f"Audio file exceeds 50 MB limit ({len(content)} bytes)")
    wrapper, _ = _load_asr(model)
    waveform, sr, path = _load_asr_audio(content, "tr")
    try:
        return {"text": wrapper.transcribe(waveform)}
    finally:
        _cleanup(path)
