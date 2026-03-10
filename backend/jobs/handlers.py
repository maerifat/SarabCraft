import base64
import io
import json
import logging
import time
from typing import Any

import numpy as np
import soundfile as sf

from backend.jobs.core import (
    append_job_event,
    cancel_job,
    complete_job,
    fail_job,
    is_cancel_requested,
    load_artifact_bytes,
    store_bytes_artifact,
    update_job_progress,
)
from backend.audio_attack_catalog import (
    audio_attack_requires_target_text,
    audio_attack_target_text,
    available_audio_attack_names,
    canonicalize_audio_attack_name,
    evaluate_audio_source_success,
)
from backend.routes import attacks as image_routes
from backend.routes import audio_attacks
from backend.routes import batch as batch_routes
from backend.routes import benchmark as benchmark_routes
from backend.routes.history import save_entry
from backend.models.registry import (
    TASK_ASR,
    TASK_AUDIO_CLASSIFICATION,
    TASK_IMAGE_CLASSIFICATION,
    list_source_models,
    resolve_source_model,
    snapshot_display_name,
    snapshot_entry,
    snapshot_model_ref,
)
from config import ATTACK_REGISTRY
from models.audio_loader import load_audio_model
from models.asr_loader import load_asr_model, WHISPER_SAMPLE_RATE
from models.loader import load_model
from utils.asr_utils import compute_snr, compute_wer, load_audio_for_asr, waveform_to_numpy as asr_waveform_to_numpy
from utils.attack_names import canonicalize_attack_name, normalize_attack_payload
from utils.audio import get_audio_predictions, load_audio, parse_target_label, waveform_to_numpy
from utils.image import get_predictions, preprocess_image, tensor_to_pil
from utils.metrics import compute_metrics
from attacks.audio_router import run_audio_attack_method
from attacks.hidden_command import hidden_command_attack, hidden_command_pgd
from attacks.over_the_air_attack import over_the_air_attack
from attacks.psychoacoustic_attack import psychoacoustic_transcription_attack
from attacks.router import AttackCancelledError, run_attack_method
from attacks.speech_jamming import speech_jamming_band_noise, speech_jamming_untargeted
from attacks.transcription_attack import targeted_transcription_attack, targeted_transcription_pgd
from attacks.universal_muting import apply_universal_segment, generate_training_waveforms, universal_muting_attack

logger = logging.getLogger("mlsec.jobs.handlers")

JOB_DEFINITIONS = {
    "image_attack": {"domain": "image", "title": "Image Attack", "resume_supported": False},
    "audio_classification": {"domain": "audio", "title": "Audio Classification Attack", "resume_supported": False},
    "asr_transcription": {"domain": "audio", "title": "Targeted Transcription Attack", "resume_supported": False},
    "asr_hidden_command": {"domain": "audio", "title": "Hidden Command Attack", "resume_supported": False},
    "asr_universal_muting": {"domain": "audio", "title": "Universal Muting Attack", "resume_supported": False},
    "asr_psychoacoustic": {"domain": "audio", "title": "Psychoacoustic Attack", "resume_supported": False},
    "asr_over_the_air": {"domain": "audio", "title": "Over-the-Air Attack", "resume_supported": False},
    "asr_speech_jamming": {"domain": "audio", "title": "Speech Jamming Attack", "resume_supported": False},
    "asr_ua3": {"domain": "audio", "title": "UA3 Attack", "resume_supported": False},
    "batch_attack": {"domain": "image", "title": "Batch Attack", "resume_supported": True},
    "image_robustness": {"domain": "image", "title": "Image Robustness Comparison", "resume_supported": True},
    "audio_robustness": {"domain": "audio", "title": "Audio Robustness Comparison", "resume_supported": True},
    "benchmark": {"domain": "mixed", "title": "Attack Benchmark", "resume_supported": True},
}


def get_job_definition(kind: str) -> dict | None:
    return JOB_DEFINITIONS.get(kind)


def _fields(job: dict) -> dict:
    return ((job.get("request") or {}).get("fields") or {})


def _files(job: dict) -> dict:
    return ((job.get("request") or {}).get("files") or {})


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() == "null":
        return None
    if text.startswith("{") and text.endswith("}") or text.startswith("[") and text.endswith("]"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    try:
        if text.startswith("-"):
            if text[1:].isdigit():
                return int(text)
        elif text.isdigit():
            return int(text)
        return float(text)
    except ValueError:
        return text


def _field(job: dict, key: str, default: Any = None) -> Any:
    fields = _fields(job)
    value = fields.get(key)
    if value is None:
        return default
    return value


def _field_float(job: dict, key: str, default: float) -> float:
    value = _field(job, key, default)
    return float(_coerce_scalar(value))


def _field_int(job: dict, key: str, default: int) -> int:
    value = _field(job, key, default)
    return int(float(_coerce_scalar(value)))


def _field_bool(job: dict, key: str, default: bool) -> bool:
    value = _field(job, key, default)
    coerced = _coerce_scalar(value)
    if isinstance(coerced, bool):
        return coerced
    if isinstance(coerced, (int, float)):
        return bool(coerced)
    return str(coerced).lower() in {"1", "true", "yes", "on"}


def _artifact_ref(job: dict, key: str) -> dict:
    ref = _files(job).get(key)
    if not ref:
        raise ValueError(f"Missing required artifact: {key}")
    if isinstance(ref, list):
        raise ValueError(f"Artifact {key} is multi-valued; expected one file")
    return ref


def _artifact_refs(job: dict, key: str) -> list[dict]:
    refs = _files(job).get(key)
    if not refs:
        raise ValueError(f"Missing required artifact list: {key}")
    if isinstance(refs, list):
        return refs
    return [refs]


def _download_one(job: dict, key: str) -> tuple[bytes, dict]:
    ref = _artifact_ref(job, key)
    return load_artifact_bytes(ref["storage_key"]), ref


def _download_many(job: dict, key: str) -> list[tuple[bytes, dict]]:
    out = []
    for ref in _artifact_refs(job, key):
        out.append((load_artifact_bytes(ref["storage_key"]), ref))
    return out


def _safe_error_message(kind: str) -> str:
    return f"{kind} failed. Check server logs for details."


def _persist_summary_artifact(job_id: str, result: dict) -> None:
    store_bytes_artifact(
        job_id,
        "result-summary",
        json.dumps(result, default=str).encode("utf-8"),
        filename="summary.json",
        mime_type="application/json",
    )


def _emit_init(job: dict, payload: dict) -> None:
    append_job_event(job["job_id"], "init", payload)


def _emit_progress(job: dict, payload: dict) -> None:
    append_job_event(job["job_id"], "progress", payload)


def _emit_result(job: dict, payload: dict) -> None:
    append_job_event(job["job_id"], "result", payload)


def _emit_summary(job: dict, payload: dict) -> None:
    append_job_event(job["job_id"], "summary", payload)


def _check_cancel(job: dict, partial_result: dict | None = None) -> bool:
    if not is_cancel_requested(job["job_id"]):
        return False
    append_job_event(job["job_id"], "error", {"message": "Job cancelled"})
    cancel_job(job["job_id"], result=partial_result)
    return True


def _write_wav_bytes(waveform_tensor, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, asr_waveform_to_numpy(waveform_tensor), sample_rate, format="WAV")
    return buf.getvalue()


def _field_json(job: dict, key: str, default: Any) -> Any:
    value = _coerce_scalar(_field(job, key, default))
    if isinstance(value, (dict, list)):
        return value
    return default


def _model_snapshot(job: dict, key: str = "model") -> dict | None:
    value = _field_json(job, f"{key}_snapshot_json", {})
    return value if isinstance(value, dict) and value else None


def _model_snapshot_list(job: dict, key: str) -> list[dict]:
    value = _field_json(job, key, [])
    return [item for item in value if isinstance(item, dict)]


def _resolve_job_model(job: dict, *, key: str, domain: str, task: str, default: str) -> tuple[str, dict | None]:
    snapshot = _model_snapshot(job, key)
    if snapshot and snapshot.get("model_ref"):
        return str(snapshot["model_ref"]), snapshot
    value = str(_field(job, key, default))
    entry = resolve_source_model(value, domain=domain, task=task)
    if not entry or not entry.get("model_ref"):
        raise ValueError(f"Unknown model: {value}")
    return str(entry["model_ref"]), snapshot_entry(entry)


def _run_image_attack(job: dict) -> dict:
    input_bytes, _ = _download_one(job, "input_file")
    target_bytes, _ = _download_one(job, "target_file")
    attack = canonicalize_attack_name(str(_field(job, "attack", "PGD")))
    model_name, model_snapshot = _resolve_job_model(
        job,
        key="model",
        domain="image",
        task=TASK_IMAGE_CLASSIFICATION,
        default="microsoft/resnet-50",
    )
    epsilon = _field_float(job, "epsilon", 16.0)
    iterations = _field_int(job, "iterations", 40)

    if attack not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {attack}")

    input_img = image_routes._parse_image(input_bytes)
    target_img = image_routes._parse_image(target_bytes)
    mdl, _ = load_model(model_name, progress=None)
    input_tensor = preprocess_image(input_img, model_name)
    target_tensor = preprocess_image(target_img, model_name)
    orig_preds, orig_class, _ = get_predictions(mdl, input_tensor)
    _, target_class, target_idx = get_predictions(mdl, target_tensor)

    attack_params = {}
    for key, value in _fields(job).items():
        if key in {"model", "attack", "epsilon", "iterations", "ensemble_models", "ensemble_mode"}:
            continue
        attack_params[key] = _coerce_scalar(value)
    attack_params["ensemble_mode"] = str(_field(job, "ensemble_mode", "Simultaneous")).lower()

    ensemble_models = image_routes._load_ensemble_models(
        _field(job, "ensemble_models"),
        model_name,
        ensemble_model_snapshots=_model_snapshot_list(job, "ensemble_model_snapshots_json"),
    )

    update_job_progress(
        job["job_id"],
        current=0,
        total=1,
        message=f"Running {attack} on {snapshot_display_name(model_snapshot, model_name)}",
    )
    if _check_cancel(job):
        return None
    try:
        adv_tensor = run_attack_method(
            attack,
            mdl,
            input_tensor,
            target_idx,
            epsilon / 255.0,
            iterations,
            attack_params,
            ensemble_models=ensemble_models or None,
            should_cancel=lambda: is_cancel_requested(job["job_id"]),
        )
    except AttackCancelledError:
        _check_cancel(job)
        return None
    if _check_cancel(job):
        return None
    adv_preds, adv_class, _ = get_predictions(mdl, adv_tensor)
    success = adv_class == target_class
    metrics = compute_metrics(input_tensor, adv_tensor)

    adversarial_b64 = image_routes._encode_image(tensor_to_pil(adv_tensor))
    perturbation_b64 = image_routes._build_perturbation_image(adv_tensor, input_tensor)
    result = {
        "adversarial_b64": adversarial_b64,
        "perturbation_b64": perturbation_b64,
        "original_class": orig_class,
        "adversarial_class": adv_class,
        "target_class": target_class,
        "original_preds": orig_preds,
        "adversarial_preds": adv_preds,
        "status": f"SUCCESS: classified as {adv_class}" if success else f"Partial: {adv_class} (target: {target_class})",
        "success": success,
        "metrics": metrics,
    }

    adv_artifact = store_bytes_artifact(
        job["job_id"],
        "adversarial-image",
        base64.b64decode(adversarial_b64),
        filename="adversarial.png",
        mime_type="image/png",
    )
    pert_artifact = store_bytes_artifact(
        job["job_id"],
        "perturbation-image",
        base64.b64decode(perturbation_b64),
        filename="perturbation.png",
        mime_type="image/png",
    )
    result["artifacts"] = [adv_artifact, pert_artifact]

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
        "adversarial_b64": adversarial_b64,
        "perturbation_b64": perturbation_b64,
        "ensemble_count": len(ensemble_models),
        "ensemble_models": _model_snapshot_list(job, "ensemble_model_snapshots_json"),
    })

    _persist_summary_artifact(job["job_id"], result)
    return result


def _run_audio_classification(job: dict) -> dict:
    content, _ = _download_one(job, "audio_file")
    model_name, model_snapshot = _resolve_job_model(
        job,
        key="model",
        domain="audio",
        task=TASK_AUDIO_CLASSIFICATION,
        default="",
    )
    attack = canonicalize_attack_name(str(_field(job, "attack", "PGD")))
    target_label = str(_field(job, "target_label"))
    epsilon = _field_float(job, "epsilon", 0.01)
    iterations = _field_int(job, "iterations", 40)
    alpha = _field_float(job, "alpha", 1.0)
    momentum_decay = _field_float(job, "momentum_decay", 1.0)
    overshoot = _field_float(job, "overshoot", 0.02)
    cw_confidence = _field_float(job, "cw_confidence", 0.0)
    cw_lr = _field_float(job, "cw_lr", 0.01)
    cw_c = _field_float(job, "cw_c", 1.0)

    wrapper, _, expected_sr = load_audio_model(model_name, progress=None)
    path = audio_attacks._save_temp(content, "job_ac")
    path = audio_attacks._convert_to_wav_if_needed(path)
    try:
        waveform, sr = load_audio(path, target_sr=expected_sr)
    finally:
        audio_attacks._cleanup(path)

    update_job_progress(
        job["job_id"],
        current=0,
        total=1,
        message=f"Running {attack} on {snapshot_display_name(model_snapshot, model_name)}",
    )

    target_idx = parse_target_label(target_label)
    attack_params = {
        "alpha": alpha,
        "momentum_decay": momentum_decay,
        "random_start": True,
        "overshoot": overshoot,
        "cw_confidence": cw_confidence,
        "cw_lr": cw_lr,
        "cw_c": cw_c,
    }
    adv = run_audio_attack_method(attack, wrapper, waveform, target_idx, epsilon, iterations, attack_params)
    adv_preds, adv_class, _ = get_audio_predictions(wrapper, adv)
    orig_preds, orig_class, _ = get_audio_predictions(wrapper, waveform)
    adv_np = waveform_to_numpy(adv)
    buf = io.BytesIO()
    sf.write(buf, adv_np, sr, format="WAV")
    wav_b64 = base64.b64encode(buf.getvalue()).decode()

    result = {
        "adversarial_wav_b64": wav_b64,
        "sample_rate": expected_sr,
        "original_class": orig_class,
        "adversarial_class": adv_class,
        "target_idx": target_idx,
        "success": adv_class == wrapper.config.id2label.get(target_idx, f"class_{target_idx}"),
    }
    artifact = store_bytes_artifact(
        job["job_id"],
        "adversarial-audio",
        buf.getvalue(),
        filename="adversarial.wav",
        mime_type="audio/wav",
    )
    result["artifacts"] = [artifact]
    _persist_summary_artifact(job["job_id"], result)
    return result


def _load_asr_wrapper(model_key: str):
    entry = resolve_source_model(model_key, domain="audio", task=TASK_ASR)
    if not entry or not entry.get("model_ref"):
        raise ValueError("Invalid ASR model")
    model_name = str(entry["model_ref"])
    return load_asr_model(model_name, progress=None)


def _load_asr_from_job(job: dict) -> tuple[Any, Any, str, dict | None]:
    model_name, model_snapshot = _resolve_job_model(
        job,
        key="model",
        domain="audio",
        task=TASK_ASR,
        default="",
    )
    wrapper, processor = load_asr_model(model_name, progress=None)
    return wrapper, processor, model_name, model_snapshot


def _load_asr_audio_from_bytes(content: bytes, tag: str):
    path = audio_attacks._save_temp(content, tag)
    path = audio_attacks._convert_to_wav_if_needed(path)
    try:
        waveform, sr = load_audio_for_asr(path, target_sr=WHISPER_SAMPLE_RATE)
        return waveform, sr
    finally:
        audio_attacks._cleanup(path)


def _asr_response_with_artifact(job: dict, result: dict, waveform_key: str = "adversarial_wav_b64") -> dict:
    if result.get(waveform_key):
        artifact = store_bytes_artifact(
            job["job_id"],
            "adversarial-audio",
            base64.b64decode(result[waveform_key]),
            filename="adversarial.wav",
            mime_type="audio/wav",
        )
        result["artifacts"] = [artifact]
    _persist_summary_artifact(job["job_id"], result)
    return result


def _run_asr_transcription(job: dict) -> dict:
    content, _ = _download_one(job, "audio_file")
    target_text = str(_field(job, "target_text", "")).strip()
    epsilon = _field_float(job, "epsilon", 0.05)
    iterations = _field_int(job, "iterations", 300)
    lr = _field_float(job, "lr", 0.005)
    optimizer = str(_field(job, "optimizer", "C&W (Adam)"))
    if optimizer not in audio_attacks.ALLOWED_OPTIMIZERS:
        raise ValueError("Invalid optimizer")

    wrapper, _, model_name, model_snapshot = _load_asr_from_job(job)
    waveform, sr = _load_asr_audio_from_bytes(content, "job_trans")
    attack_fn = targeted_transcription_pgd if optimizer == "PGD (Sign-based)" else targeted_transcription_attack
    kwargs = {"alpha": lr} if optimizer == "PGD (Sign-based)" else {"lr": lr}

    update_job_progress(
        job["job_id"],
        current=0,
        total=1,
        message=f"Running targeted transcription on {snapshot_display_name(model_snapshot, model_name)}",
    )
    adv = attack_fn(wrapper, waveform, target_text, epsilon=epsilon, iterations=iterations, **kwargs)
    result = audio_attacks._asr_response(adv, waveform, wrapper, sr, target_text=target_text)
    result["wer"] = compute_wer(target_text, result.get("adversarial_text", ""))
    return _asr_response_with_artifact(job, result)


def _run_asr_hidden_command(job: dict) -> dict:
    content, _ = _download_one(job, "carrier_file")
    command_text = str(_field(job, "command_text", "")).strip()
    epsilon = _field_float(job, "epsilon", 0.1)
    iterations = _field_int(job, "iterations", 500)
    lr = _field_float(job, "lr", 0.005)
    optimizer = str(_field(job, "optimizer", "C&W (Adam)"))
    if optimizer not in audio_attacks.ALLOWED_OPTIMIZERS:
        raise ValueError("Invalid optimizer")

    wrapper, _, model_name, model_snapshot = _load_asr_from_job(job)
    waveform, sr = _load_asr_audio_from_bytes(content, "job_hidden")
    attack_fn = hidden_command_pgd if optimizer == "PGD (Sign-based)" else hidden_command_attack
    kwargs = {"alpha": lr} if optimizer == "PGD (Sign-based)" else {"lr": lr}

    update_job_progress(
        job["job_id"],
        current=0,
        total=1,
        message=f"Running hidden command attack on {snapshot_display_name(model_snapshot, model_name)}",
    )
    adv = attack_fn(wrapper, waveform, command_text, epsilon=epsilon, iterations=iterations, **kwargs)
    result = audio_attacks._asr_response(adv, waveform, wrapper, sr, command_text=command_text)
    return _asr_response_with_artifact(job, result)


def _run_asr_universal_muting(job: dict) -> dict:
    content, _ = _download_one(job, "audio_file")
    mode = str(_field(job, "mode", "Mute (Silence)"))
    target_text = str(_field(job, "target_text", "")).strip()
    segment_duration = _field_float(job, "segment_duration", 0.64)
    iterations = _field_int(job, "iterations", 300)
    lr = _field_float(job, "lr", 0.01)

    wrapper, _, model_name, model_snapshot = _load_asr_from_job(job)
    waveform, sr = _load_asr_audio_from_bytes(content, "job_muting")

    update_job_progress(
        job["job_id"],
        current=0,
        total=1,
        message=f"Running universal muting attack on {snapshot_display_name(model_snapshot, model_name)}",
    )
    training_waveforms = generate_training_waveforms(wrapper, waveform, n_augments=3)
    mode_key = "target" if mode == "Targeted Override" else "mute"
    segment = universal_muting_attack(
        wrapper,
        training_waveforms,
        segment_duration=segment_duration,
        iterations=iterations,
        lr=lr,
        mode=mode_key,
        target_text=target_text if mode_key == "target" else None,
    )
    adv = apply_universal_segment(segment, waveform)
    result = audio_attacks._asr_response(adv, waveform, wrapper, sr)
    return _asr_response_with_artifact(job, result)


def _run_asr_psychoacoustic(job: dict) -> dict:
    content, _ = _download_one(job, "audio_file")
    target_text = str(_field(job, "target_text", "")).strip()
    iterations = _field_int(job, "iterations", 500)
    lr = _field_float(job, "lr", 0.005)
    masking_weight = _field_float(job, "masking_weight", 1.0)

    wrapper, _, model_name, model_snapshot = _load_asr_from_job(job)
    waveform, sr = _load_asr_audio_from_bytes(content, "job_psy")
    update_job_progress(
        job["job_id"],
        current=0,
        total=1,
        message=f"Running psychoacoustic attack on {snapshot_display_name(model_snapshot, model_name)}",
    )
    adv = psychoacoustic_transcription_attack(
        wrapper,
        waveform,
        target_text,
        iterations=iterations,
        lr=lr,
        masking_weight=masking_weight,
    )
    result = audio_attacks._asr_response(adv, waveform, wrapper, sr)
    return _asr_response_with_artifact(job, result)


def _run_asr_over_the_air(job: dict) -> dict:
    content, _ = _download_one(job, "audio_file")
    target_text = str(_field(job, "target_text", "")).strip()
    epsilon = _field_float(job, "epsilon", 0.08)
    iterations = _field_int(job, "iterations", 500)
    lr = _field_float(job, "lr", 0.005)
    n_rooms = _field_int(job, "n_rooms", 5)
    noise_snr_db = _field_float(job, "noise_snr_db", 20.0)

    wrapper, _, model_name, model_snapshot = _load_asr_from_job(job)
    waveform, sr = _load_asr_audio_from_bytes(content, "job_ota")
    update_job_progress(
        job["job_id"],
        current=0,
        total=1,
        message=f"Running over-the-air attack on {snapshot_display_name(model_snapshot, model_name)}",
    )
    adv = over_the_air_attack(
        wrapper,
        waveform,
        target_text,
        epsilon=epsilon,
        iterations=iterations,
        lr=lr,
        n_rooms=n_rooms,
        noise_snr_db=noise_snr_db,
    )
    result = audio_attacks._asr_response(adv, waveform, wrapper, sr)
    return _asr_response_with_artifact(job, result)


def _run_asr_speech_jamming(job: dict) -> dict:
    content, _ = _download_one(job, "audio_file")
    method = str(_field(job, "method", "Untargeted (Max CE)"))
    epsilon = _field_float(job, "epsilon", 0.05)
    iterations = _field_int(job, "iterations", 300)
    lr = _field_float(job, "lr", 0.005)
    band_low_hz = _field_int(job, "band_low_hz", 300)
    band_high_hz = _field_int(job, "band_high_hz", 4000)

    wrapper, _, model_name, model_snapshot = _load_asr_from_job(job)
    waveform, sr = _load_asr_audio_from_bytes(content, "job_jam")
    update_job_progress(
        job["job_id"],
        current=0,
        total=1,
        message=f"Running speech jamming attack on {snapshot_display_name(model_snapshot, model_name)}",
    )
    if method == "Band-Limited Noise":
        adv = speech_jamming_band_noise(
            wrapper,
            waveform,
            epsilon=epsilon,
            iterations=iterations,
            lr=lr,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
        )
    else:
        adv = speech_jamming_untargeted(wrapper, waveform, epsilon=epsilon, iterations=iterations, lr=lr)
    result = audio_attacks._asr_response(adv, waveform, wrapper, sr)
    return _asr_response_with_artifact(job, result)


def _run_asr_ua3(job: dict) -> dict:
    content, _ = _download_one(job, "audio_file")
    target_text = str(_field(job, "target_text", "")).strip()
    model_selection = str(_field(job, "model_selection", "Whisper + Wav2Vec2"))
    iterations = _field_int(job, "iterations", 1500)
    epsilon = _field_float(job, "epsilon", 0.08)
    lr = _field_float(job, "lr", 0.005)

    selected = audio_attacks.UA3_MODEL_MAP.get(model_selection)
    if not selected:
        raise ValueError("Invalid UA3 model selection")

    waveform, sr = _load_asr_audio_from_bytes(content, "job_ua3")
    models = audio_attacks._get_ua3_models(selected, audio_attacks.device)
    from lab.audio_benchmark.universal_attack import _normalize_text, ua3_attack

    update_job_progress(job["job_id"], current=0, total=1, message=f"Running UA3 on {model_selection}")
    adv_waveform = ua3_attack(
        models,
        waveform,
        target_text,
        iterations=iterations,
        lr=lr,
        linf_budget=epsilon,
        use_augment=True,
    )

    target_norm = _normalize_text(target_text)
    per_model = []
    all_match = True
    for model in models:
        text = model.transcribe(adv_waveform)
        matched = _normalize_text(text) == target_norm
        if not matched:
            all_match = False
        per_model.append({"name": model.name, "text": text, "matched": matched})

    delta = adv_waveform - waveform
    snr_val = 10 * audio_attacks.torch.log10(waveform.pow(2).mean() / delta.pow(2).mean().clamp(min=1e-10)).item()
    linf_val = delta.abs().max().item()
    wav_bytes = _write_wav_bytes(adv_waveform, sr)
    result = {
        "adversarial_wav_b64": base64.b64encode(wav_bytes).decode(),
        "sample_rate": sr,
        "target_text": target_text,
        "per_model_results": per_model,
        "all_match": all_match,
        "snr_db": snr_val,
        "linf": linf_val,
        "num_models": len(models),
    }
    artifact = store_bytes_artifact(
        job["job_id"],
        "adversarial-audio",
        wav_bytes,
        filename="adversarial.wav",
        mime_type="audio/wav",
    )
    result["artifacts"] = [artifact]
    _persist_summary_artifact(job["job_id"], result)
    return result


def _batch_summary(results: list[dict], total: int, attack: str, model_name: str, target_class: str) -> dict:
    successes = sum(1 for row in results if row.get("success"))
    metric_results = [row["metrics"] for row in results if row.get("metrics")]
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


def _run_batch_attack(job: dict) -> dict | None:
    inputs = _download_many(job, "input_files")
    target_bytes, _ = _download_one(job, "target_file")
    attack = str(_field(job, "attack", "PGD"))
    model_name, model_snapshot = _resolve_job_model(
        job,
        key="model",
        domain="image",
        task=TASK_IMAGE_CLASSIFICATION,
        default="microsoft/resnet-50",
    )
    model_label = snapshot_display_name(model_snapshot, model_name) or model_name
    epsilon = _field_float(job, "epsilon", 16.0)
    iterations = _field_int(job, "iterations", 40)

    if len(inputs) > batch_routes.MAX_BATCH_IMAGES:
        raise ValueError(f"Too many images: {len(inputs)}")
    if attack not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {attack}")

    target_img = batch_routes._parse_image(target_bytes)
    mdl, _ = load_model(model_name, progress=None)
    target_tensor = preprocess_image(target_img, model_name)
    _, target_class, target_idx = get_predictions(mdl, target_tensor)

    total = len(inputs)
    current_result = normalize_attack_payload(job.get("result") or {})
    results = list(current_result.get("results") or [])
    start_index = max(job.get("progress", {}).get("current", 0), len(results))

    update_job_progress(job["job_id"], current=start_index, total=total, message="Running batch attack", result=current_result or None)
    _emit_init(job, {"total": total, "attack": attack, "model": model_label, "domain": "image", "resume_from": start_index})

    for idx in range(start_index, total):
        if _check_cancel(job, _batch_summary(results, total, attack, model_label, target_class)):
            return None

        data, ref = inputs[idx]
        entry = {"filename": ref.get("filename"), "index": idx}
        _emit_progress(job, {"index": idx, "total": total, "filename": ref.get("filename"), "status": "running"})
        try:
            img = batch_routes._parse_image(data)
            input_tensor = preprocess_image(img, model_name)
            orig_preds, orig_class, _ = get_predictions(mdl, input_tensor)
            adv_tensor = run_attack_method(
                attack,
                mdl,
                input_tensor,
                target_idx,
                epsilon / 255.0,
                iterations,
                {},
                should_cancel=lambda: is_cancel_requested(job["job_id"]),
            )
            if _check_cancel(job, _batch_summary(results, total, attack, model_label, target_class)):
                return None
            adv_preds, adv_class, _ = get_predictions(mdl, adv_tensor)
            metrics = compute_metrics(input_tensor, adv_tensor)
            success = adv_class == target_class
            entry.update({
                "success": success,
                "original_class": orig_class,
                "adversarial_class": adv_class,
                "metrics": metrics,
            })
            save_entry({
                "domain": "image",
                "attack": attack,
                "model": model_label,
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
                "batch": True,
            })
        except AttackCancelledError:
            if _check_cancel(job, _batch_summary(results, total, attack, model_label, target_class)):
                return None
            raise
        except Exception as exc:
            entry["error"] = str(exc)
            entry["success"] = False

        results.append(entry)
        partial = _batch_summary(results, total, attack, model_label, target_class)
        update_job_progress(
            job["job_id"],
            current=idx + 1,
            total=total,
            message=f"Processed {idx + 1}/{total} images",
            result=partial,
            checkpoint={"last_completed_index": idx},
        )
        _emit_result(job, entry)

    summary = _batch_summary(results, total, attack, model_label, target_class)
    _emit_summary(job, summary)
    _persist_summary_artifact(job["job_id"], summary)
    return summary


def _robustness_summary(results: list[dict], attack: str, epsilon: float, iterations: int) -> dict:
    successes = sum(1 for row in results if row.get("success"))
    return {
        "attack": attack,
        "epsilon": epsilon,
        "iterations": iterations,
        "total_models": len(results),
        "successful_transfers": successes,
        "transfer_rate": successes / len(results) if results else 0,
        "results": results,
    }


def _run_image_robustness(job: dict) -> dict | None:
    input_bytes, _ = _download_one(job, "input_file")
    target_bytes, _ = _download_one(job, "target_file")
    attack = canonicalize_attack_name(str(_field(job, "attack", "PGD")))
    epsilon = _field_float(job, "epsilon", 16.0)
    iterations = _field_int(job, "iterations", 40)

    model_keys = _coerce_scalar(_field(job, "models_json", "[]"))
    if not isinstance(model_keys, list):
        raise ValueError("models_json must be a list")
    if len(model_keys) > batch_routes.MAX_ROBUSTNESS_MODELS:
        raise ValueError(f"Too many models: {len(model_keys)}")
    model_snapshots = _model_snapshot_list(job, "models_snapshot_json")
    if not model_keys:
        model_keys = [item["id"] for item in list_source_models("image", task=TASK_IMAGE_CLASSIFICATION)[:10]]

    extra_params = _coerce_scalar(_field(job, "attack_params_json", "{}"))
    if not isinstance(extra_params, dict):
        extra_params = {}
    attack_extra = {k: v for k, v in extra_params.items() if k not in ("epsilon", "iterations")}
    input_img = batch_routes._parse_image(input_bytes)
    target_img = batch_routes._parse_image(target_bytes)

    total = len(model_keys)
    current_result = normalize_attack_payload(job.get("result") or {})
    results = list(current_result.get("results") or [])
    start_index = max(job.get("progress", {}).get("current", 0), len(results))
    update_job_progress(job["job_id"], current=start_index, total=total, message="Running robustness comparison", result=current_result or None)
    _emit_init(job, {"attack": attack, "epsilon": epsilon, "iterations": iterations, "total_models": total, "resume_from": start_index})

    for idx in range(start_index, total):
        if _check_cancel(job, _robustness_summary(results, attack, epsilon, iterations)):
            return None

        mk = model_keys[idx]
        model_snapshot = model_snapshots[idx] if idx < len(model_snapshots) else snapshot_entry(
            resolve_source_model(mk, domain="image", task=TASK_IMAGE_CLASSIFICATION)
        )
        model_name = snapshot_model_ref(model_snapshot, None)
        if not model_name:
            raise ValueError(f"Unknown model: {mk}")
        short_name = snapshot_display_name(model_snapshot, model_name) or model_name
        entry = {
            "model": snapshot_display_name(model_snapshot, model_name),
            "model_key": mk,
            "model_ref": model_name,
            "model_snapshot": model_snapshot,
            "index": idx,
        }
        _emit_progress(job, {"index": idx, "total": total, "model": short_name, "status": "running"})
        t0 = time.time()
        try:
            mdl, _ = load_model(model_name, progress=None)
            input_tensor = preprocess_image(input_img, model_name)
            target_tensor = preprocess_image(target_img, model_name)
            orig_preds, orig_class, _ = get_predictions(mdl, input_tensor)
            _, target_class, target_idx = get_predictions(mdl, target_tensor)
            adv_tensor = run_attack_method(
                attack,
                mdl,
                input_tensor,
                target_idx,
                epsilon / 255.0,
                iterations,
                attack_extra,
                should_cancel=lambda: is_cancel_requested(job["job_id"]),
            )
            if _check_cancel(job, _robustness_summary(results, attack, epsilon, iterations)):
                return None
            adv_preds, adv_class, _ = get_predictions(mdl, adv_tensor)
            metrics = compute_metrics(input_tensor, adv_tensor)
            success = adv_class == target_class
            entry.update({
                "success": success,
                "original_class": orig_class,
                "adversarial_class": adv_class,
                "target_class": target_class,
                "metrics": metrics,
                "elapsed_ms": round((time.time() - t0) * 1000),
                "top_adv_conf": max(adv_preds.values()) if adv_preds else 0,
            })
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
                "robustness_test": True,
            })
        except AttackCancelledError:
            if _check_cancel(job, _robustness_summary(results, attack, epsilon, iterations)):
                return None
            raise
        except Exception as exc:
            entry["error"] = str(exc)
            entry["success"] = False
            entry["elapsed_ms"] = round((time.time() - t0) * 1000)

        results.append(entry)
        partial = _robustness_summary(results, attack, epsilon, iterations)
        update_job_progress(
            job["job_id"],
            current=idx + 1,
            total=total,
            message=f"Tested {idx + 1}/{total} models",
            result=partial,
            checkpoint={"last_completed_index": idx},
        )
        _emit_result(job, entry)

    summary = _robustness_summary(results, attack, epsilon, iterations)
    _emit_summary(job, summary)
    _persist_summary_artifact(job["job_id"], summary)
    return summary


def _audio_robustness_summary(results: list[dict], attack: str) -> dict:
    successes = sum(1 for row in results if row.get("success"))
    return {
        "attack": attack,
        "domain": "audio",
        "total_models": len(results),
        "successful_attacks": successes,
        "attack_success_rate": successes / len(results) if results else 0,
        "results": results,
    }


def _run_audio_robustness(job: dict) -> dict | None:
    audio_bytes, _ = _download_one(job, "audio_file")
    attack = canonicalize_audio_attack_name(_field(job, "attack", "Targeted Transcription"))
    target_text = str(_field(job, "target_text", "")).strip()
    model_keys = _coerce_scalar(_field(job, "models_json", "[]"))
    if not isinstance(model_keys, list):
        raise ValueError("models_json must be a list")
    if len(model_keys) > batch_routes.MAX_AUDIO_ROBUSTNESS_MODELS:
        raise ValueError(f"Too many models: {len(model_keys)}")
    model_snapshots = _model_snapshot_list(job, "models_snapshot_json")
    if not model_keys:
        model_keys = [item["id"] for item in list_source_models("audio", task=TASK_ASR)[:4]]

    extra_params = _coerce_scalar(_field(job, "attack_params_json", "{}"))
    if not isinstance(extra_params, dict):
        extra_params = {}
    atk_key = batch_routes.AUDIO_ATK_MAP.get(attack)
    if atk_key is None:
        raise ValueError(f"Unknown audio attack: {attack}")
    effective_target_text = audio_attack_target_text(attack, target_text)

    wav_tensor, sr, raw_np = batch_routes._load_audio_bytes(audio_bytes)
    total = len(model_keys)
    current_result = job.get("result") or {}
    results = list(current_result.get("results") or [])
    start_index = max(job.get("progress", {}).get("current", 0), len(results))
    update_job_progress(job["job_id"], current=start_index, total=total, message="Running audio robustness comparison", result=current_result or None)
    _emit_init(job, {"attack": attack, "total_models": total, "domain": "audio", "params": extra_params, "resume_from": start_index})

    for idx in range(start_index, total):
        if _check_cancel(job, _audio_robustness_summary(results, attack)):
            return None

        mk = model_keys[idx]
        model_snapshot = model_snapshots[idx] if idx < len(model_snapshots) else snapshot_entry(
            resolve_source_model(mk, domain="audio", task=TASK_ASR)
        )
        model_name = snapshot_model_ref(model_snapshot, None)
        if not model_name:
            raise ValueError(f"Unknown ASR model: {mk}")
        short_name = snapshot_display_name(model_snapshot, model_name) or model_name
        entry = {
            "model": snapshot_display_name(model_snapshot, model_name),
            "model_key": mk,
            "model_ref": model_name,
            "model_snapshot": model_snapshot,
            "index": idx,
        }
        _emit_progress(job, {"index": idx, "total": total, "model": short_name, "status": "running"})
        t0 = time.time()
        try:
            adv_wav, result_text, orig_text = batch_routes._run_audio_attack_for_model(
                atk_key,
                wav_tensor,
                sr,
                mk,
                effective_target_text,
                extra_params,
            )
            snr_val = compute_snr(wav_tensor, adv_wav) if adv_wav is not None else 0
            success = evaluate_audio_source_success(
                attack,
                result_text,
                target_text=effective_target_text,
                original_text=orig_text,
            )
            adv_np = adv_wav.squeeze().cpu().numpy() if adv_wav is not None else raw_np
            adv_b64 = batch_routes._encode_wav(adv_np, sr)
            entry.update({
                "success": success,
                "original_text": orig_text or "",
                "result_text": result_text or "",
                "target_text": effective_target_text,
                "evaluation_mode": "untargeted" if atk_key == "jamming" else "targeted",
                "snr_db": round(snr_val, 1) if snr_val else 0,
                "adversarial_b64": adv_b64,
                "elapsed_ms": round((time.time() - t0) * 1000),
            })
            save_entry({
                "domain": "audio",
                "attack": attack,
                "model": snapshot_display_name(model_snapshot, model_name),
                "model_id": model_snapshot.get("id") if model_snapshot else None,
                "model_ref": model_name,
                "model_snapshot": model_snapshot,
                "success": success,
                "original_text": orig_text,
                "result_text": result_text,
                "target_text": effective_target_text,
                "robustness_test": True,
            })
        except Exception as exc:
            entry["error"] = str(exc)
            entry["success"] = False
            entry["elapsed_ms"] = round((time.time() - t0) * 1000)

        results.append(entry)
        partial = _audio_robustness_summary(results, attack)
        update_job_progress(
            job["job_id"],
            current=idx + 1,
            total=total,
            message=f"Tested {idx + 1}/{total} ASR models",
            result=partial,
            checkpoint={"last_completed_index": idx},
        )
        _emit_result(job, entry)

    summary = _audio_robustness_summary(results, attack)
    _emit_summary(job, summary)
    _persist_summary_artifact(job["job_id"], summary)
    return summary


def _run_benchmark(job: dict) -> dict | None:
    domain = str(_field(job, "domain", "image"))
    attacks = _coerce_scalar(_field(job, "attacks_json", "[]"))
    if not isinstance(attacks, list) or not attacks:
        raise ValueError("Select at least one attack")
    attacks = [canonicalize_attack_name(name) for name in attacks]

    param_mode = str(_field(job, "param_mode", "preset"))
    param_preset = str(_field(job, "param_preset", "balanced"))
    sweep_cfg = _coerce_scalar(_field(job, "param_sweep_json", "{}"))
    transfer_targets = _coerce_scalar(_field(job, "transfer_targets_json", "{}"))
    transfer_target_snapshots = _field_json(job, "transfer_targets_snapshot_json", {})
    if not isinstance(sweep_cfg, dict):
        sweep_cfg = {}
    if not isinstance(transfer_targets, dict):
        transfer_targets = {}
    if not isinstance(transfer_target_snapshots, dict):
        transfer_target_snapshots = {}

    source_model = str(_field(job, "source_model", "microsoft/resnet-50"))
    source_model_snapshot = _model_snapshot(job, "source_model")
    source_model_ref = snapshot_model_ref(source_model_snapshot, source_model) or source_model
    source_model_label = snapshot_display_name(source_model_snapshot, source_model_ref) or source_model_ref
    current_result = normalize_attack_payload(job.get("result") or {})
    results = list(current_result.get("results") or [])
    start_index = max(job.get("progress", {}).get("current", 0), len(results))

    if domain == "image":
        input_bytes, _ = _download_one(job, "input_file")
        target_bytes, _ = _download_one(job, "target_file")
        valid_attacks = [name for name in attacks if name in ATTACK_REGISTRY]
        if not valid_attacks:
            raise ValueError("No valid image attacks selected")
        combos = benchmark_routes._build_image_param_combos(valid_attacks, param_mode, param_preset, sweep_cfg)
        if len(combos) > benchmark_routes.MAX_COMBOS:
            raise ValueError(f"Too many combinations ({len(combos)})")
        total = len(combos)
        update_job_progress(job["job_id"], current=start_index, total=total, message="Running benchmark", result=current_result or None)
        _emit_init(job, {"total": total, "domain": "image", "attacks": valid_attacks, "source_model": source_model_label, "resume_from": start_index})

        input_img = benchmark_routes._parse_image(input_bytes)
        target_img = benchmark_routes._parse_image(target_bytes)
        remaining = combos[start_index:]
        for local_idx, row in enumerate(
            benchmark_routes._run_image_benchmark(
                input_img,
                target_img,
                source_model_ref,
                remaining,
                transfer_target_snapshots or transfer_targets,
                str(_field(job, "preprocess_mode", "exact")),
                should_cancel=lambda: is_cancel_requested(job["job_id"]),
            ),
            start=start_index,
        ):
            if _check_cancel(job, {"results": results, **benchmark_routes._build_summary(results)}):
                return None
            row["index"] = local_idx
            results.append(row)
            partial = {"results": results, **benchmark_routes._build_summary(results)}
            update_job_progress(
                job["job_id"],
                current=local_idx + 1,
                total=total,
                message=f"Completed {local_idx + 1}/{total} benchmark combinations",
                result=partial,
                checkpoint={"last_completed_index": local_idx},
            )
            _emit_result(job, row)

        if _check_cancel(job, {"results": results, **benchmark_routes._build_summary(results)}):
            return None
        summary = benchmark_routes._build_summary(results)
        summary["results"] = results
        _emit_summary(job, summary)
        _persist_summary_artifact(job["job_id"], summary)
        return summary

    if domain == "audio":
        input_bytes, _ = _download_one(job, "input_file")
        target_text = str(_field(job, "target_text", "")).strip()
        normalized_attacks = [canonicalize_audio_attack_name(name) for name in attacks]
        valid_attacks = [name for name in normalized_attacks if name in set(available_audio_attack_names())]
        if not valid_attacks:
            raise ValueError("No valid audio attacks selected")
        if any(audio_attack_requires_target_text(name) for name in valid_attacks) and not target_text:
            raise ValueError("target_text is required for targeted audio benchmark")
        combos = benchmark_routes._build_audio_param_combos(valid_attacks, param_mode, param_preset, sweep_cfg)
        if len(combos) > benchmark_routes.MAX_COMBOS:
            raise ValueError(f"Too many combinations ({len(combos)})")

        total = len(combos)
        update_job_progress(job["job_id"], current=start_index, total=total, message="Running audio benchmark", result=current_result or None)
        _emit_init(job, {"total": total, "domain": "audio", "attacks": valid_attacks, "source_model": source_model_label, "resume_from": start_index})

        remaining = combos[start_index:]
        for local_idx, row in enumerate(
            benchmark_routes._run_audio_benchmark(
                input_bytes,
                source_model_ref,
                target_text,
                remaining,
                transfer_target_snapshots or transfer_targets,
            ),
            start=start_index,
        ):
            if _check_cancel(job, {"results": results, **benchmark_routes._build_summary(results)}):
                return None
            row["index"] = local_idx
            results.append(row)
            partial = {"results": results, **benchmark_routes._build_summary(results)}
            update_job_progress(
                job["job_id"],
                current=local_idx + 1,
                total=total,
                message=f"Completed {local_idx + 1}/{total} benchmark combinations",
                result=partial,
                checkpoint={"last_completed_index": local_idx},
            )
            _emit_result(job, row)

        if _check_cancel(job, {"results": results, **benchmark_routes._build_summary(results)}):
            return None
        summary = benchmark_routes._build_summary(results)
        summary["results"] = results
        _emit_summary(job, summary)
        _persist_summary_artifact(job["job_id"], summary)
        return summary

    raise ValueError("Unsupported benchmark domain")


def run_job(job: dict) -> dict | None:
    kind = job["kind"]
    logger.info("Running job %s (%s)", job["job_id"], kind)

    try:
        if kind == "image_attack":
            result = _run_image_attack(job)
        elif kind == "audio_classification":
            result = _run_audio_classification(job)
        elif kind == "asr_transcription":
            result = _run_asr_transcription(job)
        elif kind == "asr_hidden_command":
            result = _run_asr_hidden_command(job)
        elif kind == "asr_universal_muting":
            result = _run_asr_universal_muting(job)
        elif kind == "asr_psychoacoustic":
            result = _run_asr_psychoacoustic(job)
        elif kind == "asr_over_the_air":
            result = _run_asr_over_the_air(job)
        elif kind == "asr_speech_jamming":
            result = _run_asr_speech_jamming(job)
        elif kind == "asr_ua3":
            result = _run_asr_ua3(job)
        elif kind == "batch_attack":
            result = _run_batch_attack(job)
        elif kind == "image_robustness":
            result = _run_image_robustness(job)
        elif kind == "audio_robustness":
            result = _run_audio_robustness(job)
        elif kind == "benchmark":
            result = _run_benchmark(job)
        else:
            raise ValueError(f"Unsupported job kind: {kind}")

        if result is None:
            return None
        complete_job(job["job_id"], result=result)
        return result
    except Exception:
        logger.exception("Job %s (%s) failed", job["job_id"], kind)
        append_job_event(job["job_id"], "error", {"message": _safe_error_message(kind)})
        fail_job(job["job_id"], _safe_error_message(kind))
        return None
