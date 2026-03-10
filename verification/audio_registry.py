"""
Audio verifier discovery and orchestration.

Mirrors verification/registry.py but for audio (ASR) backends.
Automatically discovers AudioVerifier subclasses and provides
run_audio_verification() for the UI.
"""

import io
import struct
import time
from typing import Optional

from verification.audio_base import (
    AudioVerifier,
    AudioVerificationResult,
    compute_wer_simple,
)

_AUDIO_REGISTRY: list = []
_AUDIO_ALL_LOADED = False


def _public_error_message(exc: Exception, fallback: str) -> str:
    if isinstance(exc, RuntimeError):
        return str(exc)
    return fallback


def register_audio(verifier_class):
    """Decorator that registers an AudioVerifier subclass."""
    if verifier_class not in _AUDIO_REGISTRY:
        _AUDIO_REGISTRY.append(verifier_class)
    return verifier_class


def _ensure_audio_loaded():
    """Import audio verifier modules so @register_audio decorators fire."""
    global _AUDIO_ALL_LOADED
    if _AUDIO_ALL_LOADED:
        return
    _AUDIO_ALL_LOADED = True
    import verification.aws_transcribe   # noqa: F401
    import verification.elevenlabs_stt   # noqa: F401


def get_all_audio_verifiers() -> list:
    _ensure_audio_loaded()
    return [cls() for cls in _AUDIO_REGISTRY]


def get_available_audio_verifiers() -> list:
    return [v for v in get_all_audio_verifiers() if v.is_available()]


def build_audio_service_status() -> list[dict]:
    """Return status info for each audio verifier (for UI display)."""
    _ensure_audio_loaded()
    result = []
    for cls in _AUDIO_REGISTRY:
        v = cls()
        result.append({
            "name": v.name,
            "service_type": v.service_type,
            "available": v.is_available(),
            "status": v.status_message(),
        })
    return result


def numpy_to_wav_bytes(audio_np, sample_rate: int) -> bytes:
    """Convert a numpy array (float32 or int16) to WAV bytes."""
    import numpy as np
    if audio_np.dtype in (np.float32, np.float64):
        audio_np = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    pcm = audio_np.tobytes()

    bits_per_sample = 16
    num_channels = 1
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(pcm)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate,
                          byte_rate, block_align, bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(pcm)))
    buf.write(pcm)
    return buf.getvalue()


def run_audio_verification(
    adversarial_audio,
    original_audio,
    sample_rate: int,
    target_text: Optional[str],
    original_text: Optional[str],
    remote_targets: list | None,
    language: str = "en-US",
    progress_callback=None,
) -> list:
    """
    Run audio transfer verification against selected ASR services.

    adversarial_audio / original_audio: numpy arrays (float32 or int16) or WAV bytes.
    Returns list of AudioVerificationResult.
    """
    import numpy as np

    _ensure_audio_loaded()
    verifiers = get_all_audio_verifiers()
    results = []

    if isinstance(adversarial_audio, np.ndarray):
        adv_wav = numpy_to_wav_bytes(adversarial_audio, sample_rate)
    else:
        adv_wav = adversarial_audio

    orig_wav = None
    if original_audio is not None:
        if isinstance(original_audio, np.ndarray):
            orig_wav = numpy_to_wav_bytes(original_audio, sample_rate)
        else:
            orig_wav = original_audio

    tasks = []
    for target in remote_targets or []:
        service_name = (target.get("settings") or {}).get("service_name") or target.get("display_name")
        display_name = target.get("display_name") or service_name
        tasks.append((display_name, service_name))
    total = max(len(tasks), 1)

    for idx, (display_name, svc_key) in enumerate(tasks):
        if progress_callback:
            progress_callback(idx / total, f"Transcribing: {display_name}...")

        verifier = next((v for v in verifiers if v.name == svc_key), None)
        if verifier is None:
            results.append(AudioVerificationResult(
                verifier_name=display_name,
                service_type="unknown",
                error=f"Audio verifier '{svc_key}' not found",
            ))
            continue

        if not verifier.is_available():
            results.append(AudioVerificationResult(
                verifier_name=display_name,
                service_type=verifier.service_type,
                error=verifier.status_message(),
            ))
            continue

        try:
            t0 = time.time()
            adv_text = verifier.transcribe(adv_wav, sample_rate, language)
            elapsed = (time.time() - t0) * 1000

            orig_transcription = (original_text or "").strip()
            if orig_wav is not None:
                orig_transcription = verifier.transcribe(orig_wav, sample_rate, language)

            target = (target_text or "").strip()
            exact_match = (adv_text.lower().strip() == target.lower()) if target else False
            contains = (target.lower() in adv_text.lower()) if target else False
            reference_text = target or orig_transcription
            wer = compute_wer_simple(reference_text, adv_text) if reference_text else None

            results.append(AudioVerificationResult(
                verifier_name=display_name,
                service_type=verifier.service_type,
                transcription=adv_text,
                original_transcription=orig_transcription,
                target_text=target,
                exact_match=exact_match,
                contains_target=contains,
                wer=wer,
                elapsed_ms=elapsed,
            ))

        except Exception as e:
            results.append(AudioVerificationResult(
                verifier_name=display_name,
                service_type=verifier.service_type,
                error=_public_error_message(e, "Audio verification failed for this target"),
                elapsed_ms=0,
            ))

    if progress_callback:
        progress_callback(1.0, "Audio verification complete")

    return results
