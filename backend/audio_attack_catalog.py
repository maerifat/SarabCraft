from __future__ import annotations


_AUDIO_ATTACK_SPECS = {
    "Targeted Transcription": {
        "key": "transcription",
        "requires_target_text": True,
        "success_mode": "targeted",
    },
    "Hidden Command": {
        "key": "hidden_command",
        "requires_target_text": True,
        "success_mode": "targeted",
    },
    "Psychoacoustic": {
        "key": "psychoacoustic",
        "requires_target_text": True,
        "success_mode": "targeted",
    },
    "Over-the-Air Robust": {
        "key": "ota",
        "requires_target_text": True,
        "success_mode": "targeted",
    },
    "Speech Jamming": {
        "key": "jamming",
        "requires_target_text": False,
        "success_mode": "untargeted",
    },
}

_AUDIO_ATTACK_ALIASES = {
    "Over-the-Air": "Over-the-Air Robust",
}


def available_audio_attack_names() -> list[str]:
    return list(_AUDIO_ATTACK_SPECS.keys())


def canonicalize_audio_attack_name(name: str | None) -> str:
    normalized = str(name or "").strip()
    return _AUDIO_ATTACK_ALIASES.get(normalized, normalized)


def resolve_audio_attack(name: str | None) -> dict | None:
    return _AUDIO_ATTACK_SPECS.get(canonicalize_audio_attack_name(name))


def audio_attack_key(name: str | None) -> str | None:
    spec = resolve_audio_attack(name)
    if spec is None:
        return None
    return str(spec["key"])


def audio_attack_requires_target_text(name: str | None) -> bool:
    spec = resolve_audio_attack(name)
    return bool(spec and spec["requires_target_text"])


def audio_attack_is_untargeted(name: str | None) -> bool:
    spec = resolve_audio_attack(name)
    return bool(spec and spec["success_mode"] == "untargeted")


def audio_attack_target_text(name: str | None, target_text: str | None) -> str:
    if not audio_attack_requires_target_text(name):
        return ""
    return str(target_text or "").strip()


def evaluate_audio_source_success(
    attack_name: str | None,
    result_text: str | None,
    target_text: str | None = "",
    original_text: str | None = "",
) -> bool:
    result = _normalize_text(result_text)
    if audio_attack_is_untargeted(attack_name):
        original = _normalize_text(original_text)
        return bool(result or original) and result != original

    target = _normalize_text(target_text)
    return bool(target) and result == target


def evaluate_audio_transfer_success(
    attack_name: str | None,
    transcription: str | None,
    target_text: str | None = "",
    original_text: str | None = "",
    original_transcription: str | None = "",
) -> bool:
    text = _normalize_text(transcription)
    if audio_attack_is_untargeted(attack_name):
        baseline = _normalize_text(original_transcription) or _normalize_text(original_text)
        return bool(text or baseline) and text != baseline

    target = _normalize_text(target_text)
    return bool(target) and (text == target or target in text)


def _normalize_text(value: str | None) -> str:
    return str(value or "").strip().lower()
