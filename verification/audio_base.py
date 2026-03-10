"""
Base classes for audio transfer verification.

AudioVerifier subclasses send adversarial audio to external ASR / audio-AI
services and return transcription results so attacks can be evaluated for
cross-model transferability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from verification.base import ConfigField


@dataclass
class AudioPrediction:
    """Single result from an audio verification backend."""
    text: str
    confidence: float = 1.0
    raw: dict = field(default_factory=dict)


@dataclass
class AudioVerificationResult:
    """Result of testing adversarial audio against one ASR backend."""
    verifier_name: str
    service_type: str
    transcription: str = ""
    original_transcription: str = ""
    target_text: str = ""
    exact_match: bool = False
    contains_target: bool = False
    wer: Optional[float] = None
    elapsed_ms: float = 0.0
    error: Optional[str] = None


def compute_wer_simple(reference: str, hypothesis: str) -> float:
    """Word Error Rate between reference and hypothesis (0.0 = perfect)."""
    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


class AudioVerifier(ABC):
    """Abstract base for audio verification backends (ASR services)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable display name."""

    @property
    @abstractmethod
    def service_type(self) -> str:
        """One of: 'local', 'api', 'cloud'."""

    @property
    def requires_config(self) -> bool:
        return len(self.get_config_schema()) > 0

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend is configured and ready."""

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, sample_rate: int, language: str = "en-US") -> str:
        """Transcribe audio bytes (WAV PCM 16-bit). Return transcribed text."""

    def get_config_schema(self) -> list:
        """Return list of ConfigField describing required env vars."""
        return []

    def status_message(self) -> str:
        """Short status string for the UI."""
        return "Ready" if self.is_available() else "Not configured"

    def detailed_status(self) -> dict:
        hb = self.heartbeat()
        avail = bool(hb.get("ok"))
        return {
            "level": hb.get("level") or ("ready" if avail else "unavailable"),
            "reason": hb.get("reason") or hb.get("message") or self.status_message(),
            "status": hb.get("status") or ("ok" if avail else "not_configured"),
        }

    def heartbeat(self) -> dict:
        avail = self.is_available()
        return {
            "ok": avail,
            "message": self.status_message(),
        }
