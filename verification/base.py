"""
Core interface for transfer verification backends.

Every verification target (local model, cloud API, etc.) implements
the Verifier abstract class.  Results are normalised into Prediction
and VerificationResult dataclasses so the UI can display them uniformly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image


@dataclass
class Prediction:
    """Single classification prediction from any backend."""
    label: str
    confidence: float
    raw: dict = field(default_factory=dict)


@dataclass
class ConfigField:
    """Describes one configuration parameter a verifier needs."""
    name: str
    env_var: str
    description: str
    required: bool = True
    secret: bool = True


@dataclass
class VerificationResult:
    """Result of testing one adversarial image against one backend."""
    verifier_name: str
    service_type: str
    predictions: list
    original_predictions: list
    target_label: Optional[str] = None
    original_label: Optional[str] = None
    matched_target: bool = False
    original_label_gone: bool = False
    confidence_drop: float = 0.0
    elapsed_ms: float = 0.0
    error: Optional[str] = None


class Verifier(ABC):
    """Abstract base for all verification backends."""

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
    def classify(self, image: Image.Image) -> list:
        """Classify a PIL image.  Return list of Prediction, best first."""

    def get_config_schema(self) -> list:
        """Return list of ConfigField describing required env vars."""
        return []

    def status_message(self) -> str:
        """Short status string for the UI."""
        return "Ready" if self.is_available() else "Not configured"

    def detailed_status(self) -> dict:
        """Return dict with level/reason for the UI service cards."""
        hb = self.heartbeat()
        avail = bool(hb.get("ok"))
        return {
            "level": hb.get("level") or ("ready" if avail else "unavailable"),
            "reason": hb.get("reason") or hb.get("message") or self.status_message(),
            "status": hb.get("status") or ("ok" if avail else "not_configured"),
        }

    def heartbeat(self) -> dict:
        """Live connectivity check. Override for cloud services."""
        avail = self.is_available()
        return {
            "ok": avail,
            "message": self.status_message(),
        }
