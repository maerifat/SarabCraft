"""
Shared data types for text adversarial attacks.

Mirrors SarabCraft's approach: lightweight dataclasses for results,
a custom error for cooperative cancellation.
"""

from dataclasses import dataclass, field
from typing import Optional


class AttackCancelledError(RuntimeError):
    """Raised when a running attack should stop due to cancellation."""


@dataclass
class TextPrediction:
    """Single classification prediction."""
    label: str
    confidence: float
    index: int = -1


@dataclass
class AttackResult:
    """Result of a text adversarial attack — assembled by result_builder, not by attacks."""
    original_text: str
    adversarial_text: str
    original_label: str
    adversarial_label: str
    original_confidence: float
    adversarial_confidence: float
    num_queries: int
    perturbation_ratio: float
    semantic_similarity: float
    success: bool
    attack_name: str
    elapsed_ms: float
    original_predictions: list = field(default_factory=list)
    adversarial_predictions: list = field(default_factory=list)
    error: Optional[str] = None
