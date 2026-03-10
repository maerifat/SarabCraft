"""
Adversarial Transfer Verification Framework.

Pluggable backends for testing adversarial images against
local models, HuggingFace API, and cloud vision services.
"""

from verification.base import Verifier, Prediction, VerificationResult, ConfigField
from verification.registry import (
    get_all_verifiers,
    get_available_verifiers,
    run_verification,
)
