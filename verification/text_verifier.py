"""
Transfer verification for text adversarial examples.

Test whether adversarial text fools models not used as the attack source.
Mirrors verification/registry.py in the main SarabCraft codebase.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger("textattack.verification")


@dataclass
class TextVerificationResult:
    """Result of verifying adversarial text against one model."""
    model_name: str
    original_label: str
    original_confidence: float
    adversarial_label: str
    adversarial_confidence: float
    transferred: bool
    confidence_drop: float


def verify_transfer(
    original_text: str,
    adversarial_text: str,
    target_model_names: list[str],
) -> list[TextVerificationResult]:
    """Test adversarial text against a set of target models.

    Args:
        original_text: the original, clean text
        adversarial_text: the adversarial text to verify
        target_model_names: list of HuggingFace model names

    Returns:
        List of TextVerificationResult, one per target model.
    """
    from models.text_loader import load_text_model, get_label_and_confidence

    results = []
    for model_name in target_model_names:
        logger.info("Verifying transfer against: %s", model_name)
        try:
            model, tokenizer = load_text_model(model_name)

            orig_label, orig_conf, _ = get_label_and_confidence(model, tokenizer, original_text)
            adv_label, adv_conf, _ = get_label_and_confidence(model, tokenizer, adversarial_text)

            transferred = adv_label != orig_label
            confidence_drop = orig_conf - adv_conf

            results.append(TextVerificationResult(
                model_name=model_name,
                original_label=orig_label,
                original_confidence=round(orig_conf, 4),
                adversarial_label=adv_label,
                adversarial_confidence=round(adv_conf, 4),
                transferred=transferred,
                confidence_drop=round(confidence_drop, 4),
            ))
        except Exception as e:
            logger.error("Failed to verify against %s: %s", model_name, e)
            results.append(TextVerificationResult(
                model_name=model_name,
                original_label="ERROR",
                original_confidence=0.0,
                adversarial_label=str(e),
                adversarial_confidence=0.0,
                transferred=False,
                confidence_drop=0.0,
            ))

    return results
