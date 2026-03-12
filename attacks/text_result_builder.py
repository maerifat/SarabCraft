"""
Post-attack result assembly.

High-level API for text attacks:
  - Calls the attack (run_text_attack returns adversarial string + query count)
  - Computes predictions on original + adversarial
  - Computes metrics (perturbation ratio, similarity, etc.)
  - Assembles AttackResult dataclass
"""

import time
import logging

from attacks.text_types import AttackResult
from attacks.text_router import run_text_attack, _TextModelWrapper
from models.text_loader import get_predictions, get_label_and_confidence
from utils.text_metrics import compute_all_metrics

logger = logging.getLogger("textattack.result_builder")


def run_and_build_result(
    attack_name: str,
    model,
    tokenizer,
    text: str,
    *,
    target_label: str = None,
    params: dict = None,
    should_cancel=None,
) -> AttackResult:
    """Run an attack and build a complete AttackResult.

    This is the high-level API for text attacks.
    It calls run_text_attack() (raw output) then adds predictions + metrics.
    """
    if params is None:
        params = {}

    # Get original predictions
    orig_preds = get_predictions(model, tokenizer, text, top_k=5)
    orig_label, orig_conf, _ = get_label_and_confidence(model, tokenizer, text)

    # Run the attack (router handles label resolution internally)
    start_ms = time.time() * 1000
    adversarial_text, query_count, resolved_target = run_text_attack(
        attack_name, model, tokenizer, text,
        target_label=target_label,
        params=params,
        should_cancel=should_cancel,
    )
    elapsed_ms = time.time() * 1000 - start_ms

    # Get adversarial predictions
    adv_preds = get_predictions(model, tokenizer, adversarial_text, top_k=5)
    adv_label, adv_conf, _ = get_label_and_confidence(model, tokenizer, adversarial_text)

    # Compute metrics
    metrics = compute_all_metrics(text, adversarial_text)

    # Determine success (use resolved label for comparison)
    if resolved_target is not None:
        success = adv_label.lower() == resolved_target.lower()
    else:
        success = adv_label != orig_label


    return AttackResult(
        original_text=text,
        adversarial_text=adversarial_text,
        original_label=orig_label,
        adversarial_label=adv_label,
        original_confidence=orig_conf,
        adversarial_confidence=adv_conf,
        num_queries=query_count,
        perturbation_ratio=metrics["perturbation_ratio"],
        semantic_similarity=metrics["semantic_similarity"],
        success=success,
        attack_name=attack_name,
        elapsed_ms=round(elapsed_ms, 1),
        original_predictions=orig_preds,
        adversarial_predictions=adv_preds,
    )
