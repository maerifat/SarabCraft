"""
Input Reduction Attack — Feng et al., 2018 (EMNLP 2018)

Pathologies of Neural Models Make Interpretations Difficult.
Deletion-only attack: iteratively removes the least important word until
the model prediction changes or a minimal sufficient input is reached.

This is the ONLY deletion-based attack — no substitution, no insertion.
It exposes models that rely on spurious correlations by showing that
removing most of the input often does not change the prediction,
revealing pathological over-confidence.

Algorithm:
  1. Rank words by importance (delete-one confidence drop)
  2. Remove the LEAST important word (lowest importance score)
  3. Re-classify reduced text
  4. If prediction flips → success (adversarial = reduced text)
  5. If prediction unchanged → repeat from step 1
  6. Stop when max_reductions reached or text is empty

Reference: TextAttack InputReductionFeng2018 recipe.
"""

import logging

logger = logging.getLogger("textattack.attacks.input_reduction")


def run_input_reduction(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_reduction_ratio: float = 0.7,
    stop_at_length: int = 1,
) -> str:
    """Input Reduction attack (Feng et al., 2018).

    Iteratively removes the least-important word until the model
    prediction changes. Exposes model reliance on spurious features.

    Args:
        model_wrapper: wrapped model with .predict() interface.
        tokenizer: HuggingFace tokenizer (unused, API compat).
        text: input text to attack.
        target_label: target class (None = untargeted, standard for this attack).
        max_reduction_ratio: max fraction of words to remove (0.7 = remove up to 70%).
        stop_at_length: minimum number of words to keep.

    Returns: adversarial text (str).
    """
    from utils.text_word_importance import delete_one_importance
    from utils.text_utils import get_words_and_spans, is_stopword

    logger.info(
        "InputReduction: starting (max_reduction=%.2f, stop_at=%d)",
        max_reduction_ratio, stop_at_length,
    )

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    orig_label, orig_conf, _ = model_wrapper.predict(text)
    max_removals = max(1, int(len(words_spans) * max_reduction_ratio))

    current_text = text
    removals = 0
    best_reduced = text
    best_impact = 0.0

    for _ in range(max_removals):
        current_spans = get_words_and_spans(current_text)
        if len(current_spans) <= stop_at_length:
            break

        importance = delete_one_importance(model_wrapper, current_text, orig_label)
        if not importance:
            break

        # Remove LEAST important word (last in descending importance list)
        # Feng et al.: remove words that contribute least to the prediction
        least_important_idx = importance[-1][0]

        words = [w for w, _, _ in current_spans]
        reduced_words = [w for i, w in enumerate(words) if i != least_important_idx]

        if not reduced_words:
            break

        reduced_text = " ".join(reduced_words)
        removals += 1

        label, conf, _ = model_wrapper.predict(reduced_text)

        if target_label is not None:
            if label.lower() == target_label.lower():
                logger.info("InputReduction: success after %d removals", removals)
                return reduced_text
        else:
            if label != orig_label:
                logger.info("InputReduction: success after %d removals", removals)
                return reduced_text

        impact = orig_conf - conf
        if impact > best_impact:
            best_impact = impact
            best_reduced = reduced_text

        current_text = reduced_text

    logger.info("InputReduction: finished (%d words removed)", removals)
    return best_reduced
