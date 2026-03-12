"""
BERT-Attack — Li et al., 2020 (arXiv:2004.09984)

Black-box word-level attack using BERT masked language model for
contextually appropriate word substitutions. Sub-word aware:
handles WordPiece tokenization correctly.
"""

import logging

logger = logging.getLogger("textattack.attacks.bert_attack")


def run_bert_attack(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_candidates: int = 48,
    similarity_threshold: float = 0.8,
    max_perturbation_ratio: float = 0.4,
) -> str:
    """BERT-Attack: uses BERT MLM for contextual word replacement.

    Returns: adversarial text (str).
    """
    from utils.text_word_importance import delete_one_importance
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, is_stopword, clean_word,
    )
    from utils.text_word_substitution import get_mlm_substitutions
    from utils.text_constraints import compute_semantic_similarity

    logger.info("BERT-Attack: starting (cands=%d, sim=%.2f)", max_candidates, similarity_threshold)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    words = [w for w, _, _ in words_spans]
    max_perturbs = max(1, int(len(words) * max_perturbation_ratio))

    orig_label, orig_conf, _ = model_wrapper.predict(text)
    importance = delete_one_importance(model_wrapper, text, orig_label)

    current_text = text
    perturbations_made = 0

    for word_idx, score in importance:
        if perturbations_made >= max_perturbs:
            break

        word = words[word_idx]
        if is_stopword(word) or len(clean_word(word)) <= 1:
            continue

        # BERT-Attack uses MLM exclusively (no external embedding)
        candidates = get_mlm_substitutions(current_text, word_idx, top_k=max_candidates)

        best_text = None
        best_impact = -1.0

        for cand in candidates:
            candidate_text = replace_word_at(current_text, word_idx, cand)

            # Semantic similarity constraint (inline)
            sim = compute_semantic_similarity(text, candidate_text)
            if sim < similarity_threshold:
                continue

            label, conf, _ = model_wrapper.predict(candidate_text)

            if target_label is not None:
                if label.lower() == target_label.lower():
                    logger.info("BERT-Attack: success at perturbation %d", perturbations_made + 1)
                    return candidate_text
            else:
                if label != orig_label:
                    logger.info("BERT-Attack: success at perturbation %d", perturbations_made + 1)
                    return candidate_text

            impact = orig_conf - conf
            if impact > best_impact:
                best_impact = impact
                best_text = candidate_text

        if best_text is not None:
            current_text = best_text
            perturbations_made += 1

    logger.info("BERT-Attack: finished (%d perturbations)", perturbations_made)
    return current_text
