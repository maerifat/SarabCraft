"""
TextFooler Attack — Jin et al., 2020 (arXiv:1907.11932)

Black-box word-level attack: ranks word importance by delete-one,
generates candidates via embedding neighbours / BERT MLM fallback,
filters by POS match + semantic similarity ≥ threshold.
"""

import logging

logger = logging.getLogger("textattack.attacks.textfooler")


def run_textfooler(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_candidates: int = 50,
    similarity_threshold: float = 0.8,
    max_perturbation_ratio: float = 0.3,
) -> str:
    """TextFooler attack.

    Returns: adversarial text (str).
    """
    from utils.text_word_importance import delete_one_importance
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, is_stopword, pos_tag_words, clean_word,
    )
    from utils.text_word_substitution import get_mlm_substitutions
    from utils.text_constraints import compute_semantic_similarity

    logger.info("TextFooler: starting (cands=%d, sim=%.2f, max_pert=%.2f)",
                max_candidates, similarity_threshold, max_perturbation_ratio)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    words = [w for w, _, _ in words_spans]
    max_perturbs = max(1, int(len(words) * max_perturbation_ratio))

    # Get original prediction
    orig_label, orig_conf, _ = model_wrapper.predict(text)
    orig_pos = pos_tag_words(words)

    # Score word importance
    importance = delete_one_importance(model_wrapper, text, orig_label)

    current_text = text
    perturbations_made = 0

    for word_idx, score in importance:
        if perturbations_made >= max_perturbs:
            break

        word = words[word_idx]
        if is_stopword(word) or len(clean_word(word)) <= 1:
            continue

        # Get substitution candidates from MLM
        candidates = get_mlm_substitutions(current_text, word_idx, top_k=max_candidates)

        # Filter by POS match
        if word_idx < len(orig_pos):
            word_pos = orig_pos[word_idx]
            filtered = []
            for cand in candidates:
                cand_pos = pos_tag_words([cand])
                if cand_pos and cand_pos[0] == word_pos:
                    filtered.append(cand)
                elif len(filtered) < max_candidates // 2:
                    # Allow some POS mismatches to have enough candidates
                    filtered.append(cand)
            candidates = filtered if filtered else candidates[:max_candidates // 2]

        # Try each candidate, pick best that passes constraints
        best_text = None
        best_impact = -1.0

        for cand in candidates:
            candidate_text = replace_word_at(current_text, word_idx, cand)

            # Check semantic similarity (inline constraint)
            sim = compute_semantic_similarity(text, candidate_text)
            if sim < similarity_threshold:
                continue

            label, conf, _ = model_wrapper.predict(candidate_text)

            if target_label is not None:
                if label.lower() == target_label.lower():
                    logger.info("TextFooler: success at perturbation %d", perturbations_made + 1)
                    return candidate_text
                # Pick candidate that increases target confidence most
                probs = model_wrapper.predict_probs(candidate_text)
                from models.text_loader import get_label_index
                target_idx = get_label_index(model_wrapper.model, target_label)
                if target_idx is not None and target_idx < len(probs):
                    impact = probs[target_idx]
                else:
                    impact = orig_conf - conf
            else:
                if label != orig_label:
                    logger.info("TextFooler: success at perturbation %d", perturbations_made + 1)
                    return candidate_text
                impact = orig_conf - conf

            if impact > best_impact:
                best_impact = impact
                best_text = candidate_text

        if best_text is not None:
            current_text = best_text
            perturbations_made += 1

    logger.info("TextFooler: finished (%d perturbations)", perturbations_made)
    return current_text
