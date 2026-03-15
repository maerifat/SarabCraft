"""
TextFooler Attack — Jin et al., 2020 (arXiv:1907.11932)

Black-box word-level attack: ranks word importance by delete-one
(two-case formula), generates candidates via counter-fitted embedding
neighbours (BERT MLM fallback), filters by word-embedding cosine ≥ δ,
strict POS match, and sentence-level semantic similarity ≥ threshold.
"""

import logging

from models.text_loader import get_label_index

logger = logging.getLogger("textattack.attacks.textfooler")


def run_textfooler(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_candidates: int = 50,
    similarity_threshold: float = 0.84,
    max_perturbation_ratio: float = 0.3,
    embedding_cos_threshold: float = 0.5,
) -> str:
    """TextFooler attack (Jin et al., 2020).

    Returns: adversarial text (str).
    """
    from utils.text_word_importance import delete_one_importance
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, is_stopword, pos_tag_words, clean_word,
    )
    from utils.text_word_substitution import get_embedding_neighbours_with_scores
    from utils.text_constraints import compute_semantic_similarity

    logger.info("TextFooler: starting (cands=%d, sim=%.2f, emb_cos=%.2f, max_pert=%.2f)",
                max_candidates, similarity_threshold, embedding_cos_threshold, max_perturbation_ratio)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    words = [w for w, _, _ in words_spans]
    max_perturbs = max(1, int(len(words) * max_perturbation_ratio))

    # Get original prediction
    orig_label, orig_conf, _ = model_wrapper.predict(text)

    # Score word importance (uses two-case formula)
    importance = delete_one_importance(model_wrapper, text, orig_label)

    current_text = text
    perturbations_made = 0

    for word_idx, score in importance:
        if perturbations_made >= max_perturbs:
            break

        word = words[word_idx]
        if is_stopword(word) or len(clean_word(word)) <= 1:
            continue

        # Get substitution candidates from counter-fitted embeddings (MLM fallback)
        candidates_with_scores = get_embedding_neighbours_with_scores(
            word, top_k=max_candidates, context_text=current_text, position=word_idx
        )

        # Word-embedding cosine pre-filter (paper: δ ≥ 0.5)
        candidates_with_scores = [
            (cand, sim) for cand, sim in candidates_with_scores
            if sim >= embedding_cos_threshold
        ]

        # Strict POS filtering — no fallback for mismatches (paper: strict)
        # Recompute POS tags on current_text words to avoid stale-tag bug
        current_words = [w for w, _, _ in get_words_and_spans(current_text)]
        current_pos = pos_tag_words(current_words)
        if word_idx < len(current_pos):
            word_pos = current_pos[word_idx]
            filtered = []
            for cand, sim in candidates_with_scores:
                cand_pos = pos_tag_words([cand])
                if cand_pos and cand_pos[0] == word_pos:
                    filtered.append(cand)
            candidates = filtered
        else:
            candidates = [cand for cand, _ in candidates_with_scores]

        # Try each candidate, pick best that passes constraints
        best_text = None
        best_impact = -1.0

        for cand in candidates:
            candidate_text = replace_word_at(current_text, word_idx, cand)

            # Check semantic similarity (sentence-level, distilUSE)
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
