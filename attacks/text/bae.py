"""
BAE Attack — Garg & Ramakrishnan, 2020 (arXiv:2004.01970)

BERT-based Adversarial Examples with four strategies:
  R  (Replace):  Mask word → fill with BERT MLM
  I  (Insert):   Insert [MASK] adjacent → fill
  R+I (Both):    Try both Replace and Insert
  D  (Delete):   Simply remove important words
"""

import logging

logger = logging.getLogger("textattack.attacks.bae")


def _insert_word(text: str, position: int, new_word: str) -> str:
    """Insert a word after the given position index."""
    from utils.text_utils import get_words_and_spans
    spans = get_words_and_spans(text)
    if position < 0 or position >= len(spans):
        return text
    _, _, end = spans[position]
    return text[:end] + " " + new_word + text[end:]


def _delete_word(text: str, position: int) -> str:
    """Delete the word at position index."""
    from utils.text_utils import get_words_and_spans
    spans = get_words_and_spans(text)
    if position < 0 or position >= len(spans):
        return text
    _, start, end = spans[position]
    result = text[:start] + text[end:]
    return " ".join(result.split())


def run_bae(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    strategy: str = "R",
    max_candidates: int = 50,
    similarity_threshold: float = 0.8,
    max_perturbation_ratio: float = 0.5,
) -> str:
    """BAE attack with configurable strategy.

    Returns: adversarial text (str).
    """
    from utils.text_word_importance import delete_one_importance
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, is_stopword, clean_word,
    )
    from utils.text_word_substitution import get_mlm_substitutions
    from utils.text_constraints import compute_semantic_similarity

    logger.info("BAE: starting (strategy=%s, cands=%d, sim=%.2f)",
                strategy, max_candidates, similarity_threshold)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    orig_label, orig_conf, _ = model_wrapper.predict(text)
    importance = delete_one_importance(model_wrapper, text, orig_label)

    current_text = text
    perturbations_made = 0
    max_perturbs = max(1, int(len(words_spans) * max_perturbation_ratio))

    for word_idx, score in importance:
        if perturbations_made >= max_perturbs:
            break

        word = get_words_and_spans(current_text)
        if word_idx >= len(word):
            continue
        word_str = word[word_idx][0]
        if is_stopword(word_str) or len(clean_word(word_str)) <= 1:
            continue

        operations = []

        # Strategy D: just delete
        if strategy in ("D",):
            candidate = _delete_word(current_text, word_idx)
            if candidate.strip():
                operations.append(candidate)

        # Strategy R: replace via MLM
        if strategy in ("R", "R+I"):
            mlm_candidates = get_mlm_substitutions(current_text, word_idx, top_k=max_candidates)
            for cand in mlm_candidates[:max_candidates]:
                operations.append(replace_word_at(current_text, word_idx, cand))

        # Strategy I: insert via MLM
        if strategy in ("I", "R+I"):
            # Insert [MASK] after word, get MLM fill
            text_with_mask = _insert_word(current_text, word_idx, "[MASK]")
            insert_candidates = get_mlm_substitutions(
                text_with_mask, word_idx + 1, top_k=max_candidates // 2,
            )
            for cand in insert_candidates:
                operations.append(_insert_word(current_text, word_idx, cand))

        # Evaluate all candidates
        best_text = None
        best_impact = -1.0

        for candidate_text in operations:
            sim = compute_semantic_similarity(text, candidate_text)
            if sim < similarity_threshold:
                continue

            label, conf, _ = model_wrapper.predict(candidate_text)

            if target_label is not None:
                if label.lower() == target_label.lower():
                    logger.info("BAE: success at perturbation %d", perturbations_made + 1)
                    return candidate_text
            else:
                if label != orig_label:
                    logger.info("BAE: success at perturbation %d", perturbations_made + 1)
                    return candidate_text

            impact = orig_conf - conf
            if impact > best_impact:
                best_impact = impact
                best_text = candidate_text

        if best_text is not None:
            current_text = best_text
            perturbations_made += 1

    logger.info("BAE: finished (%d perturbations)", perturbations_made)
    return current_text
