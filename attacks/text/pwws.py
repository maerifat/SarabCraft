"""
PWWS Attack — Ren et al., 2019 (arXiv:1907.06292)

Probability Weighted Word Saliency: scores each word by
  score = ΔP(correct_class) × P(word_saliency)
Uses WordNet synonyms filtered by POS tag for substitution.
"""

import logging

logger = logging.getLogger("textattack.attacks.pwws")


def _word_saliency(model_wrapper, text: str, word_idx: int) -> float:
    """Compute word saliency: probability change when word is removed."""
    from utils.text_utils import get_words_and_spans
    spans = get_words_and_spans(text)
    if word_idx >= len(spans):
        return 0.0

    _, start, end = spans[word_idx]
    reduced = (text[:start] + text[end:]).strip()
    reduced = " ".join(reduced.split())

    if not reduced:
        return 1.0

    orig_probs = model_wrapper.predict_probs(text)
    reduced_probs = model_wrapper.predict_probs(reduced)

    # Saliency = max change in any class probability
    saliency = max(abs(a - b) for a, b in zip(orig_probs, reduced_probs))
    return saliency


def run_pwws(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_candidates: int = 10,
) -> str:
    """PWWS attack: WordNet synonym substitution weighted by word saliency.

    Returns: adversarial text (str).
    """
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, is_stopword, clean_word, simple_pos_tag,
    )
    from utils.text_word_substitution import get_wordnet_synonyms

    logger.info("PWWS: starting (max_cands=%d)", max_candidates)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    orig_label, orig_conf, _ = model_wrapper.predict(text)
    orig_probs = model_wrapper.predict_probs(text)
    orig_class_idx = orig_probs.index(max(orig_probs))

    # Compute PWWS score for each word: ΔP × saliency
    word_scores = []
    for i, (word, _, _) in enumerate(words_spans):
        if is_stopword(word) or len(clean_word(word)) <= 1:
            continue

        pos = simple_pos_tag(word)
        synonyms = get_wordnet_synonyms(clean_word(word), pos=pos, max_candidates=max_candidates)
        if not synonyms:
            # Fall back to MLM if no WordNet synonyms
            from utils.text_word_substitution import get_mlm_substitutions
            synonyms = get_mlm_substitutions(text, i, top_k=max_candidates)

        if not synonyms:
            continue

        # Best synonym for this word: the one that maximally reduces P(correct)
        best_syn = None
        best_delta = 0.0

        for syn in synonyms:
            candidate = replace_word_at(text, i, syn)
            cand_probs = model_wrapper.predict_probs(candidate)
            delta = orig_probs[orig_class_idx] - cand_probs[orig_class_idx]
            if delta > best_delta:
                best_delta = delta
                best_syn = syn

        if best_syn is not None:
            saliency = _word_saliency(model_wrapper, text, i)
            pwws_score = best_delta * saliency
            word_scores.append((i, best_syn, pwws_score))

    # Sort by PWWS score (descending) and substitute greedily
    word_scores.sort(key=lambda x: x[2], reverse=True)

    current_text = text
    for word_idx, synonym, score in word_scores:
        # Re-parse to get updated spans after prior substitutions
        current_spans = get_words_and_spans(current_text)
        if word_idx >= len(current_spans):
            continue
        # Guard against stale indices after word-boundary changes
        if current_spans[word_idx][0] != words_spans[word_idx][0]:
            continue

        candidate = replace_word_at(current_text, word_idx, synonym)
        label, conf, _ = model_wrapper.predict(candidate)

        if target_label is not None:
            if label.lower() == target_label.lower():
                logger.info("PWWS: success")
                return candidate
        else:
            if label != orig_label:
                logger.info("PWWS: success")
                return candidate

        current_text = candidate

    logger.info("PWWS: finished")
    return current_text
