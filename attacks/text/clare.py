"""
Clare Attack — Li et al., 2021 (arXiv:2009.07502)

Contextualized Perturbation for Textual Adversarial Attack.
Uses BERT MLM for three contextual operations:
  - Replace: mask word → fill
  - Insert: insert [MASK] adjacent → fill
  - Merge: merge two adjacent words into one via MLM

Selects the perturbation that maximises label change while maintaining fluency.
"""

import logging

logger = logging.getLogger("textattack.attacks.clare")


def run_clare(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_perturbations: int = 5,
    similarity_threshold: float = 0.7,
) -> str:
    """Clare attack: contextual perturbation with Replace, Insert, Merge.

    Returns: adversarial text (str).
    """
    from utils.text_word_importance import delete_one_importance
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, is_stopword, clean_word,
    )
    from utils.text_word_substitution import get_mlm_substitutions
    from utils.text_constraints import compute_semantic_similarity

    logger.info("Clare: starting (max_pert=%d, sim=%.2f)", max_perturbations, similarity_threshold)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    orig_label, orig_conf, _ = model_wrapper.predict(text)
    importance = delete_one_importance(model_wrapper, text, orig_label)

    current_text = text
    perturbations_made = 0

    for word_idx, score in importance:
        if perturbations_made >= max_perturbations:
            break

        current_spans = get_words_and_spans(current_text)
        if word_idx >= len(current_spans):
            continue

        word_str = current_spans[word_idx][0]
        if is_stopword(word_str) or len(clean_word(word_str)) <= 1:
            continue

        all_candidates = []

        # Operation 1: Replace — mask word → MLM fill
        replace_candidates = get_mlm_substitutions(current_text, word_idx, top_k=15)
        for cand in replace_candidates:
            all_candidates.append(("replace", replace_word_at(current_text, word_idx, cand)))

        # Operation 2: Insert — insert [MASK] after word → MLM fill
        words_list = [w for w, _, _ in current_spans]
        insert_text = " ".join(words_list[:word_idx + 1] + ["[MASK]"] + words_list[word_idx + 1:])
        insert_candidates = get_mlm_substitutions(insert_text, word_idx + 1, top_k=10)
        for cand in insert_candidates:
            inserted = " ".join(words_list[:word_idx + 1] + [cand] + words_list[word_idx + 1:])
            all_candidates.append(("insert", inserted))

        # Operation 3: Merge — combine current word with next via MLM
        if word_idx + 1 < len(current_spans):
            merge_text_words = list(words_list)
            merge_text_words[word_idx] = "[MASK]"
            merge_text_words.pop(word_idx + 1)
            merge_text = " ".join(merge_text_words)
            merge_candidates = get_mlm_substitutions(merge_text, word_idx, top_k=10)
            for cand in merge_candidates:
                merged = " ".join(merge_text_words[:word_idx] + [cand] + merge_text_words[word_idx + 1:])
                all_candidates.append(("merge", merged))

        # Evaluate all candidates, select best
        best_text = None
        best_impact = -1.0

        for op_type, candidate_text in all_candidates:
            sim = compute_semantic_similarity(text, candidate_text)
            if sim < similarity_threshold:
                continue

            label, conf, _ = model_wrapper.predict(candidate_text)

            if target_label is not None:
                if label.lower() == target_label.lower():
                    logger.info("Clare: success at perturbation %d (%s)", perturbations_made + 1, op_type)
                    return candidate_text
            else:
                if label != orig_label:
                    logger.info("Clare: success at perturbation %d (%s)", perturbations_made + 1, op_type)
                    return candidate_text

            impact = orig_conf - conf
            if impact > best_impact:
                best_impact = impact
                best_text = candidate_text

        if best_text is not None:
            current_text = best_text
            perturbations_made += 1

    logger.info("Clare: finished (%d perturbations)", perturbations_made)
    return current_text
