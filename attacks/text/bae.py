"""
BAE Attack — Garg & Ramakrishnan, 2020 (arXiv:2004.01970)

BERT-based Adversarial Examples with four strategies from the paper:
  R    (Replace):  Mask word → fill with BERT MLM (Algorithm 1)
  I    (Insert):   Insert [MASK] left or right of word → fill
  R/I  (Either):   Try both R and I, pick best single operation
  R+I  (Both):     First R, then I on the replaced text (sequential)
  D    (Delete):   Remove important words (SarabCraft extension, NOT in paper)

Reference: TextAttack bae_garg_2019.py
  - Transformation: WordSwapMaskedLM(method="bae", max_candidates=50)
  - Constraints: RepeatModification, StopwordModification,
                 PartOfSpeech(allow_verb_noun_swap=True),
                 UniversalSentenceEncoder(threshold≈0.8, window_size=15,
                     compare_against_original=True,
                     skip_text_shorter_than_window=True)
  - Search: GreedyWordSwapWIR(wir_method="delete")
"""

import logging

logger = logging.getLogger("textattack.attacks.bae")


# ── Helpers: word insertion / deletion ──────────────────────────────────────

def _insert_word_after(text: str, position: int, new_word: str) -> str:
    """Insert a word after the given position index."""
    from utils.text_utils import get_words_and_spans
    spans = get_words_and_spans(text)
    if position < 0 or position >= len(spans):
        return text
    _, _, end = spans[position]
    return text[:end] + " " + new_word + text[end:]


def _insert_word_before(text: str, position: int, new_word: str) -> str:
    """Insert a word before the given position index."""
    from utils.text_utils import get_words_and_spans
    spans = get_words_and_spans(text)
    if position < 0 or position >= len(spans):
        return text
    _, start, _ = spans[position]
    return text[:start] + new_word + " " + text[start:]


def _delete_word(text: str, position: int) -> str:
    """Delete the word at position index."""
    from utils.text_utils import get_words_and_spans
    spans = get_words_and_spans(text)
    if position < 0 or position >= len(spans):
        return text
    _, start, end = spans[position]
    result = text[:start] + text[end:]
    return " ".join(result.split())


# ── Helpers: POS consistency (paper Section 3.1) ───────────────────────────

def _get_pos_tag(text: str, word_idx: int) -> str:
    """Get POS tag for word at word_idx in context of the full sentence."""
    from utils.text_utils import get_words_and_spans
    words_spans = get_words_and_spans(text)
    if word_idx < 0 or word_idx >= len(words_spans):
        return ""
    words = [w for w, _, _ in words_spans]
    try:
        import nltk
        tagged = nltk.pos_tag(words)
        return tagged[word_idx][1]
    except (ImportError, LookupError):
        from utils.text_utils import simple_pos_tag
        return simple_pos_tag(words[word_idx])


def _pos_consistent(orig_pos: str, cand_pos: str,
                    allow_verb_noun_swap: bool = True) -> bool:
    """Check POS consistency between original and candidate word.

    Matches TextAttack PartOfSpeech(allow_verb_noun_swap=True).
    """
    if orig_pos == cand_pos:
        return True
    if allow_verb_noun_swap:
        noun_tags = {"NN", "NNS", "NNP", "NNPS", "noun"}
        verb_tags = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "verb"}
        if ((orig_pos in noun_tags and cand_pos in verb_tags)
                or (orig_pos in verb_tags and cand_pos in noun_tags)):
            return True
    return False


# ── Helpers: GloVe vocabulary filter (author email, TextAttack) ────────────

def _filter_by_vocab(candidates: list[str]) -> list[str]:
    """Filter MLM candidates to whole words in embedding vocabulary.

    From author email (documented in TextAttack): 'we filter out the
    sub-words and only retain the whole words (by checking if they are
    present in the GloVe vocabulary)'.

    Graceful no-op if word vectors are unavailable.
    """
    from utils.text_word_substitution import _load_word_vectors
    vectors = _load_word_vectors()
    if vectors is None:
        return candidates
    filtered = [c for c in candidates if c.lower() in vectors]
    return filtered if filtered else candidates  # don't discard everything


# ── Helpers: candidate generation ──────────────────────────────────────────

def _generate_replace_candidates(
    current_text: str, word_idx: int, max_candidates: int,
) -> list[tuple[str, str]]:
    """Generate BAE-R candidates: mask word → MLM fill.

    Returns list of (candidate_text, "R").
    """
    from utils.text_word_substitution import get_mlm_substitutions
    from utils.text_utils import replace_word_at

    mlm_words = get_mlm_substitutions(current_text, word_idx,
                                       top_k=max_candidates)
    mlm_words = _filter_by_vocab(mlm_words)

    return [(replace_word_at(current_text, word_idx, w), "R")
            for w in mlm_words[:max_candidates]]


def _generate_insert_candidates(
    current_text: str, word_idx: int, max_candidates: int,
) -> list[tuple[str, str]]:
    """Generate BAE-I candidates: insert [MASK] left AND right → MLM fill.

    Paper: 'Insert a token to the left or right of t'.
    Returns list of (candidate_text, "I").
    """
    from utils.text_word_substitution import get_mlm_substitutions
    from utils.text_utils import get_words_and_spans

    half_k = max(1, max_candidates // 2)
    candidates = []

    # ── Right insertion ──
    text_mask_r = _insert_word_after(current_text, word_idx, "[MASK]")
    r_spans = get_words_and_spans(text_mask_r)
    mask_pos_r = word_idx + 1
    if mask_pos_r < len(r_spans):
        r_cands = get_mlm_substitutions(text_mask_r, mask_pos_r, top_k=half_k)
        r_cands = _filter_by_vocab(r_cands)
        for w in r_cands:
            candidates.append(
                (_insert_word_after(current_text, word_idx, w), "I"))

    # ── Left insertion ──
    text_mask_l = _insert_word_before(current_text, word_idx, "[MASK]")
    l_spans = get_words_and_spans(text_mask_l)
    mask_pos_l = word_idx  # [MASK] occupies position word_idx after shift
    if mask_pos_l < len(l_spans):
        l_cands = get_mlm_substitutions(text_mask_l, mask_pos_l, top_k=half_k)
        l_cands = _filter_by_vocab(l_cands)
        for w in l_cands:
            candidates.append(
                (_insert_word_before(current_text, word_idx, w), "I"))

    return candidates


# ── Helpers: candidate evaluation ──────────────────────────────────────────

def _evaluate_candidates(
    candidates: list[tuple[str, str]],
    original_text: str,
    current_text: str,
    word_idx: int,
    orig_label: str,
    orig_conf: float,
    model_wrapper,
    similarity_threshold: float,
    target_label: str = None,
) -> tuple[str | None, bool]:
    """Evaluate candidates and select the best one per the paper.

    Selection logic (Garg & Ramakrishnan, 2020):
      - If any cause misclassification → pick the one with HIGHEST
        USE similarity to the original (preserve semantics).
      - If none cause misclassification → pick the one with the MOST
        confidence drop (greedily push toward misclassification).

    POS consistency is checked only for R (replace) operations.

    Returns (best_candidate_text, is_misclassification).
    """
    from utils.text_constraints import compute_windowed_semantic_similarity

    # Pre-compute original POS for R operations
    orig_pos = _get_pos_tag(current_text, word_idx)

    flipping: list[tuple[str, float]] = []       # (text, sim_score)
    non_flipping: list[tuple[str, float]] = []   # (text, confidence_drop)

    for candidate_text, op_type in candidates:
        # POS consistency (R operations only, per paper Section 3.1)
        if op_type == "R":
            cand_pos = _get_pos_tag(candidate_text, word_idx)
            if not _pos_consistent(orig_pos, cand_pos):
                continue

        # Windowed USE similarity (compare against ORIGINAL text)
        passes, sim_score = compute_windowed_semantic_similarity(
            original_text, candidate_text, word_idx, similarity_threshold,
        )
        if not passes:
            continue

        # Classifier evaluation
        label, conf, _ = model_wrapper.predict(candidate_text)

        if target_label is not None:
            is_flip = label.lower() == target_label.lower()
        else:
            is_flip = label != orig_label

        if is_flip:
            flipping.append((candidate_text, sim_score))
        else:
            non_flipping.append((candidate_text, orig_conf - conf))

    # Selection (paper): flip → best USE similarity; no flip → most drop
    if flipping:
        flipping.sort(key=lambda x: x[1], reverse=True)
        return flipping[0][0], True

    if non_flipping:
        non_flipping.sort(key=lambda x: x[1], reverse=True)
        return non_flipping[0][0], False

    return None, False


# ── Main entry point ───────────────────────────────────────────────────────

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

    Strategies (from the paper):
      R    — Replace word via masked MLM (Algorithm 1)
      I    — Insert word via masked MLM (left or right)
      R/I  — Either replace or insert (pick best single operation)
      R+I  — First replace, then insert (sequential, both applied)
      D    — Delete important words (SarabCraft extension, NOT in paper)

    Returns: adversarial text (str).
    """
    from utils.text_word_importance import delete_one_importance
    from utils.text_utils import (
        get_words_and_spans, is_stopword, clean_word,
    )
    from utils.text_constraints import compute_windowed_semantic_similarity

    logger.info("BAE: starting (strategy=%s, cands=%d, sim=%.2f)",
                strategy, max_candidates, similarity_threshold)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    orig_label, orig_conf, _ = model_wrapper.predict(text)
    importance = delete_one_importance(model_wrapper, text, orig_label)

    # Build word-string list for identity tracking (fixes index-drift bug)
    orig_word_strings = [w for w, _, _ in words_spans]

    current_text = text
    perturbations_made = 0
    max_perturbs = max(1, int(len(words_spans) * max_perturbation_ratio))
    modified_positions: set[int] = set()  # RepeatModification

    for word_idx, score in importance:
        if perturbations_made >= max_perturbs:
            break

        # ── RepeatModification: skip already-modified positions ──
        if word_idx in modified_positions:
            continue

        # ── Re-tokenize and verify word identity (index-drift guard) ──
        current_words = get_words_and_spans(current_text)
        if word_idx >= len(current_words):
            continue
        word_str = current_words[word_idx][0]

        # Verify this is still the same word (may have shifted after edits)
        if (word_idx < len(orig_word_strings)
                and clean_word(word_str) != clean_word(orig_word_strings[word_idx])):
            continue

        if is_stopword(word_str) or len(clean_word(word_str)) <= 1:
            continue

        # ── Strategy dispatch ──

        if strategy == "D":
            # SarabCraft extension (NOT in original paper)
            candidate = _delete_word(current_text, word_idx)
            if not candidate.strip():
                continue
            passes, _ = compute_windowed_semantic_similarity(
                text, candidate, word_idx, similarity_threshold)
            if not passes:
                continue
            label, _, _ = model_wrapper.predict(candidate)
            is_flip = ((target_label is not None
                        and label.lower() == target_label.lower())
                       or (target_label is None and label != orig_label))
            current_text = candidate
            perturbations_made += 1
            modified_positions.add(word_idx)
            if is_flip:
                logger.info("BAE: success at perturbation %d (D)",
                            perturbations_made)
                return current_text

        elif strategy == "R":
            cands = _generate_replace_candidates(
                current_text, word_idx, max_candidates)
            best, is_flip = _evaluate_candidates(
                cands, text, current_text, word_idx,
                orig_label, orig_conf, model_wrapper,
                similarity_threshold, target_label)
            if best is not None:
                current_text = best
                perturbations_made += 1
                modified_positions.add(word_idx)
                if is_flip:
                    logger.info("BAE: success at perturbation %d (R)",
                                perturbations_made)
                    return current_text

        elif strategy == "I":
            cands = _generate_insert_candidates(
                current_text, word_idx, max_candidates)
            best, is_flip = _evaluate_candidates(
                cands, text, current_text, word_idx,
                orig_label, orig_conf, model_wrapper,
                similarity_threshold, target_label)
            if best is not None:
                current_text = best
                perturbations_made += 1
                modified_positions.add(word_idx)
                if is_flip:
                    logger.info("BAE: success at perturbation %d (I)",
                                perturbations_made)
                    return current_text

        elif strategy == "R/I":
            # Paper: 'Either replace token t or insert a token to the
            #         left or right of t'  — pick best single op.
            r_cands = _generate_replace_candidates(
                current_text, word_idx, max_candidates)
            i_cands = _generate_insert_candidates(
                current_text, word_idx, max_candidates)
            best, is_flip = _evaluate_candidates(
                r_cands + i_cands, text, current_text, word_idx,
                orig_label, orig_conf, model_wrapper,
                similarity_threshold, target_label)
            if best is not None:
                current_text = best
                perturbations_made += 1
                modified_positions.add(word_idx)
                if is_flip:
                    logger.info("BAE: success at perturbation %d (R/I)",
                                perturbations_made)
                    return current_text

        elif strategy == "R+I":
            # Paper: 'First replace token t, then insert a token to
            #         the left or right of t'  — sequential.
            # Step 1: Replace
            r_cands = _generate_replace_candidates(
                current_text, word_idx, max_candidates)
            r_best, r_flip = _evaluate_candidates(
                r_cands, text, current_text, word_idx,
                orig_label, orig_conf, model_wrapper,
                similarity_threshold, target_label)
            if r_best is not None:
                current_text = r_best
                perturbations_made += 1
                modified_positions.add(word_idx)
                if r_flip:
                    logger.info("BAE: success at perturbation %d (R+I/R)",
                                perturbations_made)
                    return current_text

                # Step 2: Insert on replaced text (only if R was applied)
                if perturbations_made < max_perturbs:
                    i_cands = _generate_insert_candidates(
                        current_text, word_idx, max_candidates)
                    i_best, i_flip = _evaluate_candidates(
                        i_cands, text, current_text, word_idx,
                        orig_label, orig_conf, model_wrapper,
                        similarity_threshold, target_label)
                    if i_best is not None:
                        current_text = i_best
                        perturbations_made += 1
                        if i_flip:
                            logger.info(
                                "BAE: success at perturbation %d (R+I/I)",
                                perturbations_made)
                            return current_text

    logger.info("BAE: finished (%d perturbations)", perturbations_made)
    return current_text
