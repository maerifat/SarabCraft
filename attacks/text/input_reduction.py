"""
Input Reduction — Feng et al., 2018 (EMNLP 2018)

Pathologies of Neural Models Make Interpretations Difficult.

Exact match to TextAttack InputReductionFeng2018 recipe:
  - Transformation: WordDeletion
  - Constraints: StopwordModification (stopwords protected from deletion)
  - Goal function: InputReduction (maximizable)
    → maintain original prediction while minimising word count
  - Search: GreedyWordSwapWIR(wir_method="delete")
    → rank words by leave-one-out InputReduction score,
      greedily delete in descending score order

The goal is to reduce the input as much as possible WHILE KEEPING
the model prediction unchanged.  This exposes pathological model
over-confidence: models often maintain predictions even when
input is reduced to near-nonsense.

Algorithm (matching TextAttack):
  1. Score each non-stopword by the InputReduction score when deleted
     (score = word_reduction_fraction + confidence / initial_words;
      score = 0 if deletion changes the predicted label)
  2. Sort words by score descending (best deletion candidate first)
  3. Greedily delete words in that order:
     - accept deletion only if it improves the score
       (i.e. prediction is maintained)
     - skip if prediction changes (score drops to 0)
  4. Stop when word count ≤ stop_at_length with same prediction (goal)
     or all candidates exhausted

Reference: https://arxiv.org/abs/1804.07781
TextAttack: textattack.attack_recipes.input_reduction_feng_2018
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
    """Input Reduction (Feng et al., 2018).

    Exact match to TextAttack InputReductionFeng2018 recipe.
    Iteratively removes the least important word while maintaining
    the model prediction, exposing pathological over-confidence.

    Args:
        model_wrapper: wrapped model with .predict() / .predict_probs().
        tokenizer: HuggingFace tokenizer (unused, API compat).
        text: input text to reduce.
        target_label: unused — Input Reduction is untargeted-only (API compat).
        max_reduction_ratio: max fraction of words to remove (safety bound;
            not in original TextAttack recipe, SarabCraft extension).
        stop_at_length: target minimum word count (TextAttack target_num_words).

    Returns:
        Maximally reduced text that still maintains the original prediction.
    """
    from utils.text_utils import get_words_and_spans, is_stopword

    logger.info(
        "InputReduction: starting (stop_at=%d, max_reduction=%.2f)",
        stop_at_length, max_reduction_ratio,
    )

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    words = [w for w, _, _ in words_spans]
    initial_num_words = len(words)

    orig_probs = model_wrapper.predict_probs(text)
    orig_label_idx = max(range(len(orig_probs)), key=lambda k: orig_probs[k])

    # ── InputReduction score (TextAttack InputReduction._get_score) ──────
    # score = 0 when prediction changes (_should_skip);
    # otherwise rewards fewer words + maintained confidence.
    # _is_goal_complete: same prediction AND word count ≤ stop_at_length.
    def _ir_score(reduced_words):
        if not reduced_words:
            return 0.0, False
        probs = model_wrapper.predict_probs(" ".join(reduced_words))
        pred_idx = max(range(len(probs)), key=lambda k: probs[k])
        if pred_idx != orig_label_idx:
            return 0.0, False
        cur_n = len(reduced_words)
        num_words_score = max(
            (initial_num_words - cur_n) / initial_num_words, 0,
        )
        model_score = probs[orig_label_idx]
        score = min(num_words_score + model_score / initial_num_words, 1.0)
        return score, cur_n <= stop_at_length

    # ── Candidate indices: exclude stopwords (StopwordModification) ──────
    candidate_indices = [
        i for i, w in enumerate(words) if not is_stopword(w)
    ]
    if not candidate_indices:
        return text

    # ── Word importance ranking (GreedyWordSwapWIR "delete") ─────────────
    # Delete each candidate word independently from the original text and
    # score the result.  Sort descending: highest score = least-important
    # word whose removal best maintains prediction.
    index_scores = []
    for idx in candidate_indices:
        trial = [w for j, w in enumerate(words) if j != idx]
        score, goal_complete = _ir_score(trial)
        if goal_complete:
            logger.info("InputReduction: goal reached during ranking")
            return " ".join(trial)
        index_scores.append(score)

    ranked = sorted(
        zip(candidate_indices, index_scores),
        key=lambda x: x[1],
        reverse=True,
    )
    index_order = [idx for idx, _ in ranked]

    # ── Greedy deletion search (GreedyWordSwapWIR.perform_search) ────────
    # Iterate through index_order.  For each word, try deleting it from the
    # current (progressively reduced) text.  Accept only if score improves
    # (prediction maintained); skip if prediction changed (score = 0).
    deleted: set[int] = set()
    cur_score, _ = _ir_score(words)
    max_removals = max(1, int(initial_num_words * max_reduction_ratio))

    for orig_idx in index_order:
        if len(deleted) >= max_removals:
            break

        trial_deleted = deleted | {orig_idx}
        trial_words = [w for j, w in enumerate(words) if j not in trial_deleted]
        if not trial_words:
            continue

        trial_score, goal_complete = _ir_score(trial_words)

        if trial_score > cur_score:
            deleted.add(orig_idx)
            cur_score = trial_score

            if goal_complete:
                logger.info(
                    "InputReduction: goal complete — %d → %d words",
                    initial_num_words, len(trial_words),
                )
                return " ".join(trial_words)

    final_words = [w for j, w in enumerate(words) if j not in deleted]
    logger.info(
        "InputReduction: finished — %d → %d words (removed %d)",
        initial_num_words, len(final_words), len(deleted),
    )
    return " ".join(final_words)
