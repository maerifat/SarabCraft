"""
DeepWordBug Attack — Gao et al., 2018 (arXiv:1801.04354)

Black-box character-level attack.  Fully compliant with the original paper:
  - 5 scoring strategies: combined, temporal (THS), tail (TTS), replaceone, random
  - 4 transformations: swap adjacent, random-char substitute, delete, insert
  - Constraints: StopwordModification, RepeatModification, LevenshteinEditDistance(ε=30)
  - Greedy word-by-word search
"""

import random
import logging

logger = logging.getLogger("textattack.attacks.deepwordbug")

# ── QWERTY keyboard layout for adjacent key substitution ──────────────────────
KEYBOARD_NEIGHBORS = {
    'q': 'wa', 'w': 'qes', 'e': 'wrd', 'r': 'etf', 't': 'ryg',
    'y': 'tuh', 'u': 'yij', 'i': 'uok', 'o': 'ipl', 'p': 'ol',
    'a': 'qsz', 's': 'wadxz', 'd': 'sfxce', 'f': 'dgcvr',
    'g': 'fhvbt', 'h': 'gjbny', 'j': 'hknmu', 'k': 'jlmi',
    'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv',
    'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
}

# ── Character-level perturbation functions ────────────────────────────────────
# Each returns a list with one random candidate, matching the paper's approach
# of generating one perturbation per transformation type and picking the best.


def _swap_adjacent(word: str) -> list[str]:
    """Swap two random adjacent characters."""
    if len(word) <= 1:
        return []
    i = random.randint(0, len(word) - 2)
    candidate = word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return [candidate]


def _substitute_random_char(word: str) -> list[str]:
    """Replace a random character with a random letter."""
    if len(word) <= 1:
        return []
    i = random.randint(0, len(word) - 1)
    new_char = random.choice("abcdefghijklmnopqrstuvwxyz")
    candidate = word[:i] + new_char + word[i + 1:]
    return [candidate]


def _delete_char(word: str) -> list[str]:
    """Delete a random character."""
    if len(word) <= 1:
        return []
    i = random.randint(0, len(word) - 1)
    candidate = word[:i] + word[i + 1:]
    return [candidate]


def _insert_char(word: str) -> list[str]:
    """Insert a random character at a random position."""
    if len(word) <= 1:
        return []
    i = random.randint(0, len(word))
    new_char = random.choice("abcdefghijklmnopqrstuvwxyz")
    candidate = word[:i] + new_char + word[i:]
    return [candidate]


# All four transformations from the paper
_PERTURBATION_FNS = [_swap_adjacent, _substitute_random_char, _delete_char, _insert_char]

# ── Scoring strategy registry ────────────────────────────────────────────────

SCORING_METHODS = {
    "combined":    "combined_importance",
    "temporal":    "temporal_head_importance",
    "tail":        "temporal_tail_importance",
    "replaceone":  "unk_importance",
    "random":      None,  # random ordering, no scoring needed
}


def _get_importance(model_wrapper, text: str, scoring_method: str):
    """Dispatch to the correct scoring function from text_word_importance.

    Matches the original paper's --scoring flag:
        combined | temporal | tail | replaceone | random
    """
    from utils.text_word_importance import (
        unk_importance,
        temporal_head_importance,
        temporal_tail_importance,
        combined_importance,
    )
    from utils.text_utils import get_words_and_spans

    method = scoring_method.lower()

    if method == "combined":
        return combined_importance(model_wrapper, text)
    elif method == "temporal":
        return temporal_head_importance(model_wrapper, text)
    elif method == "tail":
        return temporal_tail_importance(model_wrapper, text)
    elif method == "replaceone":
        return unk_importance(model_wrapper, text)
    elif method == "random":
        words_spans = get_words_and_spans(text)
        indices = list(range(len(words_spans)))
        random.shuffle(indices)
        return [(i, 0.0) for i in indices]
    else:
        logger.warning("Unknown scoring method '%s', falling back to 'combined'", method)
        return combined_importance(model_wrapper, text)


def run_deepwordbug(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_candidates: int = 5,
    max_perturbations: int = 5,
    max_edit_distance: int = 30,
    scoring_method: str = "combined",
) -> str:
    """DeepWordBug attack — fully compliant with Gao et al., 2018.

    Scoring (--scoring flag from original paper):
      - combined: (THS + TTS) / 2   (default, best in paper)
      - temporal: Temporal Head Score (prefix-based)
      - tail: Temporal Tail Score (suffix-based)
      - replaceone: Replace-1 ([UNK] replacement)
      - random: random word ordering

    Search: Greedy word-by-word
      1. Rank words by chosen scoring strategy.
      2. For each word in importance order, generate candidates from all 4
         character transformations, evaluate them, pick the best.
      3. Accept if it improves score; return immediately on label flip.

    Constraints:
      - StopwordModification: skip stopwords
      - RepeatModification: never modify the same word index twice
      - LevenshteinEditDistance(ε): total char edit distance ≤ max_edit_distance

    Args:
        model_wrapper: wrapped model with .predict() -> (label, conf, idx)
        tokenizer: HuggingFace tokenizer (unused, kept for API compat)
        text: input text to attack
        target_label: target class name (None = untargeted)
        max_candidates: kept for API compat
        max_perturbations: max number of words to perturb
        max_edit_distance: Levenshtein ε (default 30, per paper)
        scoring_method: one of "combined", "temporal", "tail", "replaceone", "random"

    Returns:
        adversarial text (str).
    """
    from utils.text_utils import get_words_and_spans, replace_word_at, is_stopword
    from utils.text_metrics import char_edit_distance

    logger.info(
        "DeepWordBug: starting (scoring=%s, max_pert=%d, max_edit_dist=%d)",
        scoring_method, max_perturbations, max_edit_distance,
    )

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    # ── Word Importance Ranking ──────────────────────────────────────────
    importance = _get_importance(model_wrapper, text, scoring_method)

    # Get original prediction
    orig_label, orig_conf, _ = model_wrapper.predict(text)

    current_text = text
    perturbations_made = 0
    modified_indices = set()  # RepeatModification constraint

    for word_idx, score in importance:
        if perturbations_made >= max_perturbations:
            break

        # RepeatModification: skip already-modified words
        if word_idx in modified_indices:
            continue

        word = words_spans[word_idx][0]

        # StopwordModification: skip stopwords and very short words
        if is_stopword(word) or len(word) <= 2:
            continue

        # ── Generate candidates from all 4 transformations ───────────────
        candidates = []
        current_spans = get_words_and_spans(current_text)
        if word_idx >= len(current_spans):
            break
        # Guard against stale indices after word-boundary changes
        current_word = current_spans[word_idx][0]
        if current_word != word:
            continue

        for perturb_fn in _PERTURBATION_FNS:
            perturbed_words = perturb_fn(current_word)
            for pw in perturbed_words:
                candidate = replace_word_at(current_text, word_idx, pw)

                # LevenshteinEditDistance constraint
                if char_edit_distance(text, candidate) > max_edit_distance:
                    continue

                candidates.append(candidate)

        if not candidates:
            continue

        # ── Evaluate all candidates, pick the best ───────────────────────
        best_text = None
        best_score = -float("inf")

        for candidate in candidates:
            label, conf, _ = model_wrapper.predict(candidate)

            if target_label is not None:
                # Targeted: want target class confidence to increase
                if label.lower() == target_label.lower():
                    logger.info("DeepWordBug: success at perturbation %d", perturbations_made + 1)
                    return candidate
                candidate_score = conf if label.lower() == target_label.lower() else -conf
            else:
                # Untargeted: want original class confidence to drop
                if label != orig_label:
                    logger.info("DeepWordBug: success at perturbation %d", perturbations_made + 1)
                    return candidate
                candidate_score = orig_conf - conf

            if candidate_score > best_score:
                best_score = candidate_score
                best_text = candidate

        # Accept only if it improves the score (greedy)
        if best_text is not None and best_score > 0:
            current_text = best_text
            perturbations_made += 1
            modified_indices.add(word_idx)

    logger.info("DeepWordBug: finished (%d perturbations)", perturbations_made)
    return current_text
