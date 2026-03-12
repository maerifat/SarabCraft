"""
Pruthi2019 Attack — Pruthi et al., 2019 (arXiv:1905.11268)

Simulates common typos: swap adjacent characters, delete characters,
insert characters, and substitute with adjacent QWERTY keyboard keys.
Practical real-world character-level attack.
"""

import random
import logging

logger = logging.getLogger("textattack.attacks.pruthi2019")


# QWERTY keyboard layout for adjacent key substitution
QWERTY_NEIGHBORS = {
    'q': ['w', 'a'], 'w': ['q', 'e', 's'], 'e': ['w', 'r', 'd'], 'r': ['e', 't', 'f'],
    't': ['r', 'y', 'g'], 'y': ['t', 'u', 'h'], 'u': ['y', 'i', 'j'], 'i': ['u', 'o', 'k'],
    'o': ['i', 'p', 'l'], 'p': ['o', 'l'],
    'a': ['q', 's', 'z'], 's': ['w', 'a', 'd', 'z', 'x'], 'd': ['e', 's', 'f', 'x', 'c'],
    'f': ['r', 'd', 'g', 'c', 'v'], 'g': ['t', 'f', 'h', 'v', 'b'], 'h': ['y', 'g', 'j', 'b', 'n'],
    'j': ['u', 'h', 'k', 'n', 'm'], 'k': ['i', 'j', 'l', 'm'], 'l': ['o', 'k', 'p'],
    'z': ['a', 's', 'x'], 'x': ['s', 'd', 'z', 'c'], 'c': ['d', 'f', 'x', 'v'],
    'v': ['f', 'g', 'c', 'b'], 'b': ['g', 'h', 'v', 'n'], 'n': ['h', 'j', 'b', 'm'],
    'm': ['j', 'k', 'n'],
}


def _swap_adjacent(word: str) -> list[str]:
    """Generate all possible adjacent character swaps."""
    candidates = []
    for i in range(len(word) - 1):
        swapped = word[:i] + word[i+1] + word[i] + word[i+2:]
        candidates.append(swapped)
    return candidates


def _delete_char(word: str) -> list[str]:
    """Generate all possible single character deletions."""
    candidates = []
    for i in range(len(word)):
        deleted = word[:i] + word[i+1:]
        if deleted:  # Don't return empty string
            candidates.append(deleted)
    return candidates


def _insert_char(word: str) -> list[str]:
    """Generate character insertions (duplicate adjacent char)."""
    candidates = []
    for i in range(len(word)):
        # Insert same character
        inserted = word[:i] + word[i] + word[i:]
        candidates.append(inserted)
    return candidates


def _substitute_qwerty(word: str) -> list[str]:
    """Generate QWERTY keyboard adjacent key substitutions."""
    candidates = []
    for i, char in enumerate(word):
        if char.lower() in QWERTY_NEIGHBORS:
            for neighbor in QWERTY_NEIGHBORS[char.lower()]:
                # Preserve case
                if char.isupper():
                    neighbor = neighbor.upper()
                substituted = word[:i] + neighbor + word[i+1:]
                candidates.append(substituted)
    return candidates


def run_pruthi2019(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_perturbations: int = 1,
) -> str:
    """Pruthi2019 typo-based attack.

    Returns: adversarial text (str).
    """
    from utils.text_utils import get_words_and_spans, replace_word_at, is_stopword, clean_word

    logger.info("Pruthi2019: starting (max_pert=%d)", max_perturbations)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    words = [w for w, _, _ in words_spans]
    orig_label, orig_conf, _ = model_wrapper.predict(text)

    # Resolve target label
    resolved_target = target_label
    if target_label is not None:
        from models.text_loader import resolve_target_label
        resolved_target = resolve_target_label(model_wrapper.model, target_label) or target_label

    # Identify mutable positions (skip stopwords and short words)
    mutable_positions = []
    for i, word in enumerate(words):
        if not is_stopword(word) and len(clean_word(word)) >= 3:
            mutable_positions.append(i)

    if not mutable_positions:
        return text

    # Generate typo candidates for each word
    def generate_typos(word: str) -> list[str]:
        """Generate all typo variants for a word."""
        candidates = []
        candidates.extend(_swap_adjacent(word))
        candidates.extend(_delete_char(word))
        candidates.extend(_insert_char(word))
        candidates.extend(_substitute_qwerty(word))
        return list(set(candidates))  # Remove duplicates

    # Greedy search: try each position, pick best
    current_text = text
    perturbations_made = 0

    for _ in range(max_perturbations):
        best_candidate = None
        best_score = 0.0
        best_position = None

        for pos in mutable_positions:
            word = words[pos]
            typos = generate_typos(word)

            for typo in typos:
                candidate_text = replace_word_at(current_text, pos, typo)
                probs = model_wrapper.predict_probs(candidate_text)
                predicted_idx = probs.index(max(probs))
                id2label = getattr(model_wrapper.model.config, 'id2label', {})
                predicted_label = id2label.get(predicted_idx, str(predicted_idx))

                # Score based on target
                if resolved_target is not None:
                    # Targeted: maximize target class probability
                    from models.text_loader import get_label_index
                    target_idx = get_label_index(model_wrapper.model, resolved_target)
                    if target_idx is not None and target_idx < len(probs):
                        score = probs[target_idx]
                    else:
                        score = max(probs) if predicted_label.lower() == resolved_target.lower() else 0.0
                else:
                    # Untargeted: minimize original class probability
                    orig_probs = model_wrapper.predict_probs(text)
                    orig_idx = orig_probs.index(max(orig_probs))
                    score = 1.0 - probs[orig_idx]

                if score > best_score:
                    best_score = score
                    best_candidate = candidate_text
                    best_position = pos

        # Check if we found improvement
        if best_candidate is None:
            break

        # Check success
        _, _, pred_idx = model_wrapper.predict(best_candidate)
        id2label = getattr(model_wrapper.model.config, 'id2label', {})
        pred_label = id2label.get(pred_idx, str(pred_idx))

        if resolved_target is not None:
            if pred_label.lower() == resolved_target.lower():
                logger.info("Pruthi2019: success with %d perturbations", perturbations_made + 1)
                return best_candidate
        else:
            if pred_label != orig_label:
                logger.info("Pruthi2019: success with %d perturbations", perturbations_made + 1)
                return best_candidate

        # Update for next iteration
        current_text = best_candidate
        words = [w for w, _, _ in get_words_and_spans(current_text)]
        perturbations_made += 1

        # Remove the perturbed position from mutable list
        if best_position in mutable_positions:
            mutable_positions.remove(best_position)

        if not mutable_positions:
            break

    logger.info("Pruthi2019: finished with %d perturbations", perturbations_made)
    return current_text
