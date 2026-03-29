"""
Pruthi2019 Attack — Pruthi et al., 2019 (arXiv:1905.11268)

Faithful to the original paper "Combating Adversarial Misspellings with
Robust Word Recognition" and the official danishpruthi/Adversarial-Misspellings
implementation.

Four character-level typo operations — swap, drop, add, keyboard — applied
only to *internal* characters (first and last character of each word are
always preserved, matching the psycholinguistic constraint from the paper).

Matches the TextAttack Pruthi2019 recipe:
  CompositeTransformation(swap, delete, insert, qwerty) with
  skip_first_char=True, skip_last_char=True, random_one=False.
  Constraints: MinWordLength(4), StopwordModification, MaxWordsPerturbed,
  RepeatModification.  Goal: UntargetedClassification.  Search: GreedySearch.

Extension: targeted attack support via target_label (not in original paper).
"""

import logging
import string

logger = logging.getLogger("textattack.attacks.pruthi2019")

_ALPHABETS = list(string.ascii_lowercase)

# ── QWERTY keyboard adjacency (4-directional: up/down/left/right) ─────────
# Generated from the official danishpruthi/Adversarial-Misspellings
# get_keyboard_neighbors() which uses dx=[-1,1,0,0] dy=[0,0,-1,1] on:
#   row 0: q w e r t y u i o p
#   row 1: a s d f g h j k l
#   row 2: z x c v b n m
QWERTY_NEIGHBORS = {
    'q': ['w', 'a'],
    'w': ['q', 'e', 's'],
    'e': ['w', 'r', 'd'],
    'r': ['e', 't', 'f'],
    't': ['r', 'y', 'g'],
    'y': ['t', 'u', 'h'],
    'u': ['y', 'i', 'j'],
    'i': ['u', 'o', 'k'],
    'o': ['i', 'p', 'l'],
    'p': ['o'],
    'a': ['q', 's', 'z'],
    's': ['w', 'a', 'd', 'x'],
    'd': ['e', 's', 'f', 'c'],
    'f': ['r', 'd', 'g', 'v'],
    'g': ['t', 'f', 'h', 'b'],
    'h': ['y', 'g', 'j', 'n'],
    'j': ['u', 'h', 'k', 'm'],
    'k': ['i', 'j', 'l'],
    'l': ['o', 'k'],
    'z': ['a', 'x'],
    'x': ['s', 'z', 'c'],
    'c': ['d', 'x', 'v'],
    'v': ['f', 'c', 'b'],
    'b': ['g', 'v', 'n'],
    'n': ['h', 'b', 'm'],
    'm': ['j', 'n'],
}

MIN_WORD_LENGTH = 4


def _swap_adjacent(word: str) -> list[str]:
    """Swap two adjacent internal characters (official: swap_one_attack).

    Preserves first and last character: range(1, len-2).
    """
    candidates = []
    for i in range(1, len(word) - 2):
        swapped = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        candidates.append(swapped)
    return candidates


def _delete_char(word: str) -> list[str]:
    """Delete one internal character (official: drop_one_attack).

    Preserves first and last character: range(1, len-1).
    """
    candidates = []
    for i in range(1, len(word) - 1):
        deleted = word[:i] + word[i + 1:]
        candidates.append(deleted)
    return candidates


def _insert_char(word: str) -> list[str]:
    """Insert a random character from a-z at each internal position
    (official: add_one_attack).

    Preserves first and last character: positions range(1, len).
    Inserts every letter of the alphabet at each valid position.
    """
    candidates = []
    for i in range(1, len(word)):
        for alpha in _ALPHABETS:
            inserted = word[:i] + alpha + word[i:]
            candidates.append(inserted)
    return candidates


def _substitute_qwerty(word: str) -> list[str]:
    """Substitute one internal character with an adjacent QWERTY key
    (official: key_one_attack).

    Preserves first and last character: range(1, len-1).
    Uses 4-directional (up/down/left/right) keyboard adjacency.
    """
    candidates = []
    for i in range(1, len(word) - 1):
        char = word[i]
        key = char.lower()
        if key in QWERTY_NEIGHBORS:
            for neighbor in QWERTY_NEIGHBORS[key]:
                if char.isupper():
                    neighbor = neighbor.upper()
                substituted = word[:i] + neighbor + word[i + 1:]
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

    Faithful to the original paper and TextAttack Pruthi2019 recipe:
      1. Identify mutable words (length >= 4, not stopwords).
      2. For each perturbation budget step, generate ALL typo candidates
         for every mutable word (swap + drop + add + keyboard, all with
         skip_first_char / skip_last_char).
      3. Greedy search: pick the single candidate that best advances the
         attack objective.  Uses batched model inference for throughput.
      4. Stop on misclassification (untargeted) or target reached (targeted),
         or when budget is exhausted.
      5. RepeatModification: each word is perturbed at most once.

    Returns: adversarial text (str).
    """
    from utils.text_utils import get_words_and_spans, replace_word_at, is_stopword, clean_word

    logger.info("Pruthi2019: starting (max_pert=%d)", max_perturbations)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    words = [w for w, _, _ in words_spans]
    orig_label, orig_conf, _ = model_wrapper.predict(text)

    resolved_target = target_label
    if target_label is not None:
        from models.text_loader import resolve_target_label
        resolved_target = resolve_target_label(model_wrapper.model, target_label) or target_label

    # MinWordLength(4) + StopwordModification — matches TextAttack constraints
    mutable_positions = []
    for i, word in enumerate(words):
        if not is_stopword(word) and len(clean_word(word)) >= MIN_WORD_LENGTH:
            mutable_positions.append(i)

    if not mutable_positions:
        return text

    def generate_typos(word: str) -> list[str]:
        """Generate all typo variants for a word (all four operations)."""
        candidates = []
        candidates.extend(_swap_adjacent(word))
        candidates.extend(_delete_char(word))
        candidates.extend(_insert_char(word))
        candidates.extend(_substitute_qwerty(word))
        return list(set(candidates))

    # Pre-compute original class index for untargeted scoring
    orig_class_idx = None
    target_idx = None
    if resolved_target is None:
        orig_probs = model_wrapper.predict_probs(text)
        orig_class_idx = orig_probs.index(max(orig_probs))
    else:
        from models.text_loader import get_label_index
        target_idx = get_label_index(model_wrapper.model, resolved_target)

    # Greedy search with batched inference
    current_text = text
    perturbations_made = 0

    for _ in range(max_perturbations):
        # Build full candidate list across all mutable positions
        candidate_texts = []
        candidate_positions = []

        for pos in mutable_positions:
            word = words[pos]
            typos = generate_typos(word)
            for typo in typos:
                candidate_texts.append(replace_word_at(current_text, pos, typo))
                candidate_positions.append(pos)

        if not candidate_texts:
            break

        # Batched model query — single call scores all candidates
        all_probs = model_wrapper.predict_probs_batch(candidate_texts)

        # Score each candidate
        best_idx = None
        best_score = 0.0

        for ci, probs in enumerate(all_probs):
            if resolved_target is not None:
                if target_idx is not None and target_idx < len(probs):
                    score = probs[target_idx]
                else:
                    predicted_idx = probs.index(max(probs))
                    id2label = getattr(model_wrapper.model.config, 'id2label', {})
                    predicted_label = id2label.get(predicted_idx, str(predicted_idx))
                    score = max(probs) if predicted_label.lower() == resolved_target.lower() else 0.0
            else:
                score = 1.0 - probs[orig_class_idx]

            if score > best_score:
                best_score = score
                best_idx = ci

        if best_idx is None:
            break

        best_candidate = candidate_texts[best_idx]
        best_position = candidate_positions[best_idx]

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

        current_text = best_candidate
        words = [w for w, _, _ in get_words_and_spans(current_text)]
        perturbations_made += 1

        # RepeatModification: don't perturb the same word twice
        if best_position in mutable_positions:
            mutable_positions.remove(best_position)

        if not mutable_positions:
            break

    logger.info("Pruthi2019: finished with %d perturbations", perturbations_made)
    return current_text
