"""
DeepWordBug Attack — Gao et al., 2018 (arXiv:1801.04354)

Black-box character-level attack. Scores word importance by delete-one
confidence drop, then applies character perturbations (swap adjacent,
substitute nearby-key, delete, insert) to top-k important words.
"""

import random
import logging

logger = logging.getLogger("textattack.attacks.deepwordbug")

# Keyboard proximity map for character substitution
KEYBOARD_NEIGHBORS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "sfcer", "e": "rdws",
    "f": "dgcvr", "g": "fhbvt", "h": "gjbny", "i": "ujko", "j": "hknmu",
    "k": "jlmi", "l": "kop", "m": "njk", "n": "bhjm", "o": "iplk",
    "p": "ol", "q": "wa", "r": "etdf", "s": "adwxz", "t": "rfgy",
    "u": "yhjik", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "xsa",
}


def _swap_adjacent(word: str) -> str:
    """Swap two adjacent characters."""
    if len(word) < 2:
        return word
    i = random.randint(0, len(word) - 2)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _substitute_nearby_key(word: str) -> str:
    """Replace a character with a nearby keyboard key."""
    if not word:
        return word
    chars = list(word)
    i = random.randint(0, len(chars) - 1)
    c = chars[i].lower()
    neighbors = KEYBOARD_NEIGHBORS.get(c, "")
    if neighbors:
        chars[i] = random.choice(neighbors)
    return "".join(chars)


def _delete_char(word: str) -> str:
    """Delete a random character."""
    if len(word) <= 1:
        return word
    i = random.randint(0, len(word) - 1)
    return word[:i] + word[i + 1:]


def _insert_char(word: str) -> str:
    """Insert a random character at a random position."""
    i = random.randint(0, len(word))
    c = random.choice("abcdefghijklmnopqrstuvwxyz")
    return word[:i] + c + word[i:]


_PERTURBATION_FNS = [_swap_adjacent, _substitute_nearby_key, _delete_char, _insert_char]


def run_deepwordbug(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_candidates: int = 5,
    max_perturbations: int = 5,
) -> str:
    """DeepWordBug attack.

    Returns: adversarial text (str).
    """
    from utils.text_word_importance import delete_one_importance
    from utils.text_utils import get_words_and_spans, is_stopword

    logger.info("DeepWordBug: starting (max_pert=%d, max_cand=%d)", max_perturbations, max_candidates)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    # Score word importance
    importance = delete_one_importance(model_wrapper, text)

    # Get original prediction
    orig_label, orig_conf, _ = model_wrapper.predict(text)

    current_text = text
    perturbations_made = 0

    for word_idx, score in importance:
        if perturbations_made >= max_perturbations:
            break

        word = words_spans[word_idx][0]
        if is_stopword(word) or len(word) <= 2:
            continue

        # Try multiple perturbation types, pick best
        best_text = None
        best_conf_drop = 0.0

        for _ in range(max_candidates):
            perturb_fn = random.choice(_PERTURBATION_FNS)
            perturbed_word = perturb_fn(word)

            # Reconstruct text with perturbed word
            from utils.text_utils import replace_word_at
            # Re-parse current text to get updated spans
            current_spans = get_words_and_spans(current_text)
            if word_idx >= len(current_spans):
                break
            # Guard against stale indices after word-boundary changes
            if current_spans[word_idx][0] != words_spans[word_idx][0]:
                continue

            candidate = replace_word_at(current_text, word_idx, perturbed_word)
            label, conf, _ = model_wrapper.predict(candidate)

            if target_label is not None:
                # Targeted: want target class confidence to increase
                if label.lower() == target_label.lower():
                    logger.info("DeepWordBug: success at perturbation %d", perturbations_made + 1)
                    return candidate
            else:
                # Untargeted: want original class confidence to drop
                if label != orig_label:
                    logger.info("DeepWordBug: success at perturbation %d", perturbations_made + 1)
                    return candidate

            conf_drop = orig_conf - conf
            if conf_drop > best_conf_drop:
                best_conf_drop = conf_drop
                best_text = candidate

        if best_text is not None:
            current_text = best_text
            perturbations_made += 1

    logger.info("DeepWordBug: finished (%d perturbations)", perturbations_made)
    return current_text
