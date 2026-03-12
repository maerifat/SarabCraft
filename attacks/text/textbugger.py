"""
TextBugger Attack — Li et al., 2019 (arXiv:1812.05271)

Black-box character-level attack with five perturbation strategies:
  1. Insert space into word
  2. Delete a character
  3. Swap two adjacent characters
  4. Substitute with visually similar (homoglyph) character
  5. Substitute with nearby keyboard key
"""

import random
import logging

logger = logging.getLogger("textattack.attacks.textbugger")

# Homoglyph table: visually similar Unicode characters
HOMOGLYPHS = {
    "a": ["à", "á", "â", "ã", "ä", "å", "ɑ", "а"],  # last is Cyrillic
    "b": ["Ꮟ", "Ь", "ƅ"],
    "c": ["ϲ", "с", "ⅽ"],  # Cyrillic с
    "d": ["ԁ", "ⅾ"],
    "e": ["è", "é", "ê", "ë", "е", "ɛ"],  # Cyrillic е
    "f": ["ẝ"],
    "g": ["ɡ", "ǥ"],
    "h": ["һ", "ℎ"],  # Cyrillic һ
    "i": ["ì", "í", "î", "ï", "і", "ɪ"],  # Cyrillic і
    "j": ["ϳ", "ј"],
    "k": ["κ", "к"],
    "l": ["ⅼ", "ℓ", "ӏ"],
    "m": ["ⅿ", "м"],
    "n": ["ո", "ν"],
    "o": ["ò", "ó", "ô", "õ", "ö", "о", "ο"],  # Cyrillic о, Greek ο
    "p": ["р", "ρ"],  # Cyrillic р, Greek ρ
    "q": ["ԛ"],
    "r": ["г"],
    "s": ["ѕ", "ꜱ"],  # Cyrillic ѕ
    "t": ["τ", "т"],
    "u": ["ù", "ú", "û", "ü", "υ"],
    "v": ["ν", "ⅴ"],
    "w": ["ԝ", "ω"],
    "x": ["х", "ⅹ"],  # Cyrillic х
    "y": ["у", "ỿ"],  # Cyrillic у
    "z": ["ᴢ"],
}


def _insert_space(word: str) -> str:
    """Insert a zero-width space or regular space inside the word."""
    if len(word) < 2:
        return word
    i = random.randint(1, len(word) - 1)
    return word[:i] + " " + word[i:]


def _delete_char(word: str) -> str:
    """Delete a random character (not first or last to maintain readability)."""
    if len(word) <= 2:
        return word
    i = random.randint(1, len(word) - 2)
    return word[:i] + word[i + 1:]


def _swap_adjacent(word: str) -> str:
    """Swap two adjacent characters."""
    if len(word) < 2:
        return word
    i = random.randint(0, len(word) - 2)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _substitute_homoglyph(word: str) -> str:
    """Replace a character with a visually similar Unicode character."""
    chars = list(word)
    # Find replaceable positions
    positions = [i for i, c in enumerate(chars) if c.lower() in HOMOGLYPHS]
    if not positions:
        return _swap_adjacent(word)  # fallback
    i = random.choice(positions)
    candidates = HOMOGLYPHS[chars[i].lower()]
    chars[i] = random.choice(candidates)
    return "".join(chars)


def _substitute_nearby_key(word: str) -> str:
    """Replace a character with a nearby keyboard key."""
    from attacks.text.deepwordbug import KEYBOARD_NEIGHBORS
    chars = list(word)
    i = random.randint(0, len(chars) - 1)
    neighbors = KEYBOARD_NEIGHBORS.get(chars[i].lower(), "")
    if neighbors:
        chars[i] = random.choice(neighbors)
    return "".join(chars)


_PERTURBATION_FNS = [
    _insert_space, _delete_char, _swap_adjacent,
    _substitute_homoglyph, _substitute_nearby_key,
]


def run_textbugger(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_perturbations: int = 5,
) -> str:
    """TextBugger attack.

    Returns: adversarial text (str).
    """
    from utils.text_word_importance import delete_one_importance
    from utils.text_utils import get_words_and_spans, replace_word_at, is_stopword

    logger.info("TextBugger: starting (max_pert=%d)", max_perturbations)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    importance = delete_one_importance(model_wrapper, text)
    orig_label, orig_conf, _ = model_wrapper.predict(text)

    current_text = text
    perturbations_made = 0

    for word_idx, score in importance:
        if perturbations_made >= max_perturbations:
            break

        word = words_spans[word_idx][0]
        if is_stopword(word) or len(word) <= 2:
            continue

        # Try each perturbation type, pick the one with greatest impact
        best_text = None
        best_impact = -1.0

        for perturb_fn in _PERTURBATION_FNS:
            perturbed_word = perturb_fn(word)
            current_spans = get_words_and_spans(current_text)
            if word_idx >= len(current_spans):
                break
            # Guard against stale indices after word-boundary changes
            if current_spans[word_idx][0] != words_spans[word_idx][0]:
                continue

            candidate = replace_word_at(current_text, word_idx, perturbed_word)
            label, conf, _ = model_wrapper.predict(candidate)

            if target_label is not None:
                if label.lower() == target_label.lower():
                    logger.info("TextBugger: success at perturbation %d", perturbations_made + 1)
                    return candidate
            else:
                if label != orig_label:
                    logger.info("TextBugger: success at perturbation %d", perturbations_made + 1)
                    return candidate

            impact = orig_conf - conf
            if impact > best_impact:
                best_impact = impact
                best_text = candidate

        if best_text is not None:
            current_text = best_text
            perturbations_made += 1

    logger.info("TextBugger: finished (%d perturbations)", perturbations_made)
    return current_text
