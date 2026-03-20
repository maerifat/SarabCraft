"""
DeepWordBug Attack — Gao et al., 2018 (arXiv:1801.04354)

Black-box character-level attack.  Faithful to the original paper and official
QData/deepWordBug implementation:
  - 5 scoring strategies: combined, temporal (THS), tail (TTS), replaceone, random
  - 5 transformations: swap, flip (substitute), remove (delete), insert, homoglyph
  - Attack: score words → rank by importance → apply chosen transformer to top ε words
  - ε = max_perturbations (number of words to perturb, official: --power)
"""

import random
import logging

logger = logging.getLogger("textattack.attacks.deepwordbug")

# ── Homoglyph mapping ────────────────────────────────────────────────────────
# Exact mapping from official QData/deepWordBug transformer.py
HOMOGLYPHS = {
    '-': '˗', '9': '৭', '8': 'Ȣ', '7': '𝟕', '6': 'б', '5': 'Ƽ',
    '4': 'Ꮞ', '3': 'Ʒ', '2': 'ᒿ', '1': 'l', '0': 'O', "'": '`',
    'a': 'ɑ', 'b': 'Ь', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝚏',
    'g': 'ɡ', 'h': 'հ', 'i': 'і', 'j': 'ϳ', 'k': '𝒌', 'l': 'ⅼ',
    'm': 'ｍ', 'n': 'ո', 'o': 'о', 'p': 'р', 'q': 'ԛ', 'r': 'ⲅ',
    's': 'ѕ', 't': '𝚝', 'u': 'ս', 'v': 'ѵ', 'w': 'ԝ', 'x': '×',
    'y': 'у', 'z': 'ᴢ',
}

# ── Character-level perturbation functions ────────────────────────────────────
# Each matches the corresponding function in the official transformer.py.


def _swap_adjacent(word: str) -> list[str]:
    """Swap two random adjacent characters.  (official: swap)"""
    if len(word) <= 1:
        return []
    i = random.randint(0, len(word) - 2)
    candidate = word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return [candidate]


def _substitute_char(word: str) -> list[str]:
    """Replace one random character with a *different* random letter.

    Matches official flip() — avoids replacing with the same character:
        rletter = randint(0,24)+97; if rletter >= letter: rletter += 1
    """
    if not word:
        return []
    i = random.randint(0, len(word) - 1)
    letter = ord(word[i].lower())
    rletter = random.randint(0, 24) + 97          # 25 values: a(97)…y(121)
    if rletter >= letter:
        rletter += 1                                # skip original letter
    candidate = word[:i] + chr(rletter) + word[i + 1:]
    return [candidate]


def _delete_char(word: str) -> list[str]:
    """Delete one random character.  (official: remove)"""
    if len(word) <= 1:
        return []
    i = random.randint(0, len(word) - 1)
    candidate = word[:i] + word[i + 1:]
    return [candidate]


def _insert_char(word: str) -> list[str]:
    """Insert a random character at a random position.  (official: insert)"""
    i = random.randint(0, len(word))
    new_char = chr(97 + random.randint(0, 25))
    candidate = word[:i] + new_char + word[i:]
    return [candidate]


def _homoglyph(word: str) -> list[str]:
    """Replace one random character with its Unicode homoglyph.

    Exact mapping from official QData/deepWordBug transformer.py.
    This is the most effective transformer in the paper.
    """
    if not word:
        return []
    i = random.randint(0, len(word) - 1)
    ch = word[i]
    replacement = HOMOGLYPHS.get(ch, HOMOGLYPHS.get(ch.lower(), ch))
    candidate = word[:i] + replacement + word[i + 1:]
    if candidate == word:
        return []
    return [candidate]


# ── Transformer registry matching the official --transformer flag ─────────────

TRANSFORMER_REGISTRY = {
    "swap":      _swap_adjacent,
    "flip":      _substitute_char,
    "remove":    _delete_char,
    "insert":    _insert_char,
    "homoglyph": _homoglyph,
}

# ── Scoring strategy registry ────────────────────────────────────────────────

SCORING_METHODS = {
    "combined":    "combined_importance",
    "temporal":    "temporal_head_importance",
    "tail":        "temporal_tail_importance",
    "replaceone":  "_replaceone_importance",
    "random":      None,
}


def _replaceone_importance(model_wrapper, text: str) -> list[tuple[int, float]]:
    """Replace-1 score — Gao et al., 2018 (official: scoring.replaceone).

    Replaces each word with [UNK] and measures the confidence drop of the
    original predicted class.  No label-flip bonus, no stopword filtering —
    faithful to the official scoring.replaceone() which uses NLL loss.

    Ranking equivalence: official sorts by NLL (higher = more important);
    we sort by P_drop = P_Y(orig) − P_Y(with_UNK) (higher = more important).
    Both produce the same word ranking since P_Y(orig) is constant.
    """
    from utils.text_utils import get_words_and_spans

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return []

    _, _, orig_label_idx = model_wrapper.predict(text)
    orig_probs = model_wrapper.predict_probs(text)
    orig_conf = orig_probs[orig_label_idx] if orig_label_idx < len(orig_probs) else 0.0

    scores = []
    for i, (word, start, end) in enumerate(words_spans):
        replaced = text[:start] + "[UNK]" + text[end:]
        rep_probs = model_wrapper.predict_probs(replaced)
        rep_conf = rep_probs[orig_label_idx] if orig_label_idx < len(rep_probs) else 0.0
        importance = orig_conf - rep_conf
        scores.append((i, importance))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def _get_importance(model_wrapper, text: str, scoring_method: str):
    """Dispatch to the correct scoring function.

    Matches the original paper's --scoring flag:
        combined | temporal | tail | replaceone | random
    """
    from utils.text_word_importance import (
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
        return _replaceone_importance(model_wrapper, text)
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
    max_perturbations: int = 5,
    scoring_method: str = "combined",
    transformer: str = "homoglyph",
) -> str:
    """DeepWordBug attack — faithful to Gao et al., 2018.

    Algorithm (matches official QData/deepWordBug attack.py → attackword()):
      1. Score all words using chosen scoring strategy.
      2. Rank words by importance (descending).
      3. Apply chosen transformer to the top ε words.
      4. Return perturbed text.

    Scoring (--scoring flag from original paper):
      - combined:   (THS + TTS) / 2   (default, best in paper)
      - temporal:   Temporal Head Score (prefix-based)
      - tail:       Temporal Tail Score (suffix-based)
      - replaceone: Replace-1 ([UNK] replacement, confidence drop)
      - random:     random word ordering

    Transformers (--transformer flag from original paper):
      - homoglyph: Unicode visual lookalike (default, most effective in paper)
      - swap:      swap two adjacent characters
      - flip:      substitute one character with a different random letter
      - remove:    delete one random character
      - insert:    insert one random character

    Args:
        model_wrapper: wrapped model with .predict() → (label, conf, idx)
        tokenizer: HuggingFace tokenizer (unused, kept for API compat)
        text: input text to attack
        target_label: target class name (unused, kept for API compat)
        max_perturbations: ε — number of words to perturb (paper: --power)
        scoring_method: one of "combined", "temporal", "tail", "replaceone", "random"
        transformer: one of "homoglyph", "swap", "flip", "remove", "insert"

    Returns:
        adversarial text (str).
    """
    from utils.text_utils import get_words_and_spans, replace_words_at

    logger.info(
        "DeepWordBug: starting (scoring=%s, transformer=%s, power=%d)",
        scoring_method, transformer, max_perturbations,
    )

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    # Resolve transformer function
    transform_fn = TRANSFORMER_REGISTRY.get(transformer.lower())
    if transform_fn is None:
        logger.warning("Unknown transformer '%s', falling back to 'homoglyph'", transformer)
        transform_fn = _homoglyph

    # ── Step 1: Score & rank words ───────────────────────────────────────
    importance = _get_importance(model_wrapper, text, scoring_method)

    # ── Step 2: Apply transformer to top ε words ─────────────────────────
    # Matches official attackword(): iterate words in importance order,
    # apply ONE chosen transformer to each, up to max_perturbations words.
    # No per-word candidate evaluation, no greedy accept-if-improves.
    replacements = {}
    perturbations = 0

    for word_idx, score in importance:
        if perturbations >= max_perturbations:
            break

        word = words_spans[word_idx][0]
        candidates = transform_fn(word)
        if not candidates:
            continue

        replacements[word_idx] = candidates[0]
        perturbations += 1

    # ── Step 3: Apply all replacements, return result ────────────────────
    if not replacements:
        logger.info("DeepWordBug: no perturbations applied")
        return text

    adversarial = replace_words_at(text, replacements)
    logger.info("DeepWordBug: finished (%d perturbations)", perturbations)
    return adversarial
