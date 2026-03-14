"""
TextBugger Attack — Li et al., 2018 (arXiv:1812.05271)

Hybrid character-level and word-level attack with five perturbation strategies:
  Character-level (4 bugs, Sub-C covers two sub-variants):
    1. Insert: insert a space into the word
    2. Delete: delete a random character (not first/last)
    3. Swap: swap two adjacent characters (not first/last)
    4. Sub-C: substitute with visually similar character (homoglyphs)
            OR substitute with keyboard-adjacent character (keyboard typos)
  Word-level:
    5. Sub-W: substitute with nearest GloVe embedding neighbour (k=5)

Supports both white-box (gradient-based) and black-box (query-based) modes.
Black-box mode includes sentence-level importance ranking for multi-sentence
inputs, as specified in the original paper.
"""

import random
import logging

logger = logging.getLogger("textattack.attacks.textbugger")

# Homoglyph table: visually similar Unicode characters
# Includes ASCII-based substitutions explicitly cited in the paper (Li et al., 2018)
HOMOGLYPHS = {
    "a": ["@", "à", "á", "â", "ã", "ä", "å", "ɑ", "а"],  # ASCII @ + Unicode variants
    "b": ["Ꮟ", "Ь", "ƅ"],
    "c": ["ϲ", "с", "ⅽ"],  # Cyrillic с
    "d": ["ԁ", "ⅾ"],
    "e": ["è", "é", "ê", "ë", "е", "ɛ"],  # Cyrillic е
    "f": ["ẝ"],
    "g": ["ɡ", "ǥ"],
    "h": ["һ", "ℎ"],  # Cyrillic һ
    "i": ["1", "!", "ì", "í", "î", "ï", "і", "ɪ"],  # ASCII 1, ! + Unicode variants
    "j": ["ϳ", "ј"],
    "k": ["κ", "к"],
    "l": ["1", "I", "ⅼ", "ℓ", "ӏ"],  # ASCII 1, I + Unicode variants
    "m": ["ⅿ", "м"],
    "n": ["ո", "ν"],
    "o": ["0", "ò", "ó", "ô", "õ", "ö", "о", "ο"],  # ASCII 0 + Unicode variants
    "p": ["р", "ρ"],  # Cyrillic р, Greek ρ
    "q": ["ԛ"],
    "r": ["г"],
    "s": ["$", "ѕ", "ꜱ"],  # ASCII $ + Cyrillic ѕ
    "t": ["τ", "т"],
    "u": ["ù", "ú", "û", "ü", "υ"],
    "v": ["ν", "ⅴ"],
    "w": ["ԝ", "ω"],
    "x": ["х", "ⅹ"],  # Cyrillic х
    "y": ["у", "ỿ"],  # Cyrillic у
    "z": ["ᴢ"],
}


def _insert_space(word: str) -> str:
    """Bug 1 — Insert: insert a space inside the word."""
    if len(word) < 2:
        return word
    i = random.randint(1, len(word) - 1)
    return word[:i] + " " + word[i:]


def _delete_char(word: str) -> str:
    """Bug 2 — Delete: delete a random character (not first or last)."""
    if len(word) <= 2:
        return word
    i = random.randint(1, len(word) - 2)
    return word[:i] + word[i + 1:]


def _swap_adjacent(word: str) -> str:
    """Bug 3 — Swap: swap two adjacent characters (not first or last).

    Following typoglycemia principle: humans rely on boundary anchors.
    Paper explicitly states: "not first or last" to maintain stealthiness.
    """
    if len(word) <= 3:
        return word  # Cannot swap without touching boundaries
    # Exclude index 0 and (len-2) to preserve first and last letters
    i = random.randint(1, len(word) - 3)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _substitute_homoglyph(word: str) -> str:
    """Bug 4a — Sub-C (homoglyph): replace a character with a visually similar one."""
    chars = list(word)
    # Find replaceable positions
    positions = [i for i, c in enumerate(chars) if c.lower() in HOMOGLYPHS]
    if not positions:
        return _swap_adjacent(word)  # fallback
    i = random.choice(positions)
    candidates = HOMOGLYPHS[chars[i].lower()]
    chars[i] = random.choice(candidates)
    return "".join(chars)


# Keyboard adjacency map for typo-based substitutions
# Based on QWERTY keyboard layout
KEYBOARD_ADJACENT = {
    "q": ["w", "a"],
    "w": ["q", "e", "s"],
    "e": ["w", "r", "d"],
    "r": ["e", "t", "f"],
    "t": ["r", "y", "g"],
    "y": ["t", "u", "h"],
    "u": ["y", "i", "j"],
    "i": ["u", "o", "k"],
    "o": ["i", "p", "l"],
    "p": ["o", "l"],
    "a": ["q", "s", "z"],
    "s": ["a", "w", "d", "x"],
    "d": ["s", "e", "f", "c"],
    "f": ["d", "r", "g", "v"],
    "g": ["f", "t", "h", "b"],
    "h": ["g", "y", "j", "n"],
    "j": ["h", "u", "k", "m"],
    "k": ["j", "i", "l"],
    "l": ["k", "o", "p"],
    "z": ["a", "x"],
    "x": ["z", "s", "c"],
    "c": ["x", "d", "v"],
    "v": ["c", "f", "b"],
    "b": ["v", "g", "n"],
    "n": ["b", "h", "m"],
    "m": ["n", "j"],
}


def _substitute_nearby_key(word: str) -> str:
    """Bug 4b — Sub-C (keyboard): replace a character with a keyboard-adjacent one.

    Implements keyboard-based typo substitution as explicitly described in the paper:
    'replacing m with n' (adjacent keys on QWERTY layout).
    """
    chars = list(word)
    # Find replaceable positions (characters with keyboard neighbors)
    positions = [i for i, c in enumerate(chars) if c.lower() in KEYBOARD_ADJACENT]
    if not positions:
        return _swap_adjacent(word)  # fallback
    i = random.choice(positions)
    candidates = KEYBOARD_ADJACENT[chars[i].lower()]
    # Preserve case
    replacement = random.choice(candidates)
    if chars[i].isupper():
        replacement = replacement.upper()
    chars[i] = replacement
    return "".join(chars)


def _substitute_word_embedding(word: str, max_candidates: int = 5) -> list[str]:
    """Bug 5 — Sub-W: replace word with nearest embedding neighbours (paper: GloVe, k=5).

    Uses embedding-based nearest neighbours as the primary method, matching the
    paper specification.  Falls back to WordNet+MLM when embeddings unavailable.

    Returns a list of synonym candidates.
    """
    from utils.text_word_substitution import get_embedding_neighbours

    # Primary: embedding neighbours (GloVe / counter-fitted via gensim)
    candidates = get_embedding_neighbours(word, top_k=max_candidates)

    if candidates:
        return candidates[:max_candidates]

    # Fallback: WordNet + MLM (only if embeddings produced nothing)
    from utils.text_word_substitution import get_wordnet_synonyms, get_mlm_substitutions
    from utils.text_utils import simple_pos_tag

    pos = simple_pos_tag(word)
    candidates = get_wordnet_synonyms(word, pos=pos, max_candidates=max_candidates)

    if not candidates:
        text = f"The {word} is important."
        candidates = get_mlm_substitutions(text, position=1, top_k=max_candidates)

    return candidates[:max_candidates]


# Character-level perturbations: Insert, Delete, Swap, Sub-C (homoglyphs + keyboard)
# Paper defines Sub-C as ONE bug with two sub-variants:
#   - visually similar characters (homoglyphs)
#   - keyboard-adjacent characters (typos)
# Both variants are tried and the best candidate is selected.
_CHARACTER_PERTURBATION_FNS = [
    _insert_space, _delete_char, _swap_adjacent, _substitute_homoglyph, _substitute_nearby_key,
]


def run_textbugger(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_perturbations: int = 5,
    mode: str = "black-box",
    strategy: str = "combined",
    similarity_threshold: float = 0.8,
    max_queries: int = 5000,
    seed: "int | None" = None,
) -> str:
    """TextBugger attack — Li et al., 2018 (arXiv:1812.05271).

    Implements the complete TextBugger algorithm with all 5 perturbations:
      Character-level (4 bugs, Sub-C has two sub-variants):
        1. Insert: insert a space into the word
        2. Delete: delete a random character (not first/last)
        3. Swap: swap two adjacent characters (not first/last)
        4. Sub-C: substitute with visually similar character (homoglyphs)
                OR substitute with keyboard-adjacent character (keyboard typos)
      Word-level (1 bug):
        5. Sub-W: substitute with nearest embedding neighbour (paper: GloVe, k=5)

    Supports both white-box (gradient-based) and black-box (query-based) modes.
    Black-box mode includes sentence-level importance ranking for multi-sentence
    inputs, as described in Section IV of the paper.

    Args:
        model_wrapper: wrapped model with .predict() method
        tokenizer: HuggingFace tokenizer
        text: input text to attack
        target_label: target class name (None = untargeted)
        max_perturbations: maximum words to perturb
        mode: "white-box" (gradient-based) or "black-box" (query-based) importance scoring
        strategy: "bug" (char-level: Insert/Delete/Swap/Sub-C),
                  "word" (word-level: Sub-W only),
                  "combined" (all 5 perturbations)
        similarity_threshold: minimum semantic similarity (paper default: 0.8)
        max_queries: maximum model queries allowed
        seed: random seed for reproducibility (None = non-deterministic)

    Returns:
        adversarial text (str).
    """
    from utils.text_word_importance import delete_one_importance, sentence_importance
    from utils.text_utils import get_words_and_spans, replace_word_at, is_stopword

    # Reproducibility
    if seed is not None:
        random.seed(seed)

    logger.info("TextBugger: starting (mode=%s, strategy=%s, max_pert=%d)",
                mode, strategy, max_perturbations)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    # ── Word importance scoring ──────────────────────────────────────────
    if mode == "white-box":
        from utils.text_word_importance import gradient_importance
        importance = gradient_importance(model_wrapper.model, tokenizer, text)
        logger.info("TextBugger: using white-box gradient importance")
    else:
        # Black-box: sentence importance → word importance (paper Section IV-B)
        sent_scores = sentence_importance(model_wrapper, text)
        if len(sent_scores) > 1:
            # Multi-sentence: focus on the most important sentence
            top_sent_idx, top_sent_text, _ = sent_scores[0]
            logger.info("TextBugger: black-box sentence importance → sentence %d", top_sent_idx)
            importance = delete_one_importance(model_wrapper, top_sent_text)
            # Offset word indices to match positions in the full text
            sent_words = get_words_and_spans(top_sent_text)
            full_words = get_words_and_spans(text)
            # Find where the sentence's words start in the full text
            offset = 0
            for i, (fw, fs, fe) in enumerate(full_words):
                if sent_words and fw == sent_words[0][0] and fs >= full_words[0][1]:
                    # Check if this is the right sentence by matching spans
                    if text[fs:fe] == top_sent_text[sent_words[0][1]:sent_words[0][2]]:
                        offset = i
                        break
            importance = [(idx + offset, score) for idx, score in importance]
        else:
            importance = delete_one_importance(model_wrapper, text)
        logger.info("TextBugger: using black-box delete-one importance")

    orig_label, orig_conf, _ = model_wrapper.predict(text)

    current_text = text
    perturbations_made = 0
    perturbed_indices: set[int] = set()  # RepeatModification guard

    for word_idx, score in importance:
        if perturbations_made >= max_perturbations:
            break

        # RepeatModification: skip already-perturbed words
        if word_idx in perturbed_indices:
            continue

        if word_idx >= len(words_spans):
            continue

        word = words_spans[word_idx][0]
        if is_stopword(word) or len(word) <= 2:
            continue

        # Try each perturbation type, pick the one with greatest impact
        best_text = None
        best_impact = -1.0

        # Character-level bugs: Insert, Delete, Swap, Sub-C (homoglyphs + keyboard)
        if strategy in ["bug", "combined"]:
            for perturb_fn in _CHARACTER_PERTURBATION_FNS:
                perturbed_word = perturb_fn(word)
                current_spans = get_words_and_spans(current_text)
                if word_idx >= len(current_spans):
                    break
                # Guard against stale indices after word-boundary changes
                if current_spans[word_idx][0] != words_spans[word_idx][0]:
                    continue

                candidate = replace_word_at(current_text, word_idx, perturbed_word)

                # Semantic similarity constraint
                from utils.text_constraints import compute_semantic_similarity
                if compute_semantic_similarity(text, candidate) < similarity_threshold:
                    continue

                # Query budget constraint
                if model_wrapper.query_count >= max_queries:
                    logger.warning("TextBugger: query budget exhausted (%d queries)", max_queries)
                    return current_text

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

        # Word-level: Sub-W — embedding neighbours (paper: GloVe, k=5)
        if strategy in ["word", "combined"]:
            synonyms = _substitute_word_embedding(word, max_candidates=5)

            for synonym in synonyms:
                current_spans = get_words_and_spans(current_text)
                if word_idx >= len(current_spans):
                    break
                # Guard against stale indices
                if current_spans[word_idx][0] != words_spans[word_idx][0]:
                    continue

                candidate = replace_word_at(current_text, word_idx, synonym)

                # Semantic similarity constraint
                from utils.text_constraints import compute_semantic_similarity
                if compute_semantic_similarity(text, candidate) < similarity_threshold:
                    continue

                # Query budget constraint
                if model_wrapper.query_count >= max_queries:
                    logger.warning("TextBugger: query budget exhausted (%d queries)", max_queries)
                    return current_text

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
            perturbed_indices.add(word_idx)

    logger.info("TextBugger: finished (%d perturbations)", perturbations_made)
    return current_text
