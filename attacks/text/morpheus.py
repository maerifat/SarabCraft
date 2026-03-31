"""
MorpheuS Attack — Tan et al., 2020 (arXiv:2009.11112)

It's Morphin' Time! Combating Linguistic Discrimination with
Inflectional Perturbations.

Inflectional morphology attack: perturbs words by changing their
grammatical inflection (verb tense, noun number, adjective degree)
rather than substituting with synonyms or introducing typos.

Key distinction from other attacks:
  - Character attacks (DeepWordBug, Pruthi): random noise, misspellings
  - Word attacks (TextFooler, BERT-Attack): semantic substitution
  - MorpheuS: linguistically principled form changes that preserve
    the stem but alter grammatical properties

Examples:
  - "walked" → "walks" / "walking" / "walk" (verb tense)
  - "cats" → "cat" (noun number)
  - "better" → "good" / "best" (adjective degree)
  - "happily" → "happier" / "happiest" (adverb form)

Uses NLTK lemmatization + rule-based inflection generation.
"""

import logging
import random

logger = logging.getLogger("textattack.attacks.morpheus")

# ── Inflection rules by POS category ────────────────────────────────────

_VERB_INFLECTIONS = {
    "VB":  lambda w: _verb_forms(w),     # base → all forms
    "VBD": lambda w: _verb_forms(w),     # past → all forms
    "VBG": lambda w: _verb_forms(w),     # gerund → all forms
    "VBN": lambda w: _verb_forms(w),     # past participle → all forms
    "VBP": lambda w: _verb_forms(w),     # present non-3sg → all forms
    "VBZ": lambda w: _verb_forms(w),     # present 3sg → all forms
    "verb": lambda w: _verb_forms(w),    # heuristic fallback tag
}

_NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS", "noun"}
_ADJ_TAGS = {"JJ", "JJR", "JJS", "adj"}
_ADV_TAGS = {"RB", "RBR", "RBS", "adv"}


def _verb_forms(word: str) -> list[str]:
    """Generate verb inflection candidates."""
    w = word.lower()
    forms = set()

    # Try to get lemma via NLTK
    try:
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        lemma = wnl.lemmatize(w, pos='v')
    except (ImportError, LookupError):
        lemma = w

    # Generate inflected forms from lemma
    forms.add(lemma)                           # base: "walk"

    # -s (3rd person singular present)
    if lemma.endswith(("s", "sh", "ch", "x", "z")):
        forms.add(lemma + "es")
    elif lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        forms.add(lemma[:-1] + "ies")
    else:
        forms.add(lemma + "s")

    # -ing (present participle)
    if lemma.endswith("e") and not lemma.endswith("ee"):
        forms.add(lemma[:-1] + "ing")
    elif lemma.endswith("ie"):
        forms.add(lemma[:-2] + "ying")
    elif (len(lemma) >= 3 and lemma[-1] not in "aeiouwxy"
          and lemma[-2] in "aeiou" and lemma[-3] not in "aeiou"):
        forms.add(lemma + lemma[-1] + "ing")
    else:
        forms.add(lemma + "ing")

    # -ed (past tense / past participle)
    if lemma.endswith("e"):
        forms.add(lemma + "d")
    elif lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        forms.add(lemma[:-1] + "ied")
    elif (len(lemma) >= 3 and lemma[-1] not in "aeiouwxy"
          and lemma[-2] in "aeiou" and lemma[-3] not in "aeiou"):
        forms.add(lemma + lemma[-1] + "ed")
    else:
        forms.add(lemma + "ed")

    forms.discard(w)
    return [f for f in forms if f]


def _noun_forms(word: str) -> list[str]:
    """Generate noun number variants (singular ↔ plural)."""
    w = word.lower()
    forms = set()

    try:
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        lemma = wnl.lemmatize(w, pos='n')
    except (ImportError, LookupError):
        lemma = w

    forms.add(lemma)  # singular

    # Pluralize
    if lemma.endswith(("s", "sh", "ch", "x", "z")):
        forms.add(lemma + "es")
    elif lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        forms.add(lemma[:-1] + "ies")
    elif lemma.endswith("f"):
        forms.add(lemma[:-1] + "ves")
    elif lemma.endswith("fe"):
        forms.add(lemma[:-2] + "ves")
    else:
        forms.add(lemma + "s")

    forms.discard(w)
    return [f for f in forms if f]


def _adj_forms(word: str) -> list[str]:
    """Generate adjective degree variants (positive/comparative/superlative)."""
    w = word.lower()
    forms = set()

    try:
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        lemma = wnl.lemmatize(w, pos='a')
    except (ImportError, LookupError):
        lemma = w

    forms.add(lemma)  # positive

    if len(lemma) <= 7:  # short adjectives use -er/-est
        if lemma.endswith("e"):
            forms.add(lemma + "r")
            forms.add(lemma + "st")
        elif lemma.endswith("y") and len(lemma) > 1:
            forms.add(lemma[:-1] + "ier")
            forms.add(lemma[:-1] + "iest")
        else:
            forms.add(lemma + "er")
            forms.add(lemma + "est")

    # Long adjectives use more/most (not a word form change, skip)
    forms.discard(w)
    return [f for f in forms if f]


def _adv_forms(word: str) -> list[str]:
    """Generate adverb variants."""
    w = word.lower()
    forms = set()

    # -ly adverbs: try base adjective
    if w.endswith("ly") and len(w) > 3:
        base = w[:-2]
        if base.endswith("i"):
            base = base[:-1] + "y"
        forms.add(base)
        forms.update(_adj_forms(base))

    forms.discard(w)
    return [f for f in forms if f]


def run_morpheus(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_perturbation_ratio: float = 0.3,
    similarity_threshold: float = 0.8,
) -> str:
    """MorpheuS attack: inflectional morphology perturbations.

    Changes grammatical inflection of words (verb tense, noun number,
    adjective degree) to test whether models are sensitive to
    linguistically valid form variations.

    Args:
        model_wrapper: wrapped model with .predict() interface.
        tokenizer: HuggingFace tokenizer (unused, API compat).
        text: input text to attack.
        target_label: target class (None = untargeted).
        max_perturbation_ratio: max fraction of words to inflect.
        similarity_threshold: min semantic similarity for result.

    Returns: adversarial text (str).
    """
    from utils.text_utils import get_words_and_spans, replace_word_at, pos_tag_words, is_stopword
    from utils.text_constraints import compute_semantic_similarity

    logger.info(
        "MorpheuS: starting (max_pert=%.2f, sim=%.2f)",
        max_perturbation_ratio, similarity_threshold,
    )

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    words = [w for w, _, _ in words_spans]
    pos_tags = pos_tag_words(words)
    max_perturbs = max(1, int(len(words) * max_perturbation_ratio))

    orig_label, orig_conf, _ = model_wrapper.predict(text)

    # Build candidates for each position
    position_candidates: list[tuple[int, list[str]]] = []

    for i, (word, pos) in enumerate(zip(words, pos_tags)):
        if is_stopword(word) or len(word) <= 2:
            continue

        # Strip trailing punctuation for inflection generation
        clean = word.rstrip(".,!?;:'\"()[]{}").lower()
        if len(clean) <= 2:
            continue

        inflections = []
        if pos in _VERB_INFLECTIONS:
            inflections = _verb_forms(clean)
        elif pos in _NOUN_TAGS:
            inflections = _noun_forms(clean)
        elif pos in _ADJ_TAGS:
            inflections = _adj_forms(clean)
        elif pos in _ADV_TAGS:
            inflections = _adv_forms(clean)
        elif pos == "other":
            # Heuristic fallback missed POS — try all types and take any hits
            for gen in (_verb_forms, _noun_forms, _adj_forms):
                inflections = gen(clean)
                if inflections:
                    break

        # Re-attach any trailing punctuation that was on the original word
        suffix = word[len(word.rstrip(".,!?;:'\"()[]{}")):]
        if suffix and inflections:
            inflections = [f + suffix for f in inflections]

        if inflections:
            position_candidates.append((i, inflections))

    if not position_candidates:
        logger.info("MorpheuS: no inflectable words found")
        return text

    # Score each position by importance (delete-one confidence drop)
    from utils.text_word_importance import delete_one_importance
    importance = delete_one_importance(model_wrapper, text, orig_label)
    importance_map = {idx: score for idx, score in importance}

    # Sort positions by importance (most important first)
    position_candidates.sort(
        key=lambda x: importance_map.get(x[0], 0.0), reverse=True,
    )

    current_text = text
    perturbations_made = 0
    modified_indices: set[int] = set()

    for word_idx, inflections in position_candidates:
        if perturbations_made >= max_perturbs:
            break

        if word_idx in modified_indices:
            continue

        best_text = None
        best_impact = -1.0

        for inflected in inflections:
            # Preserve original capitalization
            orig_word = words[word_idx]
            if orig_word[0].isupper():
                inflected = inflected.capitalize()
            if orig_word.isupper():
                inflected = inflected.upper()

            candidate_text = replace_word_at(current_text, word_idx, inflected)

            sim = compute_semantic_similarity(current_text, candidate_text)
            if sim < similarity_threshold:
                continue

            label, conf, _ = model_wrapper.predict(candidate_text)

            if target_label is not None:
                if label.lower() == target_label.lower():
                    logger.info("MorpheuS: success at perturbation %d (%s→%s)",
                                perturbations_made + 1, orig_word, inflected)
                    return candidate_text
            else:
                if label != orig_label:
                    logger.info("MorpheuS: success at perturbation %d (%s→%s)",
                                perturbations_made + 1, orig_word, inflected)
                    return candidate_text

            impact = orig_conf - conf
            if impact > best_impact:
                best_impact = impact
                best_text = candidate_text

        if best_text is not None:
            current_text = best_text
            perturbations_made += 1
            modified_indices.add(word_idx)

    logger.info("MorpheuS: finished (%d perturbations)", perturbations_made)
    return current_text
