"""
PWWS Attack — Ren et al., 2019 (arXiv:1907.06292)

Probability Weighted Word Saliency: scores each word by
  H(x, x_i*, w_i) = ΔP(w_i, w_i*) × softmax(S(x, w_i))    [Eq.7]
where:
  S(x, w_i) = P(y|x) − P(y|x_\\w_i)                         [Eq.6]
  softmax(S)_i = exp(S_i) / Σ exp(S_j)                       [Eq.8]
  ΔP(w_i, w*) = P(y|x) − P(y|x[w_i→w*])                     [Eq.4]
  w_i* = argmax_{w∈L_i} ΔP(w_i, w)                           [Eq.5]

Uses WordNet synonyms filtered by Penn Treebank POS tag for substitution.
"""

import logging
import numpy as np

logger = logging.getLogger("textattack.attacks.pwws")


# ── Penn Treebank POS whitelist ──────────────────────────────────────────────
# From official PWWS code (paraphrase.py: supported_pos_tags).
# Only words with these POS tags are eligible for perturbation.
SUPPORTED_POS_TAGS = frozenset({
    'CC',    # Coordinating conjunction
    'JJ',    # Adjective
    'JJR',   # Adjective, comparative
    'JJS',   # Adjective, superlative
    'NN',    # Noun, singular or mass
    'NNS',   # Noun, plural
    'NNP',   # Proper noun, singular
    'NNPS',  # Proper noun, plural
    'RB',    # Adverb
    'RBR',   # Adverb, comparative
    'RBS',   # Adverb, superlative
    'VB',    # Verb, base form
    'VBD',   # Verb, past tense
    'VBG',   # Verb, gerund or present participle
    'VBN',   # Verb, past participle
    'VBP',   # Verb, non-3rd person singular present
    'VBZ',   # Verb, 3rd person singular present
})


# ── POS helpers ──────────────────────────────────────────────────────────────

def _pos_tag_words(words: list[str]) -> list[str]:
    """POS-tag a word list using NLTK (Penn Treebank tagset).

    Falls back to 'NN' for every word when NLTK is unavailable so that
    synonym generation can still proceed (albeit with reduced accuracy).
    """
    try:
        import nltk
        tagged = nltk.pos_tag(words)
        return [tag for _, tag in tagged]
    except (ImportError, LookupError):
        logger.warning("NLTK pos_tag unavailable; defaulting all words to NN")
        return ['NN'] * len(words)


def _get_wordnet_pos(penn_tag: str):
    """Map a Penn Treebank POS tag to a WordNet POS constant."""
    from nltk.corpus import wordnet as wn
    prefix = penn_tag[0].lower()
    if prefix == 'j':
        return wn.ADJ
    elif prefix == 'v':
        return wn.VERB
    elif prefix == 'n':
        return wn.NOUN
    elif prefix == 'r':
        return wn.ADV
    return None


# ── Synonym pre-filter ───────────────────────────────────────────────────────
# Matches official paraphrase.py _synonym_prefilter_fn():
#   reject multi-word (>2 tokens), same lemma, POS mismatch, word "be".

def _synonym_prefilter(candidate_text: str, original_text: str,
                       candidate_tag: str, original_tag: str) -> bool:
    """Return True if *candidate_text* passes all official pre-filter checks."""
    # 1. Reject phrases with more than 2 tokens
    if len(candidate_text.split()) > 2:
        return False

    # 2. Reject same lemma
    try:
        from nltk.stem import WordNetLemmatizer
        _wnl = WordNetLemmatizer()
        if _wnl.lemmatize(candidate_text.lower()) == _wnl.lemmatize(original_text.lower()):
            return False
    except ImportError:
        if candidate_text.lower() == original_text.lower():
            return False

    # 3. Reject POS-tag mismatch (fine-grained Penn Treebank)
    if candidate_tag and original_tag and candidate_tag != original_tag:
        return False

    # 4. Reject if the original word is "be"
    if original_text.lower() == 'be':
        return False

    return True


# ── Synonym candidate generation ─────────────────────────────────────────────

def _generate_synonym_candidates(word: str, penn_tag: str,
                                 max_candidates: int = 50) -> list[str]:
    """Generate WordNet synonym candidates matching the official algorithm.

    Mirrors _generate_synonym_candidates() in paraphrase.py:
      1. Skip words whose POS is not in SUPPORTED_POS_TAGS.
      2. Look up WordNet synsets filtered by WordNet POS.
      3. Apply _synonym_prefilter to every lemma.
    """
    if penn_tag not in SUPPORTED_POS_TAGS:
        return []

    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        logger.debug("NLTK WordNet unavailable")
        return []

    wn_pos = _get_wordnet_pos(penn_tag)
    synsets = wn.synsets(word.lower(), pos=wn_pos) if wn_pos else wn.synsets(word.lower())

    candidates: list[str] = []
    seen: set[str] = set()

    for synset in synsets:
        for lemma in synset.lemmas():
            name = lemma.name().replace('_', ' ')
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)

            # POS-tag the candidate in isolation (mirrors official:
            # spacy_synonym = nlp(wordnet_synonym.name())[0])
            cand_tags = _pos_tag_words([name])
            cand_tag = cand_tags[0] if cand_tags else ''

            if _synonym_prefilter(name, word, cand_tag, penn_tag):
                candidates.append(name)
                if len(candidates) >= max_candidates:
                    return candidates

    return candidates


# ── Word saliency ────────────────────────────────────────────────────────────

def _compute_word_saliency(model_wrapper, text: str, word_idx: int,
                           orig_probs: list, true_class_idx: int) -> float:
    """Compute S(x, w_i) = P(y|x) − P(y|x_\\w_i)  [Eq.6].

    Measures the drop in *true-class* probability when word w_i is removed.
    The official code zeros the word embedding vector; for transformer models
    in a black-box text API, word deletion is the standard adaptation (also
    used by the TextAttack reference).
    """
    from utils.text_utils import get_words_and_spans

    spans = get_words_and_spans(text)
    if word_idx >= len(spans):
        return 0.0

    _, start, end = spans[word_idx]
    reduced = (text[:start] + text[end:]).strip()
    reduced = " ".join(reduced.split())

    if not reduced:
        # Single-word text: full probability is attributable to this word.
        return orig_probs[true_class_idx]

    reduced_probs = model_wrapper.predict_probs(reduced)

    # Eq.(6): change in true-class probability only.
    return orig_probs[true_class_idx] - reduced_probs[true_class_idx]


# ── Main entry point ─────────────────────────────────────────────────────────

def run_pwws(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_candidates: int = 10,
) -> str:
    """PWWS attack: WordNet synonym substitution weighted by word saliency.

    Full algorithm (Ren et al., 2019):
      1. POS-tag all words (Penn Treebank); filter by SUPPORTED_POS_TAGS.
      2. Compute word saliency S(x, w_i) for eligible words  [Eq.6].
      3. Softmax-normalize the saliency vector  [Eq.8].
      4. For each eligible word, find best synonym
         w_i* = argmax_{w∈L_i} ΔP  [Eq.4-5].
      5. Compute H(x, x_i*, w_i) = ΔP × softmax(S)_i  [Eq.7].
      6. Sort words by H descending; substitute greedily until
         the classifier prediction flips.

    Returns: adversarial text (str).
    """
    from utils.text_utils import get_words_and_spans, replace_word_at

    logger.info("PWWS: starting (max_cands=%d)", max_candidates)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    orig_label, orig_conf, _ = model_wrapper.predict(text)
    orig_probs = model_wrapper.predict_probs(text)
    orig_class_idx = orig_probs.index(max(orig_probs))

    words = [w for w, _, _ in words_spans]

    # ── Step 1: POS-tag all words (Penn Treebank tagset) ─────────────────
    pos_tags = _pos_tag_words(words)

    # ── Step 2: Compute word saliency for all eligible words [Eq.6] ──────
    eligible: list[tuple[int, float, str]] = []   # (word_idx, saliency, tag)
    for i in range(len(words)):
        if pos_tags[i] not in SUPPORTED_POS_TAGS:
            continue
        saliency = _compute_word_saliency(
            model_wrapper, text, i, orig_probs, orig_class_idx,
        )
        eligible.append((i, saliency, pos_tags[i]))

    if not eligible:
        return text

    # ── Step 3: Softmax normalization of saliency [Eq.8] ─────────────────
    saliency_arr = np.array([s for _, s, _ in eligible])
    # Numerical-stability shift (does not change softmax output).
    exp_s = np.exp(saliency_arr - np.max(saliency_arr))
    softmax_saliency = exp_s / np.sum(exp_s)

    # ── Steps 4–5: Find best synonym per word & compute H-score [Eq.7] ──
    word_scores: list[tuple[int, str, float]] = []
    for elig_idx, (word_idx, _sal, penn_tag) in enumerate(eligible):
        word = words[word_idx]
        synonyms = _generate_synonym_candidates(
            word, penn_tag, max_candidates=max_candidates,
        )
        if not synonyms:
            continue

        # w_i* = argmax ΔP  [Eq.5].
        # Start at -inf so that a candidate is always selected (matches
        # official code which picks sorted_candidates.pop() unconditionally).
        best_syn = None
        best_delta = float('-inf')

        for syn in synonyms:
            candidate = replace_word_at(text, word_idx, syn)
            cand_probs = model_wrapper.predict_probs(candidate)
            delta = orig_probs[orig_class_idx] - cand_probs[orig_class_idx]
            if delta > best_delta:
                best_delta = delta
                best_syn = syn

        if best_syn is not None:
            # H(x, x_i*, w_i) = ΔP × softmax(S)_i  [Eq.7]
            h_score = best_delta * softmax_saliency[elig_idx]
            word_scores.append((word_idx, best_syn, h_score))

    # ── Step 6: Sort by H descending; greedy substitution ────────────────
    word_scores.sort(key=lambda x: x[2], reverse=True)

    current_text = text
    for word_idx, synonym, _score in word_scores:
        # Re-parse to get updated spans after prior substitutions.
        current_spans = get_words_and_spans(current_text)
        if word_idx >= len(current_spans):
            continue
        # Guard against stale indices after word-boundary changes.
        if current_spans[word_idx][0] != words_spans[word_idx][0]:
            continue

        candidate = replace_word_at(current_text, word_idx, synonym)
        label, conf, _ = model_wrapper.predict(candidate)

        if target_label is not None:
            if label.lower() == target_label.lower():
                logger.info("PWWS: success")
                return candidate
        else:
            if label != orig_label:
                logger.info("PWWS: success")
                return candidate

        current_text = candidate

    logger.info("PWWS: finished")
    return current_text
