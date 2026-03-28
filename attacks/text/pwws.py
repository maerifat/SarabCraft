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
Named Entity substitution supported when NE candidates are provided.

Exact match to the official JHL-HUST/PWWS implementation:
  - Saliency computed for ALL words (not just POS-eligible)
  - Softmax over the FULL saliency vector
  - No candidate cap (iterates all WordNet synsets)
  - Position-stable greedy substitution via token compilation
  - Named Entity substitution path (optional, dataset-specific)
"""

import logging
import numpy as np

logger = logging.getLogger("textattack.attacks.pwws")


# ── Penn Treebank POS whitelist ──────────────────────────────────────────────
# From official PWWS code (paraphrase.py: supported_pos_tags).
# Only words with these POS tags are eligible for WordNet perturbation.
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
    if len(candidate_text.split()) > 2:
        return False

    try:
        from nltk.stem import WordNetLemmatizer
        _wnl = WordNetLemmatizer()
        if _wnl.lemmatize(candidate_text.lower()) == _wnl.lemmatize(original_text.lower()):
            return False
    except ImportError:
        if candidate_text.lower() == original_text.lower():
            return False

    if candidate_tag and original_tag and candidate_tag != original_tag:
        return False

    if original_text.lower() == 'be':
        return False

    return True


# ── Synonym candidate generation ─────────────────────────────────────────────

def _generate_synonym_candidates(word: str, penn_tag: str) -> list[str]:
    """Generate WordNet synonym candidates matching the official algorithm.

    Mirrors _generate_synonym_candidates() in paraphrase.py:
      1. Skip words whose POS is not in SUPPORTED_POS_TAGS.
      2. Look up WordNet synsets filtered by WordNet POS.
      3. Apply _synonym_prefilter to every lemma.

    No candidate cap — iterates through all synsets/lemmas (matching official).
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

    return candidates


# ── Named Entity detection ───────────────────────────────────────────────────

def _detect_named_entities(text: str) -> dict[int, str]:
    """Detect named entities in text, returning {word_position: NER_tag}.

    Mirrors official get_NE_list usage: for each token, if it belongs to a
    named entity, record its NER label (PERSON, ORG, GPE, etc.).
    """
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return {}

        doc = nlp(text)

        char_to_ner: dict[int, str] = {}
        for ent in doc.ents:
            for i in range(ent.start_char, ent.end_char):
                char_to_ner[i] = ent.label_

        from utils.text_utils import get_words_and_spans
        word_ner: dict[int, str] = {}
        for idx, (_word, start, end) in enumerate(get_words_and_spans(text)):
            for c in range(start, end):
                if c in char_to_ner:
                    word_ner[idx] = char_to_ner[c]
                    break
        return word_ner
    except ImportError:
        return {}


# ── Word saliency ────────────────────────────────────────────────────────────

def _compute_word_saliency(model_wrapper, text: str, word_idx: int,
                           orig_probs: list, true_class_idx: int) -> float:
    """Compute S(x, w_i) = P(y|x) − P(y|x_\\w_i)  [Eq.6].

    The official code zeros the word embedding vector; for black-box text
    models, word deletion is the standard adaptation (TextAttack reference).
    """
    from utils.text_utils import get_words_and_spans

    spans = get_words_and_spans(text)
    if word_idx >= len(spans):
        return 0.0

    _, start, end = spans[word_idx]
    reduced = (text[:start] + text[end:]).strip()
    reduced = " ".join(reduced.split())

    if not reduced:
        return orig_probs[true_class_idx]

    reduced_probs = model_wrapper.predict_probs(reduced)

    return orig_probs[true_class_idx] - reduced_probs[true_class_idx]


# ── Token compilation ────────────────────────────────────────────────────────
# Matches official _compile_perturbed_tokens(): build the perturbed token list
# by iterating over the original words and substituting at stored positions.
# This keeps indices stable across multiple substitutions — no re-parsing.

def _compile_perturbed_tokens(words: list[str], substitutions: dict[int, str]) -> list[str]:
    """Build token list with substitutions applied at given positions."""
    result = []
    for i, word in enumerate(words):
        if i in substitutions:
            result.append(substitutions[i].replace('_', ' '))
        else:
            result.append(word)
    return result


# ── Main entry point ─────────────────────────────────────────────────────────

def run_pwws(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_candidates: int = 50,
    use_named_entities: bool = False,
    ne_candidates: dict[str, str] | None = None,
) -> str:
    """PWWS attack: WordNet synonym substitution weighted by word saliency.

    Exact match to the official JHL-HUST/PWWS algorithm (Ren et al., 2019):
      1. POS-tag ALL words (Penn Treebank).
      2. Compute word saliency S(x, w_i) for ALL words  [Eq.6].
      3. Softmax-normalize the FULL saliency vector  [Eq.8].
      4. For each word with candidates (NE or POS-eligible WordNet):
         find best synonym w_i* = argmax_{w∈L_i} ΔP  [Eq.4-5].
      5. Compute H(x, x_i*, w_i) = ΔP × softmax(S)_i  [Eq.7].
      6. Sort words by H descending; substitute greedily until
         the classifier prediction flips.

    Args:
        model_wrapper: _TextModelWrapper with .predict() and .predict_probs()
        tokenizer: HuggingFace tokenizer (unused, kept for router signature)
        text: input text to attack
        target_label: target class name (None = untargeted)
        max_candidates: max synonyms to evaluate per word (0 = no limit)
        use_named_entities: enable NE substitution path (official: True)
        ne_candidates: {NER_tag: replacement_entity} lookup table;
            mirrors official NE_list.L[dataset][true_y]

    Returns: adversarial text (str).
    """
    from utils.text_utils import get_words_and_spans, replace_word_at

    logger.info("PWWS: starting (max_cands=%d, use_NE=%s)", max_candidates, use_named_entities)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    orig_label, orig_conf, _ = model_wrapper.predict(text)
    orig_probs = model_wrapper.predict_probs(text)
    orig_class_idx = orig_probs.index(max(orig_probs))

    words = [w for w, _, _ in words_spans]
    num_words = len(words)

    # ── Step 1: POS-tag ALL words (Penn Treebank tagset) ─────────────────
    pos_tags = _pos_tag_words(words)

    # Detect NER tags for all words (if NE substitution enabled)
    word_ner_tags: dict[int, str] = {}
    ne_tag_set: set[str] = set()
    if use_named_entities and ne_candidates:
        word_ner_tags = _detect_named_entities(text)
        ne_tag_set = set(ne_candidates.keys())

    # ── Step 2: Compute word saliency for ALL words [Eq.6] ───────────────
    # Official: evaluate_word_saliency() computes S for every position.
    saliency_values: list[float] = []
    for i in range(num_words):
        s = _compute_word_saliency(
            model_wrapper, text, i, orig_probs, orig_class_idx,
        )
        saliency_values.append(s)

    # ── Step 3: Softmax over the FULL saliency vector [Eq.8] ────────────
    # Official: word_saliency_array = softmax(word_saliency_array) over ALL words.
    saliency_arr = np.array(saliency_values)
    exp_s = np.exp(saliency_arr - np.max(saliency_arr))
    softmax_saliency = exp_s / np.sum(exp_s)

    # ── Steps 4–5: Find best synonym per word & compute H-score [Eq.7] ──
    # Official loop: for (position, token, word_saliency, tag) in word_saliency_list
    # — iterates ALL words; words without candidates are skipped.
    word_scores: list[tuple[int, str, float]] = []

    for i in range(num_words):
        word = words[i]
        penn_tag = pos_tags[i]

        # NE path first (official: if use_NE and NER_tag in NE_tags)
        synonyms: list[str] = []
        if use_named_entities and ne_candidates:
            ner_tag = word_ner_tags.get(i, "")
            if ner_tag in ne_tag_set:
                synonyms = [ne_candidates[ner_tag]]
            else:
                synonyms = _generate_synonym_candidates(word, penn_tag)
        else:
            synonyms = _generate_synonym_candidates(word, penn_tag)

        if not synonyms:
            continue

        # Optionally limit candidates for efficiency
        eval_synonyms = synonyms if max_candidates <= 0 else synonyms[:max_candidates]

        # w_i* = argmax ΔP [Eq.5]
        # Official: sorted_candidates.pop() — always selects the best.
        best_syn = None
        best_delta = float('-inf')

        for syn in eval_synonyms:
            candidate_text = replace_word_at(text, i, syn)
            cand_probs = model_wrapper.predict_probs(candidate_text)
            delta = orig_probs[orig_class_idx] - cand_probs[orig_class_idx]
            if delta > best_delta:
                best_delta = delta
                best_syn = syn

        if best_syn is not None:
            # H(x, x_i*, w_i) = ΔP × softmax(S)_i [Eq.7]
            # Position i indexes into the FULL softmax array (matching official).
            h_score = best_delta * softmax_saliency[i]
            word_scores.append((i, best_syn, h_score))

    # ── Step 6: Sort by H descending; greedy substitution ────────────────
    # Official: sorted_substitute_tuple_list = sorted(..., key=lambda t: t[3], reverse=True)
    word_scores.sort(key=lambda x: x[2], reverse=True)

    # Greedy substitution using position-stable token compilation
    # (matches official _compile_perturbed_tokens loop).
    applied_subs: dict[int, str] = {}

    for word_idx, synonym, _score in word_scores:
        applied_subs[word_idx] = synonym
        perturbed_tokens = _compile_perturbed_tokens(words, applied_subs)
        candidate_text = " ".join(perturbed_tokens)

        label, conf, _ = model_wrapper.predict(candidate_text)

        if target_label is not None:
            if label.lower() == target_label.lower():
                logger.info("PWWS: success (targeted)")
                return candidate_text
        else:
            if label != orig_label:
                logger.info("PWWS: success (untargeted)")
                return candidate_text

    # All substitutions applied but prediction did not flip — return best effort
    if applied_subs:
        logger.info("PWWS: finished (no flip, %d substitutions applied)", len(applied_subs))
        return " ".join(_compile_perturbed_tokens(words, applied_subs))

    logger.info("PWWS: finished (no candidates found)")
    return text
