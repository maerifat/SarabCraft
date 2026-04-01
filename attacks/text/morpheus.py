"""
MorpheuS Attack — Tan et al., 2020 (ACL, arXiv:2005.04364)

It's Morphin' Time! Combating Linguistic Discrimination with
Inflectional Perturbations.

Exact reimplementation of the official Salesforce algorithm
(github.com/salesforce/morpheus), adapted for text classification.

Algorithm (Algorithm 1 — official pseudocode):
  1. Tokenize and POS-tag (NLTK, universal tagset)
  2. Generate inflection candidates per position (lemminflect)
     - Only NOUN, VERB, ADJ are inflectable
     - POS-constrained inflections (constrain_pos=True)
  3. FORWARD PASS: iterate left-to-right through positions
     - At each position, try all inflections
     - Pick the inflection that maximally reduces evaluation metric
     - Apply it (greedy commitment)
  4. BACKWARD PASS: same procedure, right-to-left
  5. Return the better of forward and backward results

No word importance ranking, no semantic similarity constraint,
no perturbation budget — matching the official algorithm exactly.

References:
  Paper — https://aclanthology.org/2020.acl-main.263/
  Code  — https://github.com/salesforce/morpheus
"""

import logging
import random

logger = logging.getLogger("textattack.attacks.morpheus")

# Inflectable universal POS categories.
# Exact match: MorpheusBase.get_inflections() → have_inflections
_HAVE_INFLECTIONS = frozenset({"NOUN", "VERB", "ADJ"})

_PUNCT_CHARS = ".,!?;:'\"()[]{}"


def _get_universal_pos_tags(words: list[str]) -> list[str]:
    """POS-tag words using NLTK with universal tagset.

    Exact match: official uses nltk.pos_tag(tokens, tagset='universal').
    """
    try:
        import nltk
        tagged = nltk.pos_tag(words, tagset='universal')
        # Official: replace POS with '.' for words containing '&'
        return [
            '.' if '&' in w else tag
            for (w, tag) in tagged
        ]
    except (ImportError, LookupError):
        from utils.text_utils import simple_pos_tag
        _map = {
            "noun": "NOUN", "verb": "VERB", "adj": "ADJ",
            "adv": "ADV", "other": "X",
        }
        return [_map.get(simple_pos_tag(w), "X") for w in words]


def _get_token_inflections(
    orig_tokens: list[str],
    pos_tags: list[str],
    constrain_pos: bool = True,
) -> list[tuple[int, list[str]]]:
    """Get inflection candidates for all inflectable positions.

    Exact match with official MorpheusBase.get_inflections().
    """
    import lemminflect

    token_inflections = []

    for i, word in enumerate(orig_tokens):
        lemmas = lemminflect.getAllLemmas(word)
        if lemmas and pos_tags[i] in _HAVE_INFLECTIONS:
            # Lemma selection (exact match with official)
            if pos_tags[i] in lemmas:
                lemma = lemmas[pos_tags[i]][0]
            else:
                lemma = random.choice(list(lemmas.values()))[0]

            # Get inflections (exact match with official)
            if constrain_pos:
                inflections_dict = lemminflect.getAllInflections(
                    lemma, upos=pos_tags[i],
                )
            else:
                inflections_dict = lemminflect.getAllInflections(lemma)

            # Flatten and deduplicate (exact match with official)
            inflections = list(set(
                infl
                for tup in inflections_dict.values()
                for infl in tup
            ))

            # Shuffle (exact match with official random.shuffle)
            random.shuffle(inflections)
            token_inflections.append((i, inflections))

    return token_inflections


def _search_classification(
    model_wrapper,
    orig_tokenized: list[str],
    token_inflections: list[tuple[int, list[str]]],
    orig_label_idx: int,
    orig_conf_on_label: float,
    backward: bool = False,
) -> tuple[list[str], float, int]:
    """Greedy sequential search — line-for-line match of official search_nmt().

    Adaptation: instead of get_bleu(perturbed, reference) returning BLEU,
    we use predict_probs(perturbed)[orig_label_idx] returning confidence
    on the original label.  Lower = better attack in both cases.
    """
    perturbed_tokenized = list(orig_tokenized)
    best_conf = orig_conf_on_label
    num_queries = 0

    inflections_order = list(token_inflections)
    if backward:
        inflections_order = list(reversed(inflections_order))

    for pos, inflections in inflections_order:
        best_infl = orig_tokenized[pos]  # Default: original word

        for infl in inflections:
            perturbed_tokenized[pos] = infl
            perturbed = " ".join(perturbed_tokenized)

            probs = model_wrapper.predict_probs(perturbed)
            num_queries += 1

            curr_conf = (
                probs[orig_label_idx]
                if orig_label_idx < len(probs)
                else 1.0
            )
            if curr_conf < best_conf:
                best_conf = curr_conf
                best_infl = infl

        perturbed_tokenized[pos] = best_infl

    return perturbed_tokenized, best_conf, num_queries


def run_morpheus(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    constrain_pos: bool = True,
) -> str:
    """MorpheuS attack — exact Algorithm 1 from Tan et al., ACL 2020.

    Adapted for classification: minimises P(y_orig | x') instead of BLEU.

    Args:
        model_wrapper: wrapped model with .predict() / .predict_probs().
        tokenizer: HuggingFace tokenizer (unused, API compat).
        text: input text to attack.
        target_label: target class (None = untargeted).
        constrain_pos: restrict inflections to matching POS (default True,
            matching official constrain_pos parameter).

    Returns: adversarial text (str).
    """
    from utils.text_utils import get_words_and_spans

    logger.info("MorpheuS: starting attack")

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    # Separate trailing punctuation for clean lemminflect lookup,
    # then reattach after inflection (adapter for regex tokeniser vs
    # official Moses tokeniser which splits punctuation into tokens).
    raw_tokens = [w for w, _, _ in words_spans]
    clean_tokens = [w.rstrip(_PUNCT_CHARS) for w in raw_tokens]
    suffixes = [
        w[len(c):] if c else ""
        for w, c in zip(raw_tokens, clean_tokens)
    ]
    clean_tokens = [c if c else w for c, w in zip(clean_tokens, raw_tokens)]

    # POS tag (NLTK, universal tagset — matching official)
    pos_tags = _get_universal_pos_tags(clean_tokens)

    # Original prediction (1 query — matching official)
    orig_label, orig_conf, orig_label_idx = model_wrapper.predict(text)
    orig_probs = model_wrapper.predict_probs(text)
    orig_conf_on_label = (
        orig_probs[orig_label_idx]
        if orig_label_idx < len(orig_probs)
        else orig_conf
    )

    # Build inflection candidates (exact match with official)
    token_inflections_clean = _get_token_inflections(
        clean_tokens, pos_tags, constrain_pos,
    )

    # Reattach trailing punctuation to inflected forms
    token_inflections: list[tuple[int, list[str]]] = []
    for pos, inflections in token_inflections_clean:
        s = suffixes[pos]
        if s:
            inflections = [infl + s for infl in inflections]
        if inflections:
            token_inflections.append((pos, inflections))

    if not token_inflections:
        logger.info("MorpheuS: no inflectable words found")
        return text

    # ── Forward search (matching official) ──────────────────────────────
    fwd_tokens, fwd_conf, fwd_q = _search_classification(
        model_wrapper, raw_tokens, token_inflections,
        orig_label_idx, orig_conf_on_label, backward=False,
    )
    forward_text = " ".join(fwd_tokens)

    # Early return if forward achieved label flip
    # (matches official: if forward_bleu == 0: return)
    fwd_label, _, _ = model_wrapper.predict(forward_text)
    if target_label is not None:
        if fwd_label.lower() == target_label.lower():
            logger.info("MorpheuS: success on forward pass")
            return forward_text
    else:
        if fwd_label != orig_label:
            logger.info("MorpheuS: success on forward pass")
            return forward_text

    # ── Backward search (matching official) ─────────────────────────────
    bwd_tokens, bwd_conf, bwd_q = _search_classification(
        model_wrapper, raw_tokens, token_inflections,
        orig_label_idx, orig_conf_on_label, backward=True,
    )
    backward_text = " ".join(bwd_tokens)

    # Return the better result (matching official:
    #   if forward_bleu < backward_bleu: return forward)
    if fwd_conf <= bwd_conf:
        logger.info(
            "MorpheuS: returning forward result (conf=%.4f)", fwd_conf,
        )
        return forward_text
    else:
        logger.info(
            "MorpheuS: returning backward result (conf=%.4f)", bwd_conf,
        )
        return backward_text
