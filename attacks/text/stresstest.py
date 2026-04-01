"""
StressTest — Naik et al., 2018 (COLING 2018)

Stress Test Evaluation for Natural Language Inference.

Exact match to the six stress tests from the original paper and official
implementation (AbhilashaRavichander/NLI_StressTest), organised into
three classes:

  Competence Tests (reasoning ability):
    - antonymy:        WordNet antonym substitution with Lesk WSD (Section 3.1)
    - numerical:       numerical quantity modification (Section 3.1)

  Distraction Tests (robustness to shallow cues):
    - word_overlap:    append "and true is true" (Section 3.2)
    - negation:        append "and false is not true" (Section 3.2)
    - length_mismatch: append "and true is true" ×5 (Section 3.2)

  Noise Tests (robustness to noisy input):
    - spelling_error:  adjacent char swap / keyboard substitution (Section 3.3)

Propositional logic guarantee for distraction tests (Section 3.2):
  (p ⇒ h) ⟹ (p ∧ True ⇒ h) — appending a tautology preserves the label.

Originally designed for NLI premise-hypothesis pairs; adapted here for
general text classification by applying perturbations to the input directly.

Official code: https://github.com/AbhilashaRavichander/NLI_StressTest
Reference:     https://arxiv.org/abs/1806.00692
"""

import logging
import random
import re

logger = logging.getLogger("textattack.attacks.stresstest")


# ── QWERTY keyboard adjacency map (Section 3.3) ─────────────────────────
# Exact match to official make_grammar_adv_samples_jsonl.py keyboard_char_dict.
# Horizontal-only adjacency (2–3 neighbours per key), NOT the full QWERTY grid.

_KEYBOARD_ADJ = {
    'a': ['s'],        'b': ['v', 'n'],    'c': ['x', 'v'],    'd': ['s', 'f'],
    'e': ['r', 'w'],   'f': ['g', 'd'],    'g': ['f', 'h'],    'h': ['g', 'j'],
    'i': ['u', 'o'],   'j': ['h', 'k'],    'k': ['j', 'l'],    'l': ['k'],
    'm': ['n'],        'n': ['m', 'b'],    'o': ['i', 'p'],    'p': ['o'],
    'q': ['w'],        'r': ['t', 'e'],    's': ['d', 'a'],    't': ['r', 'y'],
    'u': ['y', 'i'],   'v': ['c', 'b'],    'w': ['e', 'q'],    'x': ['z', 'c'],
    'y': ['t', 'u'],   'z': ['x'],
}


# ── Antonymy blacklist (make_antonym_adv_samples.py) ─────────────────────
# Exact match to the official blacklist_words list — semantically ambiguous
# words whose WordNet antonyms are unreliable or misleading.

_ANTONYMY_BLACKLIST = {
    "here", "goodness", "yes", "no", "decision", "growing", "priority",
    "cheers", "volume", "right", "left", "goods", "addition", "income",
    "indecision", "there", "parent", "being", "parents", "lord", "lady",
    "put", "capital", "lowercase", "unions",
}


# ── Distraction strategies exempt from similarity checking ───────────────
# Appending a tautology is mathematically guaranteed to preserve meaning
# (Section 3.2): (p ⇒ h) ⟹ (p ∧ True ⇒ h).  Similarity checking should
# not reject these, especially length_mismatch which appends ×5.

_DISTRACTION_STRATEGIES = {"word_overlap", "negation", "length_mismatch"}


# ── Strategy generators ─────────────────────────────────────────────────

def _gen_antonymy(text: str, count: int) -> list[str]:
    """Antonymy (Section 3.1): substitute with WordNet antonym.

    Exact match to official make_antonym_adv_samples.py:
      1. For each word (not in blacklist), perform WSD via Lesk
         algorithm WITHOUT a POS constraint
      2. Accept only senses with POS 's' (adjective satellite)
         or 'n' (noun) — matching official: best_sense.pos() in ('s','n')
      3. Retrieve antonyms from the disambiguated sense's lemmas
      4. Filter multi-word antonyms (containing '_') and "civilian"
      5. Substitute the word with its antonym
    """
    try:
        from nltk.corpus import wordnet as wn
        from nltk.wsd import lesk
    except ImportError:
        logger.debug("StressTest/antonymy: NLTK not available")
        return []

    # Ensure required NLTK resources are available
    try:
        wn.synsets("test")
    except LookupError:
        import nltk
        for res in ["wordnet", "omw-1.4"]:
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass

    words = text.split()
    context = [w.lower() for w in words]

    pool: list[tuple[int, str]] = []
    for i, word in enumerate(words):
        word_lower = word.lower()

        # Official blacklist check
        if word_lower in _ANTONYMY_BLACKLIST:
            continue

        # Official: lesk(sentence, each_word) without POS constraint
        best_sense = lesk(context, word_lower)
        if best_sense is None:
            continue

        # Official: only adjective satellite ('s') and noun ('n')
        if best_sense.pos() not in ('s', 'n'):
            continue

        for lemma in best_sense.lemmas():
            for ant in lemma.antonyms():
                name = ant.name()
                # Official: filter multi-word antonyms and "civilian"
                if "_" in name or name == "civilian":
                    continue
                if name.lower() != word_lower:
                    pool.append((i, name))

    if not pool:
        return []

    random.shuffle(pool)
    results: list[str] = []
    for idx, ant in pool[:count]:
        new_words = list(words)
        new_words[idx] = ant
        results.append(" ".join(new_words))
    return results


def _gen_numerical(text: str, count: int) -> list[str]:
    """Numerical (Section 3.1): modify numerical quantities.

    Exact match to official quant_example_gen.py methodology:

      Entailed form (get_entailed_hypothesis):
        1. Change the first digit to a random 1–9 (different from original)
        2. If old_digit < new_digit → prepend "less than" + new_num
           Else → prepend "more than" + new_num

      Contradictory form (get_contradictory_hypothesis), 50/50:
        A) Replace first digit with a random different digit (number swap)
        B) Prepend "more than" or "less than" to the ORIGINAL number
    """
    matches = list(re.finditer(r'\b(\d+)\b', text))
    if not matches:
        return []

    results: list[str] = []
    for match in random.sample(matches, min(count, len(matches))):
        number_str = match.group()
        old_digit = int(number_str[0])

        # Official: generate a different first digit (randint(1,9))
        new_digit = random.randint(1, 9)
        while new_digit == old_digit:
            new_digit = random.randint(1, 9)
        new_num = str(new_digit) + number_str[1:]

        if random.random() < 0.5:
            # Entailed form (get_entailed_hypothesis)
            # Official logic: if old_digit < new_digit → "less than"
            if old_digit < new_digit:
                prefix = "less than "
            else:
                prefix = "more than "
            new_text = text[:match.start()] + prefix + new_num + text[match.end():]
        else:
            # Contradictory form (get_contradictory_hypothesis)
            if random.random() < 0.5:
                # Official: replace number with new number (first digit changed)
                new_text = text[:match.start()] + new_num + text[match.end():]
            else:
                # Official: prepend "more/less than" to ORIGINAL number
                prefix = random.choice(["more than ", "less than "])
                new_text = text[:match.start()] + prefix + number_str + text[match.end():]

        results.append(new_text)
    return results


def _gen_word_overlap(text: str, _count: int) -> list[str]:
    """Word Overlap (Section 3.2): append 'and true is true'.

    Exact match to official make_distraction_adv_samples_jsonl.py
    with tautology string " and true is true".

    Propositional logic: (p ∧ True = p), so appending a tautology
    preserves the entailment relation while increasing word overlap.
    """
    return [f"{text} and true is true"]


def _gen_negation(text: str, _count: int) -> list[str]:
    """Negation (Section 3.2): append 'and false is not true'.

    Exact match to official make_distraction_adv_samples_jsonl.py
    with tautology string " and false is not true".

    A tautology containing the negation word 'not', testing whether
    models are distracted by negation cues.
    """
    return [f"{text} and false is not true"]


def _gen_length_mismatch(text: str, _count: int) -> list[str]:
    """Length Mismatch (Section 3.2): append 'and true is true' five times.

    Exact match to official make_distraction_adv_samples_jsonl.py
    with tautology string " and true is true" repeated 5 times.

    Creates significant length disparity to test whether models are
    biased by input length.
    """
    return [f"{text}" + " and true is true" * 5]


def _gen_spelling_error(text: str, count: int) -> list[str]:
    """Spelling Error (Section 3.3): character-level perturbation.

    Exact match to official make_grammar_adv_samples_jsonl.py:

      Two perturbation types (Section 3.3 describes both):
        1. perturb_word_swap: for 2-char words → reverse word;
           for longer words → swap char at random position [0, len-2]
           with the next char (note: position 0 IS included)
        2. perturb_word_kb: replace a random character with an adjacent
           QWERTY keyboard character from the official keyboard_char_dict
           (case preserved: official inline code title-cases the perturbed
           word when the original was capitalised in the sentence)

      Word selection: any word with len > 1 (official code).
    """
    words = text.split()
    if not words:
        return []

    # Official: any word with len > 1
    eligible = [i for i, w in enumerate(words) if len(w) > 1]
    if not eligible:
        return []

    results: list[str] = []
    for _ in range(min(count, len(eligible))):
        idx = random.choice(eligible)
        new_words = list(words)
        word = words[idx]

        if random.random() < 0.5:
            # perturb_word_swap (official make_grammar_adv_samples_jsonl.py)
            if len(word) == 2:
                # Official: reverse 2-char words
                new_words[idx] = word[::-1]
            else:
                chars = list(word)
                # Official: int(np.random.uniform(0, len(word) - 1))
                # → random index in [0, len-2], position 0 IS included
                pos = random.randint(0, len(chars) - 2)
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                new_words[idx] = "".join(chars)
        else:
            # perturb_word_kb (official make_grammar_adv_samples_jsonl.py)
            chars = list(word)
            acceptable = [j for j, c in enumerate(chars)
                          if c.lower() in _KEYBOARD_ADJ]
            if not acceptable:
                continue
            char_idx = random.choice(acceptable)
            c = chars[char_idx].lower()
            replacement = random.choice(_KEYBOARD_ADJ[c])
            if chars[char_idx].isupper():
                replacement = replacement.upper()
            chars[char_idx] = replacement
            new_words[idx] = "".join(chars)

        if new_words[idx] != words[idx]:
            results.append(" ".join(new_words))
    return results


# ── Strategy dispatch ────────────────────────────────────────────────────

_STRATEGY_GENERATORS = {
    # Competence Tests (Section 3.1)
    "antonymy":        _gen_antonymy,
    "numerical":       _gen_numerical,
    # Distraction Tests (Section 3.2)
    "word_overlap":    _gen_word_overlap,
    "negation":        _gen_negation,
    "length_mismatch": _gen_length_mismatch,
    # Noise Tests (Section 3.3)
    "spelling_error":  _gen_spelling_error,
}


def run_stresstest(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    strategies: str = "all",
    candidates_per_strategy: int = 5,
    similarity_threshold: float = 0.5,
) -> str:
    """StressTest (Naik et al., COLING 2018).

    Exact match to the six stress tests from the original paper and
    official implementation (AbhilashaRavichander/NLI_StressTest).
    Tests model reasoning ability (antonymy, numerical), robustness to
    shallow distractions (word overlap, negation, length mismatch),
    and noise tolerance (spelling errors).

    Args:
        model_wrapper: wrapped model with .predict() interface.
        tokenizer: HuggingFace tokenizer (unused, API compat).
        text: input text to test.
        target_label: target class (None = untargeted).
        strategies: comma-separated strategy names or "all".
        candidates_per_strategy: max candidates for multi-candidate
            strategies (antonymy, numerical, spelling_error).
            Distraction tests produce exactly one candidate each.
        similarity_threshold: minimum semantic similarity to accept.
            Distraction tests are exempt from this check because
            appending a tautology is guaranteed to preserve meaning
            (Section 3.2 propositional logic guarantee).

    Returns: adversarial text (str).
    """
    from utils.text_constraints import compute_semantic_similarity

    logger.info(
        "StressTest: starting (strategies=%s, per_strategy=%d, sim=%.2f)",
        strategies, candidates_per_strategy, similarity_threshold,
    )

    orig_label, orig_conf, _ = model_wrapper.predict(text)

    if strategies == "all":
        active = list(_STRATEGY_GENERATORS.keys())
    else:
        active = [s.strip() for s in strategies.split(",")
                  if s.strip() in _STRATEGY_GENERATORS]
        if not active:
            active = list(_STRATEGY_GENERATORS.keys())

    all_candidates: list[tuple[str, str]] = []
    for strategy in active:
        gen_fn = _STRATEGY_GENERATORS[strategy]
        for cand in gen_fn(text, candidates_per_strategy):
            all_candidates.append((cand, strategy))

    if not all_candidates:
        logger.info("StressTest: no candidates generated")
        return text

    best_text = text
    best_impact = 0.0

    for candidate_text, strategy in all_candidates:
        # Distraction tests are exempt from similarity checking:
        # appending a tautology is mathematically guaranteed to
        # preserve meaning (Section 3.2 propositional logic).
        if strategy not in _DISTRACTION_STRATEGIES:
            sim = compute_semantic_similarity(text, candidate_text)
            if sim < similarity_threshold:
                continue

        label, conf, _ = model_wrapper.predict(candidate_text)

        if target_label is not None:
            if label.lower() == target_label.lower():
                logger.info("StressTest: success via %s strategy", strategy)
                return candidate_text
        else:
            if label != orig_label:
                logger.info("StressTest: success via %s strategy", strategy)
                return candidate_text

        impact = orig_conf - conf
        if impact > best_impact:
            best_impact = impact
            best_text = candidate_text

    logger.info("StressTest: finished (%d candidates evaluated)", len(all_candidates))
    return best_text
