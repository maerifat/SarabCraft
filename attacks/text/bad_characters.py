"""
Bad Characters — Boucher et al., 2021 (arXiv:2106.09898, IEEE S&P 2022)

"Bad Characters: Imperceptible NLP Attacks."  Imperceptible, visually-identical
adversarial perturbations built from Unicode control/encoding tricks that flip
model predictions while leaving the rendered text unchanged to a human reader.

Faithful to the official QData/TextAttack BadCharacters2021 recipe and the four
imperceptible transformations it composes with the DifferentialEvolution search:

  - homoglyphs   (WordSwapHomoglyphSwap):      swap a char for a Unicode look-alike
  - invisible    (WordSwapInvisibleCharacters): inject zero-width Unicode chars
  - deletions    (WordSwapDeletions):          insert <char><BKSP> pairs (net no-op
                                               when rendered, but present in bytes)
  - reorderings  (WordSwapReorderings):        Unicode BiDi override swaps that render
                                               identically but reorder the code points

Search: the paper uses black-box Differential Evolution over a perturbation vector
bounded by the number of allowed perturbations.  We reproduce that here — a real
differential-evolution optimiser (DE/rand/1/bin) over integer-decoded perturbation
vectors, scored by the victim model's confidence in the original class (untargeted)
or the target class (targeted), with early exit on a successful label flip.  This
matches scipy.optimize.differential_evolution semantics used by the official
WordSwapDifferentialEvolution base class.

Reference: https://github.com/QData/TextAttack — bad_characters_2021.py and
transformations/word_swaps/word_swap_{homoglyph_swap,invisible_characters,
deletions,reorderings}.py
"""

import logging
import random
from dataclasses import dataclass

logger = logging.getLogger("textattack.attacks.bad_characters")


# ── Homoglyph map (official TextAttack WordSwapHomoglyphSwap `homos`) ─────────
HOMOGLYPHS = {
    "-": "˗", "9": "৭", "8": "Ȣ", "7": "𝟕", "6": "б", "5": "Ƽ",
    "4": "Ꮞ", "3": "Ʒ", "2": "ᒿ", "1": "l", "0": "O", "'": "`",
    "a": "ɑ", "b": "Ь", "c": "ϲ", "d": "ԁ", "e": "е", "f": "𝚏",
    "g": "ɡ", "h": "հ", "i": "і", "j": "ϳ", "k": "𝒌", "l": "ⅼ",
    "m": "ｍ", "n": "ո", "o": "о", "p": "р", "q": "ԛ", "r": "ⲅ",
    "s": "ѕ", "t": "𝚝", "u": "ս", "v": "ѵ", "w": "ԝ", "x": "×",
    "y": "у", "z": "ᴢ",
}

# Zero-width invisible characters (WordSwapInvisibleCharacters.invisible_chars)
INVISIBLE_CHARS = ["\u200b", "\u200c", "\u200d"]

# Backspace control char used by WordSwapDeletions
DEL_CHR = chr(0x8)
INS_CHR_MIN = ord("!")
INS_CHR_MAX = ord("~")

# BiDi control characters used by WordSwapReorderings
_PDF = chr(0x202C)
_LRI = chr(0x2066)
_RLI = chr(0x2067)
_LRO = chr(0x202D)
_RLO = chr(0x202E)
_PDI = chr(0x2069)


def _natural(x: float) -> int:
    """Round a float to the nearest non-negative integer (official _natural)."""
    return max(0, round(float(x)))


# ── Perturbation appliers ─────────────────────────────────────────────────────
# Each takes the original string + a list of (integer) gene values and returns a
# perturbed string.  These mirror the official apply_perturbation() methods, which
# decode a flat perturbation vector into positioned character edits.


def _apply_homoglyphs(text: str, positions: list[int]) -> str:
    """Replace characters at the chosen positions with their homoglyphs.

    Mirrors WordSwapHomoglyphSwap.apply_perturbation: a precomputed glyph map of
    (index, homoglyph_char) is indexed by the perturbation genes.
    """
    glyph_map = [(i, HOMOGLYPHS[ch]) for i, ch in enumerate(text) if ch in HOMOGLYPHS]
    if not glyph_map:
        return text
    candidate = list(text)
    for p in positions:
        if 0 <= p < len(glyph_map):
            i, char = glyph_map[p]
            candidate[i] = char
    return "".join(candidate)


def _apply_invisible(text: str, genes: list[int]) -> str:
    """Inject invisible characters.  Genes come in (char_choice, index) pairs.

    Mirrors WordSwapInvisibleCharacters.apply_perturbation.
    """
    candidate = list(text)
    for i in range(0, len(genes) - 1, 2):
        inp_index = genes[i + 1]
        if inp_index >= 0:
            inv_char = INVISIBLE_CHARS[genes[i] % len(INVISIBLE_CHARS)]
            candidate = candidate[:inp_index] + [inv_char] + candidate[inp_index:]
    return "".join(candidate)


def _apply_deletions(text: str, genes: list[int]) -> str:
    """Insert <printable-char><BACKSPACE> pairs.  Genes: (index, char_code) pairs.

    Mirrors WordSwapDeletions.apply_perturbation, including the running index
    shift as characters are inserted.
    """
    candidate = list(text)
    genes = list(genes)
    for i in range(0, len(genes) - 1, 2):
        idx = _natural(genes[i])
        char = chr(_clamp(genes[i + 1], INS_CHR_MIN, INS_CHR_MAX))
        idx = min(idx, len(candidate))
        candidate = candidate[:idx] + [char, DEL_CHR] + candidate[idx:]
        for j in range(i, len(genes), 2):
            genes[j] += 2
    return "".join(candidate)


@dataclass
class _Swap:
    one: str
    two: str


def _apply_swaps(elements) -> str:
    """Recursively encode BiDi swaps (official WordSwapReorderings._apply_swaps)."""
    res = ""
    for el in elements:
        if isinstance(el, _Swap):
            res += _apply_swaps(
                [_LRO, _LRI, _RLO, _LRI, el.one, _PDI, _LRI, el.two, _PDI, _PDF, _PDI, _PDF]
            )
        elif isinstance(el, str):
            res += el
    return res


def _apply_reorderings(text: str, positions: list[int]) -> str:
    """Swap adjacent characters via BiDi overrides so the string renders identically
    but its underlying code-point order is changed.

    Mirrors WordSwapReorderings.apply_perturbation.
    """
    if len(text) < 2:
        return text
    elements: list = list(text)
    for p in positions:
        if p < 0:
            continue
        idx = p % (len(elements) - 1) if len(elements) >= 2 else 0
        if idx + 1 < len(elements) and isinstance(elements[idx], str) and isinstance(elements[idx + 1], str):
            elements[idx : idx + 2] = [_Swap(elements[idx], elements[idx + 1])]
    return _apply_swaps(elements)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(v))))


# ── Perturbation-type configuration ───────────────────────────────────────────
# For each type: how many genes encode one perturbation, and integer bounds per
# gene (matching the official _get_bounds), plus the applier.


def _config_for(perturbation_type: str, text: str):
    n = len(text)
    if perturbation_type == "homoglyphs":
        num_glyphs = sum(1 for ch in text if ch in HOMOGLYPHS)
        # one gene per perturbation: index into the glyph map ([-1, num_glyphs-1])
        return 1, [(-1, max(0, num_glyphs - 1))], _apply_homoglyphs
    if perturbation_type == "invisible":
        # two genes: (invisible-char choice, insertion index)
        return 2, [(0, len(INVISIBLE_CHARS) - 1), (-1, max(0, n - 1))], _apply_invisible
    if perturbation_type == "deletions":
        # two genes: (insertion index, printable char code)
        return 2, [(-1, max(0, n - 1)), (INS_CHR_MIN, INS_CHR_MAX)], _apply_deletions
    if perturbation_type == "reorderings":
        # one gene: swap position ([-1, n-2])
        return 1, [(-1, max(0, n - 2))], _apply_reorderings
    raise ValueError(f"Invalid perturbation_type: {perturbation_type!r}")


# ── Differential Evolution search (DE/rand/1/bin) ─────────────────────────────


def run_bad_characters(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    perturbation_type: str = "homoglyphs",
    max_perturbs: int = 1,
    popsize: int = 32,
    maxiter: int = 10,
    seed: int | None = None,
) -> str:
    """Bad Characters imperceptible attack (Boucher et al., 2021).

    Applies one of four imperceptible Unicode perturbation families and searches
    for a perturbation that flips (untargeted) or reaches (targeted) the model
    prediction, using black-box Differential Evolution — faithful to the official
    TextAttack BadCharacters2021 recipe (DifferentialEvolution search method).

    Args:
        model_wrapper: wrapped model with .predict() → (label, conf, idx) and
                       .predict_probs() → list[float].
        tokenizer: HuggingFace tokenizer (unused, kept for API compatibility).
        text: input text to attack.
        target_label: target class name (None = untargeted).
        perturbation_type: "homoglyphs" | "invisible" | "deletions" | "reorderings".
        max_perturbs: maximum number of imperceptible perturbations per input.
        popsize: DE population size (paper/official default: 32).
        maxiter: DE maximum generations (paper/official default: 10).
        seed: optional RNG seed for reproducibility.

    Returns:
        adversarial text (str) — visually identical to the input, differing only
        in imperceptible Unicode code points.
    """
    ptype = perturbation_type.lower()
    logger.info(
        "BadCharacters: starting (type=%s, max_perturbs=%d, popsize=%d, maxiter=%d)",
        ptype, max_perturbs, popsize, maxiter,
    )

    if not text:
        return text

    rng = random.Random(seed)

    orig_label, orig_conf, orig_idx = model_wrapper.predict(text)

    # Resolve the class index we are optimising the objective against.
    from models.text_loader import get_label_index

    if target_label is not None:
        target_idx = get_label_index(model_wrapper.model, target_label)
        if target_idx is None:
            target_idx = orig_idx
        maximize = True   # push probability of target class UP
    else:
        target_idx = orig_idx
        maximize = False  # push probability of original class DOWN

    try:
        genes_per, bounds_template, applier = _config_for(ptype, text)
    except ValueError as e:
        logger.warning("BadCharacters: %s — falling back to homoglyphs", e)
        ptype = "homoglyphs"
        genes_per, bounds_template, applier = _config_for(ptype, text)

    dim = genes_per * max_perturbs
    bounds = (bounds_template * max_perturbs)[:dim]
    if not bounds:
        return text

    def _decode(vec: list[float]) -> list[int]:
        return [_clamp(v, lo, hi) for v, (lo, hi) in zip(vec, bounds)]

    best_adv = {"text": text, "success": False}

    def _objective(vec: list[float]) -> float:
        """Lower is better (scipy DE minimises). Returns target-class probability
        for untargeted (minimise orig prob) or its negation for targeted."""
        genes = _decode(vec)
        candidate = applier(text, genes)
        if candidate == text:
            return 1.0  # no-op perturbation — worst score
        label, _conf, _idx = model_wrapper.predict(candidate)

        # Success check with early exit signalling via best_adv
        if target_label is not None:
            if label.lower() == target_label.lower():
                best_adv["text"], best_adv["success"] = candidate, True
        else:
            if label != orig_label:
                best_adv["text"], best_adv["success"] = candidate, True

        probs = model_wrapper.predict_probs(candidate)
        p = probs[target_idx] if target_idx < len(probs) else 0.0
        return -p if maximize else p

    # ── Initialise population uniformly within bounds ────────────────────────
    def _rand_individual() -> list[float]:
        return [rng.uniform(lo, hi) for (lo, hi) in bounds]

    population = [_rand_individual() for _ in range(max(4, popsize))]
    fitness = [_objective(ind) for ind in population]
    if best_adv["success"]:
        logger.info("BadCharacters: success during init")
        return best_adv["text"]

    best_i = min(range(len(population)), key=lambda i: fitness[i])
    F, CR = 0.7, 0.9  # DE mutation factor + crossover rate (standard DE/rand/1/bin)

    for gen in range(maxiter):
        for i in range(len(population)):
            # DE/rand/1: pick three distinct others
            choices = [j for j in range(len(population)) if j != i]
            a, b, c = rng.sample(choices, 3)
            mutant = [
                population[a][k] + F * (population[b][k] - population[c][k])
                for k in range(dim)
            ]
            # Binomial crossover
            r = rng.randrange(dim)
            trial = [
                mutant[k] if (rng.random() < CR or k == r) else population[i][k]
                for k in range(dim)
            ]
            # Clip to bounds
            trial = [min(max(trial[k], bounds[k][0]), bounds[k][1]) for k in range(dim)]

            f_trial = _objective(trial)
            if best_adv["success"]:
                logger.info("BadCharacters: success at generation %d", gen + 1)
                return best_adv["text"]

            if f_trial <= fitness[i]:
                population[i], fitness[i] = trial, f_trial
                if f_trial < fitness[best_i]:
                    best_i = i

    # No label flip — return the lowest-objective (most-perturbing) candidate.
    best_genes = _decode(population[best_i])
    fallback = applier(text, best_genes)
    logger.info(
        "BadCharacters: finished without flip (best-effort candidate, type=%s)", ptype
    )
    return fallback if fallback != text else best_adv["text"]
