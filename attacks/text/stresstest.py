"""
StressTest Attack — Naik et al., 2018 (NAACL 2018)

Stress Test Evaluation for Natural Language Inference.
Concatenation-based attack that appends distraction sentences to the input
to test whether models are distracted by irrelevant information.

Five distraction strategies:
  - Tautology: append logically vacuous statements ("and true is true")
  - Negation: append contradictory noise ("and false is not true")
  - Overlap: duplicate a random chunk of the input as suffix
  - Length: pad with neutral filler sentences
  - Noise: append random unrelated but grammatical sentences

Originally designed for NLI, but generalises to any text classifier:
models that rely on shallow heuristics (keyword matching, length bias)
are vulnerable.
"""

import logging
import random

logger = logging.getLogger("textattack.attacks.stresstest")

# ── Tautology templates ──────────────────────────────────────────────────

_TAUTOLOGIES = [
    "and true is true",
    "and false is not true",
    "and if true then true",
    "and it is true that true is true",
    "and true is not false",
    "and false is false",
    "and the fact is that the fact is true",
    "and one is one",
    "and water is water",
    "and if a then a",
]

# ── Negation distraction sentences ───────────────────────────────────────

_NEGATION_DISTRACTORS = [
    "and false is not true",
    "and it is not the case that false is true",
    "and not false",
    "and nothing is not something",
    "and the negation of false is true",
    "and true is never false",
]

# ── Neutral filler sentences (length padding) ────────────────────────────

_NEUTRAL_FILLERS = [
    "The weather is nice today.",
    "It was a normal day.",
    "People are walking outside.",
    "The sky is blue.",
    "Someone left a message.",
    "There is a table in the room.",
    "The clock shows the time.",
    "A bird is sitting on a branch.",
    "The store is open.",
    "Traffic is moving slowly.",
    "A man reads a newspaper.",
    "The music is playing softly.",
    "Children are playing in the park.",
    "The coffee is hot.",
    "A woman carries a bag.",
]

# ── Noise: grammatical but unrelated sentences ───────────────────────────

_NOISE_SENTENCES = [
    "Penguins live in Antarctica.",
    "The speed of light is approximately 300,000 km/s.",
    "Mount Everest is the tallest mountain.",
    "The Pacific Ocean is the largest ocean.",
    "Shakespeare wrote many plays.",
    "Water boils at 100 degrees Celsius.",
    "The Earth orbits the Sun.",
    "Elephants are the largest land animals.",
    "The Amazon is the longest river.",
    "Gold is a chemical element.",
    "Honey never spoils.",
    "Octopuses have three hearts.",
    "Bananas are technically berries.",
    "Venus is the hottest planet.",
    "Sound travels faster in water than in air.",
]


def _gen_tautology(text: str, count: int) -> list[str]:
    results = []
    for taut in random.sample(_TAUTOLOGIES, min(count, len(_TAUTOLOGIES))):
        results.append(f"{text} {taut}")
    return results


def _gen_negation_distraction(text: str, count: int) -> list[str]:
    results = []
    for neg in random.sample(_NEGATION_DISTRACTORS, min(count, len(_NEGATION_DISTRACTORS))):
        results.append(f"{text} {neg}")
    return results


def _gen_overlap(text: str, count: int) -> list[str]:
    words = text.split()
    if len(words) < 3:
        return [f"{text} {text}"]
    results = []
    for _ in range(count):
        chunk_len = random.randint(2, max(2, len(words) // 2))
        start = random.randint(0, max(0, len(words) - chunk_len))
        chunk = " ".join(words[start:start + chunk_len])
        results.append(f"{text} {chunk}")
    return results


def _gen_length(text: str, count: int) -> list[str]:
    results = []
    for _ in range(count):
        n_fillers = random.randint(1, 3)
        fillers = " ".join(random.sample(_NEUTRAL_FILLERS, min(n_fillers, len(_NEUTRAL_FILLERS))))
        results.append(f"{text} {fillers}")
    return results


def _gen_noise(text: str, count: int) -> list[str]:
    results = []
    for _ in range(count):
        n_noise = random.randint(1, 2)
        noise = " ".join(random.sample(_NOISE_SENTENCES, min(n_noise, len(_NOISE_SENTENCES))))
        results.append(f"{text} {noise}")
    return results


_STRATEGY_GENERATORS = {
    "tautology": _gen_tautology,
    "negation": _gen_negation_distraction,
    "overlap": _gen_overlap,
    "length": _gen_length,
    "noise": _gen_noise,
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
    """StressTest attack: distraction-based input concatenation.

    Appends tautologies, negations, overlap chunks, filler sentences, or
    noise to test whether the model is distracted by irrelevant input.
    Lower similarity threshold since appending content naturally reduces
    sentence similarity.

    Args:
        model_wrapper: wrapped model with .predict() interface.
        tokenizer: HuggingFace tokenizer (unused, API compat).
        text: input text to attack.
        target_label: target class (None = untargeted).
        strategies: comma-separated list or "all".
        candidates_per_strategy: number of candidates per strategy.
        similarity_threshold: minimum semantic similarity.

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

    random.shuffle(all_candidates)

    best_text = text
    best_impact = 0.0

    for candidate_text, strategy in all_candidates:
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
