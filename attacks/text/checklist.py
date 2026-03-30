"""
CheckList Attack — Ribeiro et al., 2020 (ACL 2020)

Beyond Accuracy: Behavioral Testing of NLP Models with CheckList.
Template-based linguistic perturbations that test specific model
capabilities via controlled transformations:

  - Negation: inject/remove negation ("is good" → "is not good")
  - Taxonomy (Hypernym/Hyponym): replace entities with related terms
  - NER swap: replace named entities with others of the same type
  - Contraction: expand/contract ("isn't" → "is not", vice versa)
  - Number: perturb numeric values (add/subtract/multiply)
  - Temporal: swap temporal references ("yesterday" → "tomorrow")

Unlike embedding-based attacks, CheckList tests whether models rely on
shallow heuristics rather than true language understanding. Each
perturbation targets a specific linguistic capability.
"""

import logging
import random
import re

logger = logging.getLogger("textattack.attacks.checklist")

# ── Negation templates ────────────────────────────────────────────────────

_NEGATION_PAIRS = [
    (r"\bis not\b", "is"),
    (r"\bis\b", "is not"),
    (r"\bwas not\b", "was"),
    (r"\bwas\b", "was not"),
    (r"\bare not\b", "are"),
    (r"\bare\b", "are not"),
    (r"\bwere not\b", "were"),
    (r"\bwere\b", "were not"),
    (r"\bdo not\b", "do"),
    (r"\bdon't\b", "do"),
    (r"\bdo\b", "do not"),
    (r"\bdoes not\b", "does"),
    (r"\bdoesn't\b", "does"),
    (r"\bdoes\b", "does not"),
    (r"\bdid not\b", "did"),
    (r"\bdidn't\b", "did"),
    (r"\bdid\b", "did not"),
    (r"\bwill not\b", "will"),
    (r"\bwon't\b", "will"),
    (r"\bwill\b", "will not"),
    (r"\bcan not\b", "can"),
    (r"\bcan't\b", "can"),
    (r"\bcannot\b", "can"),
    (r"\bcan\b", "can not"),
    (r"\bcould not\b", "could"),
    (r"\bcouldn't\b", "could"),
    (r"\bcould\b", "could not"),
    (r"\bshould not\b", "should"),
    (r"\bshouldn't\b", "should"),
    (r"\bshould\b", "should not"),
    (r"\bwould not\b", "would"),
    (r"\bwouldn't\b", "would"),
    (r"\bwould\b", "would not"),
    (r"\bhave not\b", "have"),
    (r"\bhaven't\b", "have"),
    (r"\bhas not\b", "has"),
    (r"\bhasn't\b", "has"),
    (r"\bhad not\b", "had"),
    (r"\bhadn't\b", "had"),
    (r"\bnever\b", "always"),
    (r"\balways\b", "never"),
    (r"\bnobody\b", "somebody"),
    (r"\bsomebody\b", "nobody"),
    (r"\bnothing\b", "something"),
    (r"\bsomething\b", "nothing"),
    (r"\bnowhere\b", "somewhere"),
    (r"\bsomewhere\b", "nowhere"),
]

# ── Contraction pairs ────────────────────────────────────────────────────

_CONTRACTION_MAP = {
    "isn't": "is not", "is not": "isn't",
    "aren't": "are not", "are not": "aren't",
    "wasn't": "was not", "was not": "wasn't",
    "weren't": "were not", "were not": "weren't",
    "don't": "do not", "do not": "don't",
    "doesn't": "does not", "does not": "doesn't",
    "didn't": "did not", "did not": "didn't",
    "won't": "will not", "will not": "won't",
    "wouldn't": "would not", "would not": "wouldn't",
    "couldn't": "could not", "could not": "couldn't",
    "shouldn't": "should not", "should not": "shouldn't",
    "can't": "can not", "can not": "can't",
    "haven't": "have not", "have not": "haven't",
    "hasn't": "has not", "has not": "hasn't",
    "hadn't": "had not", "had not": "hadn't",
    "I'm": "I am", "I am": "I'm",
    "you're": "you are", "you are": "you're",
    "he's": "he is", "he is": "he's",
    "she's": "she is", "she is": "she's",
    "it's": "it is", "it is": "it's",
    "we're": "we are", "we are": "we're",
    "they're": "they are", "they are": "they're",
    "I've": "I have", "I have": "I've",
    "you've": "you have", "you have": "you've",
    "we've": "we have", "we have": "we've",
    "they've": "they have", "they have": "they've",
    "I'd": "I would", "I would": "I'd",
    "you'd": "you would", "you would": "you'd",
    "he'd": "he would", "he would": "he'd",
    "she'd": "she would", "she would": "she'd",
    "we'd": "we would", "we would": "we'd",
    "they'd": "they would", "they would": "they'd",
    "I'll": "I will", "I will": "I'll",
    "you'll": "you will", "you will": "you'll",
    "he'll": "he will", "he will": "he'll",
    "she'll": "she will", "she will": "she'll",
    "we'll": "we will", "we will": "we'll",
    "they'll": "they will", "they will": "they'll",
}

# ── Temporal swaps ────────────────────────────────────────────────────────

_TEMPORAL_SWAPS = {
    "yesterday": "tomorrow", "tomorrow": "yesterday",
    "today": "yesterday", "last": "next", "next": "last",
    "past": "future", "future": "past",
    "before": "after", "after": "before",
    "morning": "evening", "evening": "morning",
    "old": "new", "new": "old",
    "early": "late", "late": "early",
    "recently": "soon", "soon": "recently",
    "already": "yet", "yet": "already",
    "ago": "from now",
}

# ── Taxonomy: common hypernym/hyponym swaps ──────────────────────────────

_TAXONOMY_SWAPS = {
    "dog": ["animal", "puppy", "canine", "pet", "hound"],
    "cat": ["animal", "kitten", "feline", "pet"],
    "car": ["vehicle", "automobile", "sedan", "truck"],
    "truck": ["vehicle", "car", "van", "lorry"],
    "house": ["building", "home", "dwelling", "residence"],
    "movie": ["film", "show", "picture", "feature"],
    "book": ["novel", "text", "volume", "publication"],
    "man": ["person", "human", "individual", "guy"],
    "woman": ["person", "human", "individual", "lady"],
    "boy": ["child", "kid", "youth", "lad"],
    "girl": ["child", "kid", "youth", "lass"],
    "city": ["town", "metropolis", "municipality", "place"],
    "food": ["meal", "dish", "cuisine", "snack"],
    "water": ["liquid", "drink", "beverage", "fluid"],
    "phone": ["device", "smartphone", "mobile", "cellphone"],
    "computer": ["machine", "device", "laptop", "PC"],
    "good": ["great", "excellent", "fine", "decent", "nice"],
    "bad": ["poor", "terrible", "awful", "horrible", "dreadful"],
    "happy": ["glad", "pleased", "joyful", "content", "cheerful"],
    "sad": ["unhappy", "depressed", "sorrowful", "gloomy", "miserable"],
    "big": ["large", "huge", "enormous", "massive", "great"],
    "small": ["tiny", "little", "miniature", "compact", "petite"],
    "fast": ["quick", "rapid", "swift", "speedy"],
    "slow": ["sluggish", "gradual", "unhurried", "leisurely"],
}

# ── Number perturbation ──────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"\b(\d+(?:\.\d+)?)\b")


def _perturb_number(match: re.Match) -> str:
    """Perturb a numeric value by a small random factor."""
    val = float(match.group(1))
    if val == 0:
        return str(random.randint(1, 10))
    op = random.choice(["add", "sub", "mult"])
    if op == "add":
        result = val + random.uniform(1, max(val * 0.3, 1))
    elif op == "sub":
        result = val - random.uniform(1, max(val * 0.3, 1))
    else:
        result = val * random.uniform(0.5, 1.5)
    if "." not in match.group(1):
        return str(int(round(result)))
    return f"{result:.2f}"


# ── Perturbation generators ──────────────────────────────────────────────

def _gen_negation(text: str) -> list[str]:
    results = []
    text_lower = text.lower()
    for pattern, replacement in _NEGATION_PAIRS:
        if re.search(pattern, text_lower):
            candidate = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
            if candidate != text:
                results.append(candidate)
    return results


def _gen_contraction(text: str) -> list[str]:
    results = []
    for src, tgt in _CONTRACTION_MAP.items():
        pattern = re.compile(re.escape(src), re.IGNORECASE)
        if pattern.search(text):
            candidate = pattern.sub(tgt, text, count=1)
            if candidate != text:
                results.append(candidate)
    return results


def _gen_temporal(text: str) -> list[str]:
    results = []
    for src, tgt in _TEMPORAL_SWAPS.items():
        pattern = re.compile(r"\b" + re.escape(src) + r"\b", re.IGNORECASE)
        if pattern.search(text):
            candidate = pattern.sub(tgt, text, count=1)
            if candidate != text:
                results.append(candidate)
    return results


def _gen_taxonomy(text: str) -> list[str]:
    results = []
    words = text.split()
    for i, word in enumerate(words):
        clean = word.lower().strip(".,!?;:'\"()[]{}").strip()
        if clean in _TAXONOMY_SWAPS:
            for replacement in _TAXONOMY_SWAPS[clean]:
                new_words = list(words)
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                new_words[i] = replacement + word[len(clean):]
                candidate = " ".join(new_words)
                if candidate != text:
                    results.append(candidate)
    return results


def _gen_number(text: str) -> list[str]:
    if not _NUMBER_RE.search(text):
        return []
    results = []
    for _ in range(5):
        candidate = _NUMBER_RE.sub(_perturb_number, text)
        if candidate != text:
            results.append(candidate)
    return results


_PERTURBATION_GENERATORS = {
    "negation": _gen_negation,
    "contraction": _gen_contraction,
    "temporal": _gen_temporal,
    "taxonomy": _gen_taxonomy,
    "number": _gen_number,
}


def run_checklist(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    perturbation_types: str = "all",
    max_candidates: int = 50,
    similarity_threshold: float = 0.7,
) -> str:
    """CheckList-style behavioral perturbation attack.

    Generates template-based linguistic perturbations (negation, contraction,
    temporal, taxonomy, number), filters by semantic similarity, and returns
    the first candidate that flips the model prediction.

    Args:
        model_wrapper: wrapped model with .predict() interface.
        tokenizer: HuggingFace tokenizer (unused, API compat).
        text: input text to attack.
        target_label: target class (None = untargeted).
        perturbation_types: comma-separated list or "all".
        max_candidates: maximum candidates to evaluate.
        similarity_threshold: minimum semantic similarity.

    Returns: adversarial text (str).
    """
    from utils.text_constraints import compute_semantic_similarity

    logger.info(
        "CheckList: starting (types=%s, max_cands=%d, sim=%.2f)",
        perturbation_types, max_candidates, similarity_threshold,
    )

    orig_label, orig_conf, _ = model_wrapper.predict(text)

    # Select perturbation types
    if perturbation_types == "all":
        active_types = list(_PERTURBATION_GENERATORS.keys())
    else:
        active_types = [t.strip() for t in perturbation_types.split(",")
                        if t.strip() in _PERTURBATION_GENERATORS]
        if not active_types:
            active_types = list(_PERTURBATION_GENERATORS.keys())

    # Generate all candidates across all perturbation types
    all_candidates: list[tuple[str, str]] = []  # (text, type)
    for ptype in active_types:
        gen_fn = _PERTURBATION_GENERATORS[ptype]
        for cand in gen_fn(text):
            all_candidates.append((cand, ptype))

    if not all_candidates:
        logger.info("CheckList: no perturbations applicable to input")
        return text

    random.shuffle(all_candidates)
    all_candidates = all_candidates[:max_candidates]

    best_text = text
    best_impact = 0.0

    for candidate_text, ptype in all_candidates:
        sim = compute_semantic_similarity(text, candidate_text)
        if sim < similarity_threshold:
            continue

        label, conf, _ = model_wrapper.predict(candidate_text)

        if target_label is not None:
            if label.lower() == target_label.lower():
                logger.info("CheckList: success via %s perturbation", ptype)
                return candidate_text
        else:
            if label != orig_label:
                logger.info("CheckList: success via %s perturbation", ptype)
                return candidate_text

        impact = orig_conf - conf
        if impact > best_impact:
            best_impact = impact
            best_text = candidate_text

    logger.info("CheckList: finished (%d candidates evaluated)", len(all_candidates))
    return best_text
