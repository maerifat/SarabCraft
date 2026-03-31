"""
CheckList — Behavioral Testing of NLP Models
Ribeiro et al., ACL 2020 (Best Paper Award)

Faithful implementation of the CheckList framework as described in
"Beyond Accuracy: Behavioral Testing of NLP Models with CheckList."

Framework components (matching the original API):
  - Expect:    expectation functions (eq, inv, monotonic) with aggregation
  - Perturb:   perturbation generators (negation, entity swap, typos, etc.)
  - Editor:    template-based test generation with lexicon fill-in
  - MFT:       Minimum Functionality Test  (unit tests with expected labels)
  - INV:       Invariance Test             (prediction must not change)
  - DIR:       Directional Expectation Test (prediction must change predictably)
  - TestSuite: grouped execution, aggregation, capability×type reporting

SarabCraft integration via run_checklist() attack entry point.
"""

import logging
import random
import re
import itertools
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable, Union

logger = logging.getLogger("textattack.attacks.checklist")


# ═══════════════════════════════════════════════════════════════════════════
# Lazy-loaded NLP tools
# ═══════════════════════════════════════════════════════════════════════════

_nlp = None


def _get_nlp():
    """Lazy-load spaCy for NER and dependency parsing."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        return _nlp
    except (ImportError, OSError):
        logger.warning("spaCy not available; NER-based perturbations will use lexicon fallback")
        return None


_mlm_model = None
_mlm_tokenizer = None


def _get_mlm():
    """Lazy-load RoBERTa for masked language model suggestions."""
    global _mlm_model, _mlm_tokenizer
    if _mlm_model is not None:
        return _mlm_model, _mlm_tokenizer
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        import torch
        _mlm_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        _mlm_model = AutoModelForMaskedLM.from_pretrained("roberta-base")
        _mlm_model.eval()
        if torch.cuda.is_available():
            _mlm_model.to("cuda")
        return _mlm_model, _mlm_tokenizer
    except Exception as e:
        logger.warning("RoBERTa MLM not available; {mask} template suggestions disabled: %s", e)
        return None, None


# ═══════════════════════════════════════════════════════════════════════════
# Lexicons — built-in word lists for template fill-in
#
# Mirrors the original CheckList lexicons sourced from Wikidata.
# ═══════════════════════════════════════════════════════════════════════════

LEXICONS = {
    "male": [
        "James", "John", "Robert", "Michael", "William", "David", "Richard",
        "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew",
        "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
        "Kenneth", "Kevin", "Brian", "George", "Timothy", "Edward",
    ],
    "female": [
        "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
        "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty",
        "Margaret", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily",
        "Donna", "Michelle", "Carol", "Amanda", "Melissa", "Deborah", "Laura",
    ],
    "last_name": [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
        "Lee", "Thompson", "White", "Harris", "Clark", "Lewis",
    ],
    "country": [
        "United States", "Canada", "Mexico", "Brazil", "Argentina", "France",
        "Germany", "Italy", "Spain", "United Kingdom", "China", "Japan",
        "India", "Australia", "Russia", "South Korea", "South Africa",
        "Nigeria", "Egypt", "Turkey", "Sweden", "Norway", "Netherlands",
    ],
    "city": [
        "New York", "Los Angeles", "Chicago", "Houston", "London", "Paris",
        "Berlin", "Tokyo", "Sydney", "Toronto", "Mumbai", "Beijing",
        "Moscow", "Cairo", "Lagos", "Mexico City", "São Paulo",
        "Buenos Aires", "Seoul", "Istanbul", "Rome", "Madrid", "Amsterdam",
    ],
    "nationality": [
        "American", "Canadian", "Mexican", "Brazilian", "Argentine", "French",
        "German", "Italian", "Spanish", "British", "Chinese", "Japanese",
        "Indian", "Australian", "Russian", "Korean", "South African",
        "Nigerian", "Egyptian", "Turkish", "Swedish", "Norwegian", "Dutch",
    ],
    "religion": [
        "Christianity", "Islam", "Judaism", "Buddhism", "Hinduism",
        "Sikhism", "Atheism",
    ],
    "religion_adj": [
        "Christian", "Muslim", "Jewish", "Buddhist", "Hindu", "Sikh", "Atheist",
    ],
    "profession": [
        "doctor", "lawyer", "teacher", "engineer", "scientist", "nurse",
        "artist", "writer", "musician", "chef", "pilot", "firefighter",
        "police officer", "accountant", "architect", "dentist", "professor",
        "journalist", "pharmacist", "electrician", "plumber", "mechanic",
    ],
    "positive_adj": [
        "good", "great", "excellent", "wonderful", "fantastic", "amazing",
        "brilliant", "outstanding", "superb", "terrific", "marvelous",
        "delightful", "pleasant", "lovely", "beautiful", "perfect",
    ],
    "negative_adj": [
        "bad", "terrible", "awful", "horrible", "dreadful", "poor",
        "lousy", "atrocious", "abysmal", "pathetic", "miserable",
        "disgusting", "unpleasant", "ugly", "hideous", "appalling",
    ],
    "neutral_adj": [
        "normal", "regular", "ordinary", "standard", "typical", "average",
        "common", "usual", "moderate", "plain",
    ],
    "positive_verb": [
        "love", "like", "enjoy", "admire", "appreciate", "adore",
        "cherish", "treasure", "value", "relish",
    ],
    "negative_verb": [
        "hate", "dislike", "despise", "loathe", "detest", "abhor",
    ],
}

LEXICONS["first_name"] = LEXICONS["male"] + LEXICONS["female"]


# ═══════════════════════════════════════════════════════════════════════════
# Contraction maps
# ═══════════════════════════════════════════════════════════════════════════

_EXPANSION_MAP = {
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "won't": "will not", "wouldn't": "would not",
    "couldn't": "could not", "shouldn't": "should not",
    "can't": "cannot", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "I'm": "I am", "you're": "you are",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "they're": "they are", "I've": "I have",
    "you've": "you have", "we've": "we have", "they've": "they have",
    "I'd": "I would", "you'd": "you would", "he'd": "he would",
    "she'd": "she would", "we'd": "we would", "they'd": "they would",
    "I'll": "I will", "you'll": "you will", "he'll": "he will",
    "she'll": "she will", "we'll": "we will", "they'll": "they will",
}

_CONTRACTION_MAP = {v: k for k, v in _EXPANSION_MAP.items()}
_CONTRACTION_MAP["can not"] = "can't"


# ═══════════════════════════════════════════════════════════════════════════
# Negation patterns (regex fallback when spaCy is unavailable)
# ═══════════════════════════════════════════════════════════════════════════

_NEGATION_ADD_PATTERNS = [
    (r"\b(is)\b", r"\1 not"), (r"\b(are)\b", r"\1 not"),
    (r"\b(was)\b", r"\1 not"), (r"\b(were)\b", r"\1 not"),
    (r"\b(do)\b", r"\1 not"), (r"\b(does)\b", r"\1 not"),
    (r"\b(did)\b", r"\1 not"), (r"\b(will)\b", r"\1 not"),
    (r"\b(would)\b", r"\1 not"), (r"\b(could)\b", r"\1 not"),
    (r"\b(should)\b", r"\1 not"), (r"\b(can)\b", r"\1 not"),
    (r"\b(have)\b", r"\1 not"), (r"\b(has)\b", r"\1 not"),
    (r"\b(had)\b", r"\1 not"),
]

_NEGATION_REMOVE_PATTERNS = [
    (r"\bis not\b", "is"), (r"\bisn't\b", "is"),
    (r"\bare not\b", "are"), (r"\baren't\b", "are"),
    (r"\bwas not\b", "was"), (r"\bwasn't\b", "was"),
    (r"\bwere not\b", "were"), (r"\bweren't\b", "were"),
    (r"\bdo not\b", "do"), (r"\bdon't\b", "do"),
    (r"\bdoes not\b", "does"), (r"\bdoesn't\b", "does"),
    (r"\bdid not\b", "did"), (r"\bdidn't\b", "did"),
    (r"\bwill not\b", "will"), (r"\bwon't\b", "will"),
    (r"\bwould not\b", "would"), (r"\bwouldn't\b", "would"),
    (r"\bcould not\b", "could"), (r"\bcouldn't\b", "could"),
    (r"\bshould not\b", "should"), (r"\bshouldn't\b", "should"),
    (r"\bcan not\b", "can"), (r"\bcannot\b", "can"), (r"\bcan't\b", "can"),
    (r"\bhave not\b", "have"), (r"\bhaven't\b", "have"),
    (r"\bhas not\b", "has"), (r"\bhasn't\b", "has"),
    (r"\bhad not\b", "had"), (r"\bhadn't\b", "had"),
    (r"\bnever\b", "always"), (r"\bnobody\b", "somebody"),
    (r"\bnothing\b", "something"), (r"\bnowhere\b", "somewhere"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Taxonomy and temporal swap tables
# ═══════════════════════════════════════════════════════════════════════════

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

_NUMBER_RE = re.compile(r"\b(\d+(?:\.\d+)?)\b")


# ═══════════════════════════════════════════════════════════════════════════
# Expect — Expectation functions for test evaluation
#
# Faithfully implements the CheckList Expect API:
#   single()    — wrap per-example expectation
#   pairwise()  — wrap (original, perturbed) comparison
#   testcase()  — wrap full-testcase function
#   eq()        — prediction == expected label          (MFT default)
#   inv()       — prediction unchanged under perturbation (INV default)
#   monotonic() — confidence changes monotonically       (DIR common)
#   aggregate() — combine per-example results
#
# Return semantics:  >0 = passed,  <=0 = failed,  None = N/A
# ═══════════════════════════════════════════════════════════════════════════


class Expect:
    """CheckList expectation functions."""

    @staticmethod
    def single(fn):
        """Wrap per-example fn(x, pred, conf, label, meta) → float|bool|None."""
        def wrapper(xs, preds, confs, labels=None, meta=None):
            results = []
            for i in range(len(xs)):
                label = labels[i] if labels is not None and i < len(labels) else None
                m = meta[i] if meta is not None and i < len(meta) else None
                results.append(fn(xs[i], preds[i], confs[i], label, m))
            return results
        return wrapper

    @staticmethod
    def pairwise(fn):
        """Wrap pairwise fn(orig_pred, pred, orig_conf, conf, label, meta).

        First element is the original (returned as None); subsequent are
        compared against it.
        """
        def wrapper(xs, preds, confs, labels=None, meta=None):
            results = [None]
            for i in range(1, len(xs)):
                label = labels[i] if labels is not None and i < len(labels) else None
                m = meta[i] if meta is not None and i < len(meta) else None
                results.append(fn(preds[0], preds[i], confs[0], confs[i], label, m))
            return results
        return wrapper

    @staticmethod
    def testcase(fn):
        """Wrap fn operating on the entire testcase (identity wrapper)."""
        return fn

    @staticmethod
    def eq(val=None):
        """Prediction must equal val (or the label if val is None).

        Default expectation for MFT.
        """
        def fn(x, pred, conf, label, meta):
            expected = val if val is not None else label
            if expected is None:
                return None
            return 1.0 if str(pred).lower().strip() == str(expected).lower().strip() else -1.0
        return Expect.single(fn)

    @staticmethod
    def inv(tolerance=0.1):
        """Prediction must remain unchanged from the original.

        Default expectation for INV.  When array/dict confidences are
        available, also fails if any class probability shifts by more
        than ``tolerance`` (matching the original CheckList behaviour).
        """
        def fn(orig_pred, pred, orig_conf, conf, label, meta):
            if isinstance(orig_conf, (list, tuple)) and isinstance(conf, (list, tuple)):
                max_diff = max(abs(a - b) for a, b in zip(orig_conf, conf))
                if max_diff > tolerance:
                    return -max_diff
            if str(orig_pred).lower().strip() != str(pred).lower().strip():
                return -1.0
            return 1.0
        return Expect.pairwise(fn)

    @staticmethod
    def monotonic(label=None, increasing=True, tolerance=0.1):
        """Confidence for ``label`` must monotonically increase/decrease.

        Commonly used for DIR tests.
        """
        def fn(orig_pred, pred, orig_conf, conf, label_arg, meta):
            if label is not None and isinstance(orig_conf, dict):
                oc = orig_conf.get(label, 0.0)
                nc = conf.get(label, 0.0)
            else:
                oc = float(orig_conf)
                nc = float(conf)
            if increasing:
                return 1.0 if nc >= oc - tolerance else -(oc - nc)
            return 1.0 if nc <= oc + tolerance else -(nc - oc)
        return Expect.pairwise(fn)

    @staticmethod
    def combine_and(fn1, fn2):
        """Both expectation functions must pass."""
        def wrapper(xs, preds, confs, labels=None, meta=None):
            r1 = fn1(xs, preds, confs, labels, meta)
            r2 = fn2(xs, preds, confs, labels, meta)
            return [
                None if a is None or b is None else min(a, b)
                for a, b in zip(r1, r2)
            ]
        return wrapper

    @staticmethod
    def combine_or(fn1, fn2):
        """At least one expectation function must pass."""
        def wrapper(xs, preds, confs, labels=None, meta=None):
            r1 = fn1(xs, preds, confs, labels, meta)
            r2 = fn2(xs, preds, confs, labels, meta)
            out = []
            for a, b in zip(r1, r2):
                if a is None and b is None:
                    out.append(None)
                elif a is None:
                    out.append(b)
                elif b is None:
                    out.append(a)
                else:
                    out.append(max(a, b))
            return out
        return wrapper

    @staticmethod
    def aggregate(results, agg_fn="all"):
        """Aggregate per-example results into testcase pass/fail.

        agg_fn:
          'all'              — every example must pass
          'all_except_first' — skip the first (original) element
          callable           — custom aggregation
        """
        if callable(agg_fn):
            return agg_fn(results)

        filtered = [(i, r) for i, r in enumerate(results) if r is not None]
        if agg_fn == "all_except_first":
            filtered = [(i, r) for i, r in filtered if i > 0]
        if not filtered:
            return None

        fails = [r for _, r in filtered if r <= 0]
        return min(fails) if fails else min(r for _, r in filtered)


# ═══════════════════════════════════════════════════════════════════════════
# Perturb — Perturbation functions for test generation
#
# Faithfully implements the CheckList Perturb API:
#   perturb()             — generic harness: apply fn across a dataset
#   change_names()        — NER-aware PERSON entity replacement
#   change_location()     — NER-aware GPE/LOC entity replacement
#   change_number()       — numeric value perturbation (±20 %)
#   add_typos()           — adjacent-character swap
#   add_negation()        — inject negation into text
#   remove_negation()     — strip negation from text
#   contractions()        — toggle between expanded / contracted forms
#   expand_contractions() — expand only
#   contract()            — contract only
#   punctuation()         — toggle trailing punctuation
#   strip_punctuation()   — remove trailing punctuation
# ═══════════════════════════════════════════════════════════════════════════


class Perturb:
    """CheckList perturbation functions."""

    # ── generic harness ────────────────────────────────────────────────

    @staticmethod
    def perturb(data, perturb_fn, keep_original=True, nsamples=None):
        """Apply *perturb_fn* to each item in *data*.

        Args:
            data: list of texts (str) or spaCy Docs.
            perturb_fn: fn(text_or_doc) → None (skip) or list[str].
            keep_original: if True, each entry is [original, perturbed…].
            nsamples: subsample result to this many entries.

        Returns:
            dict with ``'data'`` key containing list-of-lists.
        """
        results = []
        for d in data:
            text = d.text if hasattr(d, "text") else d
            perturbed = perturb_fn(d)
            if perturbed is None or len(perturbed) == 0:
                continue
            if keep_original:
                results.append([text] + list(perturbed))
            else:
                results.append(list(perturbed))
        if nsamples and len(results) > nsamples:
            results = random.sample(results, nsamples)
        return {"data": results}

    # ── NER-aware perturbations ────────────────────────────────────────

    @staticmethod
    def change_names(text, n=3, first_only=False, last_only=False):
        """Replace PERSON entities with alternative names.

        Uses spaCy NER when available; falls back to lexicon-based matching.
        """
        nlp = _get_nlp()
        source = text.text if hasattr(text, "text") else text
        spans = []

        if nlp is not None:
            doc = nlp(source) if isinstance(text, str) else text
            spans = [(e.start_char, e.end_char, e.text)
                     for e in doc.ents if e.label_ == "PERSON"]
        else:
            all_names = set(LEXICONS["first_name"] + LEXICONS["last_name"])
            for name in all_names:
                for m in re.finditer(r"\b" + re.escape(name) + r"\b", source):
                    spans.append((m.start(), m.end(), m.group()))
        if not spans:
            return None

        ret = []
        for _ in range(n):
            new_text, offset = source, 0
            for start, end, orig_name in spans:
                parts = orig_name.split()
                if last_only and len(parts) > 1:
                    rep = random.choice(LEXICONS["last_name"])
                elif first_only or len(parts) == 1:
                    pool = (LEXICONS["male"] if orig_name in LEXICONS["male"]
                            else LEXICONS["female"] if orig_name in LEXICONS["female"]
                            else LEXICONS["first_name"])
                    rep = random.choice(pool)
                else:
                    rep = f"{random.choice(LEXICONS['first_name'])} {random.choice(LEXICONS['last_name'])}"
                new_text = new_text[:start + offset] + rep + new_text[end + offset:]
                offset += len(rep) - (end - start)
            if new_text != source:
                ret.append(new_text)
        return ret or None

    @staticmethod
    def change_location(text, n=3):
        """Replace GPE / LOC entities with alternatives."""
        nlp = _get_nlp()
        source = text.text if hasattr(text, "text") else text
        spans = []

        if nlp is not None:
            doc = nlp(source) if isinstance(text, str) else text
            spans = [(e.start_char, e.end_char, e.text)
                     for e in doc.ents if e.label_ in ("GPE", "LOC")]
        else:
            for loc in set(LEXICONS["country"] + LEXICONS["city"]):
                for m in re.finditer(r"\b" + re.escape(loc) + r"\b", source):
                    spans.append((m.start(), m.end(), m.group()))
        if not spans:
            return None

        cities, countries = LEXICONS["city"], LEXICONS["country"]
        ret = []
        for _ in range(n):
            new_text, offset = source, 0
            for start, end, orig in spans:
                pool = ([c for c in cities if c != orig] if orig in cities
                        else [c for c in countries if c != orig] if orig in countries
                        else [c for c in cities + countries if c != orig])
                if not pool:
                    continue
                rep = random.choice(pool)
                new_text = new_text[:start + offset] + rep + new_text[end + offset:]
                offset += len(rep) - (end - start)
            if new_text != source:
                ret.append(new_text)
        return ret or None

    # ── Value perturbations ────────────────────────────────────────────

    @staticmethod
    def change_number(text, n=3):
        """Replace numeric values with nearby alternatives (±20 %)."""
        source = text.text if hasattr(text, "text") else text
        if not _NUMBER_RE.search(source):
            return None

        def _perturb(match):
            val = float(match.group(1))
            if val == 0:
                return str(random.randint(1, 10))
            delta = val * random.uniform(0.05, 0.2)
            nv = val + random.choice([-1, 1]) * delta
            if "." not in match.group(1):
                return str(max(0, int(round(nv))))
            return f"{nv:.2f}"

        ret = []
        for _ in range(n):
            candidate = _NUMBER_RE.sub(_perturb, source)
            if candidate != source:
                ret.append(candidate)
        return ret or None

    @staticmethod
    def add_typos(text, n=1):
        """Swap adjacent characters to simulate keyboard typos."""
        source = text.text if hasattr(text, "text") else text
        words = source.split()
        content = [(i, w) for i, w in enumerate(words) if len(w) > 2]
        if not content:
            return None

        ret = []
        for _ in range(n):
            idx, word = random.choice(content)
            pos = random.randint(0, len(word) - 2)
            new_word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
            new_words = list(words)
            new_words[idx] = new_word
            candidate = " ".join(new_words)
            if candidate != source:
                ret.append(candidate)
        return ret or None

    # ── Negation ───────────────────────────────────────────────────────

    @staticmethod
    def add_negation(text):
        """Inject negation into text.

        Uses spaCy dependency parsing to find the root verb when available;
        falls back to regex-based insertion.
        """
        source = text.text if hasattr(text, "text") else text
        negation_markers = {"not", "n't", "never", "no", "neither",
                            "nor", "nobody", "nothing", "nowhere"}
        lower_words = source.lower().split()
        if any(w in negation_markers or w.endswith("n't") for w in lower_words):
            return None

        nlp = _get_nlp()
        if nlp is not None:
            doc = nlp(source) if isinstance(text, str) else text
            for token in doc:
                if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
                    aux_words = {"is", "are", "was", "were", "do", "does", "did",
                                 "will", "would", "could", "should", "can",
                                 "has", "have", "had"}
                    if token.text.lower() in aux_words:
                        result = source[:token.idx] + token.text + " not" + source[token.idx + len(token.text):]
                    else:
                        result = source[:token.idx] + "do not " + token.text + source[token.idx + len(token.text):]
                    return [result]

        for pattern, replacement in _NEGATION_ADD_PATTERNS:
            if re.search(pattern, source, re.IGNORECASE):
                candidate = re.sub(pattern, replacement, source, count=1, flags=re.IGNORECASE)
                if candidate != source:
                    return [candidate]
        return None

    @staticmethod
    def remove_negation(text):
        """Strip negation from text (handles contractions)."""
        source = text.text if hasattr(text, "text") else text
        for pattern, replacement in _NEGATION_REMOVE_PATTERNS:
            if re.search(pattern, source, re.IGNORECASE):
                candidate = re.sub(pattern, replacement, source, count=1, flags=re.IGNORECASE)
                if candidate != source:
                    return [candidate]
        return None

    # ── Contractions ───────────────────────────────────────────────────

    @staticmethod
    def contractions(text):
        """Both expand and contract contractions in the text."""
        source = text.text if hasattr(text, "text") else text
        ret = []
        expanded = Perturb.expand_contractions(source)
        contracted = Perturb.contract(source)
        if expanded:
            ret.extend(expanded)
        if contracted:
            ret.extend(contracted)
        return ret or None

    @staticmethod
    def expand_contractions(text):
        """Expand contractions ("can't" → "cannot")."""
        source = text.text if hasattr(text, "text") else text
        ret = []
        for contraction, expansion in _EXPANSION_MAP.items():
            pattern = re.compile(re.escape(contraction), re.IGNORECASE)
            if pattern.search(source):
                candidate = pattern.sub(
                    lambda m, exp=expansion: exp.capitalize() if m.group()[0].isupper() else exp,
                    source, count=1,
                )
                if candidate != source:
                    ret.append(candidate)
        return ret or None

    @staticmethod
    def contract(text):
        """Contract expanded forms ("cannot" → "can't")."""
        source = text.text if hasattr(text, "text") else text
        ret = []
        for expanded, contraction in _CONTRACTION_MAP.items():
            pattern = re.compile(r"\b" + re.escape(expanded) + r"\b", re.IGNORECASE)
            if pattern.search(source):
                candidate = pattern.sub(
                    lambda m, con=contraction: con.capitalize() if m.group()[0].isupper() else con,
                    source, count=1,
                )
                if candidate != source:
                    ret.append(candidate)
        return ret or None

    # ── Punctuation ────────────────────────────────────────────────────

    @staticmethod
    def punctuation(text):
        """Add or remove trailing punctuation."""
        source = (text.text if hasattr(text, "text") else text).rstrip()
        ret = []
        if source and source[-1] in ".!?":
            ret.append(source[:-1].rstrip())
            for p in ".!?":
                if source[-1] != p:
                    ret.append(source[:-1] + p)
        else:
            for p in ".!?":
                ret.append(source + p)
        return ret or None

    @staticmethod
    def strip_punctuation(text):
        """Remove trailing punctuation."""
        source = (text.text if hasattr(text, "text") else text).rstrip()
        if source and source[-1] in ".!?,;:":
            return [source[:-1].rstrip()]
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Editor — Template-based test case generation
#
# Faithfully implements the CheckList Editor API:
#   template()  — fill {placeholder} with lexicons / kwargs, Cartesian product
#   suggest()   — use masked LM for {mask} fill-in (optional, requires MLM)
#
# Template syntax:
#   {name}   → fill from lexicons or kwargs
#   {a:name} → fill + prepend "a" / "an"
#   {mask}   → fill via masked language model (RoBERTa)
# ═══════════════════════════════════════════════════════════════════════════

_TEMPLATE_RE = re.compile(r"\{([^}]+)\}")


class Editor:
    """CheckList template editor with lexicon fill-in and MLM suggestions."""

    def __init__(self, language="english"):
        self.language = language
        self.lexicons = dict(LEXICONS)

    def template(self, template_str, nsamples=None, remove_duplicates=True,
                 labels=None, **kwargs):
        """Generate test data by filling template placeholders.

        Args:
            template_str: text with ``{placeholder}`` slots.
            nsamples: subsample to this many generated texts.
            remove_duplicates: deduplicate generated texts.
            labels: ground-truth label(s) for generated examples.
            **kwargs: custom fill-in values (e.g. ``adj=['good','bad']``).

        Returns:
            dict with ``'data'``, ``'meta'``, and optionally ``'labels'`` keys.
        """
        placeholders = _TEMPLATE_RE.findall(template_str)
        if not placeholders:
            return {"data": [template_str], "meta": [{}]}

        fill_values: dict[str, list[str]] = {}
        for ph in placeholders:
            key = ph[2:] if ph.startswith("a:") else ph
            if key == "mask":
                suggestions = self._suggest_mask(template_str, **kwargs)
                fill_values[ph] = suggestions if suggestions else ["[UNK]"]
            elif key in kwargs:
                vals = kwargs[key]
                fill_values[ph] = list(vals) if isinstance(vals, (list, tuple)) else [vals]
            elif key in self.lexicons:
                fill_values[ph] = list(self.lexicons[key])
            else:
                raise ValueError(f"Unknown placeholder '{key}': not in lexicons or kwargs")

        keys = list(dict.fromkeys(placeholders))
        all_values = [fill_values[k] for k in keys]

        data: list[str] = []
        meta: list[dict] = []
        for combo in itertools.product(*all_values):
            txt = template_str
            m: dict = {}
            for ph, val in zip(keys, combo):
                key = ph[2:] if ph.startswith("a:") else ph
                if ph.startswith("a:"):
                    article = "an" if val[0].lower() in "aeiou" else "a"
                    txt = txt.replace("{" + ph + "}", f"{article} {val}", 1)
                else:
                    txt = txt.replace("{" + ph + "}", val, 1)
                m[key] = val
            data.append(txt)
            meta.append(m)

        if remove_duplicates:
            seen: set[str] = set()
            udata, umeta = [], []
            for d, mv in zip(data, meta):
                if d not in seen:
                    seen.add(d)
                    udata.append(d)
                    umeta.append(mv)
            data, meta = udata, umeta

        if nsamples and len(data) > nsamples:
            indices = random.sample(range(len(data)), nsamples)
            data = [data[i] for i in indices]
            meta = [meta[i] for i in indices]

        result: dict = {"data": data, "meta": meta}
        if labels is not None:
            result["labels"] = [labels] * len(data) if not isinstance(labels, list) else list(labels)
        return result

    def suggest(self, template_str, top_k=10, **kwargs):
        """Return ranked MLM suggestions for {mask} positions."""
        return self._suggest_mask(template_str, top_k=top_k, **kwargs)

    def _suggest_mask(self, template_str, top_k=10, **kwargs):
        model, tokenizer = _get_mlm()
        if model is None:
            return None
        import torch

        filled = template_str.replace("{mask}", tokenizer.mask_token)
        for ph in _TEMPLATE_RE.findall(filled):
            key = ph[2:] if ph.startswith("a:") else ph
            if key == "mask":
                continue
            if key in kwargs:
                val = kwargs[key][0] if isinstance(kwargs[key], list) else kwargs[key]
            elif key in self.lexicons:
                val = self.lexicons[key][0]
            else:
                val = f"[{key}]"
            if ph.startswith("a:"):
                article = "an" if val[0].lower() in "aeiou" else "a"
                filled = filled.replace("{" + ph + "}", f"{article} {val}", 1)
            else:
                filled = filled.replace("{" + ph + "}", val, 1)

        device = next(model.parameters()).device
        inputs = tokenizer(filled, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        if len(mask_positions[1]) == 0:
            return None
        with torch.no_grad():
            outputs = model(**inputs)
        mask_idx = mask_positions[1][0]
        top_ids = outputs.logits[0, mask_idx].topk(top_k * 3).indices.tolist()

        suggestions = []
        for tid in top_ids:
            word = tokenizer.decode([tid]).strip()
            if word and not word.startswith("##") and len(word) > 1 and word.isalpha():
                suggestions.append(word)
            if len(suggestions) >= top_k:
                break
        return suggestions or None


# ═══════════════════════════════════════════════════════════════════════════
# Test Types — MFT, INV, DIR
#
# Three core CheckList behavioural test types:
#   MFT  — unit tests with expected labels  (agg='all')
#   INV  — prediction must not change        (agg='all_except_first')
#   DIR  — prediction must change predictably (agg='all_except_first')
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TestStats:
    """Aggregated statistics for a single test run."""
    testcases: int = 0
    testcases_run: int = 0
    after_filtering: int = 0
    fails: int = 0

    @property
    def fail_rate(self) -> float:
        return (100.0 * self.fails / self.after_filtering) if self.after_filtering else 0.0


class AbstractTest:
    """Base class for all CheckList test types."""

    def __init__(self, data, expect_fn, *, labels=None, name=None,
                 capability=None, description=None, agg_fn="all",
                 test_type="abstract"):
        if isinstance(data, dict) and "data" in data:
            self.data = data["data"]
            self.meta = data.get("meta")
            if labels is None and "labels" in data:
                labels = data["labels"]
        else:
            self.data = list(data)
            self.meta = None

        self.expect_fn = expect_fn
        self.labels = labels
        self.name = name or "Unnamed test"
        self.capability = capability
        self.description = description
        self.agg_fn = agg_fn
        self.test_type = test_type

        self.results: Optional[list] = None
        self.result_details: Optional[list] = None
        self.stats: Optional[TestStats] = None
        self._preds: Optional[list] = None
        self._confs: Optional[list] = None

    # ── execution ──────────────────────────────────────────────────────

    def run(self, predict_fn, overwrite=False, verbose=True, n=None):
        """Run this test against *predict_fn(list[str]) → (preds, confs)*."""
        if self.results is not None and not overwrite:
            return

        test_data = self.data
        if n and len(test_data) > n:
            test_data = random.sample(test_data, n)

        texts: list[str] = []
        bounds: list[tuple[int, int]] = []
        for tc in test_data:
            s = len(texts)
            if isinstance(tc, list):
                texts.extend(tc)
            else:
                texts.append(tc)
            bounds.append((s, len(texts)))

        if not texts:
            return

        preds, confs = predict_fn(texts)
        self._preds, self._confs = preds, confs
        self.results = []
        self.result_details = []

        for tc_idx, (s, e) in enumerate(bounds):
            tc_texts = texts[s:e]
            tc_preds = preds[s:e]
            tc_confs = confs[s:e]

            tc_labels = None
            if self.labels is not None:
                if isinstance(self.labels, list) and tc_idx < len(self.labels):
                    lv = self.labels[tc_idx]
                    tc_labels = lv if isinstance(lv, list) else [lv] * len(tc_texts)
                elif not isinstance(self.labels, list):
                    tc_labels = [self.labels] * len(tc_texts)

            per_example = self.expect_fn(tc_texts, tc_preds, tc_confs, tc_labels, None)
            agg = Expect.aggregate(per_example, self.agg_fn)
            self.results.append(agg)
            self.result_details.append({
                "texts": tc_texts, "preds": tc_preds, "confs": tc_confs,
                "per_example": per_example, "aggregate": agg,
            })

        non_none = [r for r in self.results if r is not None]
        self.stats = TestStats(
            testcases=len(self.data),
            testcases_run=len(test_data),
            after_filtering=len(non_none),
            fails=sum(1 for r in non_none if r <= 0),
        )
        if verbose:
            self._print_summary()

    # ── reporting ──────────────────────────────────────────────────────

    def _print_summary(self, n_examples=3):
        if self.stats is None:
            logger.info("%s: not yet run", self.name)
            return
        logger.info(
            "%s (%s) — cases: %d | fails: %d (%.1f%%)",
            self.name, self.test_type.upper(),
            self.stats.testcases_run, self.stats.fails, self.stats.fail_rate,
        )
        if self.result_details:
            fail_details = [d for d in self.result_details
                            if d["aggregate"] is not None and d["aggregate"] <= 0]
            for det in fail_details[:n_examples]:
                if len(det["texts"]) == 1:
                    logger.info("  FAIL: '%s' → pred=%s conf=%.3f",
                                det["texts"][0], det["preds"][0], det["confs"][0])
                else:
                    logger.info("  FAIL: '%s' → '%s' | pred: %s→%s",
                                det["texts"][0], det["texts"][-1],
                                det["preds"][0], det["preds"][-1])

    def summary(self, n=3):
        """Print and return structured summary."""
        self._print_summary(n_examples=n)
        return self.form_test_info()

    def form_test_info(self):
        """Structured test info matching the original CheckList format."""
        if self.stats is None:
            return {"name": self.name, "type": self.test_type, "status": "not_run"}
        return {
            "name": self.name,
            "description": self.description,
            "capability": self.capability,
            "type": self.test_type,
            "stats": {
                "testcases": self.stats.testcases,
                "testcases_run": self.stats.testcases_run,
                "after_filtering": self.stats.after_filtering,
                "nfailed": self.stats.fails,
                "npassed": self.stats.after_filtering - self.stats.fails,
                "fail_rate": round(self.stats.fail_rate, 2),
            },
        }

    def form_testcases(self):
        """Structured testcase results matching the original CheckList format."""
        if self.result_details is None:
            return []
        cases = []
        for det in self.result_details:
            tc = {"succeed": 1 if (det["aggregate"] is not None and det["aggregate"] > 0) else 0,
                  "examples": []}
            for i in range(len(det["texts"])):
                ex: dict = {"new": {"text": det["texts"][i],
                                    "pred": str(det["preds"][i]),
                                    "conf": float(det["confs"][i])}}
                if i > 0:
                    ex["old"] = {"text": det["texts"][0],
                                 "pred": str(det["preds"][0]),
                                 "conf": float(det["confs"][0])}
                tc["examples"].append(ex)
            cases.append(tc)
        return cases


class MFT(AbstractTest):
    """Minimum Functionality Test.

    Simple examples with expected labels — the NLP equivalent of unit tests.
    Default: Expect.eq(), agg='all'.
    """
    def __init__(self, data, labels=None, expect=None, *, name=None,
                 capability=None, description=None):
        super().__init__(
            data, expect or Expect.eq(), labels=labels, name=name,
            capability=capability, description=description,
            agg_fn="all", test_type="mft",
        )


class INV(AbstractTest):
    """Invariance Test.

    Label-preserving perturbations — prediction must stay the same.
    Data: ``[[original, perturbed1, …], …]``.
    Default: Expect.inv(), agg='all_except_first'.
    """
    def __init__(self, data, expect=None, *, name=None,
                 capability=None, description=None):
        super().__init__(
            data, expect or Expect.inv(), name=name,
            capability=capability, description=description,
            agg_fn="all_except_first", test_type="inv",
        )


class DIR(AbstractTest):
    """Directional Expectation Test.

    Perturbations with a known directional effect — predictions must
    change accordingly.  Expectation **must** be provided explicitly.
    Data: ``[[original, perturbed1, …], …]``.
    Default agg='all_except_first'.
    """
    def __init__(self, data, expect, *, name=None,
                 capability=None, description=None):
        if expect is None:
            raise ValueError("DIR tests require an explicit expectation "
                             "(e.g. Expect.monotonic())")
        super().__init__(
            data, expect, name=name,
            capability=capability, description=description,
            agg_fn="all_except_first", test_type="dir",
        )


# ═══════════════════════════════════════════════════════════════════════════
# TestSuite — Container for behavioural tests
#
# Groups tests by capability, executes against a model, and produces
# structured reports including per-test stats and capability×type matrix.
# ═══════════════════════════════════════════════════════════════════════════


class TestSuite:
    """Collection of CheckList behavioural tests."""

    def __init__(self, name=None):
        self.name = name or "CheckList Suite"
        self.tests: dict[str, AbstractTest] = {}

    def add(self, test, name=None, overwrite=False):
        tname = name or test.name
        if tname in self.tests and not overwrite:
            raise ValueError(f"Test '{tname}' exists. Use overwrite=True.")
        test.name = tname
        self.tests[tname] = test

    def run(self, predict_fn, overwrite=False, verbose=True, n=None):
        """Run every test in the suite against *predict_fn*."""
        for tname, test in self.tests.items():
            if verbose:
                logger.info("Running: %s", tname)
            test.run(predict_fn, overwrite=overwrite, verbose=verbose, n=n)

    def summary(self, types=None, capabilities=None):
        """Print suite-level summary grouped by capability.

        Returns dict[capability] → list[test_info].
        """
        by_cap: dict[str, list] = defaultdict(list)
        for test in self.tests.values():
            if types and test.test_type not in types:
                continue
            if capabilities and test.capability not in capabilities:
                continue
            by_cap[test.capability or "Uncategorized"].append(test.form_test_info())

        total_tests = total_fails = total_cases = 0
        logger.info("=" * 60)
        logger.info("Suite: %s", self.name)
        logger.info("=" * 60)
        for cap, infos in sorted(by_cap.items()):
            logger.info("\n[%s]", cap)
            for info in infos:
                s = info.get("stats", {})
                total_tests += 1
                total_fails += s.get("nfailed", 0)
                total_cases += s.get("after_filtering", 0)
                logger.info("  %s (%s): %d/%d failed (%.1f%%)",
                            info["name"], info["type"].upper(),
                            s.get("nfailed", 0), s.get("after_filtering", 0),
                            s.get("fail_rate", 0))
        rate = (100.0 * total_fails / total_cases) if total_cases else 0.0
        logger.info("\nOverall: %d tests, %d cases, %d failures (%.1f%%)",
                    total_tests, total_cases, total_fails, rate)
        logger.info("=" * 60)
        return dict(by_cap)

    def summary_matrix(self):
        """Capability × test-type matrix of failure rates."""
        matrix: dict[str, dict[str, Optional[float]]] = defaultdict(dict)
        for test in self.tests.values():
            cap = test.capability or "Uncategorized"
            s = test.form_test_info().get("stats", {})
            matrix[cap][test.test_type] = s.get("fail_rate")
        return dict(matrix)

    def get_failures(self):
        """Collect all failing testcases across the suite."""
        failures = []
        for test in self.tests.values():
            if test.result_details is None:
                continue
            for det in test.result_details:
                if det["aggregate"] is not None and det["aggregate"] <= 0:
                    failures.append({
                        "test_name": test.name, "test_type": test.test_type,
                        "capability": test.capability,
                        "texts": det["texts"], "preds": det["preds"],
                        "confs": det["confs"], "per_example": det["per_example"],
                    })
        return failures

    def form_suite_info(self):
        return {
            "name": self.name,
            "tests": {n: t.form_test_info() for n, t in self.tests.items()},
            "matrix": self.summary_matrix(),
        }

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# SarabCraft Attack Entry Point
#
# Integrates the CheckList framework with SarabCraft's text-attack pipeline.
# Given a single input text it:
#   1. Generates perturbations via Perturb
#   2. Builds INV + DIR behavioural tests from the perturbations
#   3. Runs the full suite against the model
#   4. Extracts adversarial examples from test failures
#      (INV violations → model changed prediction on meaning-preserving edit)
#   5. Returns the best adversarial text for SarabCraft's result builder
# ═══════════════════════════════════════════════════════════════════════════

_CAPABILITY_MAP = {
    "negation": "Negation", "contraction": "Robustness",
    "temporal": "Temporal", "taxonomy": "Taxonomy",
    "number": "Robustness", "typo": "Robustness",
    "punctuation": "Robustness", "entity_name": "NER",
    "entity_location": "NER",
}


def _gen_taxonomy_perturbations(text):
    source = text.text if hasattr(text, "text") else text
    results = []
    words = source.split()
    for i, word in enumerate(words):
        clean = word.lower().strip(".,!?;:'\"()[]{}").strip()
        if clean in _TAXONOMY_SWAPS:
            for rep in _TAXONOMY_SWAPS[clean]:
                nw = list(words)
                if word[0].isupper():
                    rep = rep.capitalize()
                nw[i] = rep + word[len(clean):]
                candidate = " ".join(nw)
                if candidate != source:
                    results.append(candidate)
    return results or None


def _gen_temporal_perturbations(text):
    source = text.text if hasattr(text, "text") else text
    results = []
    for src, tgt in _TEMPORAL_SWAPS.items():
        pat = re.compile(r"\b" + re.escape(src) + r"\b", re.IGNORECASE)
        if pat.search(source):
            candidate = pat.sub(tgt, source, count=1)
            if candidate != source:
                results.append(candidate)
    return results or None


def run_checklist(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    test_types: str = "all",
    perturbation_types: str = "all",
    max_test_cases: int = 50,
    similarity_threshold: float = 0.7,
) -> str:
    """CheckList behavioural testing attack.

    Builds a CheckList TestSuite from the input text, runs MFT / INV / DIR
    tests, and returns the most effective adversarial example found among
    behavioural-test failures.

    Args:
        model_wrapper: wrapped model with .predict() interface.
        tokenizer: HuggingFace tokenizer (unused — API compat).
        text: input text to attack.
        target_label: target class (None = untargeted).
        test_types: comma-separated "mft,inv,dir" or "all".
        perturbation_types: comma-separated perturbation names or "all".
        max_test_cases: max perturbations per test.
        similarity_threshold: min semantic similarity to accept a candidate.

    Returns:
        adversarial text (str).
    """
    from utils.text_constraints import compute_semantic_similarity

    logger.info(
        "CheckList: starting (types=%s, perturbations=%s, max=%d, sim=%.2f)",
        test_types, perturbation_types, max_test_cases, similarity_threshold,
    )

    def predict_fn(texts):
        preds, confs = [], []
        for t in texts:
            label, conf, _ = model_wrapper.predict(t)
            preds.append(label)
            confs.append(conf)
        return preds, confs

    orig_label, orig_conf, _ = model_wrapper.predict(text)

    active_types = ({"mft", "inv", "dir"} if test_types == "all"
                    else {t.strip().lower() for t in test_types.split(",")
                          if t.strip().lower() in ("mft", "inv", "dir")} or {"mft", "inv", "dir"})

    all_ptypes = ["negation", "contraction", "temporal", "taxonomy",
                  "number", "typo", "punctuation", "entity_name", "entity_location"]
    active_ptypes = (all_ptypes if perturbation_types == "all"
                     else [t.strip() for t in perturbation_types.split(",")
                           if t.strip() in all_ptypes] or all_ptypes)

    suite = TestSuite(name=f"CheckList: {text[:50]}...")

    def _filter(candidates):
        return [c for c in candidates
                if compute_semantic_similarity(text, c) >= similarity_threshold]

    # ── INV tests ──────────────────────────────────────────────────────
    if "inv" in active_types:
        inv_generators = {
            "contraction": Perturb.contractions,
            "typo": lambda t: Perturb.add_typos(t, n=5),
            "punctuation": Perturb.punctuation,
            "entity_name": lambda t: Perturb.change_names(t, n=5),
            "entity_location": lambda t: Perturb.change_location(t, n=5),
            "taxonomy": _gen_taxonomy_perturbations,
            "number": lambda t: Perturb.change_number(t, n=5),
        }
        for ptype, gen_fn in inv_generators.items():
            if ptype not in active_ptypes:
                continue
            raw = gen_fn(text)
            if not raw:
                continue
            filtered = _filter(raw)[:max_test_cases]
            if filtered:
                data = [[text, p] for p in filtered]
                suite.add(INV(data, name=f"INV:{ptype}",
                              capability=_CAPABILITY_MAP.get(ptype, "Other")))

    # ── DIR tests ──────────────────────────────────────────────────────
    if "dir" in active_types:
        dir_generators = {
            "negation": (Perturb.add_negation,
                         Expect.monotonic(increasing=False, tolerance=0.1)),
            "temporal": (_gen_temporal_perturbations,
                         Expect.monotonic(increasing=False, tolerance=0.2)),
        }
        for ptype, (gen_fn, expect_fn) in dir_generators.items():
            if ptype not in active_ptypes:
                continue
            raw = gen_fn(text)
            if not raw:
                continue
            filtered = _filter(raw)[:max_test_cases]
            if filtered:
                data = [[text, p] for p in filtered]
                suite.add(DIR(data, expect=expect_fn, name=f"DIR:{ptype}",
                              capability=_CAPABILITY_MAP.get(ptype, "Other")))

    # ── MFT tests ──────────────────────────────────────────────────────
    if "mft" in active_types and "negation" in active_ptypes:
        neg_removed = Perturb.remove_negation(text)
        neg_added = Perturb.add_negation(text)
        mft_data, mft_labels = [], []
        if neg_removed:
            for nr in neg_removed[:max_test_cases]:
                mft_data.append(nr)
                mft_labels.append(orig_label)
        if neg_added:
            for na in neg_added[:max_test_cases]:
                mft_data.append(na)
                mft_labels.append(orig_label)
        if mft_data:
            suite.add(MFT(mft_data, labels=mft_labels,
                          name="MFT:negation_sanity",
                          capability="Negation"))

    if not suite.tests:
        logger.info("CheckList: no applicable perturbations for input")
        return text

    suite.run(predict_fn, verbose=True)

    # ── extract adversarial examples from failures ─────────────────────
    best_text = text
    best_impact = 0.0

    for failure in suite.get_failures():
        for ftxt, fpred, fconf in zip(failure["texts"], failure["preds"], failure["confs"]):
            if ftxt == text:
                continue
            if target_label is not None:
                if str(fpred).lower() == target_label.lower():
                    logger.info("CheckList: success via %s (%s)",
                                failure["test_name"], failure["test_type"])
                    return ftxt
            else:
                if str(fpred) != str(orig_label):
                    logger.info("CheckList: success via %s (%s)",
                                failure["test_name"], failure["test_type"])
                    return ftxt
            impact = orig_conf - float(fconf)
            if impact > best_impact:
                best_impact = impact
                best_text = ftxt

    for test in suite.tests.values():
        if test.result_details is None:
            continue
        for det in test.result_details:
            for dtxt, dpred, dconf in zip(det["texts"], det["preds"], det["confs"]):
                if dtxt == text:
                    continue
                if target_label is not None and str(dpred).lower() == target_label.lower():
                    return dtxt
                if target_label is None and str(dpred) != str(orig_label):
                    impact = orig_conf - float(dconf)
                    if impact > best_impact:
                        best_impact = impact
                        best_text = dtxt

    logger.info("CheckList: finished (%d tests, %d failures)",
                len(suite.tests), len(suite.get_failures()))
    return best_text
