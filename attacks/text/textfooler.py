"""
TextFooler Attack — Jin et al., 2020 (arXiv:1907.11932)

Black-box word-level attack: ranks word importance by delete-one
(two-case formula), generates candidates via counter-fitted embedding
neighbours (BERT MLM fallback), filters by word-embedding cosine ≥ δ,
POS match (allow_verb_noun_swap=True), and sentence-level USE angular
similarity ≥ threshold.  Exact match to TextAttack TextFoolerJin2019
recipe and the original public implementation.
"""

import logging
import math

from models.text_loader import get_label_index

logger = logging.getLogger("textattack.attacks.textfooler")

# ---------------------------------------------------------------------------
# Official TextFooler stopword list from the public implementation
# (https://github.com/jind11/TextFooler) — matches TextAttack
# StopwordModification(stopwords=...) in the TextFoolerJin2019 recipe.
# ---------------------------------------------------------------------------
TEXTFOOLER_STOPWORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again",
    "against", "ain", "all", "almost", "alone", "along", "already", "also",
    "although", "am", "among", "amongst", "an", "and", "another", "any",
    "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren",
    "aren't", "around", "as", "at", "back", "been", "before", "beforehand",
    "behind", "being", "below", "beside", "besides", "between", "beyond",
    "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't",
    "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down",
    "due", "during", "either", "else", "elsewhere", "empty", "enough",
    "even", "ever", "everyone", "everything", "everywhere", "except",
    "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn",
    "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself",
    "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into",
    "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter",
    "latterly", "least", "ll", "may", "me", "meanwhile", "mightn",
    "mightn't", "mine", "more", "moreover", "most", "mostly", "must",
    "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't",
    "neither", "never", "nevertheless", "next", "no", "nobody", "none",
    "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off",
    "on", "once", "one", "only", "onto", "or", "other", "others",
    "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please",
    "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn",
    "shouldn't", "somehow", "something", "sometime", "somewhere", "such",
    "t", "than", "that", "that'll", "the", "their", "theirs", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "this", "those",
    "through", "throughout", "thru", "thus", "to", "too", "toward",
    "towards", "under", "unless", "until", "up", "upon", "used", "ve",
    "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what",
    "whatever", "when", "whence", "whenever", "where", "whereafter",
    "whereas", "whereby", "wherein", "whereupon", "wherever", "whether",
    "which", "while", "whither", "who", "whoever", "whole", "whom", "whose",
    "why", "with", "within", "without", "won", "won't", "would", "wouldn",
    "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves",
])

# TextAttack threshold: the original TextFooler code forgets to divide the
# angle by pi.  TextAttack corrects this: 1 - 0.5/pi ≈ 0.840845057.
USE_ANGULAR_THRESHOLD = 0.840845057
USE_WINDOW_SIZE = 15

_VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "verb"}
_NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS", "noun"}


def _is_textfooler_stopword(word: str) -> bool:
    return word.lower().strip(".,!?;:'\"()[]{}") in TEXTFOOLER_STOPWORDS


def _angular_sim(cos_sim: float) -> float:
    """Cosine → angular similarity: 1 - arccos(cos) / π."""
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return 1.0 - math.acos(cos_sim) / math.pi


def _pos_compatible(orig_tag: str, cand_tag: str) -> bool:
    """POS compatibility with noun↔verb interchange (paper: allow_verb_noun_swap=True)."""
    if orig_tag == cand_tag:
        return True
    orig_v = orig_tag in _VERB_TAGS
    orig_n = orig_tag in _NOUN_TAGS
    cand_v = cand_tag in _VERB_TAGS
    cand_n = cand_tag in _NOUN_TAGS
    return (orig_v and cand_n) or (orig_n and cand_v)


def _windowed_cosine(text_a: str, text_b: str, word_pos: int,
                     window_size: int, sim_fn) -> float:
    """USE windowed comparison: extract `window_size` words around `word_pos`."""
    from utils.text_utils import get_words_and_spans

    words_a = [w for w, _, _ in get_words_and_spans(text_a)]
    words_b = [w for w, _, _ in get_words_and_spans(text_b)]

    half = window_size // 2
    start = max(0, word_pos - half)
    end = start + window_size
    if end > len(words_a):
        end = len(words_a)
        start = max(0, end - window_size)

    win_a = " ".join(words_a[start:end])
    cand_end = min(len(words_b), end + max(0, len(words_b) - len(words_a)))
    win_b = " ".join(words_b[start:cand_end])

    if not win_a.strip() or not win_b.strip():
        return 0.0
    return sim_fn(win_a, win_b)


def run_textfooler(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_candidates: int = 50,
    similarity_threshold: float = USE_ANGULAR_THRESHOLD,
    max_perturbation_ratio: float = 0.3,
    embedding_cos_threshold: float = 0.5,
) -> str:
    """TextFooler attack (Jin et al., 2020).

    Returns: adversarial text (str).
    """
    from utils.text_word_importance import delete_one_importance
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, pos_tag_words, clean_word,
    )
    from utils.text_word_substitution import get_embedding_neighbours_with_scores
    from utils.text_constraints import compute_semantic_similarity

    logger.info("TextFooler: starting (cands=%d, ang_sim=%.4f, emb_cos=%.2f, "
                "max_pert=%.2f)", max_candidates, similarity_threshold,
                embedding_cos_threshold, max_perturbation_ratio)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    words = [w for w, _, _ in words_spans]
    max_perturbs = max(1, int(len(words) * max_perturbation_ratio))

    orig_label, orig_conf, _ = model_wrapper.predict(text)

    importance = delete_one_importance(model_wrapper, text, orig_label)

    current_text = text
    perturbations_made = 0
    modified_indices: set[int] = set()

    for word_idx, score in importance:
        if perturbations_made >= max_perturbs:
            break

        # RepeatModification: never modify the same word twice
        if word_idx in modified_indices:
            continue

        word = words[word_idx]

        # StopwordModification: official TextFooler stopword list
        if _is_textfooler_stopword(word) or len(clean_word(word)) <= 1:
            continue

        # Counter-fitted embedding neighbours (MLM fallback when unavailable)
        candidates_with_scores = get_embedding_neighbours_with_scores(
            word, top_k=max_candidates, context_text=current_text,
            position=word_idx,
        )

        # WordEmbeddingDistance(min_cos_sim=0.5) — per-word cosine filter
        candidates_with_scores = [
            (cand, sim) for cand, sim in candidates_with_scores
            if sim >= embedding_cos_threshold
        ]

        # PartOfSpeech(allow_verb_noun_swap=True)
        current_words = [w for w, _, _ in get_words_and_spans(current_text)]
        current_pos = pos_tag_words(current_words)
        if word_idx < len(current_pos):
            word_pos = current_pos[word_idx]
            filtered = []
            for cand, sim in candidates_with_scores:
                cand_pos = pos_tag_words([cand])
                if cand_pos and _pos_compatible(word_pos, cand_pos[0]):
                    filtered.append(cand)
            candidates = filtered
        else:
            candidates = [cand for cand, _ in candidates_with_scores]

        best_text = None
        best_impact = -1.0

        for cand in candidates:
            candidate_text = replace_word_at(current_text, word_idx, cand)

            # UniversalSentenceEncoder angular similarity
            #   compare_against_original=False → compare against current_text
            #   window_size=15, skip_text_shorter_than_window=True
            num_current_words = len(current_words)
            if num_current_words >= USE_WINDOW_SIZE:
                cos_sim = _windowed_cosine(
                    current_text, candidate_text, word_idx,
                    USE_WINDOW_SIZE, compute_semantic_similarity,
                )
                ang_sim = _angular_sim(cos_sim)
                if ang_sim < similarity_threshold:
                    continue
            # else: skip USE constraint for short texts (paper behaviour)

            label, conf, _ = model_wrapper.predict(candidate_text)

            if target_label is not None:
                if label.lower() == target_label.lower():
                    logger.info("TextFooler: success at perturbation %d",
                                perturbations_made + 1)
                    return candidate_text
                probs = model_wrapper.predict_probs(candidate_text)
                target_idx = get_label_index(model_wrapper.model, target_label)
                if target_idx is not None and target_idx < len(probs):
                    impact = probs[target_idx]
                else:
                    impact = orig_conf - conf
            else:
                if label != orig_label:
                    logger.info("TextFooler: success at perturbation %d",
                                perturbations_made + 1)
                    return candidate_text
                impact = orig_conf - conf

            if impact > best_impact:
                best_impact = impact
                best_text = candidate_text

        if best_text is not None:
            current_text = best_text
            perturbations_made += 1
            modified_indices.add(word_idx)

    logger.info("TextFooler: finished (%d perturbations)", perturbations_made)
    return current_text
