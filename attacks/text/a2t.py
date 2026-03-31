"""
A2T — Yoo & Qi, 2021 (arXiv:2109.00544)

Towards Improving Adversarial Training of NLP Models.

Exact match with TextAttack A2TYoo2021 recipe:
  - Search: GreedyWordSwapWIR(wir_method="gradient")
  - Default variant: WordSwapEmbedding(max_candidates=20)
    + WordEmbeddingDistance(min_cos_sim=0.8)
  - MLM variant (mlm=True): WordSwapMaskedLM(method="bae",
    max_candidates=20, min_confidence=0.0, model="bert-base-uncased")
  - SBERT("stsb-distilbert-base", threshold=0.9, metric="cosine",
    window_size=15, compare_against_original=True)
  - PartOfSpeech(allow_verb_noun_swap=False)
  - MaxModificationRate(max_rate=0.1, min_threshold=4)
  - RepeatModification, StopwordModification
  - UntargetedClassification goal (targeted accepted as extension)
  - On success: returns candidate with highest similarity score

Reference: TextAttack A2TYoo2021 recipe
  https://github.com/QData/TextAttack/blob/master/textattack/attack_recipes/a2t_yoo_2021.py
"""

import logging

from models.text_loader import get_label_index

logger = logging.getLogger("textattack.attacks.a2t")

_A2T_SBERT_MODEL = "stsb-distilbert-base"
_a2t_sbert = None
_a2t_sbert_loaded = False

_SBERT_WINDOW_SIZE = 15

_UNIVERSAL_POS_MAP = {
    "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB",
    "VBP": "VERB", "VBZ": "VERB",
    "NN": "NOUN", "NNS": "NOUN", "NNP": "NOUN", "NNPS": "NOUN",
    "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
    "RB": "ADV", "RBR": "ADV", "RBS": "ADV", "WRB": "ADV",
    "DT": "DET", "PDT": "DET", "WDT": "DET",
    "IN": "ADP",
    "CC": "CCONJ",
    "PRP": "PRON", "PRP$": "PRON", "WP": "PRON", "WP$": "PRON",
    "EX": "PRON",
    "TO": "PART", "RP": "PART",
    "CD": "NUM",
    "MD": "AUX",
    "UH": "INTJ",
    "FW": "X", "SYM": "SYM",
    "verb": "VERB", "noun": "NOUN", "adj": "ADJ", "adv": "ADV",
    "other": "X",
}


def _get_a2t_sbert():
    """Lazy-load SBERT for A2T similarity constraint."""
    global _a2t_sbert, _a2t_sbert_loaded
    if _a2t_sbert_loaded:
        return _a2t_sbert

    _a2t_sbert_loaded = True
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading A2T SBERT: %s", _A2T_SBERT_MODEL)
        _a2t_sbert = SentenceTransformer(_A2T_SBERT_MODEL)
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; "
            "A2T SBERT similarity constraint will fail closed"
        )
    return _a2t_sbert


def _sbert_cosine_sim(text_a: str, text_b: str) -> float:
    """Cosine similarity via stsb-distilbert-base (raw cosine, no angular)."""
    model = _get_a2t_sbert()
    if model is None:
        return 0.0

    embeddings = model.encode([text_a, text_b], convert_to_tensor=True)
    from torch.nn.functional import cosine_similarity
    return cosine_similarity(
        embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
    ).item()


def _windowed_text(words: list[str], index: int, window_size: int) -> str:
    """Extract text window centred on index.

    Matches TextAttack AttackedText.text_window_around_index().
    """
    half = (window_size - 1) // 2
    if index - half < 0:
        start = 0
        end = min(window_size, len(words))
    elif index + half >= len(words):
        start = max(0, len(words) - window_size)
        end = len(words)
    else:
        start = index - half
        end = index + half + 1
    return " ".join(words[start:end])


def _a2t_sbert_check(
    original_text: str,
    candidate_text: str,
    word_idx: int,
    threshold: float,
) -> tuple[bool, float]:
    """Windowed SBERT cosine similarity check.

    Matches TextAttack SBERT(model_name="stsb-distilbert-base",
    threshold=0.9, metric="cosine", window_size=15,
    skip_text_shorter_than_window=False, compare_against_original=True).
    """
    from utils.text_utils import get_words_and_spans

    orig_words = [w for w, _, _ in get_words_and_spans(original_text)]
    cand_words = [w for w, _, _ in get_words_and_spans(candidate_text)]

    orig_window = _windowed_text(orig_words, word_idx, _SBERT_WINDOW_SIZE)
    cand_window = _windowed_text(cand_words, word_idx, _SBERT_WINDOW_SIZE)

    if not orig_window.strip() or not cand_window.strip():
        return False, 0.0

    sim = _sbert_cosine_sim(orig_window, cand_window)
    return sim >= threshold, sim


def _to_universal_pos(tag: str) -> str:
    """Map Penn Treebank / heuristic POS to universal tagset."""
    return _UNIVERSAL_POS_MAP.get(tag, tag)


def run_a2t(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    mlm: bool = False,
    max_candidates: int = 20,
    similarity_threshold: float = 0.9,
    max_modification_rate: float = 0.1,
    min_threshold: int = 4,
    embedding_cos_threshold: float = 0.8,
) -> str:
    """A2T attack (Yoo & Qi, 2021).

    Exact match with TextAttack A2TYoo2021 recipe:
      - Search: GreedyWordSwapWIR(wir_method="gradient")
      - Transformation: WordSwapEmbedding (default) or WordSwapMaskedLM (mlm=True)
      - Constraints: RepeatModification, StopwordModification,
        PartOfSpeech(allow_verb_noun_swap=False),
        MaxModificationRate(max_rate=0.1, min_threshold=4),
        SBERT("stsb-distilbert-base", threshold=0.9, metric="cosine"),
        WordEmbeddingDistance(min_cos_sim=0.8) [default variant only]

    Returns: adversarial text (str).
    """
    from utils.text_word_importance import gradient_importance
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, pos_tag_words, clean_word,
        is_stopword,
    )
    from utils.text_word_substitution import (
        get_embedding_neighbours_with_scores, get_mlm_substitutions,
    )

    logger.info(
        "A2T: starting (mlm=%s, cands=%d, sim=%.2f, max_mod=%.2f)",
        mlm, max_candidates, similarity_threshold, max_modification_rate,
    )

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    words = [w for w, _, _ in words_spans]
    num_words = len(words)

    # MaxModificationRate(max_rate, min_threshold):
    # rate enforced only when num_words >= min_threshold
    if num_words >= min_threshold:
        max_perturbs = max(1, int(num_words * max_modification_rate))
    else:
        max_perturbs = num_words

    orig_label, orig_conf, orig_label_idx = model_wrapper.predict(text)
    orig_probs = model_wrapper.predict_probs(text)

    # GreedyWordSwapWIR(wir_method="gradient"): gradient-based word importance
    importance = gradient_importance(
        model_wrapper.model, model_wrapper.tokenizer, text,
    )

    orig_pos_tags = pos_tag_words(words)

    original_text = text
    current_text = text
    perturbations_made = 0
    modified_indices: set[int] = set()

    # UntargetedClassification score: 1 - P(original_class)
    cur_score = 1.0 - orig_probs[orig_label_idx]

    for word_idx, _imp_score in importance:
        if perturbations_made >= max_perturbs:
            break

        # RepeatModification
        if word_idx in modified_indices:
            continue

        if word_idx >= num_words:
            continue

        word = words[word_idx]

        # StopwordModification
        if is_stopword(word):
            continue

        if len(clean_word(word)) <= 1:
            continue

        current_words = [w for w, _, _ in get_words_and_spans(current_text)]
        if word_idx >= len(current_words):
            continue

        # --- Candidate generation ---
        if mlm:
            # A2T-MLM: WordSwapMaskedLM(method="bae", max_candidates=20)
            candidates = get_mlm_substitutions(
                current_text, position=word_idx, top_k=max_candidates,
            )
        else:
            # Default: WordSwapEmbedding(max_candidates=20)
            # + WordEmbeddingDistance(min_cos_sim=0.8)
            emb_results = get_embedding_neighbours_with_scores(
                current_words[word_idx], top_k=max_candidates,
                context_text=current_text, position=word_idx,
            )
            candidates = [
                c for c, s in emb_results if s >= embedding_cos_threshold
            ]

        if not candidates:
            continue

        # PartOfSpeech(allow_verb_noun_swap=False): universal tagset, exact match
        if word_idx < len(orig_pos_tags):
            word_upos = _to_universal_pos(orig_pos_tags[word_idx])
            filtered = []
            for cand in candidates:
                cand_tags = pos_tag_words([cand])
                if cand_tags and _to_universal_pos(cand_tags[0]) == word_upos:
                    filtered.append(cand)
            if filtered:
                candidates = filtered

        # --- Score all candidates ---
        # (text, goal_score, succeeded, sbert_sim)
        scored: list[tuple[str, float, bool, float]] = []

        for cand in candidates:
            candidate_text = replace_word_at(current_text, word_idx, cand)

            # SBERT cosine: compare_against_original=True, window_size=15
            passes, sim = _a2t_sbert_check(
                original_text, candidate_text, word_idx, similarity_threshold,
            )
            if not passes:
                continue

            probs = model_wrapper.predict_probs(candidate_text)
            pred_idx = probs.index(max(probs))

            if target_label is not None:
                t_idx = get_label_index(model_wrapper.model, target_label)
                succeeded = t_idx is not None and pred_idx == t_idx
                if succeeded:
                    cand_score = 2.0 - probs[orig_label_idx] + probs[t_idx]
                else:
                    cand_score = (
                        probs[t_idx]
                        if t_idx is not None and t_idx < len(probs)
                        else 1.0 - probs[orig_label_idx]
                    )
            else:
                # UntargetedClassification scoring
                succeeded = pred_idx != orig_label_idx
                if succeeded:
                    cand_score = 2.0 - probs[orig_label_idx] + max(probs)
                else:
                    cand_score = 1.0 - probs[orig_label_idx]

            scored.append((candidate_text, cand_score, succeeded, sim))

        if not scored:
            continue

        scored.sort(key=lambda x: x[1], reverse=True)

        best_text, best_score, best_succeeded, _ = scored[0]

        # Greedy: only accept if score improved
        if best_score <= cur_score:
            continue

        cur_score = best_score
        current_text = best_text
        perturbations_made += 1
        modified_indices.add(word_idx)

        # On success: return candidate with highest similarity among
        # all successful candidates (matching GreedyWordSwapWIR behavior)
        if best_succeeded:
            max_sim = -float("inf")
            return_text = best_text
            for cand_text, _, succeeded, sim in scored:
                if not succeeded:
                    break
                if sim > max_sim:
                    max_sim = sim
                    return_text = cand_text
            logger.info(
                "A2T: success at perturbation %d (sim=%.4f)",
                perturbations_made, max_sim,
            )
            return return_text

    logger.info("A2T: finished (%d perturbations, no success)", perturbations_made)
    return current_text
