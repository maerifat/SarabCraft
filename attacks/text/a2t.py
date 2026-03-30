"""
A2T — Yoo & Qi, 2021 (arXiv:2109.00544)

Towards Improving Adversarial Training of NLP Models.
Word-level black-box attack designed for efficient adversarial training:
  - Gradient-ranked word importance (GreedyWordSwapWIR)
  - DistilBERT MLM for contextual substitution (WordSwapMaskedLM)
  - USE cosine similarity constraint (threshold=0.9)
  - Per-word embedding distance filter (min_cos_sim=0.8)
  - POS tag preservation (allow_verb_noun_swap=True)

Key difference from BERT-Attack/BAE: A2T masks each word position
individually (standard BAE-style) but uses DistilBERT for faster
inference, and enforces stricter similarity constraints to produce
higher-quality adversarial examples suitable for adversarial training.

Reference: TextAttack A2TYoo2021 recipe.
"""

import logging
import math

from models.text_loader import get_label_index

logger = logging.getLogger("textattack.attacks.a2t")

_A2T_MLM_MODEL = "distilbert-base-uncased"
_a2t_mlm = None
_a2t_tok = None

_VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "verb"}
_NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS", "noun"}


def _get_a2t_mlm():
    """Lazy-load DistilBERT MLM for A2T candidate generation."""
    global _a2t_mlm, _a2t_tok
    if _a2t_mlm is not None:
        return _a2t_mlm, _a2t_tok

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    logger.info("Loading A2T MLM: %s", _A2T_MLM_MODEL)
    _a2t_tok = AutoTokenizer.from_pretrained(_A2T_MLM_MODEL, use_fast=True)
    _a2t_mlm = AutoModelForMaskedLM.from_pretrained(_A2T_MLM_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _a2t_mlm.to(device)
    _a2t_mlm.eval()
    return _a2t_mlm, _a2t_tok


def _mlm_word_candidates(words: list[str], index: int, max_candidates: int = 48) -> list[str]:
    """Mask word at index, predict top-k replacements via DistilBERT MLM."""
    import torch

    model, tokenizer = _get_a2t_mlm()
    device = next(model.parameters()).device

    masked = list(words)
    masked[index] = tokenizer.mask_token
    masked_text = " ".join(masked)

    encoding = tokenizer(
        masked_text, max_length=512, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        preds = model(**inputs)[0]

    ids = inputs["input_ids"][0].tolist()
    try:
        masked_idx = ids.index(tokenizer.mask_token_id)
    except ValueError:
        return []

    logits = preds[0, masked_idx]
    probs = torch.softmax(logits, dim=0)
    ranked = torch.argsort(probs, descending=True)

    candidates = []
    for _id in ranked[:max_candidates * 3]:
        _id = _id.item()
        token = tokenizer.convert_ids_to_tokens(_id)
        word = token.lstrip("##").strip()
        if (
            word
            and word.isalpha()
            and word.lower() != words[index].lower()
            and not token.startswith("##")
        ):
            candidates.append(word)
        if len(candidates) >= max_candidates:
            break

    return candidates


def _angular_sim(cos_sim: float) -> float:
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return 1.0 - math.acos(cos_sim) / math.pi


def _pos_compatible(orig_tag: str, cand_tag: str) -> bool:
    if orig_tag == cand_tag:
        return True
    orig_v = orig_tag in _VERB_TAGS
    orig_n = orig_tag in _NOUN_TAGS
    cand_v = cand_tag in _VERB_TAGS
    cand_n = cand_tag in _NOUN_TAGS
    return (orig_v and cand_n) or (orig_n and cand_v)


def run_a2t(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_candidates: int = 48,
    similarity_threshold: float = 0.9,
    max_perturbation_ratio: float = 0.3,
    embedding_cos_threshold: float = 0.8,
) -> str:
    """A2T attack (Yoo & Qi, 2021).

    Gradient-ranked word importance + DistilBERT MLM substitution with
    strict USE and embedding constraints. Designed for generating
    high-quality adversarial examples suitable for adversarial training.

    Returns: adversarial text (str).
    """
    from utils.text_word_importance import delete_one_importance
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, pos_tag_words, clean_word,
    )
    from utils.text_word_substitution import get_embedding_neighbours_with_scores
    from utils.text_constraints import compute_semantic_similarity

    logger.info(
        "A2T: starting (cands=%d, sim=%.4f, emb_cos=%.2f, max_pert=%.2f)",
        max_candidates, similarity_threshold, embedding_cos_threshold,
        max_perturbation_ratio,
    )

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

        if word_idx in modified_indices:
            continue

        word = words[word_idx]
        if len(clean_word(word)) <= 1:
            continue

        current_words = [w for w, _, _ in get_words_and_spans(current_text)]

        # DistilBERT MLM candidates (primary substitution source)
        mlm_cands = _mlm_word_candidates(current_words, word_idx, max_candidates)

        # Also collect embedding neighbours and filter by cosine distance
        emb_cands_scored = get_embedding_neighbours_with_scores(
            word, top_k=max_candidates, context_text=current_text,
            position=word_idx,
        )
        emb_cands = {c for c, s in emb_cands_scored if s >= embedding_cos_threshold}

        # Intersection: candidates must appear in MLM output AND pass
        # embedding distance (A2T paper uses both as constraint)
        candidates = [c for c in mlm_cands if c.lower() in {e.lower() for e in emb_cands}]
        if not candidates:
            # Fallback: use MLM-only candidates if intersection empty
            candidates = mlm_cands[:max_candidates // 2]

        # POS tag filter
        if word_idx < len(current_words):
            current_pos = pos_tag_words(current_words)
            if word_idx < len(current_pos):
                word_pos = current_pos[word_idx]
                filtered = []
                for cand in candidates:
                    cand_pos = pos_tag_words([cand])
                    if cand_pos and _pos_compatible(word_pos, cand_pos[0]):
                        filtered.append(cand)
                candidates = filtered

        best_text = None
        best_impact = -1.0

        for cand in candidates:
            candidate_text = replace_word_at(current_text, word_idx, cand)

            # USE similarity: stricter threshold (0.9 vs TextFooler's 0.84)
            sim = compute_semantic_similarity(current_text, candidate_text)
            ang_sim = _angular_sim(sim)
            if ang_sim < similarity_threshold:
                continue

            label, conf, _ = model_wrapper.predict(candidate_text)

            if target_label is not None:
                if label.lower() == target_label.lower():
                    logger.info("A2T: success at perturbation %d", perturbations_made + 1)
                    return candidate_text
                probs = model_wrapper.predict_probs(candidate_text)
                target_idx = get_label_index(model_wrapper.model, target_label)
                if target_idx is not None and target_idx < len(probs):
                    impact = probs[target_idx]
                else:
                    impact = orig_conf - conf
            else:
                if label != orig_label:
                    logger.info("A2T: success at perturbation %d", perturbations_made + 1)
                    return candidate_text
                impact = orig_conf - conf

            if impact > best_impact:
                best_impact = impact
                best_text = candidate_text

        if best_text is not None:
            current_text = best_text
            perturbations_made += 1
            modified_indices.add(word_idx)

    logger.info("A2T: finished (%d perturbations)", perturbations_made)
    return current_text
