"""
Kuleshov2017 Attack — Kuleshov et al., 2018 (arXiv:1707.05461)

Adversarial Examples for Natural Language Classification Problems.
Greedy word substitution attack constrained by LANGUAGE MODEL PERPLEXITY
rather than sentence-embedding similarity (USE/distilUSE).

Key difference from TextFooler/BERT-Attack: uses GPT-2 perplexity to
ensure fluency rather than Universal Sentence Encoder similarity.
This produces more grammatically natural adversarial examples.

Algorithm:
  1. Rank words by importance (delete-one confidence drop)
  2. For each word in importance order:
     a. Generate candidates via counter-fitted embeddings
     b. Filter by GPT-2 perplexity: candidate must not increase
        perplexity beyond threshold ratio vs original
     c. Select candidate with maximum confidence impact
  3. Stop when label flips or budget exhausted

Reference: TextAttack Kuleshov2017 recipe.
"""

import logging

logger = logging.getLogger("textattack.attacks.kuleshov2017")

_gpt2_model = None
_gpt2_tok = None


def _load_gpt2():
    """Lazy-load GPT-2 for perplexity scoring."""
    global _gpt2_model, _gpt2_tok
    if _gpt2_model is not None:
        return _gpt2_model, _gpt2_tok

    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    logger.info("Loading GPT-2 for perplexity scoring")
    _gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    _gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _gpt2_model.to(device)
    _gpt2_model.eval()
    return _gpt2_model, _gpt2_tok


def _compute_perplexity(text: str) -> float:
    """Compute GPT-2 perplexity for a text string.

    Lower perplexity = more fluent/natural text.
    """
    import torch
    import math

    model, tokenizer = _load_gpt2()
    device = next(model.parameters()).device

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)

    if input_ids.shape[1] == 0:
        return float("inf")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return math.exp(loss.item())


def run_kuleshov2017(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_candidates: int = 50,
    max_perplexity_ratio: float = 4.0,
    max_perturbation_ratio: float = 0.3,
    embedding_cos_threshold: float = 0.5,
) -> str:
    """Kuleshov2017 attack: greedy substitution with GPT-2 perplexity constraint.

    Args:
        model_wrapper: wrapped model with .predict() interface.
        tokenizer: HuggingFace tokenizer (unused, API compat).
        text: input text to attack.
        target_label: target class (None = untargeted).
        max_candidates: embedding neighbours per word.
        max_perplexity_ratio: max allowed perplexity increase ratio
            (candidate_ppl / original_ppl must be <= this).
        max_perturbation_ratio: max fraction of words to perturb.
        embedding_cos_threshold: min cosine similarity for candidates.

    Returns: adversarial text (str).
    """
    from utils.text_word_importance import delete_one_importance
    from utils.text_utils import get_words_and_spans, replace_word_at, clean_word
    from utils.text_word_substitution import get_embedding_neighbours_with_scores
    from models.text_loader import get_label_index

    logger.info(
        "Kuleshov2017: starting (cands=%d, ppl_ratio=%.1f, max_pert=%.2f)",
        max_candidates, max_perplexity_ratio, max_perturbation_ratio,
    )

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    words = [w for w, _, _ in words_spans]
    max_perturbs = max(1, int(len(words) * max_perturbation_ratio))

    orig_label, orig_conf, _ = model_wrapper.predict(text)
    importance = delete_one_importance(model_wrapper, text, orig_label)

    orig_ppl = _compute_perplexity(text)

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

        candidates_with_scores = get_embedding_neighbours_with_scores(
            word, top_k=max_candidates, context_text=current_text,
            position=word_idx,
        )

        candidates_with_scores = [
            (cand, sim) for cand, sim in candidates_with_scores
            if sim >= embedding_cos_threshold
        ]

        best_text = None
        best_impact = -1.0

        for cand, _ in candidates_with_scores:
            candidate_text = replace_word_at(current_text, word_idx, cand)

            # GPT-2 perplexity constraint (core Kuleshov innovation)
            cand_ppl = _compute_perplexity(candidate_text)
            if orig_ppl > 0 and cand_ppl / orig_ppl > max_perplexity_ratio:
                continue

            label, conf, _ = model_wrapper.predict(candidate_text)

            if target_label is not None:
                if label.lower() == target_label.lower():
                    logger.info("Kuleshov2017: success at perturbation %d",
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
                    logger.info("Kuleshov2017: success at perturbation %d",
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

    logger.info("Kuleshov2017: finished (%d perturbations)", perturbations_made)
    return current_text
