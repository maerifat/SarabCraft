"""
Kuleshov2017 Attack — Kuleshov et al., 2018

Adversarial Examples for Natural Language Classification Problems.

Exact match to the TextAttack Kuleshov2017 recipe
(textattack.attack_recipes.kuleshov_2017.Kuleshov2017).

Algorithm (paper Section 3, TextAttack GreedySearch):
  GreedySearch (BeamSearch, beam_width=1):
    At each step, generate all valid single-word substitutions across
    every eligible position (not-yet-modified, not-stopword, within δ).
    Evaluate each through constraints and goal function.  Accept the
    globally best-scoring candidate.  Repeat until success or no valid
    candidates remain.

Constraints (paper Section 3, exact TextAttack recipe):
  1. RepeatModification          — don't re-modify already-changed words
  2. StopwordModification        — hard-block stopwords
  3. MaxWordsPerturbed(δ = 0.5)  — at most 50 % of words may change
  4. ThoughtVector(λ₁ = 0.2, max_euclidean)  — *squared* Euclidean distance
     between sentence-level mean counter-fitted GloVe embeddings ≤ 0.2
     (Eq. 4).  TextAttack get_neg_euclidean_dist computes −Σ(e₁−e₂)²
     and checks ≥ −threshold, so the bound is on Σ(e₁−e₂)² ≤ λ₁.
  5. GPT2(λ₂ = 2.0 nats)        — per-word log-probability difference at
     the substitution position, given the original prefix  (Eq. 5)

Goal function:
  UntargetedClassification(τ = 0.7):
    Success when model confidence for the original class < τ.

Transformation:
  WordSwapEmbedding(N = 15): counter-fitted PARAGRAM-SL999 neighbours.

Paper parameters: τ = 0.7, N = 15, λ₁ = 0.2, δ = 0.5, λ₂ = 2 nats.

Paper: https://openreview.net/forum?id=r1QZ3zbAZ
TextAttack recipe: textattack.attack_recipes.kuleshov_2017.Kuleshov2017
"""

import logging
import math

import torch

logger = logging.getLogger("textattack.attacks.kuleshov2017")

_gpt2_model = None
_gpt2_tok = None


def _load_gpt2():
    """Lazy-load GPT-2 for per-word log-probability scoring (Eq. 5)."""
    global _gpt2_model, _gpt2_tok
    if _gpt2_model is not None:
        return _gpt2_model, _gpt2_tok

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    logger.info("Loading GPT-2 for LM constraint")
    _gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    _gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _gpt2_model.to(device)
    _gpt2_model.eval()
    return _gpt2_model, _gpt2_tok


def _get_logits_at_index(prefix: str, words: list[str]) -> list[float]:
    """GPT-2 logits for several next-words sharing the same prefix.

    Matches TextAttack GPT2.get_log_probs_at_index(): tokenise the prefix,
    one forward pass, look up the logit for each word's first sub-word token
    at the last position.  Prefix is always from the ORIGINAL text
    (compare_against_original=True in the recipe).

    Since log(softmax(xᵢ)) - log(softmax(xⱼ)) = xᵢ - xⱼ, comparing raw
    logits is equivalent to comparing log-probabilities for the difference
    check, matching the TextAttack LanguageModelConstraint implementation.
    """
    if not prefix or not any(c.isalpha() for c in prefix):
        return [0.0] * len(words)

    model, tokenizer = _load_gpt2()
    device = next(model.parameters()).device

    prefix_ids = tokenizer.encode(prefix)
    if not prefix_ids:
        return [0.0] * len(words)

    with torch.no_grad():
        last_logits = model(
            torch.tensor([prefix_ids], device=device),
        ).logits[0, -1]

    out: list[float] = []
    for w in words:
        ids = tokenizer.encode(w)
        out.append(last_logits[ids[0]].item() if ids else 0.0)
    return out


def _thought_vector(words: list[str], wv) -> torch.Tensor | None:
    """Mean of counter-fitted word embeddings (TextAttack ThoughtVector).

    Out-of-vocabulary words are silently skipped, matching official code.
    """
    vecs = []
    for w in words:
        try:
            vecs.append(torch.tensor(wv[w.lower()], dtype=torch.float32))
        except KeyError:
            continue
    if not vecs:
        return None
    return torch.stack(vecs).mean(dim=0)


def run_kuleshov2017(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_candidates: int = 15,
    max_log_prob_diff: float = 2.0,
    max_perturbation_ratio: float = 0.5,
    thought_vector_threshold: float = 0.2,
    target_max_score: float = 0.7,
) -> str:
    """Kuleshov2017 attack — exact match to TextAttack recipe.

    Args:
        model_wrapper: wrapped model with .predict() / .predict_probs().
        tokenizer: HuggingFace tokenizer (unused, API compat).
        text: input text to attack.
        target_label: target class (None = untargeted, paper default).
        max_candidates: embedding neighbours per word (paper N = 15).
        max_log_prob_diff: GPT-2 log-prob tolerance (paper λ₂ = 2.0 nats).
        max_perturbation_ratio: max fraction of words to perturb (paper δ = 0.5).
        thought_vector_threshold: max squared Euclidean thought-vector dist (paper λ₁ = 0.2).
        target_max_score: confidence threshold for success (paper τ = 0.7).

    Returns: adversarial text (str).
    """
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, clean_word, is_stopword,
    )
    from utils.text_word_substitution import (
        get_embedding_neighbours, _load_word_vectors,
    )
    from models.text_loader import get_label_index

    logger.info(
        "Kuleshov2017: starting (N=%d, λ₂=%.1f, δ=%.2f, λ₁=%.2f, τ=%.2f)",
        max_candidates, max_log_prob_diff, max_perturbation_ratio,
        thought_vector_threshold, target_max_score,
    )

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    orig_words = [w for w, _, _ in words_spans]
    num_words = len(orig_words)
    max_perturbs = max(1, math.ceil(num_words * max_perturbation_ratio))

    _, _, orig_label_idx = model_wrapper.predict(text)

    # ThoughtVector: pre-compute original sentence vector (Eq. 4)
    wv = _load_word_vectors()
    orig_tv = _thought_vector(orig_words, wv) if wv is not None else None
    if wv is None:
        logger.warning(
            "Counter-fitted embeddings unavailable — ThoughtVector constraint "
            "disabled (install gensim and download paragramcf vectors)"
        )

    current_text = text
    modified_indices: set[int] = set()

    # ── GreedySearch (BeamSearch, beam_width=1) ─────────────────────
    # Matches TextAttack BeamSearch.perform_search(): each step generates
    # ALL valid single-word substitutions, evaluates ALL through constraints
    # and goal function, then selects the globally best-scoring candidate.
    # search_over is set when ANY candidate achieves the goal; the best
    # result is returned (not the first successful one).
    while len(modified_indices) < max_perturbs:
        best_text: str | None = None
        best_score = -float("inf")
        best_idx = -1
        search_over = False

        cur_spans = get_words_and_spans(current_text)
        cur_words = [w for w, _, _ in cur_spans]

        for word_idx in range(len(cur_words)):
            # RepeatModification
            if word_idx in modified_indices:
                continue
            # StopwordModification (hard block, matching TextAttack recipe)
            if is_stopword(cur_words[word_idx]):
                continue
            if len(clean_word(cur_words[word_idx])) <= 1:
                continue

            # WordSwapEmbedding(max_candidates=N)
            candidates = get_embedding_neighbours(
                cur_words[word_idx], top_k=max_candidates,
                context_text=current_text, position=word_idx,
            )
            if not candidates:
                continue

            # GPT-2 LM: one forward pass per position (Eq. 5)
            # Prefix is always from the ORIGINAL text
            # (compare_against_original=True)
            prefix = (
                " ".join(orig_words[:word_idx])
                if word_idx < len(orig_words) else ""
            )
            orig_word = (
                orig_words[word_idx] if word_idx < len(orig_words) else ""
            )
            logits = _get_logits_at_index(prefix, [orig_word] + candidates)
            orig_logit = logits[0]
            cand_logit_map = dict(zip(candidates, logits[1:]))

            for cand in candidates:
                # GPT-2 LM constraint (Eq. 5, λ₂)
                # Matches LanguageModelConstraint._check_constraint:
                #   reject if cand_logit <= orig_logit - max_log_prob_diff
                if cand_logit_map[cand] <= orig_logit - max_log_prob_diff:
                    continue

                cand_text = replace_word_at(current_text, word_idx, cand)

                # ThoughtVector constraint (Eq. 4, λ₁, max_euclidean)
                # Always compared against ORIGINAL text
                # (compare_against_original=True).
                # TextAttack get_neg_euclidean_dist = −Σ(e₁−e₂)²;
                # check: −Σ(e₁−e₂)² ≥ −λ₁  ⟹  Σ(e₁−e₂)² ≤ λ₁
                if orig_tv is not None:
                    cand_tv = _thought_vector(cand_text.split(), wv)
                    if cand_tv is not None:
                        diff = orig_tv - cand_tv
                        sq_euclidean = torch.sum(diff * diff).item()
                        if sq_euclidean > thought_vector_threshold:
                            continue

                # Goal function evaluation
                cand_probs = model_wrapper.predict_probs(cand_text)
                orig_class_prob = (
                    cand_probs[orig_label_idx]
                    if orig_label_idx < len(cand_probs)
                    else 1.0
                )

                if target_label is None:
                    # UntargetedClassification(target_max_score=τ)
                    score = 1.0 - orig_class_prob
                    if orig_class_prob < target_max_score:
                        search_over = True
                else:
                    t_idx = get_label_index(model_wrapper.model, target_label)
                    if t_idx is not None and t_idx < len(cand_probs):
                        cand_label, _, _ = model_wrapper.predict(cand_text)
                        if cand_label.lower() == target_label.lower():
                            search_over = True
                        score = cand_probs[t_idx]
                    else:
                        score = 1.0 - orig_class_prob

                if score > best_score:
                    best_score = score
                    best_text = cand_text
                    best_idx = word_idx

        if best_text is None:
            break

        current_text = best_text
        modified_indices.add(best_idx)

        if search_over:
            logger.info(
                "Kuleshov2017: success (%d perturbations)",
                len(modified_indices),
            )
            return current_text

    logger.info(
        "Kuleshov2017: finished (%d perturbations)", len(modified_indices),
    )
    return current_text
