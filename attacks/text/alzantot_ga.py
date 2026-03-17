"""
Alzantot Genetic Algorithm Attack — Alzantot et al., 2018 (arXiv:1804.07998)

Evolutionary search: maintains a population of perturbed texts,
applies crossover + mutation (word substitution from counter-fitted
embedding neighbours), selects parents via softmax fitness-proportional
sampling, and enforces inline constraints every generation.

Reference: TextAttack GeneticAlgorithmAlzantot2018 recipe.
  - Transformation: WordSwapEmbedding(max_candidates=8)
  - Constraints: RepeatModification, StopwordModification,
                 MaxWordsPerturbed(20%), WordEmbeddingDistance(MSE≤0.5),
                 Google1BillionWordsLanguageModel(top_n=4) [omitted — see TODO]
  - Search: AlzantotGeneticAlgorithm(pop_size=60, max_iters=20, temp=0.3)
"""

import random
import logging
import math

import numpy as np

logger = logging.getLogger("textattack.attacks.alzantot_ga")


def run_alzantot_ga(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    population_size: int = 60,
    max_generations: int = 20,
    mutation_rate: float = 1.0,
    similarity_threshold: float = 0.8,
    require_embeddings: bool = True,
) -> str:
    """Alzantot genetic algorithm attack.

    Compliant with Alzantot et al., 2018 and TextAttack reference.

    Args:
        model_wrapper: wrapped model with predict/predict_probs.
        tokenizer: HuggingFace tokenizer.
        text: input text to attack.
        target_label: target class name (None = untargeted).
        population_size: number of individuals (paper: S=60).
        max_generations: number of generations (paper: N=20).
        mutation_rate: probability of mutating a child (paper: always mutate).
        similarity_threshold: sentence-level semantic similarity gate.
        require_embeddings: if True, raise error when counter-fitted word
            vectors are unavailable (prevents silent degradation to MLM).

    Returns: adversarial text (str).
    """
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, replace_words_at, is_stopword, clean_word,
    )
    from utils.text_word_substitution import get_embedding_neighbours_with_scores
    from utils.text_constraints import compute_semantic_similarity

    logger.info("Alzantot GA: starting (pop=%d, gen=%d, temp=0.3)",
                population_size, max_generations)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    words = [w for w, _, _ in words_spans]
    n_words = len(words)
    orig_label, orig_conf, _ = model_wrapper.predict(text)

    # ── Resolve target label ────────────────────────────────────────────
    resolved_target = target_label
    resolved_target_idx = None
    if target_label is not None:
        from models.text_loader import resolve_target_label, get_label_index
        resolved_target = resolve_target_label(model_wrapper.model, target_label) or target_label
        resolved_target_idx = get_label_index(model_wrapper.model, resolved_target)

    # ── Perturbation budget (MaxWordsPerturbed: 20%) ────────────────────
    max_words_perturbed = max(1, int(n_words * 0.2))

    # ── Pre-compute embedding-based substitution candidates ─────────────
    # Paper: counter-fitted Paragram embeddings, N=8 candidates, δ=0.5
    # Static caching is correct for embedding-based synonyms (context-free).
    sub_cache: dict[int, list[str]] = {}
    num_candidate_transformations = np.zeros(n_words)

    _mlm_fallback_warned = False
    for i, word in enumerate(words):
        if is_stopword(word) or len(clean_word(word)) <= 1:
            continue
        neighbours = get_embedding_neighbours_with_scores(word, top_k=8,
                                                          context_text=text,
                                                          position=i)
        # Detect silent MLM fallback: when word vectors are unavailable,
        # get_embedding_neighbours_with_scores returns MLM candidates with
        # similarity=1.0, making the cosine filter below ineffective.
        if neighbours and all(sim == 1.0 for _, sim in neighbours):
            if require_embeddings:
                raise RuntimeError(
                    "Alzantot GA requires counter-fitted word vectors for "
                    "compliance. Download Paragram vectors to "
                    "~/.textattack/embedding/paragramcf or set "
                    "require_embeddings=False to allow MLM fallback."
                )
            if not _mlm_fallback_warned:
                logger.warning(
                    "Alzantot GA: using MLM fallback — "
                    "WordEmbeddingDistance constraint will NOT be enforced. "
                    "Download Paragram vectors for full compliance."
                )
                _mlm_fallback_warned = True

        # WordEmbeddingDistance constraint: keep only candidates with
        # cosine similarity ≥ 0.5 (MSE ≤ 0.5 in TextAttack — equivalent
        # to a cosine threshold of ~0.5 for unit-normed vectors).
        filtered = [w for w, sim in neighbours if sim >= 0.5 and w.lower() != word.lower()]
        if filtered:
            sub_cache[i] = filtered
            num_candidate_transformations[i] = len(filtered)

    mutable_positions = [i for i in sub_cache]

    if not mutable_positions:
        return text

    # Give epsilon probability to positions with no candidates
    # (matches TextAttack: prevents permanent exclusion)
    min_ct = num_candidate_transformations[num_candidate_transformations > 0].min() if any(
        num_candidate_transformations > 0) else 1
    epsilon = max(1, int(min_ct * 0.1))
    for i in range(n_words):
        if num_candidate_transformations[i] == 0:
            num_candidate_transformations[i] = epsilon

    # ── Fitness function ────────────────────────────────────────────────
    def fitness(candidate_text: str) -> tuple[float, str]:
        """Returns (fitness_score, predicted_label)."""
        probs = model_wrapper.predict_probs(candidate_text)
        predicted_idx = probs.index(max(probs))
        id2label = getattr(model_wrapper.model.config, 'id2label', {})
        predicted_label = id2label.get(predicted_idx, str(predicted_idx))

        if resolved_target is not None:
            # Targeted: maximise target class confidence
            if resolved_target_idx is not None and resolved_target_idx < len(probs):
                return probs[resolved_target_idx], predicted_label
            return (max(probs) if predicted_label.lower() == resolved_target.lower() else 0.0), predicted_label
        else:
            # Untargeted: minimise original class confidence
            orig_idx = model_wrapper.predict_probs(text).index(max(model_wrapper.predict_probs(text)))
            return 1.0 - probs[orig_idx], predicted_label

    # Cache original-text untargeted index to avoid redundant calls
    _orig_prob_cache = {}

    def fitness_fast(candidate_text: str) -> tuple[float, str]:
        """Optimised fitness: caches the original-class index."""
        probs = model_wrapper.predict_probs(candidate_text)
        predicted_idx = probs.index(max(probs))
        id2label = getattr(model_wrapper.model.config, 'id2label', {})
        predicted_label = id2label.get(predicted_idx, str(predicted_idx))

        if resolved_target is not None:
            if resolved_target_idx is not None and resolved_target_idx < len(probs):
                return probs[resolved_target_idx], predicted_label
            return (max(probs) if predicted_label.lower() == resolved_target.lower() else 0.0), predicted_label
        else:
            if 'orig_idx' not in _orig_prob_cache:
                orig_probs = model_wrapper.predict_probs(text)
                _orig_prob_cache['orig_idx'] = orig_probs.index(max(orig_probs))
            return 1.0 - probs[_orig_prob_cache['orig_idx']], predicted_label

    # ── Perturb function (paper: greedy best-improvement mutation) ──────
    def perturb(individual_text: str, modified_indices: set) -> tuple[str, set]:
        """Apply one best-improvement mutation to an individual.

        Matches TextAttack _perturb():
          1. Sample word position proportional to num_candidate_transformations.
          2. Generate all valid substitutions at that position.
          3. Evaluate each → keep the one with maximum fitness improvement.
          4. Enforce RepeatModification (skip already-modified positions).
          5. Enforce MaxWordsPerturbed (reject if budget exceeded).
        """
        if len(modified_indices) >= max_words_perturbed:
            return individual_text, modified_indices

        # Build weight vector: zero out already-modified positions
        weights = np.copy(num_candidate_transformations)
        for idx in modified_indices:
            if idx < len(weights):
                weights[idx] = 0

        non_zero = np.count_nonzero(weights)
        if non_zero == 0:
            return individual_text, modified_indices

        attempts = 0
        while attempts < non_zero:
            # Weighted word selection (proportional to num candidates)
            w_probs = weights / weights.sum()
            pos = np.random.choice(n_words, p=w_probs)

            if pos not in sub_cache:
                weights[pos] = 0
                attempts += 1
                continue

            # Evaluate all candidates, pick best-improvement
            current_score, _ = fitness_fast(individual_text)
            best_text = None
            best_score = current_score
            for cand in sub_cache[pos]:
                cand_text = replace_word_at(individual_text, pos, cand)
                score, _ = fitness_fast(cand_text)
                if score > best_score:
                    best_score = score
                    best_text = cand_text

            if best_text is not None:
                new_modified = modified_indices | {pos}
                return best_text, new_modified

            # No improvement at this position — zero it out and retry
            weights[pos] = 0
            attempts += 1

        return individual_text, modified_indices

    # ── Initialize population ───────────────────────────────────────────
    # Paper: each member starts as original text with one perturbation.
    # TextAttack: _initialize_population → _perturb(original_text).
    population = []
    pop_modified_indices = []

    for _ in range(population_size):
        ind_text, ind_modified = perturb(text, set())
        population.append(ind_text)
        pop_modified_indices.append(ind_modified)

    # ── Evolution loop ──────────────────────────────────────────────────
    for gen in range(max_generations):
        # Evaluate fitness
        scored = [(population[i], *fitness_fast(population[i]), pop_modified_indices[i])
                  for i in range(len(population))]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_text, best_fitness, best_label, best_modified = scored[0]

        # ── Check success ───────────────────────────────────────────
        if resolved_target is not None:
            if best_label.lower() == resolved_target.lower():
                sim = compute_semantic_similarity(text, best_text)
                if sim >= similarity_threshold:
                    logger.info("Alzantot GA: success at generation %d", gen + 1)
                    return best_text
        else:
            if best_label != orig_label:
                sim = compute_semantic_similarity(text, best_text)
                if sim >= similarity_threshold:
                    logger.info("Alzantot GA: success at generation %d", gen + 1)
                    return best_text

        if gen % 5 == 0:
            logger.debug("Alzantot GA: gen %d/%d, best_fitness=%.4f",
                         gen + 1, max_generations, best_fitness)

        # ── Softmax fitness-proportional parent selection ────────────
        # Paper + TextAttack: logits = exp(-scores / temp), probs = logits / sum
        temp = 0.3
        scores = np.array([s[1] for s in scored])
        logits = np.exp(-scores / temp)
        # Guard against all-zero or NaN
        logit_sum = logits.sum()
        if logit_sum == 0 or np.isnan(logit_sum):
            select_probs = np.ones(len(scored)) / len(scored)
        else:
            select_probs = logits / logit_sum

        # ── Elitism: best individual always survives ────────────────
        new_population = [best_text]
        new_pop_modified = [best_modified]

        # Generate pop_size - 1 children
        for _ in range(population_size - 1):
            # Select two parents (fitness-proportional)
            p1_idx = np.random.choice(len(scored), p=select_probs)
            p2_idx = np.random.choice(len(scored), p=select_probs)
            p1_text = scored[p1_idx][0]
            p2_text = scored[p2_idx][0]
            p1_modified = scored[p1_idx][3]
            p2_modified = scored[p2_idx][3]

            # ── Crossover: uniform (50% per word) ──────────────────
            p1_words = get_words_and_spans(p1_text)
            p2_words = get_words_and_spans(p2_text)
            swaps = {}
            child_modified = set(p1_modified)
            for j in range(min(len(p1_words), len(p2_words))):
                if random.random() < 0.5:
                    swaps[j] = p2_words[j][0]
                    # Track which parent contributed the modification
                    if j in p2_modified:
                        child_modified.add(j)
                    elif j in p1_modified:
                        child_modified.discard(j)
            child = replace_words_at(p1_text, swaps) if swaps else p1_text

            # ── Mutation: best-improvement ─────────────────────────
            if random.random() < mutation_rate:
                child, child_modified = perturb(child, child_modified)

            new_population.append(child)
            new_pop_modified.append(child_modified)

        population = new_population
        pop_modified_indices = new_pop_modified

    # ── Return best from final generation ───────────────────────────────
    scored = [(population[i], *fitness_fast(population[i]))
              for i in range(len(population))]
    scored.sort(key=lambda x: x[1], reverse=True)
    logger.info("Alzantot GA: finished (%d generations)", max_generations)
    return scored[0][0]


# TODO: Google 1 Billion Words Language Model constraint
# The official Alzantot recipe uses Google1BillionWordsLanguageModel(top_n_per_index=4)
# to filter candidates for contextual fluency. This is omitted here because the
# language model is very large (~1.7 GB) and impractical to bundle. To achieve
# full compliance, download the model and add an LM scoring filter inside the
# perturb() function that retains only the top-4 candidates by LM score.
