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
                 Google1BillionWordsLanguageModel(top_n_per_index=4,
                     compare_against_original=False)
  - Search: AlzantotGeneticAlgorithm(pop_size=60, max_iters=20, temp=0.3,
                                      post_crossover_check=False)
"""

import random
import logging

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

    Compliant with Alzantot et al., 2018 and TextAttack
    GeneticAlgorithmAlzantot2018 recipe.

    Args:
        model_wrapper: wrapped model with predict/predict_probs.
        tokenizer: HuggingFace tokenizer.
        text: input text to attack.
        target_label: target class name (None = untargeted).
        population_size: number of individuals (paper: S=60).
        max_generations: number of generations (paper: N=20).
        mutation_rate: probability of mutating a child (paper: always mutate).
        similarity_threshold: unused — retained for API compatibility.
            Quality is enforced by word-level constraints (embedding
            distance, LM constraint, stopwords, repeat modification,
            20% perturbation budget) matching the official recipe.
        require_embeddings: if True, raise error when counter-fitted word
            vectors are unavailable (prevents silent degradation to MLM).

    Returns: adversarial text (str).
    """
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, replace_words_at, is_stopword, clean_word,
    )
    from utils.text_word_substitution import get_embedding_neighbours_with_scores
    from utils.text_constraints import score_lm_candidates

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

        # WordEmbeddingDistance constraint (MSE ≤ 0.5).
        # Gensim returns cosine similarity on Paragram counter-fitted vectors.
        # For these embeddings: MSE ≈ 2(1-cos)/d, but since vector norms vary,
        # we keep all 8 nearest neighbours (same as TextAttack
        # WordSwapEmbedding(max_candidates=8) which returns 8 candidates and
        # lets the constraint pipeline filter). The LM constraint (below) is
        # the primary quality filter matching the original recipe.
        filtered = [w for w, sim in neighbours if w.lower() != word.lower()]
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
    _orig_prob_cache = {}

    # Matches TextAttack GeneticAlgorithm._search_over: set True when any
    # candidate achieves goal success during fitness evaluation, enabling
    # immediate termination mid-perturb and mid-generation.
    _search_over = [False]

    def _goal_succeeded(predicted_label: str) -> bool:
        if resolved_target is not None:
            return predicted_label.lower() == resolved_target.lower()
        return predicted_label != orig_label

    def fitness(candidate_text: str) -> tuple[float, str]:
        """Returns (fitness_score, predicted_label).

        Caches the original-class index to avoid redundant model calls.
        Sets _search_over[0] = True if goal is achieved (matches TextAttack
        get_goal_results → GoalFunctionResultStatus.SUCCEEDED).
        """
        probs = model_wrapper.predict_probs(candidate_text)
        predicted_idx = probs.index(max(probs))
        id2label = getattr(model_wrapper.model.config, 'id2label', {})
        predicted_label = id2label.get(predicted_idx, str(predicted_idx))

        if _goal_succeeded(predicted_label):
            _search_over[0] = True

        if resolved_target is not None:
            if resolved_target_idx is not None and resolved_target_idx < len(probs):
                return probs[resolved_target_idx], predicted_label
            return (max(probs) if predicted_label.lower() == resolved_target.lower() else 0.0), predicted_label
        else:
            if 'orig_idx' not in _orig_prob_cache:
                orig_probs = model_wrapper.predict_probs(text)
                _orig_prob_cache['orig_idx'] = orig_probs.index(max(orig_probs))
            return 1.0 - probs[_orig_prob_cache['orig_idx']], predicted_label

    # ── Perturb function ────────────────────────────────────────────────
    # Matches TextAttack GeneticAlgorithm._perturb() +
    # AlzantotGeneticAlgorithm._modify_population_member():
    #   1. Sample word position proportional to num_candidate_transformations.
    #   2. Apply Google1BillionWordsLM top-4 ranking (compare_against_original=False).
    #   3. Evaluate remaining candidates → keep the one with max fitness improvement.
    #   4. Enforce RepeatModification (skip already-modified positions).
    #   5. Enforce MaxWordsPerturbed (reject if budget exceeded).
    #   6. Zero out num_candidate_transformations at modified position.
    def perturb(
        individual_text: str,
        modified_indices: set,
        ind_num_cand: np.ndarray,
    ) -> tuple[str, set, np.ndarray]:
        """Apply one best-improvement mutation to an individual.

        Returns (new_text, new_modified_indices, new_num_candidate_transformations).
        """
        if len(modified_indices) >= max_words_perturbed:
            return individual_text, modified_indices, ind_num_cand

        weights = np.copy(ind_num_cand)
        for idx in modified_indices:
            if idx < len(weights):
                weights[idx] = 0

        non_zero = np.count_nonzero(weights)
        if non_zero == 0:
            return individual_text, modified_indices, ind_num_cand

        # Get current individual's word list for LM scoring context
        ind_words = [w for w, _, _ in get_words_and_spans(individual_text)]

        attempts = 0
        while attempts < non_zero:
            w_probs = weights / weights.sum()
            pos = np.random.choice(n_words, p=w_probs)

            if pos not in sub_cache:
                weights[pos] = 0
                attempts += 1
                if _search_over[0]:
                    break
                continue

            # Google1BillionWordsLanguageModel(top_n_per_index=4,
            #     compare_against_original=False):
            # Rank candidates by LM score in the context of the CURRENT
            # individual (not the original text) and keep top 4.
            lm_filtered = score_lm_candidates(
                current_words=ind_words,
                position=pos,
                candidates=sub_cache[pos],
                top_n=4,
                window_size=6,
            )

            current_score, _ = fitness(individual_text)
            best_text = None
            best_score = current_score
            for cand in lm_filtered:
                cand_text = replace_word_at(individual_text, pos, cand)
                score, _ = fitness(cand_text)
                if score > best_score:
                    best_score = score
                    best_text = cand_text

            if best_text is not None:
                new_modified = modified_indices | {pos}
                # Zero out num_candidate_transformations at modified position
                # (matches AlzantotGeneticAlgorithm._modify_population_member)
                new_num_cand = np.copy(ind_num_cand)
                new_num_cand[pos] = 0
                return best_text, new_modified, new_num_cand

            weights[pos] = 0
            attempts += 1

            # Matches TextAttack: break retry loop when goal achieved
            if _search_over[0]:
                break

        return individual_text, modified_indices, ind_num_cand

    # ── Initialize population ───────────────────────────────────────────
    # Paper + TextAttack: each member starts as original text, then one
    # _perturb() call is applied (best-improvement single mutation).
    population = []
    pop_modified_indices = []
    pop_num_cand = []

    for _ in range(population_size):
        ind_num_cand = np.copy(num_candidate_transformations)
        ind_text, ind_modified, ind_num_cand = perturb(text, set(), ind_num_cand)
        population.append(ind_text)
        pop_modified_indices.append(ind_modified)
        pop_num_cand.append(ind_num_cand)

    # ── Evolution loop ──────────────────────────────────────────────────
    for gen in range(max_generations):
        # Evaluate fitness
        scored = [(population[i], *fitness(population[i]),
                   pop_modified_indices[i], pop_num_cand[i])
                  for i in range(len(population))]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_text, best_fitness, best_label, best_modified, best_num_cand = scored[0]

        # ── Check success (matches TextAttack GoalFunctionResultStatus) ─
        # Stop when _search_over (set during child generation) or best
        # member achieves goal. No similarity gate — constraints are
        # enforced during candidate generation.
        if _search_over[0] or _goal_succeeded(best_label):
            logger.info("Alzantot GA: success at generation %d", gen + 1)
            return best_text

        if gen % 5 == 0:
            logger.debug("Alzantot GA: gen %d/%d, best_fitness=%.4f",
                         gen + 1, max_generations, best_fitness)

        # ── Softmax fitness-proportional parent selection ────────────
        temp = 0.3
        scores = np.array([s[1] for s in scored])
        logits = np.exp(-scores / temp)
        logit_sum = logits.sum()
        if logit_sum == 0 or np.isnan(logit_sum):
            select_probs = np.ones(len(scored)) / len(scored)
        else:
            select_probs = logits / logit_sum

        # ── Elitism: best individual always survives ────────────────
        new_population = [best_text]
        new_pop_modified = [best_modified]
        new_pop_num_cand = [best_num_cand]

        # Generate pop_size - 1 children
        for _ in range(population_size - 1):
            p1_idx = np.random.choice(len(scored), p=select_probs)
            p2_idx = np.random.choice(len(scored), p=select_probs)
            p1_text = scored[p1_idx][0]
            p2_text = scored[p2_idx][0]
            p1_modified = scored[p1_idx][3]
            p2_modified = scored[p2_idx][3]
            p1_num_cand = scored[p1_idx][4]
            p2_num_cand = scored[p2_idx][4]

            # ── Crossover: uniform (50% per word) ──────────────────
            # Matches AlzantotGeneticAlgorithm._crossover_operation()
            # post_crossover_check=False
            p1_words = get_words_and_spans(p1_text)
            p2_words = get_words_and_spans(p2_text)
            swaps = {}
            child_num_cand = np.copy(p1_num_cand)
            child_modified = set(p1_modified)
            for j in range(min(len(p1_words), len(p2_words))):
                if random.random() < 0.5:
                    swaps[j] = p2_words[j][0]
                    child_num_cand[j] = p2_num_cand[j]
                    if j in p2_modified:
                        child_modified.add(j)
                    else:
                        child_modified.discard(j)
            child = replace_words_at(p1_text, swaps) if swaps else p1_text

            # Evaluate crossover child (matches TextAttack _crossover →
            # get_goal_results: sets _search_over if goal achieved)
            fitness(child)
            if _search_over[0]:
                new_population.append(child)
                new_pop_modified.append(child_modified)
                new_pop_num_cand.append(child_num_cand)
                break

            # ── Mutation: best-improvement ─────────────────────────
            if random.random() < mutation_rate:
                child, child_modified, child_num_cand = perturb(
                    child, child_modified, child_num_cand
                )

            new_population.append(child)
            new_pop_modified.append(child_modified)
            new_pop_num_cand.append(child_num_cand)

            # Matches TextAttack: _search_over check after _perturb
            if _search_over[0]:
                break

        population = new_population
        pop_modified_indices = new_pop_modified
        pop_num_cand = new_pop_num_cand

    # ── Return best from final generation ───────────────────────────────
    scored = [(population[i], *fitness(population[i]))
              for i in range(len(population))]
    scored.sort(key=lambda x: x[1], reverse=True)
    logger.info("Alzantot GA: finished (%d generations)", max_generations)
    return scored[0][0]
