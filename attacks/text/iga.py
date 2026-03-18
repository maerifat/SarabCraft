"""
IGA (Improved Genetic Algorithm) — Wang et al., 2019 (arXiv:1909.06723)

Reimplementation of TextAttack ImprovedGeneticAlgorithm + IGAWang2019 recipe.

Key improvements over Alzantot GA:
  1. Population initialized by replacing each word by its optimal synonym
  2. max_replace_times_per_index (λ=5): words can be replaced multiple times;
     remaining replacement budget used as word selection probability
  3. Single-point crossover (vs uniform): random cut point, concat fragments

Inherited GA mechanics (from TextAttack GeneticAlgorithm base):
  - Softmax fitness-proportional parent selection (T=0.3)
  - Best-improvement mutation (_perturb)
  - Single-elite preservation (population = [best] + children)

Reference: TextAttack IGAWang2019 recipe.
  - Transformation: WordSwapEmbedding(max_candidates=50)
  - Constraints: StopwordModification, MaxWordsPerturbed(20%),
                 WordEmbeddingDistance(MSE≤0.5, compare_against_original=False)
  - Search: ImprovedGeneticAlgorithm(pop_size=60, max_iters=20,
            max_replace_times_per_index=5, post_crossover_check=False)
"""

import random
import logging
import math

import numpy as np

logger = logging.getLogger("textattack.attacks.iga")


def run_iga(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    population_size: int = 60,
    max_generations: int = 20,
    max_replace_times_per_index: int = 5,
    max_perturbation_ratio: float = 0.2,
    temp: float = 0.3,
    require_embeddings: bool = False,
) -> str:
    """Improved Genetic Algorithm attack.

    Compliant with Wang et al., 2019 and TextAttack IGAWang2019 recipe.

    Args:
        model_wrapper: wrapped model with predict/predict_probs.
        tokenizer: HuggingFace tokenizer.
        text: input text to attack.
        target_label: target class name (None = untargeted).
        population_size: number of individuals (paper: S=60).
        max_generations: number of generations (paper: M=20).
        max_replace_times_per_index: max times each word can be replaced (paper: λ=5).
        max_perturbation_ratio: max fraction of words to perturb (paper: 20%).
        temp: softmax temperature for parent selection (paper: T=0.3).
        require_embeddings: if True, raise error when counter-fitted word
            vectors are unavailable (prevents silent degradation to MLM).

    Returns: adversarial text (str).
    """
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, replace_words_at, is_stopword, clean_word,
    )
    from utils.text_word_substitution import get_embedding_neighbours_with_scores

    logger.info("IGA: starting (pop=%d, gen=%d, temp=%.1f, λ=%d)",
                population_size, max_generations, temp, max_replace_times_per_index)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    words = [w for w, _, _ in words_spans]
    n_words = len(words)
    max_words_perturbed = max(1, int(math.ceil(n_words * max_perturbation_ratio)))

    orig_label, orig_conf, _ = model_wrapper.predict(text)

    # ── Resolve target label ────────────────────────────────────────────
    resolved_target = target_label
    resolved_target_idx = None
    if target_label is not None:
        from models.text_loader import resolve_target_label, get_label_index
        resolved_target = resolve_target_label(model_wrapper.model, target_label) or target_label
        resolved_target_idx = get_label_index(model_wrapper.model, resolved_target)

    # ── Embedding-based substitution candidates ─────────────────────────
    # Paper: WordSwapEmbedding(max_candidates=50), counter-fitted Paragram
    # WordEmbeddingDistance(max_mse_dist=0.5, compare_against_original=False):
    #   candidates filtered relative to CURRENT word (not original).
    # Word-based cache: keyed by cleaned word so re-substituted words
    # get fresh candidates, matching compare_against_original=False.
    _candidate_cache: dict[str, list[str]] = {}
    _mlm_fallback_warned = False

    def get_candidates(word: str) -> list[str]:
        """Get filtered embedding neighbours for a word (cached by word)."""
        nonlocal _mlm_fallback_warned
        key = clean_word(word)
        if not key or len(key) <= 1:
            return []
        if key in _candidate_cache:
            return _candidate_cache[key]

        neighbours = get_embedding_neighbours_with_scores(key, top_k=50)

        # Detect silent MLM fallback (MLM returns sim=1.0 for all)
        if neighbours and all(sim == 1.0 for _, sim in neighbours):
            if require_embeddings:
                raise RuntimeError(
                    "IGA requires counter-fitted word vectors for compliance. "
                    "Download Paragram vectors to ~/.textattack/embedding/paragramcf "
                    "or set require_embeddings=False to allow MLM fallback."
                )
            if not _mlm_fallback_warned:
                logger.warning(
                    "IGA: using MLM fallback — WordEmbeddingDistance constraint "
                    "will NOT be enforced."
                )
                _mlm_fallback_warned = True

        # WordEmbeddingDistance: cosine ≥ 0.5 (≈ MSE ≤ 0.5)
        filtered = [w for w, sim in neighbours if sim >= 0.5 and w.lower() != key]
        _candidate_cache[key] = filtered
        return filtered

    # ── Fitness function ────────────────────────────────────────────────
    _orig_idx_cache = {}

    def fitness(candidate_text: str) -> tuple[float, str]:
        """Returns (fitness_score, predicted_label).

        Untargeted: score = 1 − P(y_orig), want to maximise.
        Targeted:   score = P(y_target), want to maximise.
        """
        probs = model_wrapper.predict_probs(candidate_text)
        predicted_idx = probs.index(max(probs))
        id2label = getattr(model_wrapper.model.config, 'id2label', {})
        predicted_label = id2label.get(predicted_idx, str(predicted_idx))

        if resolved_target is not None:
            if resolved_target_idx is not None and resolved_target_idx < len(probs):
                return probs[resolved_target_idx], predicted_label
            return (max(probs) if predicted_label.lower() == resolved_target.lower() else 0.0), predicted_label
        else:
            if 'idx' not in _orig_idx_cache:
                orig_probs = model_wrapper.predict_probs(text)
                _orig_idx_cache['idx'] = orig_probs.index(max(orig_probs))
            return 1.0 - probs[_orig_idx_cache['idx']], predicted_label

    def is_success(label: str) -> bool:
        if resolved_target is not None:
            return label.lower() == resolved_target.lower()
        return label != orig_label

    # ── Best-improvement perturbation ───────────────────────────────────
    # Matches TextAttack GeneticAlgorithm._perturb():
    #   1. Select word weighted by num_replacements_left
    #   2. Get all valid substitutions for the CURRENT word at that position
    #   3. Evaluate each → keep best-improvement
    #   4. If no improvement, zero weight and retry another position

    def perturb(ind_text, nrl, modified_indices, index=None):
        """Apply one best-improvement mutation.

        Args:
            ind_text: current text of the individual.
            nrl: per-word num_replacements_left array.
            modified_indices: set of word indices modified vs original.
            index: if specified, perturb at this specific index only.

        Returns: (text, nrl, modified_indices) — updated.
        """
        weights = nrl.astype(float).copy()

        # StopwordModification: zero out stopwords and short words
        for i in range(min(len(weights), n_words)):
            w = words[i]
            if is_stopword(w) or len(clean_word(w)) <= 1:
                weights[i] = 0

        # MaxWordsPerturbed: if at budget, can only re-modify existing
        if len(modified_indices) >= max_words_perturbed:
            for i in range(len(weights)):
                if i not in modified_indices:
                    weights[i] = 0

        non_zero = int(np.count_nonzero(weights))
        if non_zero == 0:
            return ind_text, nrl, modified_indices

        current_score, _ = fitness(ind_text)
        cur_words_spans = get_words_and_spans(ind_text)

        attempts = 0
        while attempts < non_zero:
            if index is not None:
                idx = index
            else:
                w_sum = weights.sum()
                if w_sum == 0:
                    break
                w_probs = weights / w_sum
                idx = np.random.choice(len(weights), p=w_probs)

            if idx >= len(cur_words_spans):
                weights[idx] = 0
                attempts += 1
                if index is not None:
                    break
                continue

            # Get candidates for the CURRENT word (compare_against_original=False)
            cur_word = cur_words_spans[idx][0]
            candidates = get_candidates(cur_word)

            if not candidates:
                weights[idx] = 0
                attempts += 1
                if index is not None:
                    break
                continue

            # Evaluate all candidates, pick best-improvement
            best_text = None
            best_score = current_score
            for cand in candidates:
                cand_text = replace_word_at(ind_text, idx, cand)
                score, _ = fitness(cand_text)
                if score > best_score:
                    best_score = score
                    best_text = cand_text

            if best_text is not None:
                new_nrl = nrl.copy()
                new_nrl[idx] -= 1
                new_mod = modified_indices | {idx}
                return best_text, new_nrl, new_mod

            # No improvement — zero out and try another position
            weights[idx] = 0
            attempts += 1

            if index is not None:
                break

        return ind_text, nrl, modified_indices

    # ── Initialize population ───────────────────────────────────────────
    # IGA: each member = original text with ONE word replaced by its
    # optimal synonym (best-improvement at specific index).
    # Creates up to n_words members, truncated to population_size.
    # Matches TextAttack ImprovedGeneticAlgorithm._initialize_population()

    logger.info("IGA: initializing population (one member per word, up to %d)", population_size)

    init_nrl = np.full(n_words, max_replace_times_per_index, dtype=int)

    population = []       # list of texts
    pop_nrl = []           # list of num_replacements_left arrays
    pop_modified = []      # list of modified_indices sets
    pop_scores = []        # list of fitness scores
    pop_labels = []        # list of predicted labels

    for idx in range(n_words):
        member_text, member_nrl, member_mod = perturb(
            text, init_nrl.copy(), set(), index=idx
        )
        score, label = fitness(member_text)

        population.append(member_text)
        pop_nrl.append(member_nrl)
        pop_modified.append(member_mod)
        pop_scores.append(score)
        pop_labels.append(label)

        # Early exit if already successful
        if is_success(label):
            logger.info("IGA: success during initialization at word %d", idx)
            return member_text

    # Truncate to population_size
    population = population[:population_size]
    pop_nrl = pop_nrl[:population_size]
    pop_modified = pop_modified[:population_size]
    pop_scores = pop_scores[:population_size]
    pop_labels = pop_labels[:population_size]
    pop_size = len(population)

    if pop_size == 0:
        return text

    # ── Evolution loop ──────────────────────────────────────────────────
    current_score = 0.0  # initial_result.score (original text score)

    for gen in range(max_generations):
        # Sort population by fitness (descending)
        order = sorted(range(pop_size), key=lambda i: pop_scores[i], reverse=True)
        population = [population[i] for i in order]
        pop_nrl = [pop_nrl[i] for i in order]
        pop_modified = [pop_modified[i] for i in order]
        pop_scores = [pop_scores[i] for i in order]
        pop_labels = [pop_labels[i] for i in order]

        # Check success (best individual)
        if is_success(pop_labels[0]):
            logger.info("IGA: success at generation %d", gen + 1)
            return population[0]

        # Give-up check (TextAttack: give_up_if_no_improvement=False by default)
        if pop_scores[0] > current_score:
            current_score = pop_scores[0]

        if gen % 5 == 0:
            logger.debug("IGA: gen %d/%d, best_fitness=%.4f",
                         gen + 1, max_generations, pop_scores[0])

        # ── Softmax fitness-proportional parent selection ────────────
        # Matches TextAttack: logits = exp(-scores / temp)
        scores_arr = np.array(pop_scores, dtype=float)
        logits = np.exp(-scores_arr / temp)
        logit_sum = logits.sum()
        if logit_sum == 0 or np.isnan(logit_sum):
            select_probs = np.ones(pop_size) / pop_size
        else:
            select_probs = logits / logit_sum

        # Pre-sample all parent indices (matches TextAttack)
        parent1_indices = np.random.choice(pop_size, size=pop_size - 1, p=select_probs)
        parent2_indices = np.random.choice(pop_size, size=pop_size - 1, p=select_probs)

        # ── Generate children ───────────────────────────────────────
        children = []
        children_nrl = []
        children_modified = []
        children_scores = []
        children_labels = []

        for c_idx in range(pop_size - 1):
            p1_i = parent1_indices[c_idx]
            p2_i = parent2_indices[c_idx]

            # ── Single-point crossover (IGA innovation) ─────────────
            # Random cut point; words [0..cut) from parent1, [cut..end) from parent2
            p1_words_spans = get_words_and_spans(population[p1_i])
            p2_words_spans = get_words_and_spans(population[p2_i])
            child_n = min(len(p1_words_spans), len(p2_words_spans))

            if child_n > 0:
                crossover_point = random.randint(0, child_n - 1)
            else:
                crossover_point = 0

            # Build child: parent1 words [0..cut), parent2 words [cut..end)
            child_nrl = pop_nrl[p1_i].copy()
            child_mod = set(pop_modified[p1_i])
            swaps = {}
            for j in range(crossover_point, child_n):
                swaps[j] = p2_words_spans[j][0]
                child_nrl[j] = pop_nrl[p2_i][j]
                # Track modifications from parent2
                if j in pop_modified[p2_i]:
                    child_mod.add(j)
                else:
                    child_mod.discard(j)

            child_text = replace_words_at(population[p1_i], swaps) if swaps else population[p1_i]

            # ── Mutation: best-improvement perturbation ─────────────
            # TextAttack: always mutate (no mutation_rate parameter)
            child_text, child_nrl, child_mod = perturb(
                child_text, child_nrl, child_mod
            )

            score, label = fitness(child_text)

            children.append(child_text)
            children_nrl.append(child_nrl)
            children_modified.append(child_mod)
            children_scores.append(score)
            children_labels.append(label)

            # Check for early success
            if is_success(label):
                logger.info("IGA: success at generation %d (during breeding)", gen + 1)
                return child_text

        # ── Elitism: population = [best] + children ─────────────────
        # Matches TextAttack: population = [population[0]] + children
        population = [population[0]] + children
        pop_nrl = [pop_nrl[0]] + children_nrl
        pop_modified = [pop_modified[0]] + children_modified
        pop_scores = [pop_scores[0]] + children_scores
        pop_labels = [pop_labels[0]] + children_labels
        pop_size = len(population)

    # Return best from final population
    best_idx = max(range(pop_size), key=lambda i: pop_scores[i])
    logger.info("IGA: finished (%d generations)", max_generations)
    return population[best_idx]
