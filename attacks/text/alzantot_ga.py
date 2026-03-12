"""
Alzantot Genetic Algorithm Attack — Alzantot et al., 2018 (arXiv:1804.07998)

Evolutionary search: maintains a population of perturbed texts,
applies crossover + mutation (word substitution from embedding neighbours),
selects by fitness (classification confidence on target class).
"""

import random
import logging

logger = logging.getLogger("textattack.attacks.alzantot_ga")


def run_alzantot_ga(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    population_size: int = 20,
    max_generations: int = 40,
    mutation_rate: float = 0.3,
    similarity_threshold: float = 0.8,
) -> str:
    """Alzantot genetic algorithm attack.

    Returns: adversarial text (str).
    """
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, replace_words_at, is_stopword, clean_word,
    )
    from utils.text_word_substitution import get_mlm_substitutions
    from utils.text_constraints import compute_semantic_similarity

    logger.info("Alzantot GA: starting (pop=%d, gen=%d, mut=%.2f)",
                population_size, max_generations, mutation_rate)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    words = [w for w, _, _ in words_spans]
    n_words = len(words)
    orig_label, orig_conf, _ = model_wrapper.predict(text)

    # Resolve target label once (not inside fitness function)
    resolved_target = target_label
    resolved_target_idx = None
    if target_label is not None:
        from models.text_loader import resolve_target_label, get_label_index
        resolved_target = resolve_target_label(model_wrapper.model, target_label) or target_label
        resolved_target_idx = get_label_index(model_wrapper.model, resolved_target)

    # Pre-compute substitution candidates for each position
    sub_cache = {}
    mutable_positions = []
    for i, word in enumerate(words):
        if is_stopword(word) or len(clean_word(word)) <= 1:
            continue
        candidates = get_mlm_substitutions(text, i, top_k=20)
        if candidates:
            sub_cache[i] = candidates
            mutable_positions.append(i)

    if not mutable_positions:
        return text

    # Fitness function
    def fitness(candidate_text: str) -> tuple[float, str]:
        """Returns (fitness_score, predicted_label)."""
        probs = model_wrapper.predict_probs(candidate_text)
        predicted_idx = probs.index(max(probs))
        # Derive label from model config
        id2label = getattr(model_wrapper.model.config, 'id2label', {})
        predicted_label = id2label.get(predicted_idx, str(predicted_idx))

        if resolved_target is not None:
            # Targeted: maximise target class confidence
            if resolved_target_idx is not None and resolved_target_idx < len(probs):
                return probs[resolved_target_idx], predicted_label
            return (max(probs) if predicted_label.lower() == resolved_target.lower() else 0.0), predicted_label
        else:
            # Untargeted: minimise original class confidence
            orig_probs = model_wrapper.predict_probs(text)
            orig_idx = orig_probs.index(max(orig_probs))
            return 1.0 - probs[orig_idx], predicted_label

    # Initialize population with random single-word mutations
    population = []
    for _ in range(population_size):
        pos = random.choice(mutable_positions)
        cand = random.choice(sub_cache[pos])
        individual = replace_word_at(text, pos, cand)
        population.append(individual)

    for gen in range(max_generations):
        # Evaluate fitness
        scored = [(ind, *fitness(ind)) for ind in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Check if best individual succeeds
        best_text, best_fitness, best_label = scored[0]

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

        if gen % 10 == 0:
            logger.debug("Alzantot GA: gen %d/%d, best_fitness=%.4f", gen + 1, max_generations, best_fitness)

        # Selection: top half survives
        survivors = [ind for ind, _, _ in scored[:population_size // 2]]

        # Crossover + mutation to fill population
        new_population = list(survivors)
        while len(new_population) < population_size:
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)

            # Crossover: start from parent1, selectively swap words from parent2
            p1_words = get_words_and_spans(p1)
            p2_words = get_words_and_spans(p2)
            # Use parent1 as base text and swap in words from parent2
            swaps = {}
            for j in range(min(len(p1_words), len(p2_words))):
                if random.random() < 0.5:
                    swaps[j] = p2_words[j][0]
            child = replace_words_at(p1, swaps) if swaps else p1

            # Mutation
            if random.random() < mutation_rate and mutable_positions:
                pos = random.choice(mutable_positions)
                child_spans = get_words_and_spans(child)
                if pos < len(child_spans) and pos in sub_cache:
                    cand = random.choice(sub_cache[pos])
                    child = replace_word_at(child, pos, cand)

            new_population.append(child)

        population = new_population

    # Return best from final generation
    scored = [(ind, *fitness(ind)) for ind in population]
    scored.sort(key=lambda x: x[1], reverse=True)
    logger.info("Alzantot GA: finished (%d generations)", max_generations)
    return scored[0][0]
