"""
Faster Alzantot Genetic Algorithm — Jia et al., 2019 (arXiv:1909.00986)

Optimized genetic algorithm 10-20x faster than original Alzantot.
Uses MLM-based (BERT) word substitutions for faster candidate generation.
Trades algorithmic compliance for speed (see alzantot_ga.py for the
compliant implementation).
"""

import random
import logging

logger = logging.getLogger("textattack.attacks.faster_alzantot_ga")


def run_faster_alzantot_ga(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    population_size: int = 60,
    max_generations: int = 20,
    max_perturbation_ratio: float = 0.2,
) -> str:
    """Faster Alzantot genetic algorithm attack.

    Returns: adversarial text (str).
    """
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, replace_words_at, is_stopword, clean_word,
    )
    from utils.text_word_substitution import get_mlm_substitutions
    from utils.text_constraints import compute_semantic_similarity

    logger.info("Faster Alzantot GA: starting (pop=%d, gen=%d, max_pert=%.2f)",
                population_size, max_generations, max_perturbation_ratio)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    words = [w for w, _, _ in words_spans]
    n_words = len(words)
    max_perturb_words = max(1, int(n_words * max_perturbation_ratio))
    
    orig_label, orig_conf, _ = model_wrapper.predict(text)

    # Resolve target label
    resolved_target = target_label
    resolved_target_idx = None
    if target_label is not None:
        from models.text_loader import resolve_target_label, get_label_index
        resolved_target = resolve_target_label(model_wrapper.model, target_label) or target_label
        resolved_target_idx = get_label_index(model_wrapper.model, resolved_target)

    # Pre-compute substitution candidates (faster with larger candidate pool)
    sub_cache = {}
    mutable_positions = []
    for i, word in enumerate(words):
        if is_stopword(word) or len(clean_word(word)) <= 1:
            continue
        candidates = get_mlm_substitutions(text, i, top_k=50)  # More candidates
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
        id2label = getattr(model_wrapper.model.config, 'id2label', {})
        predicted_label = id2label.get(predicted_idx, str(predicted_idx))

        if resolved_target is not None:
            if resolved_target_idx is not None and resolved_target_idx < len(probs):
                return probs[resolved_target_idx], predicted_label
            return (max(probs) if predicted_label.lower() == resolved_target.lower() else 0.0), predicted_label
        else:
            orig_probs = model_wrapper.predict_probs(text)
            orig_idx = orig_probs.index(max(orig_probs))
            return 1.0 - probs[orig_idx], predicted_label

    # Initialize population with diverse mutations
    population = []
    for _ in range(population_size):
        # Random number of mutations (1 to max_perturb_words)
        num_muts = random.randint(1, min(max_perturb_words, len(mutable_positions)))
        positions = random.sample(mutable_positions, num_muts)
        
        swaps = {}
        for pos in positions:
            swaps[pos] = random.choice(sub_cache[pos])
        
        individual = replace_words_at(text, swaps)
        population.append(individual)

    best_overall = text
    best_overall_fitness = 0.0

    for gen in range(max_generations):
        # Evaluate fitness
        scored = [(ind, *fitness(ind)) for ind in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Track best
        best_text, best_fitness, best_label = scored[0]
        if best_fitness > best_overall_fitness:
            best_overall = best_text
            best_overall_fitness = best_fitness

        # Check success
        if resolved_target is not None:
            if best_label.lower() == resolved_target.lower():
                logger.info("Faster Alzantot GA: success at generation %d", gen + 1)
                return best_text
        else:
            if best_label != orig_label:
                logger.info("Faster Alzantot GA: success at generation %d", gen + 1)
                return best_text

        if gen % 5 == 0:
            logger.debug("Faster Alzantot GA: gen %d/%d, best_fitness=%.4f", 
                        gen + 1, max_generations, best_fitness)

        # Selection: top 30% survive
        elite_size = max(2, population_size // 3)
        survivors = [ind for ind, _, _ in scored[:elite_size]]

        # Generate new population
        new_population = list(survivors)
        
        while len(new_population) < population_size:
            # Tournament selection
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)

            # Crossover
            p1_words = get_words_and_spans(p1)
            p2_words = get_words_and_spans(p2)
            swaps = {}
            for j in range(min(len(p1_words), len(p2_words))):
                if random.random() < 0.5:
                    swaps[j] = p2_words[j][0]
            child = replace_words_at(p1, swaps) if swaps else p1

            # Mutation with adaptive rate
            if random.random() < 0.5 and mutable_positions:
                num_muts = random.randint(1, min(2, len(mutable_positions)))
                positions = random.sample(mutable_positions, num_muts)
                child_spans = get_words_and_spans(child)
                swaps = {}
                for pos in positions:
                    if pos < len(child_spans) and pos in sub_cache:
                        swaps[pos] = random.choice(sub_cache[pos])
                if swaps:
                    child = replace_words_at(child, swaps)

            new_population.append(child)

        population = new_population

    logger.info("Faster Alzantot GA: finished (%d generations)", max_generations)
    return best_overall
