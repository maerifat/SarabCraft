"""
IGA (Improved Genetic Algorithm) — Wang et al., 2019 (arXiv:1909.06723)

Enhanced genetic algorithm with prioritized word importance ranking
and improved search strategy for better convergence.
"""

import random
import logging

logger = logging.getLogger("textattack.attacks.iga")


def run_iga(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    population_size: int = 20,
    max_generations: int = 20,
    max_perturbation_ratio: float = 0.2,
) -> str:
    """Improved Genetic Algorithm attack.

    Returns: adversarial text (str).
    """
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, replace_words_at, is_stopword, clean_word,
    )
    from utils.text_word_substitution import get_mlm_substitutions
    from utils.text_constraints import compute_semantic_similarity

    logger.info("IGA: starting (pop=%d, gen=%d, max_pert=%.2f)",
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

    # Compute word importance scores (delete-one method)
    word_importance = []
    orig_probs = model_wrapper.predict_probs(text)
    orig_idx = orig_probs.index(max(orig_probs))
    
    for i, word in enumerate(words):
        if is_stopword(word) or len(clean_word(word)) <= 1:
            word_importance.append((i, 0.0))
            continue
        
        # Delete word and measure impact
        deleted_text = replace_word_at(text, i, "")
        deleted_probs = model_wrapper.predict_probs(deleted_text)
        
        if resolved_target is not None and resolved_target_idx is not None:
            # Importance = how much deleting increases target class
            importance = deleted_probs[resolved_target_idx] - orig_probs[resolved_target_idx]
        else:
            # Importance = how much deleting decreases original class
            importance = orig_probs[orig_idx] - deleted_probs[orig_idx]
        
        word_importance.append((i, importance))
    
    # Sort by importance (descending)
    word_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Pre-compute substitutions for top important words
    sub_cache = {}
    mutable_positions = []
    for i, importance in word_importance[:int(n_words * 0.5)]:  # Top 50% important words
        if importance <= 0:
            continue
        candidates = get_mlm_substitutions(text, i, top_k=30)
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
            return 1.0 - probs[orig_idx], predicted_label

    # Initialize population with importance-guided mutations
    population = []
    for _ in range(population_size):
        # Prioritize more important positions
        num_muts = random.randint(1, min(max_perturb_words, len(mutable_positions)))
        # Weighted sampling: higher probability for more important positions
        positions = random.sample(mutable_positions[:min(10, len(mutable_positions))], 
                                 min(num_muts, len(mutable_positions[:10])))
        
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
                logger.info("IGA: success at generation %d", gen + 1)
                return best_text
        else:
            if best_label != orig_label:
                logger.info("IGA: success at generation %d", gen + 1)
                return best_text

        if gen % 5 == 0:
            logger.debug("IGA: gen %d/%d, best_fitness=%.4f", gen + 1, max_generations, best_fitness)

        # Elitism: keep top 20%
        elite_size = max(2, population_size // 5)
        survivors = [ind for ind, _, _ in scored[:elite_size]]

        # Generate new population
        new_population = list(survivors)
        
        while len(new_population) < population_size:
            # Tournament selection (size 3)
            tournament = random.sample(scored, min(3, len(scored)))
            tournament.sort(key=lambda x: x[1], reverse=True)
            p1 = tournament[0][0]
            p2 = tournament[1][0] if len(tournament) > 1 else p1

            # Uniform crossover
            p1_words = get_words_and_spans(p1)
            p2_words = get_words_and_spans(p2)
            swaps = {}
            for j in range(min(len(p1_words), len(p2_words))):
                if random.random() < 0.5:
                    swaps[j] = p2_words[j][0]
            child = replace_words_at(p1, swaps) if swaps else p1

            # Adaptive mutation (higher rate early, lower rate later)
            mutation_rate = 0.5 * (1.0 - gen / max_generations)
            if random.random() < mutation_rate and mutable_positions:
                # Prioritize important positions
                pos = random.choice(mutable_positions[:min(5, len(mutable_positions))])
                child_spans = get_words_and_spans(child)
                if pos < len(child_spans) and pos in sub_cache:
                    child = replace_word_at(child, pos, random.choice(sub_cache[pos]))

            new_population.append(child)

        population = new_population

    logger.info("IGA: finished (%d generations)", max_generations)
    return best_overall
