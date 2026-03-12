"""
PSO (Particle Swarm Optimization) — Zang et al., 2020 (arXiv:2004.14641)

Treats adversarial text generation as combinatorial optimization.
Uses particle swarm optimization with sememe-based word substitution.
"""

import random
import logging
import numpy as np

logger = logging.getLogger("textattack.attacks.pso")


def run_pso(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    num_particles: int = 20,
    max_iterations: int = 20,
    max_perturbation_ratio: float = 0.2,
) -> str:
    """Particle Swarm Optimization attack.

    Returns: adversarial text (str).
    """
    from utils.text_utils import (
        get_words_and_spans, replace_word_at, replace_words_at, is_stopword, clean_word,
    )
    from utils.text_word_substitution import get_mlm_substitutions

    logger.info("PSO: starting (particles=%d, iter=%d, max_pert=%.2f)",
                num_particles, max_iterations, max_perturbation_ratio)

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

    # Pre-compute substitution candidates
    sub_cache = {}
    mutable_positions = []
    for i, word in enumerate(words):
        if is_stopword(word) or len(clean_word(word)) <= 1:
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
            orig_probs = model_wrapper.predict_probs(text)
            orig_idx = orig_probs.index(max(orig_probs))
            return 1.0 - probs[orig_idx], predicted_label

    # Initialize particles
    # Each particle is represented as a dict: {position_idx: candidate_idx}
    particles = []
    velocities = []
    personal_best = []
    personal_best_fitness = []
    
    for _ in range(num_particles):
        # Random initialization
        num_muts = random.randint(1, min(max_perturb_words, len(mutable_positions)))
        positions = random.sample(mutable_positions, num_muts)
        
        particle = {}
        for pos in positions:
            particle[pos] = random.randint(0, len(sub_cache[pos]) - 1)
        
        particles.append(particle)
        velocities.append({pos: 0.0 for pos in particle})
        
        # Evaluate initial fitness
        particle_text = text
        for pos, cand_idx in particle.items():
            particle_text = replace_word_at(particle_text, pos, sub_cache[pos][cand_idx])
        
        fit, _ = fitness(particle_text)
        personal_best.append(dict(particle))
        personal_best_fitness.append(fit)

    # Global best
    global_best_idx = personal_best_fitness.index(max(personal_best_fitness))
    global_best = dict(personal_best[global_best_idx])
    global_best_fitness = personal_best_fitness[global_best_idx]

    # PSO parameters
    w = 0.8  # Inertia weight
    c1 = 0.8  # Cognitive parameter
    c2 = 0.8  # Social parameter

    for iteration in range(max_iterations):
        for i in range(num_particles):
            # Update velocity and position for each dimension
            new_particle = {}
            
            # Consider all mutable positions
            for pos in mutable_positions[:min(max_perturb_words, len(mutable_positions))]:
                # Current position
                current = particles[i].get(pos, -1)
                
                # Velocity update
                r1, r2 = random.random(), random.random()
                
                cognitive = 0.0
                if pos in personal_best[i]:
                    cognitive = c1 * r1 * (personal_best[i][pos] - current if current >= 0 else 1.0)
                
                social = 0.0
                if pos in global_best:
                    social = c2 * r2 * (global_best[pos] - current if current >= 0 else 1.0)
                
                velocity = w * velocities[i].get(pos, 0.0) + cognitive + social
                velocities[i][pos] = velocity
                
                # Position update (probabilistic)
                if random.random() < abs(velocity) / 10.0:  # Normalize velocity to probability
                    if pos in global_best:
                        new_particle[pos] = global_best[pos]
                    elif pos in personal_best[i]:
                        new_particle[pos] = personal_best[i][pos]
                    else:
                        new_particle[pos] = random.randint(0, len(sub_cache[pos]) - 1)
                elif current >= 0:
                    new_particle[pos] = current

            particles[i] = new_particle

            # Evaluate fitness
            particle_text = text
            for pos, cand_idx in particles[i].items():
                if pos in sub_cache and 0 <= cand_idx < len(sub_cache[pos]):
                    particle_text = replace_word_at(particle_text, pos, sub_cache[pos][cand_idx])
            
            fit, label = fitness(particle_text)

            # Update personal best
            if fit > personal_best_fitness[i]:
                personal_best[i] = dict(particles[i])
                personal_best_fitness[i] = fit

            # Update global best
            if fit > global_best_fitness:
                global_best = dict(particles[i])
                global_best_fitness = fit

        # Check success
        best_text = text
        for pos, cand_idx in global_best.items():
            if pos in sub_cache and 0 <= cand_idx < len(sub_cache[pos]):
                best_text = replace_word_at(best_text, pos, sub_cache[pos][cand_idx])
        
        _, best_label = fitness(best_text)

        if resolved_target is not None:
            if best_label.lower() == resolved_target.lower():
                logger.info("PSO: success at iteration %d", iteration + 1)
                return best_text
        else:
            if best_label != orig_label:
                logger.info("PSO: success at iteration %d", iteration + 1)
                return best_text

        if iteration % 5 == 0:
            logger.debug("PSO: iter %d/%d, best_fitness=%.4f", 
                        iteration + 1, max_iterations, global_best_fitness)

    # Return global best
    best_text = text
    for pos, cand_idx in global_best.items():
        if pos in sub_cache and 0 <= cand_idx < len(sub_cache[pos]):
            best_text = replace_word_at(best_text, pos, sub_cache[pos][cand_idx])
    
    logger.info("PSO: finished (%d iterations)", max_iterations)
    return best_text
