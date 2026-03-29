"""
PSO (Particle Swarm Optimization) — Zang et al., 2020 (arXiv:2004.14641)

Treats adversarial text generation as combinatorial optimization.
Uses particle swarm optimization with word substitution.

Compliant reimplementation of the TextAttack ParticleSwarmOptimization
search method and PSOZang2020 recipe (https://github.com/QData/TextAttack).

Transformation: WordSwapHowNet (sememe-based) with MLM fallback.
Constraints: RepeatModification, StopwordModification.

Key hyperparameters from the paper (tuned on SST validation set):
  - ω₁ = 0.8, ω₂ = 0.2  (inertia weight, linearly decayed)
  - c1_origin = 0.8, c2_origin = 0.2  (cognitive/social coefficients)
  - V_max = 3.0  (max velocity → sigmoid probability range [0.047, 0.953])
  - pop_size = 60, max_iters = 20
"""

import copy
import logging

import numpy as np

logger = logging.getLogger("textattack.attacks.pso")

# ── PSO constants (from paper / TextAttack reference) ────────────────────────
_OMEGA_1 = 0.8
_OMEGA_2 = 0.2
_C1_ORIGIN = 0.8
_C2_ORIGIN = 0.2
_V_MAX = 3.0
_MAX_TURN_RETRIES = 20


# ── Utility helpers ──────────────────────────────────────────────────────────

def _sigmoid(x):
    """Sigmoid activation, vectorised, with safe clipping."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _equal(word_a, word_b):
    """Discrete PSO equality function (TextAttack `_equal`).

    Returns −V_max when words match  (no need to move),
            +V_max when words differ (should consider moving).
    """
    return -_V_MAX if word_a == word_b else _V_MAX


def _normalize(scores):
    """Normalize a score list into a discrete probability distribution."""
    n = np.array(scores, dtype=np.float64)
    n[n < 0] = 0
    s = np.sum(n)
    if s == 0:
        return np.ones(len(n)) / len(n)
    return n / s


# ── Main entry point ─────────────────────────────────────────────────────────

def run_pso(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    pop_size: int = 60,
    max_iters: int = 20,
    max_perturbation_ratio: float = 0.2,
    max_queries: int = 5000,
    seed: int = None,
) -> str:
    """Particle Swarm Optimization attack.

    Compliant implementation of Zang et al. (2020) following the
    TextAttack ``ParticleSwarmOptimization`` reference class and
    ``PSOZang2020`` recipe.

    Transformation: HowNet sememe-based synonyms (``WordSwapHowNet``)
    with automatic MLM fallback when the HowNet synonym bank is
    unavailable.

    Args:
        model_wrapper: wrapped HF model with predict/predict_probs.
        tokenizer: HF tokenizer.
        text: original input text.
        target_label: target class name (None = untargeted).
        pop_size: swarm population size (paper default: 60).
        max_iters: maximum PSO iterations (paper default: 20).
        max_perturbation_ratio: kept for API compatibility (not
            enforced inside the PSO loop, matching reference).
        max_queries: query budget — abort search when reached
            (mirrors TextAttack _search_over mechanism).
        seed: random seed for reproducibility (None = non-deterministic).

    Returns:
        Adversarial text (str).
    """
    from utils.text_utils import (
        get_words_and_spans, replace_words_at, is_stopword, clean_word,
    )
    from utils.text_word_substitution import (
        get_hownet_substitutions_for_text, get_mlm_substitutions,
    )

    if seed is not None:
        np.random.seed(seed)

    logger.info("PSO: starting (pop_size=%d, max_iters=%d)", pop_size, max_iters)

    # ── Tokenize ─────────────────────────────────────────────────────────
    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    original_words = [w for w, _, _ in words_spans]
    n_words = len(original_words)

    # ── Original prediction ──────────────────────────────────────────────
    orig_label, orig_conf, _ = model_wrapper.predict(text)

    resolved_target = target_label
    resolved_target_idx = None
    if target_label is not None:
        from models.text_loader import resolve_target_label, get_label_index
        resolved_target = resolve_target_label(model_wrapper.model, target_label) or target_label
        resolved_target_idx = get_label_index(model_wrapper.model, resolved_target)

    orig_probs = model_wrapper.predict_probs(text)
    orig_pred_idx = orig_probs.index(max(orig_probs))

    # ── Pre-compute substitution candidates ──────────────────────────────
    # Paper: WordSwapHowNet (sememe-based). Falls back to MLM if the
    # HowNet synonym bank is not available.
    sub_cache: dict[int, list[str]] = {}

    hownet_cache = get_hownet_substitutions_for_text(original_words)
    using_hownet = bool(hownet_cache)

    if using_hownet:
        logger.info("PSO: using HowNet sememe-based substitutions (paper-compliant)")
        for i, cands in hownet_cache.items():
            if is_stopword(original_words[i]) or len(clean_word(original_words[i])) <= 1:
                continue
            if cands:
                sub_cache[i] = cands
    else:
        logger.info("PSO: HowNet unavailable, falling back to MLM substitutions")
        for i, word in enumerate(original_words):
            if is_stopword(word) or len(clean_word(word)) <= 1:
                continue
            candidates = get_mlm_substitutions(text, i, top_k=30)
            if candidates:
                sub_cache[i] = candidates

    if not sub_cache:
        return text

    # ── Query-budget tracking (mirrors TextAttack _search_over) ──────────
    _search_over = [False]

    def _check_budget():
        if max_queries and model_wrapper.query_count >= max_queries:
            _search_over[0] = True

    # ── Helper: build text from word list ────────────────────────────────
    def _build_text(word_list):
        replacements = {}
        for i, w in enumerate(word_list):
            if w != original_words[i]:
                replacements[i] = w
        return replace_words_at(text, replacements) if replacements else text

    # ── Fitness function ─────────────────────────────────────────────────
    def _fitness(word_list):
        """Returns (score, predicted_label, is_success).

        Also checks query budget (sets _search_over, matching TextAttack
        get_goal_results behaviour).
        """
        candidate_text = _build_text(word_list)
        probs = model_wrapper.predict_probs(candidate_text)
        _check_budget()
        predicted_idx = probs.index(max(probs))
        id2label = getattr(model_wrapper.model.config, 'id2label', {})
        predicted_label = id2label.get(predicted_idx, str(predicted_idx))

        if resolved_target is not None:
            if resolved_target_idx is not None and resolved_target_idx < len(probs):
                score = probs[resolved_target_idx]
            else:
                score = max(probs) if predicted_label.lower() == resolved_target.lower() else 0.0
            success = predicted_label.lower() == resolved_target.lower()
        else:
            score = 1.0 - probs[orig_pred_idx]
            success = predicted_label != orig_label

        return score, predicted_label, success

    # ── Best-neighbor search (TextAttack _get_best_neighbors) ────────────
    def _get_best_neighbors(current_words, current_score):
        """For each word position, find the single substitution that yields
        the maximum score improvement.

        Matches TextAttack ParticleSwarmOptimization._get_best_neighbors():
        evaluates ALL candidates at ALL positions (no early exit).

        Returns:
            best_results: list[n_words] of (word_list, score, success)
            prob_list:    np.array[n_words] probability distribution
        """
        best_results = []
        score_diffs = []

        for pos in range(n_words):
            # RepeatModification + StopwordModification: skip positions
            # that are already modified or have no candidates
            if pos not in sub_cache or current_words[pos] != original_words[pos]:
                best_results.append((list(current_words), current_score, False))
                score_diffs.append(0.0)
                continue

            best_score = current_score
            best_words = list(current_words)
            best_success = False

            for cand_word in sub_cache[pos]:
                trial = list(current_words)
                trial[pos] = cand_word
                score, _, success = _fitness(trial)
                if score > best_score:
                    best_score = score
                    best_words = trial
                    best_success = success

                if _search_over[0]:
                    break

            best_results.append((best_words, best_score, best_success))
            score_diffs.append(best_score - current_score)

            if _search_over[0]:
                for _ in range(pos + 1, n_words):
                    best_results.append((list(current_words), current_score, False))
                    score_diffs.append(0.0)
                break

        return best_results, _normalize(score_diffs)

    # ── Turn operator (TextAttack _turn) ─────────────────────────────────
    def _turn(source_words, target_words, turn_prob):
        """Move from *source* towards *target* probabilistically.

        Post-turn constraint check matches TextAttack: verifies both
        StopwordModification (stopwords unchanged) and RepeatModification
        (only positions with candidates can be modified). Retries up to
        _MAX_TURN_RETRIES times.
        """
        for _ in range(_MAX_TURN_RETRIES + 1):
            new_words = list(source_words)
            for d in range(n_words):
                if np.random.uniform() < turn_prob[d]:
                    new_words[d] = target_words[d]

            if new_words == source_words:
                return new_words

            # Post-turn constraint: StopwordModification + RepeatModification
            ok = True
            for d in range(n_words):
                if new_words[d] != original_words[d]:
                    # StopwordModification: stopwords / short words must stay
                    if is_stopword(original_words[d]) or len(clean_word(original_words[d])) <= 1:
                        ok = False
                        break
                    # RepeatModification: only positions with candidates can change
                    if d not in sub_cache:
                        ok = False
                        break
            if ok:
                return new_words

        return list(source_words)

    # ── Perturb / mutate (TextAttack _perturb) ───────────────────────────
    def _perturb(particle_words, current_score):
        """Replace one word with its best-improvement neighbour.

        Samples a position from the best-neighbor probability distribution
        and adopts that neighbour's word list.

        Returns (new_word_list, new_score, success).
        """
        best_results, prob_list = _get_best_neighbors(particle_words, current_score)
        chosen_idx = np.random.choice(len(best_results), p=prob_list)
        new_words, new_score, success = best_results[chosen_idx]
        if new_words == particle_words:
            return particle_words, current_score, False
        return new_words, new_score, success

    # ══════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ══════════════════════════════════════════════════════════════════════

    orig_score, _, orig_success = _fitness(original_words)
    if orig_success:
        return text

    best_results, prob_list = _get_best_neighbors(original_words, orig_score)

    population = []
    pop_scores = []

    for _ in range(pop_size):
        chosen_idx = np.random.choice(len(best_results), p=prob_list)
        particle_words = list(best_results[chosen_idx][0])
        score = best_results[chosen_idx][1]
        population.append(particle_words)
        pop_scores.append(score)

    # Check if any initial particle already succeeds (matches TextAttack
    # post-init check: global_elite.result.goal_status == SUCCEEDED)
    for k in range(pop_size):
        _, _, success = _fitness(population[k])
        if success:
            logger.info("PSO: success during initialization")
            return _build_text(population[k])

    # Post-init _search_over check (matches TextAttack perform_search)
    if _search_over[0]:
        best_idx = int(np.argmax(pop_scores))
        logger.info("PSO: query budget exhausted during initialization")
        return _build_text(population[best_idx])

    # Velocities: [pop_size × n_words], initialised uniformly in [-V_max, V_max]
    v_init = np.random.uniform(-_V_MAX, _V_MAX, pop_size)
    velocities = np.array([
        np.full(n_words, v_init[t]) for t in range(pop_size)
    ])

    # Local elites (personal bests) — deep copies
    local_elites = [list(p) for p in population]
    local_elite_scores = list(pop_scores)

    # Global elite
    global_elite_idx = int(np.argmax(pop_scores))
    global_elite = list(population[global_elite_idx])
    global_elite_score = pop_scores[global_elite_idx]

    # ══════════════════════════════════════════════════════════════════════
    # MAIN PSO LOOP
    # ══════════════════════════════════════════════════════════════════════

    for iteration in range(max_iters):
        # ── Adaptive coefficients (linear schedule) ──────────────────────
        omega = (_OMEGA_1 - _OMEGA_2) * (max_iters - iteration) / max_iters + _OMEGA_2
        C1 = _C1_ORIGIN - iteration / max_iters * (_C1_ORIGIN - _C2_ORIGIN)
        C2 = _C2_ORIGIN + iteration / max_iters * (_C1_ORIGIN - _C2_ORIGIN)
        P1 = C1
        P2 = C2

        # ── Phase 1: Velocity & position update ─────────────────────────
        for k in range(pop_size):
            for d in range(n_words):
                velocities[k][d] = omega * velocities[k][d] + (1 - omega) * (
                    _equal(population[k][d], local_elites[k][d])
                    + _equal(population[k][d], global_elite[d])
                )

            turn_prob = _sigmoid(velocities[k])

            if np.random.uniform() < P1:
                population[k] = _turn(
                    local_elites[k], population[k], turn_prob
                )

            if np.random.uniform() < P2:
                population[k] = _turn(
                    global_elite, population[k], turn_prob
                )

        # ── Phase 2: Evaluate all particles ──────────────────────────────
        for k in range(pop_size):
            score, label, success = _fitness(population[k])
            pop_scores[k] = score
            if success:
                logger.info("PSO: success at iteration %d", iteration + 1)
                return _build_text(population[k])
            if _search_over[0]:
                break

        top_k_idx = int(np.argmax(pop_scores))
        if _search_over[0]:
            logger.info("PSO: query budget exhausted (%d queries)", model_wrapper.query_count)
            return _build_text(population[top_k_idx])

        # ── Phase 3: Mutation (TextAttack _perturb) ──────────────────────
        for k in range(pop_size):
            changed = sum(
                1 for d in range(n_words)
                if population[k][d] != original_words[d]
            )
            change_ratio = changed / n_words if n_words > 0 else 0.0
            p_change = 1.0 - 2.0 * change_ratio
            if np.random.uniform() < p_change:
                new_words, new_score, _ = _perturb(population[k], pop_scores[k])
                if new_words != population[k]:
                    population[k] = new_words
                    pop_scores[k] = new_score
            if _search_over[0]:
                break

        # Post-mutation success check (no redundant fitness call — uses
        # scores already computed by _perturb → _get_best_neighbors)
        top_k_idx = int(np.argmax(pop_scores))
        top_score, _, top_success = _fitness(population[top_k_idx])
        pop_scores[top_k_idx] = top_score

        if _search_over[0] or top_success:
            if top_success:
                logger.info("PSO: success at iteration %d (post-mutation)", iteration + 1)
                return _build_text(population[top_k_idx])
            logger.info("PSO: query budget exhausted (%d queries)", model_wrapper.query_count)
            return _build_text(global_elite)

        # ── Phase 4: Update elites ───────────────────────────────────────
        for k in range(pop_size):
            if pop_scores[k] > local_elite_scores[k]:
                local_elites[k] = list(population[k])
                local_elite_scores[k] = pop_scores[k]

        if pop_scores[top_k_idx] > global_elite_score:
            global_elite = list(population[top_k_idx])
            global_elite_score = pop_scores[top_k_idx]

        if iteration % 5 == 0:
            logger.debug("PSO: iter %d/%d, best_score=%.4f",
                         iteration + 1, max_iters, global_elite_score)

    # ── Return global best ───────────────────────────────────────────────
    logger.info("PSO: finished (%d iterations)", max_iters)
    return _build_text(global_elite)
