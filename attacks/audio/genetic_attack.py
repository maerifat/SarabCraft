"""
Black-box Genetic + Gradient-Estimation Targeted Transcription Attack.

Implements the two-stage decision-based attack of:
  Taori, Kamsetty, Chu, Vemuri (2019),
  "Targeted Adversarial Examples for Black Box Audio Systems"
  (IEEE S&P DLS Workshop).

The attacker only has access to the model's *transcription output* (and,
optionally, a scalar decoder loss / edit-distance signal) — no gradients.
The attack proceeds in two phases:

  Phase 1 (Genetic Algorithm):
    Maintain a population of candidate perturbations. Score each candidate
    by how close its transcription is to the target (negative token/char
    edit distance). Elite candidates survive; the rest are produced by
    crossover of two parents (sampled proportional to fitness) followed by
    momentum-mutation with an adaptive mutation rate. This phase drives the
    transcription until it *matches* the target text.

  Phase 2 (Gradient Estimation):
    Once the transcription matches, switch to finite-difference gradient
    estimation (NES-style antithetic sampling) to *minimize the L-inf norm*
    of the perturbation while keeping the transcription on-target, yielding
    a smaller / less perceptible perturbation.

This mirrors the original paper's design while remaining framework-agnostic:
it drives any ``wrapper`` exposing ``transcribe(waveform)`` and, when
available, ``forward_with_labels`` for the CTC/CE proxy score. Only the
decision (transcription string) is strictly required.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def _levenshtein(a, b):
    """Token/char-level Levenshtein edit distance between two strings."""
    a = a or ""
    b = b or ""
    if a == b:
        return 0
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def _score_transcription(wrapper, candidate, target_text):
    """Lower is better. Edit distance between candidate transcription and target."""
    text = wrapper.transcribe(torch.clamp(candidate, -1.0, 1.0))
    return _levenshtein(text.strip().lower(), target_text.strip().lower()), text


def _decoder_proxy_loss(wrapper, candidate, target_ids):
    """
    Optional differentiable-free proxy: the model's own CE/CTC loss toward the
    target token ids evaluated under no_grad. Used only to break ties between
    candidates with equal edit distance (finer-grained fitness).
    """
    try:
        with torch.no_grad():
            out = wrapper.forward_with_labels(torch.clamp(candidate, -1.0, 1.0), target_ids)
            return float(out.loss.item())
    except Exception:
        return 0.0


def genetic_targeted_attack(
    wrapper,
    waveform,
    target_text,
    epsilon=0.05,
    population_size=20,
    genetic_iterations=300,
    elite_frac=0.1,
    mutation_prob=0.005,
    mutation_scale=0.02,
    momentum=0.9,
    gradient_estimation_iterations=100,
    ge_samples=50,
    ge_sigma=0.002,
    ge_lr=0.002,
    progress_fn=None,
):
    """
    Two-phase black-box targeted transcription attack (Taori et al. 2019).

    Args:
        wrapper: object exposing ``transcribe(waveform)`` and (optionally)
            ``forward_with_labels`` + ``tokenize_target`` for tie-breaking.
        waveform: tensor [1, N] on device, raw audio -1..1.
        target_text: attacker-chosen transcription string.
        epsilon: L-inf bound on the perturbation.
        population_size: GA population size.
        genetic_iterations: max GA generations (phase 1).
        elite_frac: fraction of population kept as elites each generation.
        mutation_prob: base per-sample mutation probability.
        mutation_scale: std of mutation noise (scaled by epsilon).
        momentum: momentum-mutation decay (accumulates successful directions).
        gradient_estimation_iterations: NES steps (phase 2) to shrink L-inf.
        ge_samples: antithetic sample pairs per NES step.
        ge_sigma: NES sampling std.
        ge_lr: NES step size.
        progress_fn: optional callback(step, total, score).

    Returns:
        adversarial waveform tensor [1, N].
    """
    device = waveform.device
    waveform = waveform.detach().clone()
    n_samples = waveform.shape[-1]

    target_ids = None
    if hasattr(wrapper, "tokenize_target"):
        try:
            target_ids = wrapper.tokenize_target(target_text)
        except Exception:
            target_ids = None

    total_steps = genetic_iterations + gradient_estimation_iterations

    # ── Phase 1: Genetic Algorithm ────────────────────────────────────────────
    population = [
        (torch.rand_like(waveform) * 2 - 1) * epsilon for _ in range(population_size)
    ]
    momentum_buf = [torch.zeros_like(waveform) for _ in range(population_size)]

    n_elite = max(1, int(elite_frac * population_size))
    adaptive_mut = mutation_prob

    best_delta = population[0].detach().clone()
    best_score = float("inf")
    best_text = ""
    matched = False

    for gen in range(genetic_iterations):
        scored = []
        for idx, delta in enumerate(population):
            cand = waveform + delta
            dist, text = _score_transcription(wrapper, cand, target_text)
            tie = _decoder_proxy_loss(wrapper, cand, target_ids) if target_ids is not None else 0.0
            scored.append((dist, tie, idx, text))

        scored.sort(key=lambda t: (t[0], t[1]))
        top_dist, top_tie, top_idx, top_text = scored[0]

        if top_dist < best_score:
            best_score = top_dist
            best_delta = population[top_idx].detach().clone()
            best_text = top_text

        if top_dist == 0:
            matched = True
            logger.info(f"[Genetic] gen {gen}: transcription matched target — switching to gradient estimation")
            break

        # Fitness: higher for lower edit distance. Softmax over -distance.
        dists = torch.tensor([s[0] for s in scored], dtype=torch.float32, device=device)
        fitness = torch.softmax(-dists / max(1.0, dists.std().item() + 1e-6), dim=0)

        # Adaptive mutation: if the best plateaus, increase exploration.
        if gen > 0 and gen % 10 == 0:
            adaptive_mut = min(0.05, adaptive_mut * 1.2)

        new_population = []
        new_momentum = []
        # Keep elites.
        for e in range(n_elite):
            src = scored[e][2]
            new_population.append(population[src].detach().clone())
            new_momentum.append(momentum_buf[src].detach().clone())

        while len(new_population) < population_size:
            p1 = torch.multinomial(fitness, 1).item()
            p2 = torch.multinomial(fitness, 1).item()
            i1, i2 = scored[p1][2], scored[p2][2]
            mask = (torch.rand_like(waveform) < 0.5).float()
            child = mask * population[i1] + (1 - mask) * population[i2]
            child_mom = mask * momentum_buf[i1] + (1 - mask) * momentum_buf[i2]

            # Momentum mutation.
            mut_mask = (torch.rand_like(waveform) < adaptive_mut).float()
            noise = torch.randn_like(waveform) * (mutation_scale * epsilon)
            child_mom = momentum * child_mom + (1 - momentum) * noise * mut_mask
            child = child + child_mom
            child = torch.clamp(child, -epsilon, epsilon)

            new_population.append(child)
            new_momentum.append(child_mom)

        population = new_population
        momentum_buf = new_momentum

        if progress_fn and gen % 2 == 0:
            progress_fn(gen, total_steps, float(top_dist))
        if gen % 20 == 0:
            logger.info(f"[Genetic] gen {gen}/{genetic_iterations}, edit_dist={top_dist}, "
                        f"mut={adaptive_mut:.4f}, text={top_text[:40]!r}")

    # ── Phase 2: NES Gradient Estimation to minimize L-inf ────────────────────
    if matched and gradient_estimation_iterations > 0:
        delta = best_delta.detach().clone()
        for step in range(gradient_estimation_iterations):
            # Antithetic NES estimate of gradient of the (proxy) on-target loss.
            grad_est = torch.zeros_like(delta)
            for _ in range(ge_samples):
                u = torch.randn_like(delta)
                plus = torch.clamp(waveform + delta + ge_sigma * u, -1.0, 1.0)
                minus = torch.clamp(waveform + delta - ge_sigma * u, -1.0, 1.0)
                if target_ids is not None:
                    f_plus = _decoder_proxy_loss(wrapper, plus, target_ids)
                    f_minus = _decoder_proxy_loss(wrapper, minus, target_ids)
                else:
                    f_plus = _levenshtein(wrapper.transcribe(plus).strip().lower(), target_text.strip().lower())
                    f_minus = _levenshtein(wrapper.transcribe(minus).strip().lower(), target_text.strip().lower())
                grad_est += (f_plus - f_minus) * u
            grad_est /= (2 * ge_samples * ge_sigma)

            # Step toward lower on-target loss, then shrink L-inf toward zero.
            delta = delta - ge_lr * grad_est.sign()
            delta = 0.98 * delta  # gentle L-inf contraction
            delta = torch.clamp(delta, -epsilon, epsilon)

            cand = waveform + delta
            dist, text = _score_transcription(wrapper, cand, target_text)
            if dist == 0:
                best_delta = delta.detach().clone()
                best_text = text
            else:
                # Reverted too far; stop contracting.
                break

            if progress_fn:
                progress_fn(genetic_iterations + step, total_steps, float(dist))
            if step % 10 == 0:
                linf = delta.abs().max().item()
                logger.info(f"[GradEst] step {step}/{gradient_estimation_iterations}, "
                            f"on_target=True, linf={linf:.5f}")

    adv = torch.clamp(waveform + best_delta, -1.0, 1.0)
    logger.info(f"[Genetic Attack] done. matched={matched}, final_edit_dist={best_score}, "
                f"final_text={best_text[:60]!r}")
    return adv.detach()
