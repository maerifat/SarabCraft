"""
SirenAttack — Black-box Particle Swarm Optimization attack on ASR/audio.

Implements the gradient-free attack of:
  Du, Ji, Li, Shi, Tian, Guo, Beyah (2020),
  "SirenAttack: Generating Adversarial Audio for End-to-End Acoustic
  Systems" (ACM ASIACCS 2020).

SirenAttack treats the model as a black box exposing only a scalar loss
(or, for classification, class probabilities). It searches for an
adversarial perturbation using Particle Swarm Optimization (PSO):

  - A swarm of particles, each a candidate perturbation δ, moves through
    perturbation space.
  - Each particle tracks its personal best position (pbest) and the swarm
    tracks the global best (gbest).
  - Velocities are updated by inertia + cognitive pull toward pbest +
    social pull toward gbest, then positions are clamped to the L-inf ball.

Supports two objectives:
  * targeted   — minimize the model's loss toward attacker-chosen output
                 (target transcription for ASR, target class for
                 classification).
  * untargeted — maximize the loss w.r.t. the original/true output so the
                 model produces anything but the correct answer.

The fitness function is evaluated under ``torch.no_grad()`` so no gradient
information ever leaks — purely score-based (decision + confidence).
"""

import logging

import torch

logger = logging.getLogger(__name__)


def _asr_fitness(wrapper, waveform, delta, target_ids, targeted):
    """Scalar fitness for ASR: lower is better. Uses model CE/CTC loss (no grad)."""
    cand = torch.clamp(waveform + delta, -1.0, 1.0)
    with torch.no_grad():
        out = wrapper.forward_with_labels(cand, target_ids)
        loss = float(out.loss.item())
    return loss if targeted else -loss


def siren_attack_asr(
    wrapper,
    waveform,
    target_text=None,
    targeted=True,
    epsilon=0.05,
    n_particles=25,
    iterations=150,
    inertia=0.729,
    c1=1.49445,
    c2=1.49445,
    v_max_frac=0.2,
    progress_fn=None,
):
    """
    SirenAttack against an ASR wrapper using PSO (fully black-box).

    Args:
        wrapper: exposes ``forward_with_labels``, ``tokenize_target``,
            ``transcribe``.
        waveform: tensor [1, N].
        target_text: required if ``targeted`` — attacker's target transcript.
            If ``targeted`` is False, the original transcription is used as
            the anchor to move away from.
        targeted: True = force target text; False = degrade correct output.
        epsilon: L-inf perturbation bound.
        n_particles: swarm size.
        iterations: PSO iterations.
        inertia, c1, c2: PSO inertia / cognitive / social coefficients.
        v_max_frac: velocity clamp as a fraction of epsilon.
        progress_fn: optional callback(step, total, fitness).

    Returns:
        adversarial waveform tensor [1, N].
    """
    device = waveform.device
    waveform = waveform.detach().clone()

    if targeted:
        if not target_text:
            raise ValueError("target_text is required for targeted SirenAttack")
        target_ids = wrapper.tokenize_target(target_text)
    else:
        anchor = wrapper.transcribe(waveform)
        if not anchor.strip():
            anchor = "hello"
        target_ids = wrapper.tokenize_target(anchor)

    v_max = v_max_frac * epsilon

    # Initialize swarm positions (perturbations) and velocities.
    positions = [(torch.rand_like(waveform) * 2 - 1) * epsilon for _ in range(n_particles)]
    velocities = [(torch.rand_like(waveform) * 2 - 1) * v_max for _ in range(n_particles)]

    pbest = [p.detach().clone() for p in positions]
    pbest_fit = [_asr_fitness(wrapper, waveform, p, target_ids, targeted) for p in positions]

    g_idx = min(range(n_particles), key=lambda i: pbest_fit[i])
    gbest = pbest[g_idx].detach().clone()
    gbest_fit = pbest_fit[g_idx]

    for step in range(iterations):
        for i in range(n_particles):
            r1 = torch.rand_like(waveform)
            r2 = torch.rand_like(waveform)
            velocities[i] = (
                inertia * velocities[i]
                + c1 * r1 * (pbest[i] - positions[i])
                + c2 * r2 * (gbest - positions[i])
            )
            velocities[i] = torch.clamp(velocities[i], -v_max, v_max)
            positions[i] = torch.clamp(positions[i] + velocities[i], -epsilon, epsilon)

            fit = _asr_fitness(wrapper, waveform, positions[i], target_ids, targeted)
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i] = positions[i].detach().clone()
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest = positions[i].detach().clone()

        if progress_fn and step % 2 == 0:
            progress_fn(step, iterations, float(gbest_fit))
        if step % 15 == 0:
            logger.info(f"[SirenAttack ASR] step {step}/{iterations}, "
                        f"gbest_fit={gbest_fit:.4f}, targeted={targeted}")

    return torch.clamp(waveform + gbest, -1.0, 1.0).detach()


def _clf_fitness(wrapper, waveform, delta, target_idx, targeted):
    """Scalar fitness for audio classification (lower is better), no grad."""
    import torch.nn.functional as F

    cand = torch.clamp(waveform + delta, -1.0, 1.0)
    wrapper.eval()
    with torch.no_grad():
        logits = wrapper(cand).logits
        probs = F.softmax(logits, dim=1).squeeze(0)
    p_target = float(probs[target_idx].item())
    if targeted:
        # Maximize target prob → minimize (1 - p_target).
        return 1.0 - p_target
    # Untargeted: minimize the true-class prob.
    return p_target


def siren_attack_classification(
    wrapper,
    waveform,
    target_idx,
    targeted=True,
    epsilon=0.02,
    n_particles=25,
    iterations=150,
    inertia=0.729,
    c1=1.49445,
    c2=1.49445,
    v_max_frac=0.2,
    progress_fn=None,
):
    """
    SirenAttack against an audio-classification wrapper using PSO.

    For targeted attacks, ``target_idx`` is the class to force. For
    untargeted attacks it is the true/original class index to suppress.
    """
    waveform = waveform.detach().clone()
    v_max = v_max_frac * epsilon

    positions = [(torch.rand_like(waveform) * 2 - 1) * epsilon for _ in range(n_particles)]
    velocities = [(torch.rand_like(waveform) * 2 - 1) * v_max for _ in range(n_particles)]

    pbest = [p.detach().clone() for p in positions]
    pbest_fit = [_clf_fitness(wrapper, waveform, p, target_idx, targeted) for p in positions]

    g_idx = min(range(n_particles), key=lambda i: pbest_fit[i])
    gbest = pbest[g_idx].detach().clone()
    gbest_fit = pbest_fit[g_idx]

    for step in range(iterations):
        for i in range(n_particles):
            r1 = torch.rand_like(waveform)
            r2 = torch.rand_like(waveform)
            velocities[i] = (
                inertia * velocities[i]
                + c1 * r1 * (pbest[i] - positions[i])
                + c2 * r2 * (gbest - positions[i])
            )
            velocities[i] = torch.clamp(velocities[i], -v_max, v_max)
            positions[i] = torch.clamp(positions[i] + velocities[i], -epsilon, epsilon)

            fit = _clf_fitness(wrapper, waveform, positions[i], target_idx, targeted)
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i] = positions[i].detach().clone()
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest = positions[i].detach().clone()

        if progress_fn and step % 2 == 0:
            progress_fn(step, iterations, float(gbest_fit))
        if step % 15 == 0:
            logger.info(f"[SirenAttack CLF] step {step}/{iterations}, "
                        f"gbest_fit={gbest_fit:.4f}, targeted={targeted}")

    return torch.clamp(waveform + gbest, -1.0, 1.0).detach()
