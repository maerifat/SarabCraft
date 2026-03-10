"""
Targeted Transcription Attack against Whisper ASR.

Given audio that says X, produce adversarial audio that:
  - sounds like X to a human listener (small perturbation)
  - Whisper transcribes as attacker-chosen text Y

Based on Carlini & Wagner (2018) "Audio Adversarial Examples" adapted
for Whisper's encoder-decoder architecture. Uses iterative PGD-style
optimization with cross-entropy loss against target token IDs.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def targeted_transcription_attack(
    wrapper,
    waveform,
    target_text,
    epsilon=0.05,
    iterations=200,
    lr=0.005,
    progress_fn=None,
):
    """
    Iterative gradient-based attack to force Whisper to transcribe target_text.

    Args:
        wrapper: WhisperAttackWrapper (differentiable mel-spec + Whisper model)
        waveform: tensor [1, num_samples] on device, raw audio -1..1
        target_text: string the attacker wants Whisper to output
        epsilon: max perturbation amplitude (L-inf bound on waveform)
        iterations: number of optimization steps
        lr: learning rate for perturbation update
        progress_fn: optional callback(step, total, loss_val)

    Returns:
        adversarial waveform tensor [1, num_samples]
    """
    target_ids = wrapper.tokenize_target(target_text)

    waveform = waveform.detach().clone()
    delta = torch.zeros_like(waveform, requires_grad=True)

    optimizer = torch.optim.Adam([delta], lr=lr)

    best_delta = delta.detach().clone()
    best_loss = float('inf')

    for step in range(iterations):
        optimizer.zero_grad()

        adv_waveform = torch.clamp(waveform + delta, -1.0, 1.0)

        outputs = wrapper.forward_with_labels(adv_waveform, target_ids)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)
            clamped = torch.clamp(waveform + delta, -1.0, 1.0)
            delta.data.copy_(clamped - waveform)

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_delta = delta.detach().clone()

        if progress_fn and step % 5 == 0:
            progress_fn(step, iterations, loss_val)

        if step % 20 == 0:
            logger.info(f"[Transcription Attack] step {step}/{iterations}, "
                  f"loss={loss_val:.4f}, best={best_loss:.4f}")

    adv_waveform = torch.clamp(waveform + best_delta, -1.0, 1.0)
    return adv_waveform.detach()


def targeted_transcription_pgd(
    wrapper,
    waveform,
    target_text,
    epsilon=0.05,
    iterations=200,
    alpha=0.002,
    progress_fn=None,
):
    """
    PGD-style (sign-based) transcription attack.

    Uses sign of gradient (like FGSM/PGD) rather than Adam. Often converges
    faster for larger epsilon budgets but less precise for small epsilon.

    Args:
        wrapper: WhisperAttackWrapper
        waveform: tensor [1, num_samples]
        target_text: target transcription string
        epsilon: L-inf perturbation bound
        iterations: number of PGD steps
        alpha: step size per iteration
        progress_fn: optional callback(step, total, loss_val)

    Returns:
        adversarial waveform tensor [1, num_samples]
    """
    target_ids = wrapper.tokenize_target(target_text)

    waveform = waveform.detach().clone()
    delta = torch.zeros_like(waveform, requires_grad=True)

    best_delta = delta.detach().clone()
    best_loss = float('inf')

    for step in range(iterations):
        adv_waveform = torch.clamp(waveform + delta, -1.0, 1.0)

        outputs = wrapper.forward_with_labels(adv_waveform, target_ids)
        loss = outputs.loss

        loss.backward()

        with torch.no_grad():
            delta.data -= alpha * delta.grad.sign()
            delta.data.clamp_(-epsilon, epsilon)
            clamped = torch.clamp(waveform + delta, -1.0, 1.0)
            delta.data.copy_(clamped - waveform)

        if delta.grad is not None:
            delta.grad.zero_()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_delta = delta.detach().clone()

        if progress_fn and step % 5 == 0:
            progress_fn(step, iterations, loss_val)

        if step % 20 == 0:
            logger.info(f"[Transcription PGD] step {step}/{iterations}, "
                  f"loss={loss_val:.4f}, best={best_loss:.4f}")

    adv_waveform = torch.clamp(waveform + best_delta, -1.0, 1.0)
    return adv_waveform.detach()
