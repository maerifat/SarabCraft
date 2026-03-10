"""
Hidden Voice Command Attack (CommanderSong-style).

Embed a voice command into carrier audio (music, ambient noise, speech)
so that a human hears the carrier naturally, but an ASR model (Whisper)
transcribes the attacker-chosen command.

This is architecturally similar to the targeted transcription attack
but operates on non-speech carriers and typically requires larger
perturbation budgets.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def hidden_command_attack(
    wrapper,
    carrier_waveform,
    command_text,
    epsilon=0.1,
    iterations=300,
    lr=0.005,
    progress_fn=None,
):
    """
    Embed a hidden voice command into carrier audio.

    The carrier can be music, ambient sound, or any audio. The attack
    optimizes a perturbation overlay so Whisper transcribes command_text
    while a human hears the carrier.

    Args:
        wrapper: WhisperAttackWrapper
        carrier_waveform: tensor [1, num_samples] on device (music/noise)
        command_text: the voice command to embed (e.g. "call nine one one")
        epsilon: max perturbation amplitude (higher than transcription attacks
                 since carrier masks more noise)
        iterations: number of optimization steps
        lr: learning rate
        progress_fn: optional callback(step, total, loss_val)

    Returns:
        adversarial waveform tensor [1, num_samples]
    """
    target_ids = wrapper.tokenize_target(command_text)

    carrier = carrier_waveform.detach().clone()
    delta = torch.zeros_like(carrier, requires_grad=True)

    optimizer = torch.optim.Adam([delta], lr=lr)

    best_delta = delta.detach().clone()
    best_loss = float('inf')

    for step in range(iterations):
        optimizer.zero_grad()

        adv_waveform = torch.clamp(carrier + delta, -1.0, 1.0)

        outputs = wrapper.forward_with_labels(adv_waveform, target_ids)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)
            clamped = torch.clamp(carrier + delta, -1.0, 1.0)
            delta.data.copy_(clamped - carrier)

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_delta = delta.detach().clone()

        if progress_fn and step % 5 == 0:
            progress_fn(step, iterations, loss_val)

        if step % 20 == 0:
            logger.info(f"[Hidden Command] step {step}/{iterations}, "
                  f"loss={loss_val:.4f}, best={best_loss:.4f}")

    adv_waveform = torch.clamp(carrier + best_delta, -1.0, 1.0)
    return adv_waveform.detach()


def hidden_command_pgd(
    wrapper,
    carrier_waveform,
    command_text,
    epsilon=0.1,
    iterations=300,
    alpha=0.003,
    progress_fn=None,
):
    """
    PGD-style hidden command attack (sign-based steps).

    Args:
        wrapper: WhisperAttackWrapper
        carrier_waveform: tensor [1, num_samples]
        command_text: command to embed
        epsilon: L-inf bound
        iterations: number of PGD steps
        alpha: step size
        progress_fn: optional callback

    Returns:
        adversarial waveform tensor [1, num_samples]
    """
    target_ids = wrapper.tokenize_target(command_text)

    carrier = carrier_waveform.detach().clone()
    delta = torch.zeros_like(carrier, requires_grad=True)

    best_delta = delta.detach().clone()
    best_loss = float('inf')

    for step in range(iterations):
        adv_waveform = torch.clamp(carrier + delta, -1.0, 1.0)

        outputs = wrapper.forward_with_labels(adv_waveform, target_ids)
        loss = outputs.loss

        loss.backward()

        with torch.no_grad():
            delta.data -= alpha * delta.grad.sign()
            delta.data.clamp_(-epsilon, epsilon)
            clamped = torch.clamp(carrier + delta, -1.0, 1.0)
            delta.data.copy_(clamped - carrier)

        if delta.grad is not None:
            delta.grad.zero_()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_delta = delta.detach().clone()

        if progress_fn and step % 5 == 0:
            progress_fn(step, iterations, loss_val)

        if step % 20 == 0:
            logger.info(f"[Hidden Command PGD] step {step}/{iterations}, "
                  f"loss={loss_val:.4f}, best={best_loss:.4f}")

    adv_waveform = torch.clamp(carrier + best_delta, -1.0, 1.0)
    return adv_waveform.detach()
