"""
Universal Muting Attack (EMNLP 2024).

Learns a short universal adversarial audio segment (~0.5-1s) that, when
prepended to ANY speech input, causes Whisper to output only the
<|endoftext|> token -- effectively "muting" the model.

One segment works universally across all inputs without per-sample
optimization.

Based on: "Muting Whisper: A Universal Acoustic Adversarial Attack on
Speech Foundation Models" (Vyas et al., EMNLP 2024).

This implementation also supports a "targeted" mode where the universal
segment forces a specific phrase instead of muting.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def universal_muting_attack(
    wrapper,
    training_waveforms,
    segment_duration=0.64,
    iterations=200,
    lr=0.01,
    mode="mute",
    target_text=None,
    progress_fn=None,
):
    """
    Learn a universal adversarial audio segment.

    In 'mute' mode, the segment causes Whisper to output <|endoftext|>.
    In 'target' mode, it forces a specific transcription.

    Args:
        wrapper: WhisperAttackWrapper
        training_waveforms: list of tensors [1, N] to optimize over
        segment_duration: length in seconds of the adversarial segment
        iterations: optimization steps
        lr: learning rate
        mode: 'mute' or 'target'
        target_text: required if mode='target'
        progress_fn: optional callback(step, total, loss_val)

    Returns:
        universal_segment: tensor [1, segment_samples]
    """
    from models.asr_loader import WHISPER_SAMPLE_RATE

    segment_samples = int(segment_duration * WHISPER_SAMPLE_RATE)

    universal = torch.randn(1, segment_samples, device=next(wrapper.parameters()).device) * 0.01
    universal = universal.requires_grad_(True)

    optimizer = torch.optim.Adam([universal], lr=lr)

    if mode == "mute":
        eos_id = wrapper.processor.tokenizer.eos_token_id
        bos_id = wrapper.processor.tokenizer.bos_token_id
        target_ids = torch.tensor([[bos_id, eos_id]], device=universal.device)
    else:
        target_ids = wrapper.tokenize_target(target_text)

    best_segment = universal.detach().clone()
    best_loss = float('inf')

    for step in range(iterations):
        optimizer.zero_grad()

        total_loss = 0.0
        n_waveforms = len(training_waveforms)
        if n_waveforms == 0:
            raise ValueError("training_waveforms is empty, cannot optimize universal segment")
        for waveform in training_waveforms:
            prepended = torch.cat([
                torch.clamp(universal, -1.0, 1.0),
                waveform
            ], dim=-1)

            outputs = wrapper.forward_with_labels(prepended, target_ids)
            total_loss = total_loss + outputs.loss

        avg_loss = total_loss / n_waveforms
        avg_loss.backward()
        optimizer.step()

        with torch.no_grad():
            universal.data.clamp_(-0.5, 0.5)

        loss_val = avg_loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_segment = universal.detach().clone()

        if progress_fn and step % 5 == 0:
            progress_fn(step, iterations, loss_val)

        if step % 20 == 0:
            logger.info(f"[Universal Muting] step {step}/{iterations}, "
                  f"loss={loss_val:.4f}, best={best_loss:.4f}")

    return torch.clamp(best_segment, -1.0, 1.0).detach()


def apply_universal_segment(segment, waveform):
    """
    Prepend a universal adversarial segment to a waveform.

    Args:
        segment: tensor [1, S] universal adversarial audio
        waveform: tensor [1, N] speech to mute

    Returns:
        tensor [1, S+N] prepended audio
    """
    return torch.cat([segment, waveform], dim=-1)


def generate_training_waveforms(wrapper, waveform, n_augments=5):
    """
    Generate a small batch of training waveforms from a single input
    by applying minor augmentations (noise, time shift, amplitude scale).
    """
    device = waveform.device
    waveforms = [waveform.detach().clone()]

    for _ in range(n_augments - 1):
        aug = waveform.detach().clone()
        aug = aug + torch.randn_like(aug) * 0.005
        scale = 0.8 + 0.4 * torch.rand(1, device=device).item()
        aug = aug * scale
        shift = int(torch.randint(0, max(1, waveform.shape[-1] // 20), (1,)).item())
        if shift > 0:
            aug = torch.cat([torch.zeros(1, shift, device=device), aug[:, :-shift]], dim=-1)
        aug = torch.clamp(aug, -1.0, 1.0)
        waveforms.append(aug)

    return waveforms
