"""
AdvPulse — Subsecond, Synchronization-Free Universal Adversarial Perturbation.

Implements the streaming-audio attack of:
  Li, Zhang, Liu, Xu, Chen (ACM CCS 2020),
  "AdvPulse: Universal, Synchronization-free, and Targeted Audio
  Adversarial Attacks via Subsecond Perturbations".

Key properties that distinguish AdvPulse from the other attacks here:

  * Subsecond: the adversarial perturbation is a very short pulse
    (~0.1-0.5 s), far shorter than the speech utterance.
  * Synchronization-free: the attacker cannot know *when* the pulse will
    land relative to the victim's speech. We therefore train the pulse to
    be effective when injected at a *random offset* into the audio
    (Expectation-over-random-position), so it works without alignment.
  * Universal + targeted: a single pulse forces an attacker-chosen output
    (target transcription for ASR, or target class) across many inputs and
    offsets.
  * Physically robust: optionally trained under Expectation-over-
    Transformation (room impulse responses + ambient noise) so it survives
    over-the-air playback, reusing the RIR simulator.

The pulse is *added* (overlaid) into the waveform at a random position each
step, unlike the muting attack which *prepends* a segment. This models a
real-world adversary emitting a brief sound while the victim speaks.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def _inject_pulse(waveform, pulse, offset):
    """
    Overlay ``pulse`` onto ``waveform`` starting at sample ``offset``.

    Differentiable: uses zero-padding + slicing so gradients flow to pulse.
    Returns a waveform of the same length as the input.
    """
    n = waveform.shape[-1]
    p = pulse.shape[-1]
    if offset + p > n:
        offset = max(0, n - p)
    left = torch.zeros(1, offset, device=waveform.device)
    right = torch.zeros(1, max(0, n - offset - p), device=waveform.device)
    placed = torch.cat([left, pulse, right], dim=-1)[:, :n]
    return waveform + placed


def advpulse_attack(
    wrapper,
    training_waveforms,
    target_text,
    pulse_duration=0.3,
    epsilon=0.1,
    iterations=400,
    lr=0.005,
    physical=False,
    n_rooms=3,
    noise_snr_db=25.0,
    progress_fn=None,
):
    """
    Learn a subsecond, synchronization-free, universal *targeted* pulse for ASR.

    Args:
        wrapper: WhisperAttackWrapper (exposes forward_with_labels/tokenize_target).
        training_waveforms: list of tensors [1, N] to generalize across.
        target_text: attacker-chosen transcription to force.
        pulse_duration: pulse length in seconds (subsecond, e.g. 0.1-0.5).
        epsilon: L-inf bound on the pulse amplitude.
        iterations: optimization steps.
        lr: learning rate.
        physical: if True, apply Expectation-over-Transformation
            (random RIR convolution + ambient noise) for over-the-air robustness.
        n_rooms: number of synthetic rooms when ``physical``.
        noise_snr_db: ambient noise SNR when ``physical``.
        progress_fn: optional callback(step, total, loss).

    Returns:
        pulse tensor [1, pulse_samples] (the universal adversarial pulse).
    """
    from models.asr_loader import WHISPER_SAMPLE_RATE

    if not training_waveforms:
        raise ValueError("training_waveforms is empty")

    device = next(wrapper.parameters()).device
    target_ids = wrapper.tokenize_target(target_text)

    pulse_samples = int(pulse_duration * WHISPER_SAMPLE_RATE)
    pulse = (torch.randn(1, pulse_samples, device=device) * 0.01).requires_grad_(True)
    optimizer = torch.optim.Adam([pulse], lr=lr)

    rirs = None
    noise_amp = 0.0
    if physical:
        from attacks.audio.over_the_air_attack import apply_rir, generate_rir_batch
        rirs = generate_rir_batch(n_rooms=n_rooms, sr=WHISPER_SAMPLE_RATE, device=device)
        noise_amp = 10.0 ** (-noise_snr_db / 20.0)

    best_pulse = pulse.detach().clone()
    best_loss = float("inf")

    for step in range(iterations):
        optimizer.zero_grad()

        total_loss = 0.0
        for waveform in training_waveforms:
            n = waveform.shape[-1]
            max_off = max(0, n - pulse_samples)
            offset = int(torch.randint(0, max_off + 1, (1,)).item()) if max_off > 0 else 0

            clamped_pulse = torch.clamp(pulse, -epsilon, epsilon)
            adv = _inject_pulse(waveform, clamped_pulse, offset)
            adv = torch.clamp(adv, -1.0, 1.0)

            if physical:
                from attacks.audio.over_the_air_attack import apply_rir
                rir = rirs[step % len(rirs)]
                adv = apply_rir(adv, rir)
                adv = torch.clamp(adv + torch.randn_like(adv) * noise_amp, -1.0, 1.0)

            outputs = wrapper.forward_with_labels(adv, target_ids)
            total_loss = total_loss + outputs.loss

        avg_loss = total_loss / len(training_waveforms)
        avg_loss.backward()
        optimizer.step()

        with torch.no_grad():
            pulse.data.clamp_(-epsilon, epsilon)

        loss_val = avg_loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_pulse = pulse.detach().clone()

        if progress_fn and step % 5 == 0:
            progress_fn(step, iterations, loss_val)
        if step % 20 == 0:
            logger.info(f"[AdvPulse] step {step}/{iterations}, "
                        f"loss={loss_val:.4f}, best={best_loss:.4f}, "
                        f"physical={physical}")

    return torch.clamp(best_pulse, -epsilon, epsilon).detach()


def apply_advpulse(pulse, waveform, offset=None):
    """
    Inject the universal pulse into a waveform at ``offset`` (random if None).

    Returns the perturbed waveform [1, N], clamped to valid audio range.
    """
    n = waveform.shape[-1]
    p = pulse.shape[-1]
    if offset is None:
        max_off = max(0, n - p)
        offset = int(torch.randint(0, max_off + 1, (1,)).item()) if max_off > 0 else 0
    adv = _inject_pulse(waveform, pulse, offset)
    return torch.clamp(adv, -1.0, 1.0)
