"""
Speech Jamming / Denial-of-Service Attack on ASR.

Untargeted attack that maximizes ASR confusion: instead of forcing a
specific transcription, this attack degrades recognition accuracy to
near-zero by maximizing the loss against the correct transcription.
The model outputs gibberish or silence.

Two variants:
  1. Untargeted (maximize WER): push away from correct transcription
  2. Universal jamming noise: frequency-band noise overlay that
     disrupts mel-spectrogram features without being overly loud

Inspired by: "Adversarial Music: Real World Audio Adversary Against
Wake-Word Detection System" (NeurIPS 2019) adapted for full ASR.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def speech_jamming_untargeted(
    wrapper,
    waveform,
    epsilon=0.05,
    iterations=200,
    lr=0.005,
    progress_fn=None,
):
    """
    Untargeted attack: maximize loss against the original transcription.

    First transcribes the audio to get ground truth, then optimizes
    perturbation to MAXIMIZE the cross-entropy loss against those tokens,
    causing the model to output anything except the correct text.

    Args:
        wrapper: WhisperAttackWrapper
        waveform: tensor [1, N]
        epsilon: perturbation bound
        iterations: optimization steps
        lr: learning rate
        progress_fn: optional callback

    Returns:
        adversarial waveform tensor [1, N]
    """
    original_text = wrapper.transcribe(waveform)
    if not original_text.strip():
        original_text = "hello"

    original_ids = wrapper.tokenize_target(original_text)

    waveform = waveform.detach().clone()
    delta = torch.zeros_like(waveform, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    best_delta = delta.detach().clone()
    best_loss = float('-inf')

    for step in range(iterations):
        optimizer.zero_grad()

        adv_waveform = torch.clamp(waveform + delta, -1.0, 1.0)

        outputs = wrapper.forward_with_labels(adv_waveform, original_ids)
        loss = -outputs.loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)
            clamped = torch.clamp(waveform + delta, -1.0, 1.0)
            delta.data.copy_(clamped - waveform)

        neg_loss = -loss.item()
        if neg_loss > best_loss:
            best_loss = neg_loss
            best_delta = delta.detach().clone()

        if progress_fn and step % 5 == 0:
            progress_fn(step, iterations, neg_loss)

        if step % 20 == 0:
            logger.info(f"[Jamming Untargeted] step {step}/{iterations}, "
                  f"ce_loss={neg_loss:.4f} (maximizing)")

    adv_waveform = torch.clamp(waveform + best_delta, -1.0, 1.0)
    return adv_waveform.detach()


def speech_jamming_band_noise(
    wrapper,
    waveform,
    epsilon=0.05,
    iterations=200,
    lr=0.005,
    band_low_hz=300,
    band_high_hz=3400,
    progress_fn=None,
):
    """
    Band-limited jamming: optimize noise only in the speech frequency band
    (300-3400 Hz) to maximally disrupt ASR while keeping the noise
    spectrally constrained and less perceptible.

    The perturbation is shaped in the frequency domain: energy is
    concentrated in the critical speech band where formants and phonemes
    live, making it maximally disruptive per unit of noise energy.

    Args:
        wrapper: WhisperAttackWrapper
        waveform: tensor [1, N]
        epsilon: perturbation bound
        iterations: optimization steps
        lr: learning rate
        band_low_hz: lower frequency boundary
        band_high_hz: upper frequency boundary
        progress_fn: optional callback

    Returns:
        adversarial waveform tensor [1, N]
    """
    from models.asr_loader import WHISPER_SAMPLE_RATE

    original_text = wrapper.transcribe(waveform)
    if not original_text.strip():
        original_text = "hello"
    original_ids = wrapper.tokenize_target(original_text)

    waveform = waveform.detach().clone()
    device = waveform.device
    n_samples = waveform.shape[-1]

    fft_bins = n_samples // 2 + 1
    freqs = torch.linspace(0, WHISPER_SAMPLE_RATE / 2, fft_bins, device=device)

    band_mask = ((freqs >= band_low_hz) & (freqs <= band_high_hz)).float()

    delta = torch.zeros_like(waveform, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    best_delta_shaped = torch.zeros_like(waveform)
    best_loss = float('-inf')

    for step in range(iterations):
        optimizer.zero_grad()

        delta_fft = torch.fft.rfft(delta.squeeze())
        shaped_fft = delta_fft * band_mask
        shaped_delta = torch.fft.irfft(shaped_fft, n=n_samples).unsqueeze(0)

        adv_waveform = torch.clamp(waveform + shaped_delta, -1.0, 1.0)

        outputs = wrapper.forward_with_labels(adv_waveform, original_ids)
        loss = -outputs.loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)

        neg_loss = -loss.item()
        if neg_loss > best_loss:
            best_loss = neg_loss
            delta_fft_snap = torch.fft.rfft(delta.detach().squeeze())
            best_delta_shaped = torch.fft.irfft(delta_fft_snap * band_mask, n=n_samples).unsqueeze(0)
            best_delta_shaped = best_delta_shaped.clamp(-epsilon, epsilon)

        if progress_fn and step % 5 == 0:
            progress_fn(step, iterations, neg_loss)

        if step % 20 == 0:
            logger.info(f"[Band Jamming] step {step}/{iterations}, "
                  f"ce_loss={neg_loss:.4f} (maximizing), "
                  f"band={band_low_hz}-{band_high_hz}Hz")

    adv_waveform = torch.clamp(waveform + best_delta_shaped, -1.0, 1.0)
    return adv_waveform.detach()
