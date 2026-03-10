"""
Psychoacoustic Adversarial Attack (Qin et al., 2019).

Uses auditory masking thresholds to constrain perturbations to frequencies
and time regions where humans cannot perceive them. The perturbation can
have substantial energy in absolute terms but remains imperceptible because
it falls below the masking threshold created by the original audio.

Based on: "Imperceptible, Robust, and Targeted Adversarial Examples for
Automatic Speech Recognition" (Qin et al., ICML 2019).
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_masking_threshold(waveform, sr=16000, n_fft=512, hop_length=160):
    """
    Approximate auditory masking threshold from the original waveform.

    Uses a simplified psychoacoustic model: the masking threshold at each
    time-frequency bin is proportional to the power of the original signal
    in that bin, scaled by a masking factor that decreases with frequency
    distance (spreading function).

    Args:
        waveform: tensor [1, N]
        sr: sample rate
        n_fft: FFT size
        hop_length: hop length

    Returns:
        masking_threshold: tensor [1, n_fft//2+1, T] in magnitude scale
    """
    wav = waveform.detach().squeeze()

    window = torch.hann_window(n_fft, device=wav.device)
    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop_length,
                       win_length=n_fft, window=window,
                       return_complex=True)
    power = stft.abs() ** 2

    n_freq = power.shape[0]
    freqs = torch.linspace(0, sr / 2, n_freq, device=wav.device)

    bark = 13.0 * torch.atan(0.00076 * freqs) + \
           3.5 * torch.atan((freqs / 7500.0) ** 2)

    spread = torch.zeros(n_freq, n_freq, device=wav.device)
    for i in range(n_freq):
        db_diff = bark - bark[i]
        power_db = 10 * torch.log10(torch.clamp(power[i].mean(), min=1e-20))
        s = torch.where(
            db_diff < 0,
            27.0 * db_diff,
            (-27.0 + 0.37 * torch.clamp(power_db, min=0)) * db_diff
        )
        spread[i] = 10.0 ** (s / 10.0)
        spread[i] = torch.clamp(spread[i], min=1e-10, max=1.0)

    masked_power = torch.matmul(spread, power)

    abs_threshold_db = 3.64 * (freqs / 1000.0) ** (-0.8) - \
                       6.5 * torch.exp(-0.6 * (freqs / 1000.0 - 3.3) ** 2) + \
                       1e-3 * (freqs / 1000.0) ** 4
    abs_threshold_db = torch.clamp(abs_threshold_db, min=-20, max=80)
    abs_threshold = 10.0 ** (abs_threshold_db / 20.0) * 1e-5

    masking_factor = 0.1
    threshold = masking_factor * masked_power.sqrt() + abs_threshold.unsqueeze(-1)

    return threshold.unsqueeze(0)


def psychoacoustic_transcription_attack(
    wrapper,
    waveform,
    target_text,
    iterations=300,
    lr=0.005,
    masking_weight=1.0,
    sr=16000,
    n_fft=512,
    hop_length=160,
    progress_fn=None,
):
    """
    Targeted transcription attack with psychoacoustic imperceptibility.

    Instead of L-inf clipping, constrains the perturbation spectrum to stay
    below the auditory masking threshold at each time-frequency bin.

    Args:
        wrapper: WhisperAttackWrapper
        waveform: tensor [1, N]
        target_text: target transcription
        iterations: optimization steps
        lr: learning rate
        masking_weight: balance between attack loss and masking penalty
        n_fft: FFT size for masking computation
        hop_length: hop length
        progress_fn: optional callback

    Returns:
        adversarial waveform tensor [1, N]
    """
    target_ids = wrapper.tokenize_target(target_text)

    waveform = waveform.detach().clone()
    device = waveform.device

    threshold = compute_masking_threshold(
        waveform, sr=sr, n_fft=n_fft, hop_length=hop_length
    )

    delta = torch.zeros_like(waveform, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    window = torch.hann_window(n_fft, device=device)

    best_delta = delta.detach().clone()
    best_loss = float('inf')

    for step in range(iterations):
        optimizer.zero_grad()

        adv_waveform = torch.clamp(waveform + delta, -1.0, 1.0)

        outputs = wrapper.forward_with_labels(adv_waveform, target_ids)
        attack_loss = outputs.loss

        delta_stft = torch.stft(
            delta.squeeze(), n_fft=n_fft, hop_length=hop_length,
            win_length=n_fft, window=window, return_complex=True
        )
        delta_mag = delta_stft.abs().unsqueeze(0)

        min_len = min(delta_mag.shape[-1], threshold.shape[-1])
        delta_mag_t = delta_mag[..., :min_len]
        threshold_t = threshold[..., :min_len]

        exceed = F.relu(delta_mag_t - threshold_t)
        masking_loss = exceed.pow(2).mean()

        total_loss = attack_loss + masking_weight * masking_loss

        total_loss.backward()
        optimizer.step()

        loss_val = attack_loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_delta = delta.detach().clone()

        if progress_fn and step % 5 == 0:
            progress_fn(step, iterations, loss_val)

        if step % 20 == 0:
            logger.info(f"[Psychoacoustic] step {step}/{iterations}, "
                  f"attack_loss={loss_val:.4f}, mask_loss={masking_loss.item():.4f}, "
                  f"best={best_loss:.4f}")

    adv_waveform = torch.clamp(waveform + best_delta, -1.0, 1.0)
    return adv_waveform.detach()
