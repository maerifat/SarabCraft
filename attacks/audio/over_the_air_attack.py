"""
Over-the-Air Robust Adversarial Attack (Imperio 2020 style).

Generates adversarial perturbations that survive physical playback:
speaker → air → microphone. During optimization, we simulate this
channel by convolving with Room Impulse Responses (RIRs) and adding
ambient noise, so the resulting adversarial audio remains effective
after real-world acoustic distortion.

Based on: "Imperio: Robust Over-the-Air Adversarial Examples for
Automatic Speech Recognition Systems" (Schönherr et al., 2020)
and "Imperceptible, Robust, and Targeted Adversarial Examples for
ASR" (Qin et al., ICML 2019).
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


def generate_synthetic_rir(duration=0.5, sr=16000, decay=3.0, n_reflections=20,
                           device='cpu'):
    """
    Generate a synthetic room impulse response (RIR).

    Uses the image-source method approximation: a direct path impulse
    followed by exponentially decaying reflections at random delays.

    Args:
        duration: RIR length in seconds
        sr: sample rate
        decay: exponential decay rate (higher = more absorptive room)
        n_reflections: number of early reflections
        device: torch device

    Returns:
        rir: tensor [1, samples]
    """
    n_samples = int(duration * sr)
    rir = torch.zeros(1, n_samples, device=device)

    rir[0, 0] = 1.0

    for i in range(n_reflections):
        delay = int(torch.randint(1, n_samples, (1,)).item())
        amplitude = torch.rand(1, device=device).item() * np.exp(-decay * delay / n_samples)
        if torch.rand(1).item() > 0.5:
            amplitude = -amplitude
        rir[0, delay] += amplitude

    t = torch.arange(n_samples, device=device, dtype=torch.float32)
    tail = torch.randn(n_samples, device=device) * 0.01 * torch.exp(-decay * t / n_samples)
    rir[0] += tail

    rir = rir / (rir.abs().max() + 1e-8)

    return rir


def generate_rir_batch(n_rooms=5, sr=16000, device='cpu'):
    """
    Generate diverse synthetic RIRs simulating different room acoustics.
    """
    rirs = []
    configs = [
        {"duration": 0.3, "decay": 5.0, "n_reflections": 10},   # small dry room
        {"duration": 0.5, "decay": 3.0, "n_reflections": 20},   # medium office
        {"duration": 0.8, "decay": 2.0, "n_reflections": 30},   # large room
        {"duration": 0.4, "decay": 4.0, "n_reflections": 15},   # carpeted room
        {"duration": 0.6, "decay": 1.5, "n_reflections": 25},   # reverberant hall
    ]

    for i in range(min(n_rooms, len(configs))):
        cfg = configs[i]
        rir = generate_synthetic_rir(
            duration=cfg["duration"],
            sr=sr,
            decay=cfg["decay"],
            n_reflections=cfg["n_reflections"],
            device=device,
        )
        rirs.append(rir)

    return rirs


def apply_rir(waveform, rir):
    """
    Convolve waveform with a room impulse response (differentiable).
    """
    rir_1d = rir.squeeze()
    wav_1d = waveform.unsqueeze(0) if waveform.dim() == 2 else waveform.unsqueeze(0).unsqueeze(0)

    if wav_1d.dim() == 2:
        wav_1d = wav_1d.unsqueeze(0)

    rir_kernel = rir_1d.flip(0).unsqueeze(0).unsqueeze(0)
    pad = rir_kernel.shape[-1] - 1

    convolved = F.conv1d(wav_1d, rir_kernel, padding=pad)
    convolved = convolved[..., :waveform.shape[-1]]

    max_val = convolved.abs().max()
    if max_val > 0:
        convolved = convolved / max_val

    if waveform.dim() == 2:
        return convolved.squeeze(0)
    return convolved.squeeze(0).squeeze(0)


def over_the_air_attack(
    wrapper,
    waveform,
    target_text,
    epsilon=0.08,
    iterations=300,
    lr=0.005,
    n_rooms=3,
    noise_snr_db=30.0,
    progress_fn=None,
):
    """
    Robust adversarial attack that survives over-the-air playback.

    During each optimization step, the adversarial audio is convolved
    with a randomly selected RIR and ambient noise is added, so the
    perturbation learns to be robust to acoustic distortions.

    Args:
        wrapper: WhisperAttackWrapper
        waveform: tensor [1, N]
        target_text: target transcription
        epsilon: perturbation bound
        iterations: optimization steps
        lr: learning rate
        n_rooms: number of synthetic rooms to simulate
        noise_snr_db: SNR of added ambient noise
        progress_fn: optional callback

    Returns:
        adversarial waveform tensor [1, N]
    """
    target_ids = wrapper.tokenize_target(target_text)
    device = waveform.device

    waveform = waveform.detach().clone()
    delta = torch.zeros_like(waveform, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    rirs = generate_rir_batch(n_rooms=n_rooms, sr=16000, device=device)

    noise_amplitude = 10.0 ** (-noise_snr_db / 20.0)

    best_delta = delta.detach().clone()
    best_loss = float('inf')

    for step in range(iterations):
        optimizer.zero_grad()

        adv_waveform = torch.clamp(waveform + delta, -1.0, 1.0)

        rir_idx = step % len(rirs)
        rir = rirs[rir_idx]
        transmitted = apply_rir(adv_waveform, rir)

        noise = torch.randn_like(transmitted) * noise_amplitude
        transmitted = torch.clamp(transmitted + noise, -1.0, 1.0)

        outputs = wrapper.forward_with_labels(transmitted, target_ids)
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
            logger.info(f"[Over-the-Air] step {step}/{iterations}, "
                  f"loss={loss_val:.4f}, best={best_loss:.4f}, "
                  f"room={rir_idx}")

    adv_waveform = torch.clamp(waveform + best_delta, -1.0, 1.0)
    return adv_waveform.detach()
