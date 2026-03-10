"""
ASR-specific utilities for transcription attacks.

Provides WER computation, SNR measurement, transcription comparison
visualization, and waveform plotting helpers for the transcription
and hidden-command attack tabs.
"""

import numpy as np
import torch
import torchaudio
import soundfile as sf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import device


def load_audio_for_asr(filepath, target_sr=16000):
    """
    Load an audio file and resample to target sample rate.
    Uses soundfile directly since torchaudio 2.10+ removed the
    soundfile backend.

    Returns:
        (waveform_tensor [1, num_samples] on device, sample_rate int)
    """
    data, sr = sf.read(filepath, dtype="float32")
    waveform = torch.from_numpy(data)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.mean(dim=-1, keepdim=False).unsqueeze(0)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    waveform = waveform.to(device)
    return waveform, target_sr


def compute_wer(reference, hypothesis):
    """
    Compute Word Error Rate between reference and hypothesis.

    Returns:
        float WER (0.0 = perfect, 1.0 = all wrong, >1.0 = many insertions)
    """
    try:
        from jiwer import wer
        return wer(reference.lower().strip(), hypothesis.lower().strip())
    except ImportError:
        ref_words = reference.lower().strip().split()
        hyp_words = hypothesis.lower().strip().split()
        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0
        return abs(len(ref_words) - len(hyp_words)) / len(ref_words)


def compute_snr(original, adversarial):
    """
    Compute Signal-to-Noise Ratio in dB.

    Args:
        original: tensor [1, N]
        adversarial: tensor [1, N]
    Returns:
        float SNR in dB
    """
    orig = original.detach().cpu().float()
    adv = adversarial.detach().cpu().float()
    min_len = min(orig.shape[-1], adv.shape[-1])
    orig = orig[..., :min_len]
    adv = adv[..., :min_len]
    noise = adv - orig

    signal_power = (orig ** 2).mean()
    noise_power = (noise ** 2).mean()

    if noise_power < 1e-20:
        return float('inf')

    return 10.0 * torch.log10(signal_power / noise_power).item()


def waveform_to_numpy(waveform_tensor):
    """Convert waveform tensor to numpy for gr.Audio output."""
    wav = waveform_tensor.detach().cpu().squeeze().numpy()
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767).astype(np.int16)


def plot_transcription_waveforms(original, adversarial, sr,
                                  orig_text, adv_text):
    """
    Create a comparison figure: original vs adversarial waveform with
    transcription text annotations.
    """
    orig_np = original.detach().cpu().squeeze().numpy()
    adv_np = adversarial.detach().cpu().squeeze().numpy()
    duration = len(orig_np) / sr
    t = np.linspace(0, duration, len(orig_np))

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    axes[0].plot(t, orig_np, linewidth=0.4, color="#667eea")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f'Original: "{orig_text}"', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, adv_np, linewidth=0.4, color="#e53e3e")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title(f'Adversarial: "{adv_text}"', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_perturbation_detail(original, adversarial, sr, magnification=50):
    """
    Plot the perturbation (difference) between original and adversarial,
    magnified for visibility.
    """
    orig_np = original.detach().cpu().squeeze().numpy()
    adv_np = adversarial.detach().cpu().squeeze().numpy()
    diff = adv_np - orig_np
    duration = len(orig_np) / sr
    t = np.linspace(0, duration, len(orig_np))

    fig, ax = plt.subplots(1, 1, figsize=(10, 2.5))
    ax.plot(t, diff * magnification, linewidth=0.4, color="#d69e2e")
    ax.set_xlim(0, duration)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Perturbation ({magnification}x)")
    ax.set_title(f"Perturbation (magnified {magnification}x)", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_spectrogram_pair(original, adversarial, sr):
    """
    Side-by-side mel spectrogram comparison of original and adversarial.
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=400, hop_length=160, n_mels=80
    )

    orig_mel = mel_transform(original.detach().cpu().squeeze())
    adv_mel = mel_transform(adversarial.detach().cpu().squeeze())

    orig_db = torchaudio.transforms.AmplitudeToDB()(orig_mel).numpy()
    adv_db = torchaudio.transforms.AmplitudeToDB()(adv_mel).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    axes[0].imshow(orig_db, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Original Mel Spectrogram", fontsize=10)
    axes[0].set_ylabel("Mel Bin")
    axes[0].set_xlabel("Frame")

    axes[1].imshow(adv_db, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Adversarial Mel Spectrogram", fontsize=10)
    axes[1].set_ylabel("Mel Bin")
    axes[1].set_xlabel("Frame")

    fig.tight_layout()
    return fig
