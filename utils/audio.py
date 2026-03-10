"""
Audio loading, preprocessing, prediction, and visualization utilities.
"""

import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import device


def load_audio(filepath, target_sr=16000):
    """
    Load audio file and resample to target sample rate.
    Uses soundfile directly since torchaudio 2.10+ removed the
    soundfile backend.

    Returns:
        waveform tensor [1, num_samples] on device, sample_rate
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

    waveform = waveform / (waveform.abs().max() + 1e-8)

    return waveform.to(device), target_sr


def get_audio_predictions(model_wrapper, waveform, top_k=5):
    """
    Run inference on a waveform through the wrapped audio model.

    Args:
        model_wrapper: AudioModelWrapper instance
        waveform: tensor [1, num_samples]
        top_k: number of top predictions

    Returns:
        (results_dict, top_class_name, top_class_idx)
    """
    model_wrapper.eval()

    with torch.no_grad():
        outputs = model_wrapper(waveform)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        top_prob, top_idx = torch.topk(probs, min(top_k, probs.shape[1]))

    results = {}
    config = model_wrapper.config
    for i in range(top_prob.shape[1]):
        idx = int(top_idx[0][i].item())
        prob = float(top_prob[0][i].item())
        name = config.id2label.get(idx, f"class_{idx}")
        results[name] = prob

    top_class_idx = int(top_idx[0][0].item())
    top_class_name = config.id2label.get(top_class_idx, f"class_{top_class_idx}")

    return results, top_class_name, top_class_idx


def get_audio_label_choices(model_wrapper):
    """Return sorted list of (label_name, label_idx) for the model's classes."""
    config = model_wrapper.config
    labels = []
    for idx, name in sorted(config.id2label.items()):
        labels.append(f"{name} ({idx})")
    return labels


def parse_target_label(label_str):
    """Extract class index from label string like 'yes (1)'."""
    try:
        idx_str = label_str.rsplit("(", 1)[1].rstrip(")")
        return int(idx_str)
    except (IndexError, ValueError):
        return 0


def waveform_to_numpy(waveform_tensor):
    """Convert waveform tensor to numpy for audio playback."""
    wav = waveform_tensor.detach().cpu().squeeze().numpy()
    wav = np.clip(wav, -1.0, 1.0)
    return wav


def plot_waveform(waveform_tensor, sr, title="Waveform"):
    """Create a matplotlib figure of the waveform."""
    wav = waveform_tensor.detach().cpu().squeeze().numpy()
    duration = len(wav) / sr
    t = np.linspace(0, duration, len(wav))

    fig, ax = plt.subplots(1, 1, figsize=(8, 2.5))
    ax.plot(t, wav, linewidth=0.5, color="#667eea")
    ax.set_xlim(0, duration)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_perturbation(original, adversarial, sr, magnification=50):
    """Plot the perturbation (difference) between original and adversarial."""
    orig = original.detach().cpu().squeeze().numpy()
    adv = adversarial.detach().cpu().squeeze().numpy()

    min_len = min(len(orig), len(adv))
    orig, adv = orig[:min_len], adv[:min_len]

    diff = (adv - orig) * magnification
    duration = min_len / sr
    t = np.linspace(0, duration, min_len)

    fig, ax = plt.subplots(1, 1, figsize=(8, 2.5))
    ax.plot(t, diff, linewidth=0.5, color="#e53e3e")
    ax.set_xlim(0, duration)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Δ × {magnification}")
    ax.set_title(f"Perturbation ({magnification}× magnified)", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_spectrogram_comparison(original, adversarial, sr):
    """Side-by-side spectrograms of original and adversarial audio."""
    orig = original.detach().cpu().squeeze().numpy()
    adv = adversarial.detach().cpu().squeeze().numpy()
    min_len = min(len(orig), len(adv))
    orig, adv = orig[:min_len], adv[:min_len]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    ax1.specgram(orig, Fs=sr, NFFT=512, noverlap=256, cmap="viridis")
    ax1.set_title("Original", fontsize=10)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")

    ax2.specgram(adv, Fs=sr, NFFT=512, noverlap=256, cmap="viridis")
    ax2.set_title("Adversarial", fontsize=10)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")

    fig.tight_layout()
    return fig
