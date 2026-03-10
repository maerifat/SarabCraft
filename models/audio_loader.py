"""
Audio model loading, caching, and differentiable preprocessing wrappers.

The key challenge: existing attacks call model(tensor) and read outputs.logits.
AudioModelWrapper wraps an audio classifier so that attacks can pass raw waveform
tensors and get .logits back, with fully differentiable preprocessing in the
forward pass so gradients flow back to the waveform.
"""

import torch
import torch.nn as nn
import torchaudio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    ASTFeatureExtractor,
)

from config import device


audio_model_cache = {}


class AudioModelWrapper(nn.Module):
    """
    Wraps a HuggingFace audio classifier so that forward(waveform) returns
    an object with .logits, enabling reuse of all gradient-based image attacks.

    For Wav2Vec2 / HuBERT: raw waveform goes directly to the model (differentiable).
    For AST: builds a differentiable mel-spectrogram pipeline with torchaudio.
    """

    def __init__(self, model, feature_extractor, model_id, sample_rate=16000):
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        self.model_id = model_id
        self.expected_sr = sample_rate
        self._model_type = self._detect_type(model_id)
        self.config = model.config

        if self._model_type == "ast":
            self._build_ast_pipeline()

    def _detect_type(self, model_id):
        mid = model_id.lower()
        if "ast" in mid:
            return "ast"
        return "waveform"

    def _build_ast_pipeline(self):
        """Build a differentiable mel-spectrogram matching AST's expectations."""
        fe = self.feature_extractor
        n_fft = getattr(fe, "n_fft", 400)
        hop = getattr(fe, "hop_length", 160)
        n_mels = getattr(fe, "num_mel_bins", 128)
        sr = getattr(fe, "sampling_rate", 16000)
        self.expected_sr = sr

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            power=2.0,
        ).to(device)

        self.ast_mean = getattr(fe, "mean", -4.2677393)
        self.ast_std = getattr(fe, "std", 4.5689974)
        self.ast_max_length = getattr(fe, "max_length", 1024)

    def forward(self, waveform):
        """
        Args:
            waveform: tensor of shape [1, num_samples] (raw audio, float, -1..1 range)
        Returns:
            object with .logits attribute
        """
        if self._model_type == "ast":
            return self._forward_ast(waveform)
        return self._forward_waveform(waveform)

    def _forward_waveform(self, waveform):
        """Wav2Vec2 / HuBERT: feed raw waveform directly."""
        if waveform.dim() == 2:
            wav = waveform.squeeze(0)
        else:
            wav = waveform
        return self.model(input_values=wav.unsqueeze(0))

    def _forward_ast(self, waveform):
        """AST: differentiable mel-spectrogram -> model."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel = self.mel_spec(waveform)
        log_mel = torch.log(mel + 1e-10)
        log_mel = (log_mel - self.ast_mean) / (self.ast_std + 1e-8)

        spec = log_mel.squeeze(0)
        if spec.dim() == 2:
            spec = spec.transpose(0, 1)

        max_len = self.ast_max_length
        if spec.shape[0] > max_len:
            spec = spec[:max_len, :]
        elif spec.shape[0] < max_len:
            pad = torch.zeros(max_len - spec.shape[0], spec.shape[1], device=spec.device)
            spec = torch.cat([spec, pad], dim=0)

        input_values = spec.unsqueeze(0)
        return self.model(input_values=input_values)


def load_audio_model(model_name, progress=None):
    """
    Load a HuggingFace audio classification model, wrapped for attack compatibility.

    Returns:
        (AudioModelWrapper, feature_extractor, sample_rate)
    """
    global audio_model_cache

    def update_progress(value, desc):
        if progress is not None:
            try:
                progress(value, desc=desc)
            except Exception:
                pass

    if model_name in audio_model_cache:
        update_progress(1.0, f"✅ {model_name} (cached)")
        return audio_model_cache[model_name]

    update_progress(0.1, f"⬇️ Downloading {model_name}...")
    print(f"Loading audio model: {model_name}...", flush=True)

    update_progress(0.2, "📦 Loading feature extractor...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    update_progress(0.5, "📦 Loading model weights...")
    model = AutoModelForAudioClassification.from_pretrained(model_name)

    update_progress(0.8, f"🔧 Moving to {device}...")
    model = model.to(device)
    model.eval()

    sr = getattr(feature_extractor, "sampling_rate", 16000)

    wrapper = AudioModelWrapper(model, feature_extractor, model_name, sr)
    wrapper = wrapper.to(device)
    wrapper.eval()

    audio_model_cache[model_name] = (wrapper, feature_extractor, sr)

    update_progress(1.0, f"✅ {model_name} ready!")
    print(f"✅ Audio model {model_name} loaded on {device}", flush=True)

    return audio_model_cache[model_name]
