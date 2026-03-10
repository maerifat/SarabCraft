"""
ASR (speech-to-text) model loading with differentiable preprocessing.

WhisperAttackWrapper wraps a HuggingFace Whisper model so that gradient-based
attacks can optimize raw waveform perturbations while the mel-spectrogram
computation remains differentiable (via torchaudio).
"""

import torch
import torch.nn as nn
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config import device


asr_model_cache = {}

WHISPER_SAMPLE_RATE = 16000
WHISPER_N_FFT = 400
WHISPER_HOP_LENGTH = 160
WHISPER_N_MELS = 80
WHISPER_CHUNK_LENGTH = 30  # seconds


class WhisperAttackWrapper(nn.Module):
    """
    Wraps Whisper for gradient-based adversarial attacks on transcription.

    The key insight: Whisper's feature extractor uses a non-differentiable numpy
    pipeline. We replace it with torchaudio.transforms.MelSpectrogram so gradients
    flow from the cross-entropy loss all the way back to the raw waveform.
    """

    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=WHISPER_SAMPLE_RATE,
            n_fft=WHISPER_N_FFT,
            hop_length=WHISPER_HOP_LENGTH,
            n_mels=WHISPER_N_MELS,
            power=2.0,
        ).to(device)

        self._max_frames = WHISPER_CHUNK_LENGTH * WHISPER_SAMPLE_RATE // WHISPER_HOP_LENGTH

    def extract_features(self, waveform):
        """
        Differentiable log-mel spectrogram matching Whisper's expectations.

        Args:
            waveform: tensor [1, num_samples] or [num_samples], float, -1..1
        Returns:
            input_features: tensor [1, n_mels, max_frames]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel = self.mel_spec(waveform)
        log_mel = torch.log10(torch.clamp(mel, min=1e-10))
        log_mel = torch.clamp(log_mel, min=log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0

        n_frames = log_mel.shape[-1]
        if n_frames > self._max_frames:
            log_mel = log_mel[..., :self._max_frames]
        elif n_frames < self._max_frames:
            pad_size = self._max_frames - n_frames
            log_mel = torch.nn.functional.pad(log_mel, (0, pad_size), value=0.0)

        return log_mel

    def forward_with_labels(self, waveform, target_ids):
        """
        Forward pass that returns loss against target token IDs.

        Args:
            waveform: tensor [1, num_samples]
            target_ids: tensor [1, seq_len] of target token IDs
        Returns:
            outputs with .loss attribute (cross-entropy toward target)
        """
        input_features = self.extract_features(waveform)
        return self.model(input_features=input_features, labels=target_ids)

    @torch.no_grad()
    def transcribe(self, waveform):
        """
        Non-differentiable transcription using Whisper's generate().

        Args:
            waveform: tensor [1, num_samples] or numpy array
        Returns:
            transcription string
        """
        if isinstance(waveform, torch.Tensor):
            wav_np = waveform.detach().cpu().squeeze().numpy()
        else:
            wav_np = waveform.squeeze()

        inputs = self.processor(
            wav_np,
            sampling_rate=WHISPER_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device)

        predicted_ids = self.model.generate(input_features)
        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text.strip()

    def tokenize_target(self, target_text):
        """
        Tokenize target transcription for use as labels.

        Returns:
            tensor [1, seq_len] on device
        """
        target_ids = self.processor.tokenizer(
            target_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids

        bos = torch.tensor([[self.processor.tokenizer.bos_token_id]], dtype=torch.long)
        target_ids = torch.cat([bos, target_ids], dim=-1)

        return target_ids.to(device)


def load_asr_model(model_name, progress=None):
    """
    Load a Whisper model wrapped for adversarial attacks.

    Returns:
        (WhisperAttackWrapper, processor)
    """
    global asr_model_cache

    def update_progress(value, desc):
        if progress is not None:
            try:
                progress(value, desc=desc)
            except Exception:
                pass

    if model_name in asr_model_cache:
        update_progress(1.0, f"Cached: {model_name}")
        return asr_model_cache[model_name]

    update_progress(0.1, f"Downloading {model_name}...")
    print(f"Loading ASR model: {model_name}...", flush=True)

    update_progress(0.3, "Loading processor...")
    processor = WhisperProcessor.from_pretrained(model_name)

    update_progress(0.5, "Loading model weights...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    update_progress(0.8, f"Moving to {device}...")
    model = model.to(device)
    model.eval()

    wrapper = WhisperAttackWrapper(model, processor)
    wrapper = wrapper.to(device)
    wrapper.eval()

    asr_model_cache[model_name] = (wrapper, processor)

    update_progress(1.0, f"{model_name} ready!")
    print(f"ASR model {model_name} loaded on {device}", flush=True)

    return asr_model_cache[model_name]
