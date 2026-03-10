"""
UA3: Universal Audio Adversarial Attack
=======================================
Multi-architecture ensemble attack that simultaneously targets
encoder-decoder (Whisper) and CTC-based (Wav2Vec2, HuBERT) models.

Solves the fundamental problem: no existing audio attack works across
ALL ASR architectures. UA3 combines:

  1. Generation-Aligned Loss for encoder-decoder models (SA3 innovation)
     — directly optimises encoder features for autoregressive decode.
  2. CTC Loss for CTC-based models — frame-level sequence alignment.
  3. Multi-model gradient aggregation — single perturbation fools all.
  4. Audio augmentation for transfer — speed perturbation + noise
     during optimisation reduces overfitting to any single model's
     feature extraction pipeline.

The result: one perturbation that forces ANY ASR model to transcribe
the attacker's chosen text.

Maerifat 2026
"""

import re
import torch
import torch.nn.functional as F
import torchaudio


WHISPER_SR = 16000


def _normalize_text(text):
    """Strip punctuation, extra whitespace, case for robust comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Model wrappers — unified interface for different architectures
# ---------------------------------------------------------------------------

class WhisperAttackModel:
    """Wrapper for Whisper encoder-decoder models with GA loss."""

    def __init__(self, wrapper, processor, device):
        self.wrapper = wrapper
        self.processor = processor
        self.device = device
        self.arch = "encoder-decoder"
        self.name = "Whisper"
        self._target_cache = {}

    def _setup_target(self, target_text):
        if target_text in self._target_cache:
            return self._target_cache[target_text]

        tokenizer = self.processor.tokenizer
        model = self.wrapper.model

        text_ids = tokenizer(
            target_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.squeeze().tolist()
        if isinstance(text_ids, int):
            text_ids = [text_ids]
        text_ids.append(tokenizer.eos_token_id)

        sot = model.config.decoder_start_token_id
        forced = model.config.forced_decoder_ids or []
        prefix = [sot]
        for _, tid in sorted(forced, key=lambda x: x[0]):
            prefix.append(tid)
        prefix_t = torch.tensor([prefix], dtype=torch.long, device=self.device)

        target_ids = self.wrapper.tokenize_target(target_text)

        info = {
            "target_ids": target_ids,
            "target_token_ids": text_ids,
            "prefix_ids": prefix_t,
            "n_tokens": len(text_ids),
        }
        self._target_cache[target_text] = info
        return info

    def compute_loss(self, waveform, target_text, step, total_steps):
        info = self._setup_target(target_text)
        phase1_steps = total_steps // 3

        if step < phase1_steps:
            outputs = self.wrapper.forward_with_labels(waveform, info["target_ids"])
            return outputs.loss

        progress = (step - phase1_steps) / max(total_steps - phase1_steps, 1)
        n_align = max(1, min(info["n_tokens"],
                             int(1 + progress * info["n_tokens"])))
        return self._generation_aligned_loss(
            waveform, info["prefix_ids"], info["target_token_ids"], n_align
        )

    def _generation_aligned_loss(self, waveform, prefix_ids, token_ids, n_align):
        features = self.wrapper.extract_features(waveform)
        total = torch.tensor(0.0, device=self.device)
        n_align = min(n_align, len(token_ids))

        for i in range(n_align):
            dec_ids = prefix_ids.clone()
            if i > 0:
                prev = torch.tensor(
                    [token_ids[:i]], dtype=torch.long, device=self.device
                )
                dec_ids = torch.cat([dec_ids, prev], dim=1)

            out = self.wrapper.model(
                input_features=features, decoder_input_ids=dec_ids
            )
            logits_i = out.logits[0, -1, :]
            target_i = torch.tensor([token_ids[i]], device=self.device)
            total = total + F.cross_entropy(logits_i.unsqueeze(0), target_i)

        return total / n_align

    def compute_loss_batch(self, waveforms, target_texts, step, total_steps):
        """Batched Whisper loss: batch encoder + batched decoder at each token pos."""
        B = len(waveforms)
        features_list = [self.wrapper.extract_features(w) for w in waveforms]
        features_batch = torch.cat(features_list, dim=0)

        encoder = self.wrapper.model.get_encoder()
        encoder_out = encoder(features_batch).last_hidden_state

        phase1_steps = total_steps // 3
        infos = [self._setup_target(t) for t in target_texts]

        if step < phase1_steps:
            total_loss = torch.tensor(0.0, device=self.device)
            for i in range(B):
                enc_i = encoder_out[i:i+1]
                target_ids = infos[i]["target_ids"]
                dec_input = target_ids[:, :-1]
                labels = target_ids[:, 1:].clone()
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
                out = self.wrapper.model(
                    encoder_outputs=(enc_i,),
                    decoder_input_ids=dec_input,
                    labels=None,
                )
                logits = out.logits
                loss_i = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                total_loss = total_loss + loss_i
            return total_loss / B
        else:
            progress = (step - phase1_steps) / max(total_steps - phase1_steps, 1)
            encoder_outs = [encoder_out[i:i+1] for i in range(B)]
            n_aligns = []
            for info in infos:
                na = max(1, min(info["n_tokens"],
                                int(1 + progress * info["n_tokens"])))
                n_aligns.append(na)
            return self._ga_loss_batched_by_token(encoder_outs, infos, n_aligns)

    def _ga_loss_from_encoder(self, encoder_output, prefix_ids, token_ids, n_align):
        """GA loss using pre-computed encoder output (avoids re-encoding)."""
        total = torch.tensor(0.0, device=self.device)
        n_align = min(n_align, len(token_ids))

        for i in range(n_align):
            dec_ids = prefix_ids.clone()
            if i > 0:
                prev = torch.tensor(
                    [token_ids[:i]], dtype=torch.long, device=self.device
                )
                dec_ids = torch.cat([dec_ids, prev], dim=1)

            out = self.wrapper.model(
                encoder_outputs=(encoder_output,),
                decoder_input_ids=dec_ids,
            )
            logits_i = out.logits[0, -1, :]
            target_i = torch.tensor([token_ids[i]], device=self.device)
            total = total + F.cross_entropy(logits_i.unsqueeze(0), target_i)

        return total / n_align

    def _ga_loss_batched_by_token(self, encoder_outs, infos, n_aligns):
        """
        GA loss batched across samples at each token position.
        Instead of B × T sequential calls, does T batched calls of size B.
        """
        B = len(encoder_outs)
        max_align = max(n_aligns)
        total_loss = torch.tensor(0.0, device=self.device)
        count = 0

        for tok_pos in range(max_align):
            active = [j for j in range(B) if tok_pos < n_aligns[j]]
            if not active:
                break

            enc_batch = torch.cat([encoder_outs[j] for j in active], dim=0)
            dec_batch = []
            targets = []
            for j in active:
                dec_ids = infos[j]["prefix_ids"].clone()
                tids = infos[j]["target_token_ids"]
                if tok_pos > 0:
                    prev = torch.tensor(
                        [tids[:tok_pos]], dtype=torch.long, device=self.device
                    )
                    dec_ids = torch.cat([dec_ids, prev], dim=1)
                dec_batch.append(dec_ids)
                targets.append(tids[tok_pos])

            max_dec_len = max(d.shape[1] for d in dec_batch)
            pad_id = self.processor.tokenizer.pad_token_id or 0
            dec_padded = torch.full(
                (len(active), max_dec_len), pad_id,
                dtype=torch.long, device=self.device,
            )
            for k, d in enumerate(dec_batch):
                dec_padded[k, :d.shape[1]] = d[0]

            out = self.wrapper.model(
                encoder_outputs=(enc_batch,),
                decoder_input_ids=dec_padded,
            )
            logits_last = out.logits[:, -1, :]
            target_t = torch.tensor(targets, device=self.device)
            total_loss = total_loss + F.cross_entropy(logits_last, target_t, reduction="sum")
            count += len(active)

        return total_loss / max(count, 1)

    def transcribe(self, waveform):
        with torch.no_grad():
            features = self.wrapper.extract_features(waveform)
            ids = self.wrapper.model.generate(features)
            return self.processor.batch_decode(
                ids, skip_special_tokens=True
            )[0].strip()

    def transcribe_batch(self, waveforms):
        """Batch transcription for all samples."""
        texts = []
        with torch.no_grad():
            features_list = [self.wrapper.extract_features(w) for w in waveforms]
            features_batch = torch.cat(features_list, dim=0)
            ids = self.wrapper.model.generate(features_batch)
            texts = self.processor.batch_decode(ids, skip_special_tokens=True)
        return [t.strip() for t in texts]


class CTCAttackModel:
    """Wrapper for CTC-based models (Wav2Vec2, HuBERT)."""

    def __init__(self, model_name, device):
        from transformers import (
            AutoModelForCTC, AutoProcessor,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.model_name = model_name
        self.arch = "CTC"

        short = model_name.split("/")[-1]
        if "wav2vec2" in short.lower():
            self.name = "Wav2Vec2"
        elif "hubert" in short.lower():
            self.name = "HuBERT"
        else:
            self.name = short

    def _normalize(self, waveform):
        wav = waveform.squeeze()
        return ((wav - wav.mean()) / (wav.std() + 1e-7)).unsqueeze(0)

    def _encode_target(self, target_text):
        """Cache-efficient target encoding with correct tokenisation."""
        if not hasattr(self, "_target_id_cache"):
            self._target_id_cache = {}
        if target_text in self._target_id_cache:
            return self._target_id_cache[target_text]

        vocab = self.processor.tokenizer.get_vocab()
        uses_upper = any(c.isupper() for c in vocab if len(c) == 1)
        text = target_text.upper() if uses_upper else target_text.lower()

        encoded = self.processor.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )
        target_ids = encoded.input_ids.squeeze()
        if target_ids.dim() == 0:
            target_ids = target_ids.unsqueeze(0)
        target_ids = target_ids.to(self.device)
        self._target_id_cache[target_text] = target_ids
        return target_ids

    def compute_loss(self, waveform, target_text, step=0, total_steps=1000):
        normalized = self._normalize(waveform)
        logits = self.model(normalized).logits

        target_ids = self._encode_target(target_text)

        input_lengths = torch.tensor([logits.shape[1]], device=self.device)
        target_lengths = torch.tensor([target_ids.shape[0]], device=self.device)

        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        loss = F.ctc_loss(
            log_probs, target_ids.unsqueeze(0),
            input_lengths, target_lengths,
            blank=pad_id, zero_infinity=True,
        )
        return loss

    def compute_loss_batch(self, waveforms, target_texts, step=0, total_steps=1000):
        """Batched CTC loss: single forward pass for all samples."""
        B = len(waveforms)
        max_len = max(w.shape[-1] for w in waveforms)

        padded = torch.zeros(B, max_len, device=self.device)
        wav_lengths = []
        for i, w in enumerate(waveforms):
            wav = w.squeeze()
            norm = (wav - wav.mean()) / (wav.std() + 1e-7)
            L = norm.shape[0]
            padded[i, :L] = norm
            wav_lengths.append(L)

        attention_mask = torch.zeros(B, max_len, device=self.device)
        for i, L in enumerate(wav_lengths):
            attention_mask[i, :L] = 1.0

        logits = self.model(padded, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)

        pad_id = self.processor.tokenizer.pad_token_id or 0
        ratio = logits.shape[1] / max_len
        input_lengths = torch.tensor(
            [max(1, int(L * ratio)) for L in wav_lengths],
            dtype=torch.long, device=self.device,
        )

        all_target_ids = []
        target_lengths = []
        for t in target_texts:
            tids = self._encode_target(t)
            all_target_ids.append(tids)
            target_lengths.append(tids.shape[0])

        max_tgt = max(target_lengths)
        targets_padded = torch.zeros(B, max_tgt, dtype=torch.long, device=self.device)
        for i, tids in enumerate(all_target_ids):
            targets_padded[i, :tids.shape[0]] = tids

        target_lengths_t = torch.tensor(target_lengths, dtype=torch.long, device=self.device)

        loss = F.ctc_loss(
            log_probs, targets_padded,
            input_lengths, target_lengths_t,
            blank=pad_id, zero_infinity=True,
        )
        return loss

    def transcribe(self, waveform):
        with torch.no_grad():
            normalized = self._normalize(waveform)
            logits = self.model(normalized).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            text = self.processor.batch_decode(predicted_ids)[0]
        return text.strip().lower()

    def transcribe_batch(self, waveforms):
        """Batch transcription for all samples at once."""
        B = len(waveforms)
        max_len = max(w.shape[-1] for w in waveforms)
        padded = torch.zeros(B, max_len, device=self.device)
        wav_lengths = []
        for i, w in enumerate(waveforms):
            wav = w.squeeze()
            norm = (wav - wav.mean()) / (wav.std() + 1e-7)
            L = norm.shape[0]
            padded[i, :L] = norm
            wav_lengths.append(L)

        attention_mask = torch.zeros(B, max_len, device=self.device)
        for i, L in enumerate(wav_lengths):
            attention_mask[i, :L] = 1.0

        with torch.no_grad():
            logits = self.model(padded, attention_mask=attention_mask).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            texts = self.processor.batch_decode(predicted_ids)
        return [t.strip().lower() for t in texts]


# ---------------------------------------------------------------------------
# Audio augmentation for transfer robustness
# ---------------------------------------------------------------------------

def _audio_augment(waveform, sr=WHISPER_SR):
    """
    Lightweight differentiable audio augmentation.
    Simulates real-world variation to prevent overfitting to
    any single model's feature extraction pipeline.
    """
    wav = waveform
    r = torch.rand(1).item()

    if r < 0.3:
        # Speed perturbation: +-5%
        factor = 0.95 + torch.rand(1).item() * 0.10
        orig_len = wav.shape[-1]
        wav = torchaudio.functional.resample(wav, sr, int(sr * factor))
        if wav.shape[-1] > orig_len:
            wav = wav[..., :orig_len]
        elif wav.shape[-1] < orig_len:
            pad = orig_len - wav.shape[-1]
            wav = F.pad(wav, (0, pad))
    elif r < 0.5:
        # Additive Gaussian noise
        noise = torch.randn_like(wav) * 0.002
        wav = wav + noise

    return wav


# ---------------------------------------------------------------------------
# UA3: Universal Attack
# ---------------------------------------------------------------------------

def ua3_attack(
    models,
    waveform,
    target_text,
    iterations=2000,
    lr=0.005,
    linf_budget=0.08,
    use_augment=True,
    model_weights=None,
    progress_fn=None,
):
    """
    UA3: Universal Audio Adversarial Attack (v2 — stabilized).

    Key improvements over v1:
      - Per-model convergence memory: once a model has matched, track it
      - Stabilization phase: once all models have individually converged
        at some point, drop LR and use equal weights to lock in
      - Adaptive check frequency: every 10 steps once 2+ models seen
      - EMA delta tracking: smooth perturbation to reduce oscillation

    Args:
        models: list of WhisperAttackModel / CTCAttackModel instances
        waveform: tensor [1, N] on device
        target_text: attacker's target transcription
        iterations: total optimisation steps
        lr: learning rate
        linf_budget: L-inf perturbation bound
        use_augment: apply audio augmentation for transfer
        model_weights: per-model loss weights (defaults to equal)
        progress_fn: optional callback(step, total, loss)

    Returns:
        adversarial waveform tensor [1, N]
    """
    device = waveform.device
    waveform = waveform.detach().clone()
    n_models = len(models)
    target_norm = _normalize_text(target_text)

    if model_weights is None:
        model_weights = [1.0 / n_models] * n_models
    weights = list(model_weights)

    delta = torch.zeros_like(waveform, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    best_delta = delta.detach().clone()
    best_score = -1
    check_interval = 50

    # Per-model convergence tracking
    model_ever_matched = [False] * n_models
    model_match_now = [False] * n_models
    stabilizing = False
    stabilize_start = -1

    model_names = [m.name for m in models]
    print(f"  [UA3v2] Models: {model_names}", flush=True)
    print(f"  [UA3v2] Target: '{target_text}'", flush=True)
    print(f"  [UA3v2] {iterations} iters, linf={linf_budget}, "
          f"augment={use_augment}", flush=True)

    gen_texts = [""] * n_models

    for step in range(iterations):
        optimizer.zero_grad()

        adv = torch.clamp(waveform + delta, -1.0, 1.0)

        if use_augment and step > 0 and not stabilizing:
            if torch.rand(1).item() < 0.25:
                adv_aug = _audio_augment(adv)
            else:
                adv_aug = adv
        else:
            adv_aug = adv

        total_loss = torch.tensor(0.0, device=device)
        loss_parts = []
        for i, model in enumerate(models):
            loss_i = model.compute_loss(
                adv_aug, target_text, step, iterations
            )
            total_loss = total_loss + weights[i] * loss_i
            loss_parts.append(loss_i.item())

        total_loss.backward()

        if stabilizing:
            with torch.no_grad():
                if delta.grad is not None:
                    delta.grad.mul_(0.5)

        optimizer.step()

        with torch.no_grad():
            delta.data.clamp_(-linf_budget, linf_budget)
            clamped = torch.clamp(waveform + delta, -1.0, 1.0)
            delta.data.copy_(clamped - waveform)

        # Adaptive check frequency: more frequent when close
        n_ever = sum(model_ever_matched)
        if n_ever >= n_models - 1:
            cur_interval = 10
        elif n_ever >= 1:
            cur_interval = 25
        else:
            cur_interval = check_interval

        if step % cur_interval == 0 and step > 0:
            with torch.no_grad():
                test_adv = torch.clamp(waveform + delta.detach(), -1.0, 1.0)
                n_success = 0
                gen_texts = []
                for mi, model in enumerate(models):
                    text = model.transcribe(test_adv)
                    gen_texts.append(text)
                    if _normalize_text(text) == target_norm:
                        n_success += 1
                        model_match_now[mi] = True
                        model_ever_matched[mi] = True
                    else:
                        model_match_now[mi] = False

                # Score: count matches, but give encoder-decoder
                # models a bonus since they're harder to converge
                score = 0.0
                for mi in range(n_models):
                    if model_match_now[mi]:
                        bonus = 1.5 if models[mi].arch == "encoder-decoder" else 1.0
                        score += bonus

                if score > best_score:
                    best_score = score
                    best_delta = delta.detach().clone()

                if n_success == n_models:
                    print(f"  [UA3v2] ALL MODELS MATCHED at step {step}!",
                          flush=True)
                    best_delta = delta.detach().clone()
                    break

                # Enter stabilization if all models have individually
                # converged at some point but not yet simultaneously
                if (all(model_ever_matched) and not stabilizing
                        and not all(model_match_now)):
                    stabilizing = True
                    stabilize_start = step
                    weights = [1.0 / n_models] * n_models
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr * 0.5
                    print(f"  [UA3v2] STABILIZING at step {step} "
                          f"(all models seen, locking in)", flush=True)

                # Weight adjustment only in non-stabilizing mode
                if not stabilizing:
                    if any(model_match_now) and not all(model_match_now):
                        n_lagging = sum(
                            1 for c in model_match_now if not c
                        )
                        n_leading = n_models - n_lagging
                        for mi in range(n_models):
                            if model_match_now[mi]:
                                weights[mi] = 0.25 / n_leading
                            else:
                                weights[mi] = 0.75 / n_lagging

        if progress_fn and step % 5 == 0:
            progress_fn(step, iterations, total_loss.item())

        if step % 100 == 0:
            with torch.no_grad():
                sig_pow = waveform.pow(2).mean()
                noi_pow = delta.detach().pow(2).mean().clamp(min=1e-10)
                snr = 10 * torch.log10(sig_pow / noi_pow).item()

            loss_str = " + ".join(
                f"{m.name}={l:.4f}"
                for m, l in zip(models, loss_parts)
            )
            w_str = " ".join(f"w{i}={w:.2f}" for i, w in enumerate(weights))
            ever_str = "".join(
                "Y" if e else "." for e in model_ever_matched
            )
            gen_str = ""
            if step > 0 and gen_texts[0]:
                gen_str = " | " + " / ".join(
                    f"{m.name}='{t[:30]}'"
                    for m, t in zip(models, gen_texts)
                )
            phase = "STAB" if stabilizing else (
                "CE" if step < iterations // 3 else "GA"
            )
            print(
                f"  [UA3v2] step {step}/{iterations} [{phase}] | "
                f"{loss_str} | {w_str} | seen=[{ever_str}] | "
                f"SNR={snr:.1f}dB{gen_str}",
                flush=True,
            )

    # Final evaluation
    with torch.no_grad():
        adv_waveform = torch.clamp(waveform + best_delta, -1.0, 1.0)

    print(f"\n  [UA3v2] Final results:", flush=True)
    for model in models:
        text = model.transcribe(adv_waveform)
        match = "OK" if _normalize_text(text) == target_norm else "FAIL"
        print(f"    {model.name}: [{match}] '{text}'", flush=True)

    return adv_waveform.detach()


# ---------------------------------------------------------------------------
# UA3 Batched: all samples in one loop, one model copy
# ---------------------------------------------------------------------------

def ua3_attack_batched(
    models,
    waveforms,
    target_texts,
    iterations=2000,
    lr=0.005,
    linf_budget=0.08,
    use_augment=True,
):
    """
    UA3 Batched: process all samples simultaneously.

    One model copy, batched forward passes, per-sample deltas.
    Dramatically more VRAM-efficient and faster than N separate processes.

    Args:
        models: list of model wrappers (WhisperAttackModel / CTCAttackModel)
        waveforms: list of [1, T_i] tensors on device
        target_texts: list of target strings (one per sample)
        iterations: total optimisation steps
        lr: learning rate
        linf_budget: L-inf perturbation bound
        use_augment: apply audio augmentation

    Returns:
        list of adversarial waveform tensors [1, T_i]
    """
    device = waveforms[0].device
    B = len(waveforms)
    n_models = len(models)
    target_norms = [_normalize_text(t) for t in target_texts]

    waveforms = [w.detach().clone() for w in waveforms]
    lengths = [w.shape[-1] for w in waveforms]

    deltas = [torch.zeros_like(w, requires_grad=True) for w in waveforms]
    optimizer = torch.optim.Adam(deltas, lr=lr)

    best_deltas = [d.detach().clone() for d in deltas]
    best_scores = [0.0] * B
    sample_done = [False] * B

    check_interval = 50

    model_names = [m.name for m in models]
    print(f"  [UA3-batch] Models: {model_names}, Samples: {B}", flush=True)
    print(f"  [UA3-batch] {iterations} iters, linf={linf_budget}", flush=True)

    for step in range(iterations):
        if all(sample_done):
            print(f"  [UA3-batch] All {B} samples converged at step {step}!",
                  flush=True)
            break

        optimizer.zero_grad()

        advs = []
        for i in range(B):
            adv_i = torch.clamp(waveforms[i] + deltas[i], -1.0, 1.0)
            if use_augment and step > 0 and torch.rand(1).item() < 0.25:
                adv_i = _audio_augment(adv_i)
            advs.append(adv_i)

        total_loss = torch.tensor(0.0, device=device)
        loss_parts = [0.0] * n_models

        for mi, model in enumerate(models):
            if isinstance(model, CTCAttackModel):
                active_idx = [i for i in range(B) if not sample_done[i]]
                if not active_idx:
                    continue
                active_wavs = [advs[i] for i in active_idx]
                active_tgts = [target_texts[i] for i in active_idx]
                loss_m = model.compute_loss_batch(
                    active_wavs, active_tgts, step, iterations
                )
                total_loss = total_loss + loss_m
                loss_parts[mi] = loss_m.item()
            else:
                active_idx = [i for i in range(B) if not sample_done[i]]
                if not active_idx:
                    continue
                active_wavs = [advs[i] for i in active_idx]
                active_tgts = [target_texts[i] for i in active_idx]
                loss_m = model.compute_loss_batch(
                    active_wavs, active_tgts, step, iterations
                )
                total_loss = total_loss + loss_m
                loss_parts[mi] = loss_m.item()

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            for i in range(B):
                if sample_done[i]:
                    continue
                deltas[i].data.clamp_(-linf_budget, linf_budget)
                clamped = torch.clamp(waveforms[i] + deltas[i], -1.0, 1.0)
                deltas[i].data.copy_(clamped - waveforms[i])

        if step > 0 and step % check_interval == 0:
            with torch.no_grad():
                for mi, model in enumerate(models):
                    test_advs = [
                        torch.clamp(waveforms[i] + deltas[i].detach(), -1.0, 1.0)
                        for i in range(B)
                    ]
                    if hasattr(model, "transcribe_batch"):
                        texts = model.transcribe_batch(test_advs)
                    else:
                        texts = [model.transcribe(a) for a in test_advs]

                    for i in range(B):
                        if sample_done[i]:
                            continue
                        if _normalize_text(texts[i]) == target_norms[i]:
                            score_add = 1.5 if model.arch == "encoder-decoder" else 1.0
                            best_scores[i] = best_scores[i] + score_add

                for i in range(B):
                    if sample_done[i]:
                        continue
                    all_match = True
                    test_adv = torch.clamp(
                        waveforms[i] + deltas[i].detach(), -1.0, 1.0
                    )
                    for model in models:
                        text = model.transcribe(test_adv)
                        if _normalize_text(text) != target_norms[i]:
                            all_match = False
                            break
                    if all_match:
                        best_deltas[i] = deltas[i].detach().clone()
                        sample_done[i] = True
                        print(f"  [UA3-batch] Sample {i} CONVERGED at step {step}",
                              flush=True)
                    else:
                        best_deltas[i] = deltas[i].detach().clone()

        if step % 100 == 0:
            n_done = sum(sample_done)
            loss_str = " + ".join(
                f"{m.name}={l:.4f}" for m, l in zip(models, loss_parts)
            )
            with torch.no_grad():
                snrs = []
                for i in range(B):
                    sig_pow = waveforms[i].pow(2).mean()
                    noi_pow = deltas[i].detach().pow(2).mean().clamp(min=1e-10)
                    snrs.append(10 * torch.log10(sig_pow / noi_pow).item())
                avg_snr = sum(snrs) / len(snrs)

            print(
                f"  [UA3-batch] step {step}/{iterations} | {loss_str} | "
                f"done={n_done}/{B} | avg_SNR={avg_snr:.1f}dB",
                flush=True,
            )

    results = []
    for i in range(B):
        adv = torch.clamp(waveforms[i] + best_deltas[i], -1.0, 1.0)
        results.append(adv.detach())

    print(f"\n  [UA3-batch] Final results:", flush=True)
    for i in range(B):
        status_parts = []
        for model in models:
            text = model.transcribe(results[i])
            match = "OK" if _normalize_text(text) == target_norms[i] else "FAIL"
            status_parts.append(f"{model.name}=[{match}]'{text[:30]}'")
        print(f"    s{i}: {' | '.join(status_parts)}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Convenience: load model ensemble
# ---------------------------------------------------------------------------

def load_model_ensemble(model_specs, device):
    """
    Load a list of ASR models for ensemble attack.

    model_specs: list of dicts with keys:
        name: HuggingFace model name
        type: 'whisper' or 'ctc'

    Returns: list of attack model wrappers
    """
    import sys
    import os
    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    sys.path.insert(0, PROJECT_ROOT)

    models = []
    for spec in model_specs:
        mtype = spec["type"]
        mname = spec["name"]
        print(f"  Loading {mtype} model: {mname}...", flush=True)

        if mtype == "whisper":
            from models.asr_loader import load_asr_model
            wrapper, processor = load_asr_model(mname)
            models.append(WhisperAttackModel(wrapper, processor, device))
        elif mtype == "ctc":
            models.append(CTCAttackModel(mname, device))
        else:
            raise ValueError(f"Unknown model type: {mtype}")

        print(f"    Loaded {models[-1].name}", flush=True)

    return models
