"""
Whisper Task-Control (Prompt-Injection) Universal Acoustic Attack.

Whisper is a multitask speech model: a sequence of special decoder tokens
at the start of generation controls *which task* it performs —
``<|startoftranscript|> <|lang|> <|transcribe|>|<|translate|> ...``.

This attack learns a short **universal audio prefix** that, when prepended
to ANY speech input, hijacks Whisper's task selection at inference time —
e.g. forcing it to *translate* instead of *transcribe*, or to switch its
detected language — even though the user requested plain transcription and
Whisper is being run with its default (auto) decoding.

This is the acoustic analogue of a prompt-injection / task-control attack
and follows the universal-perturbation methodology of:
  Vyas, Raina, Gales (EMNLP 2024), "Muting Whisper: A Universal Acoustic
  Adversarial Attack on Speech Foundation Models",
adapted from muting to *task hijacking* by targeting the special task
tokens in the decoder prefix rather than the end-of-text token.

Only multilingual Whisper checkpoints expose the ``<|translate|>`` /
language tokens; English-only ``*.en`` models do not and will raise a
clear error.
"""

import logging

import torch

logger = logging.getLogger(__name__)


TASK_TRANSLATE = "translate"
TASK_TRANSCRIBE = "transcribe"


def _resolve_task_prefix_ids(wrapper, task, language):
    """
    Build the target decoder-prefix token ids that encode the desired task.

    Returns a 1-D python list of token ids beginning with the SOT token,
    followed by the (optional) language token and the task token.

    Raises ValueError if the tokenizer lacks the requested control tokens
    (e.g. English-only Whisper cannot ``translate``).
    """
    tok = wrapper.processor.tokenizer

    def _convert(token_str):
        tid = tok.convert_tokens_to_ids(token_str)
        unk = getattr(tok, "unk_token_id", None)
        if tid is None or tid == unk:
            return None
        return tid

    sot = _convert("<|startoftranscript|>")
    if sot is None:
        sot = getattr(tok, "bos_token_id", None)
    if sot is None:
        raise ValueError("Tokenizer has no start-of-transcript token")

    task_tok = _convert(f"<|{task}|>")
    if task_tok is None:
        raise ValueError(
            f"This Whisper checkpoint does not support task '{task}'. "
            "Use a multilingual Whisper model (not a *.en checkpoint) for task-control."
        )

    prefix = [sot]
    if language:
        lang_tok = _convert(f"<|{language}|>")
        if lang_tok is None:
            raise ValueError(
                f"Language '{language}' is not available in this tokenizer. "
                "Use a multilingual Whisper model or omit the language."
            )
        prefix.append(lang_tok)
    prefix.append(task_tok)
    return prefix


def task_control_universal_attack(
    wrapper,
    training_waveforms,
    task=TASK_TRANSLATE,
    language=None,
    segment_duration=0.64,
    iterations=250,
    lr=0.01,
    progress_fn=None,
):
    """
    Learn a universal audio prefix that forces Whisper into ``task``.

    The universal segment is optimized so that, prepended to each training
    waveform, the model's decoder prefix loss toward the *task-control token
    sequence* is minimized — biasing generation into the attacker-chosen
    task regardless of the input speech.

    Args:
        wrapper: WhisperAttackWrapper (multilingual checkpoint required).
        training_waveforms: list of tensors [1, N] to generalize across.
        task: 'translate' or 'transcribe'.
        language: optional forced language code (e.g. 'de', 'fr'); None = leave to model.
        segment_duration: length in seconds of the universal prefix.
        iterations: optimization steps.
        lr: learning rate.
        progress_fn: optional callback(step, total, loss).

    Returns:
        universal_segment tensor [1, segment_samples].
    """
    from models.asr_loader import WHISPER_SAMPLE_RATE

    if not training_waveforms:
        raise ValueError("training_waveforms is empty")

    prefix_ids = _resolve_task_prefix_ids(wrapper, task, language)
    device = next(wrapper.parameters()).device
    target_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    segment_samples = int(segment_duration * WHISPER_SAMPLE_RATE)
    universal = torch.randn(1, segment_samples, device=device) * 0.01
    universal = universal.requires_grad_(True)

    optimizer = torch.optim.Adam([universal], lr=lr)

    best_segment = universal.detach().clone()
    best_loss = float("inf")

    for step in range(iterations):
        optimizer.zero_grad()

        total_loss = 0.0
        for waveform in training_waveforms:
            prepended = torch.cat([torch.clamp(universal, -1.0, 1.0), waveform], dim=-1)
            outputs = wrapper.forward_with_labels(prepended, target_ids)
            total_loss = total_loss + outputs.loss

        avg_loss = total_loss / len(training_waveforms)
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
            logger.info(f"[Task-Control] step {step}/{iterations}, task={task}, "
                        f"loss={loss_val:.4f}, best={best_loss:.4f}")

    return torch.clamp(best_segment, -1.0, 1.0).detach()


def apply_task_control_segment(segment, waveform):
    """Prepend the universal task-control segment to a waveform."""
    return torch.cat([segment, waveform], dim=-1)
