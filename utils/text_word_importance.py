"""
Word importance scoring: determines which words to perturb first.

Two strategies:
  - delete_one_importance: black-box, measures confidence drop when word removed
  - gradient_importance: white-box, L2 norm of gradient w.r.t. each token embedding
"""

import torch
from utils.text_utils import get_words_and_spans, is_stopword


def delete_one_importance(model_wrapper, text: str, original_label: str = None) -> list[tuple[int, float]]:
    """Score each word's importance by removing it and measuring confidence drop.

    Args:
        model_wrapper: _TextModelWrapper with .predict() method
        text: original text
        original_label: if None, will be determined from model

    Returns:
        List of (word_position, importance_score) sorted by importance descending.
    """
    words_spans = get_words_and_spans(text)
    if not words_spans:
        return []

    if original_label is None:
        original_label, orig_conf, _ = model_wrapper.predict(text)
    else:
        _, orig_conf, _ = model_wrapper.predict(text)

    # Get original probability vector for two-case formula
    orig_probs = model_wrapper.predict_probs(text)
    _, _, orig_label_idx = model_wrapper.predict(text)

    scores = []
    for i, (word, start, end) in enumerate(words_spans):
        if is_stopword(word):
            scores.append((i, 0.0))
            continue

        # Remove word and re-classify
        reduced = (text[:start] + text[end:]).strip()
        reduced = " ".join(reduced.split())  # clean up double spaces
        if not reduced:
            scores.append((i, orig_conf))
            continue

        del_label, _, del_label_idx = model_wrapper.predict(reduced)
        del_probs = model_wrapper.predict_probs(reduced)

        f_y_x = orig_probs[orig_label_idx] if orig_label_idx < len(orig_probs) else orig_conf
        f_y_xw = del_probs[orig_label_idx] if orig_label_idx < len(del_probs) else 0.0

        if del_label != original_label:
            # Two-case formula (Jin et al., 2020):
            # I(w) = (F_Y(X) - F_Y(X\w)) + (F_Y'(X\w) - F_Y'(X))
            f_yp_xw = del_probs[del_label_idx] if del_label_idx < len(del_probs) else 0.0
            f_yp_x = orig_probs[del_label_idx] if del_label_idx < len(orig_probs) else 0.0
            importance = (f_y_x - f_y_xw) + (f_yp_xw - f_yp_x)
        else:
            # Simple confidence drop: I(w) = F_Y(X) - F_Y(X\w)
            importance = f_y_x - f_y_xw

        scores.append((i, importance))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def unk_importance(
    model_wrapper, text: str, unk_token: str = "[UNK]", original_label: str = None,
) -> list[tuple[int, float]]:
    """Score each word's importance by replacing it with [UNK] and measuring score change.

    Matches the official BERT-Attack ``_get_importance()`` combined formula:
      - When label does NOT change:  I(w) = F_Y(X) - F_Y(X\\w)
      - When label changes:
        I(w) = (F_Y(X) - F_Y(X\\w))
             + (F_Y'(X\\w) - F_Y'(X))
        where Y' is the argmax label after replacement.

    This combined formula gives a substantial bonus to words whose removal
    causes a label flip, matching the official vectorised formula::

        import_score = (orig_prob - leave_1_probs[:, orig_label]
                       + (leave_1_probs_argmax != orig_label).float()
                       * (leave_1_probs.max(dim=-1)[0]
                          - torch.index_select(orig_probs, 0, leave_1_probs_argmax)))

    Args:
        model_wrapper: _TextModelWrapper with .predict() and .predict_probs()
        text: original text
        unk_token: placeholder token (default "[UNK]")
        original_label: if None, will be determined from model

    Returns:
        List of (word_position, importance_score) sorted by importance descending.
    """
    words_spans = get_words_and_spans(text)
    if not words_spans:
        return []

    if original_label is None:
        original_label, orig_conf, orig_label_idx = model_wrapper.predict(text)
    else:
        _, orig_conf, orig_label_idx = model_wrapper.predict(text)

    orig_probs = model_wrapper.predict_probs(text)

    scores = []
    for i, (word, start, end) in enumerate(words_spans):
        if is_stopword(word):
            scores.append((i, 0.0))
            continue

        # Replace word with [UNK] to preserve sentence structure
        replaced = text[:start] + unk_token + text[end:]

        rep_label, _, rep_label_idx = model_wrapper.predict(replaced)
        rep_probs = model_wrapper.predict_probs(replaced)

        f_y_x = orig_probs[orig_label_idx] if orig_label_idx < len(orig_probs) else orig_conf
        f_y_xw = rep_probs[orig_label_idx] if orig_label_idx < len(rep_probs) else 0.0

        if rep_label != original_label:
            # Combined formula: label changed — add bonus
            f_yp_xw = rep_probs[rep_label_idx] if rep_label_idx < len(rep_probs) else 0.0
            f_yp_x = orig_probs[rep_label_idx] if rep_label_idx < len(orig_probs) else 0.0
            importance = (f_y_x - f_y_xw) + (f_yp_xw - f_yp_x)
        else:
            # Simple confidence drop
            importance = f_y_x - f_y_xw

        scores.append((i, importance))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def temporal_head_importance(
    model_wrapper, text: str, original_label: str = None,
) -> list[tuple[int, float]]:
    """Temporal Head Score (THS) from Gao et al., 2018.

    THS(i) = F(x_1:i) - F(x_1:i-1)

    Measures importance by comparing model confidence on the prefix up to
    position i versus the prefix up to position i-1.  Captures forward
    sequential dependency.

    Args:
        model_wrapper: _TextModelWrapper with .predict() method
        text: original text
        original_label: if None, will be determined from model

    Returns:
        List of (word_position, importance_score) sorted by importance descending.
    """
    words_spans = get_words_and_spans(text)
    if not words_spans:
        return []

    if original_label is None:
        original_label, _, orig_label_idx = model_wrapper.predict(text)
    else:
        _, _, orig_label_idx = model_wrapper.predict(text)

    words = [w for w, _, _ in words_spans]
    num_words = len(words)

    # Compute confidence on each prefix x_1:i
    prefix_confs = []
    for i in range(num_words):
        prefix = " ".join(words[: i + 1])
        if not prefix:
            prefix_confs.append(0.0)
            continue
        probs = model_wrapper.predict_probs(prefix)
        prefix_confs.append(probs[orig_label_idx] if orig_label_idx < len(probs) else 0.0)

    # THS(0) = prefix_conf[0] - 1/num_classes (uniform baseline)
    num_classes = len(model_wrapper.predict_probs(text))
    baseline = 1.0 / num_classes if num_classes > 0 else 0.0

    scores = []
    scores.append((0, prefix_confs[0] - baseline))
    for i in range(1, num_words):
        ths = prefix_confs[i] - prefix_confs[i - 1]
        scores.append((i, ths))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def temporal_tail_importance(
    model_wrapper, text: str, original_label: str = None,
) -> list[tuple[int, float]]:
    """Temporal Tail Score (TTS) from Gao et al., 2018.

    TTS(i) = F(x_i:n) - F(x_i+1:n)

    Measures importance by comparing model confidence on the suffix starting
    at position i versus the suffix starting at position i+1.  Captures
    backward sequential dependency.

    Args:
        model_wrapper: _TextModelWrapper with .predict() method
        text: original text
        original_label: if None, will be determined from model

    Returns:
        List of (word_position, importance_score) sorted by importance descending.
    """
    words_spans = get_words_and_spans(text)
    if not words_spans:
        return []

    if original_label is None:
        original_label, _, orig_label_idx = model_wrapper.predict(text)
    else:
        _, _, orig_label_idx = model_wrapper.predict(text)

    words = [w for w, _, _ in words_spans]
    num_words = len(words)

    # Compute confidence on each suffix x_i:n
    suffix_confs = []
    for i in range(num_words):
        suffix = " ".join(words[i:])
        if not suffix:
            suffix_confs.append(0.0)
            continue
        probs = model_wrapper.predict_probs(suffix)
        suffix_confs.append(probs[orig_label_idx] if orig_label_idx < len(probs) else 0.0)

    # TTS(last) = suffix_conf[last] - 1/num_classes
    num_classes = len(model_wrapper.predict_probs(text))
    baseline = 1.0 / num_classes if num_classes > 0 else 0.0

    scores = []
    for i in range(num_words - 1):
        tts = suffix_confs[i] - suffix_confs[i + 1]
        scores.append((i, tts))
    scores.append((num_words - 1, suffix_confs[num_words - 1] - baseline))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def combined_importance(
    model_wrapper, text: str, original_label: str = None,
) -> list[tuple[int, float]]:
    """Combined Score from Gao et al., 2018.

    Combined(i) = (THS(i) + TTS(i)) / 2

    Averages Temporal Head Score and Temporal Tail Score for a balanced
    importance measure.  This is the default and best-performing strategy
    in the original paper.

    Args:
        model_wrapper: _TextModelWrapper with .predict() method
        text: original text
        original_label: if None, will be determined from model

    Returns:
        List of (word_position, importance_score) sorted by importance descending.
    """
    ths_scores = temporal_head_importance(model_wrapper, text, original_label)
    tts_scores = temporal_tail_importance(model_wrapper, text, original_label)

    # Convert to dicts for easy lookup
    ths_map = dict(ths_scores)
    tts_map = dict(tts_scores)

    # All word positions
    all_positions = set(ths_map.keys()) | set(tts_map.keys())
    scores = []
    for pos in all_positions:
        combined = (ths_map.get(pos, 0.0) + tts_map.get(pos, 0.0)) / 2.0
        scores.append((pos, combined))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def gradient_importance(model, tokenizer, text: str, target_index: int = None) -> list[tuple[int, float]]:
    """Score word importance using gradient L2 norm (white-box).

    Computes per-subword-token gradient norms and aggregates them to
    word-level scores so that returned indices align with the word
    positions from ``get_words_and_spans(text)``.

    Args:
        model: HuggingFace model (requires gradient computation)
        tokenizer: HuggingFace tokenizer
        text: input text
        target_index: class index for gradient computation; if None, uses predicted class

    Returns:
        List of (word_position, importance_score) sorted by importance descending.
        Positions correspond to whitespace-word indices from get_words_and_spans.
    """
    from models.text_loader import device

    # Tokenize WITH offset mapping so we can align subwords → words
    tok_out = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512,
        return_offsets_mapping=True,
    )
    offset_mapping = tok_out.pop("offset_mapping")[0].tolist()  # list of (start, end)
    inputs = {k: v.to(device) for k, v in tok_out.items()}

    embeddings = model.get_input_embeddings()
    input_ids = inputs["input_ids"]
    embed_out = embeddings(input_ids)
    embed_out.requires_grad_(True)
    embed_out.retain_grad()

    # Forward pass through remaining layers
    outputs = model(inputs_embeds=embed_out, attention_mask=inputs.get("attention_mask"))
    logits = outputs.logits

    if target_index is None:
        target_index = logits.argmax(dim=-1).item()

    loss = logits[0, target_index]
    loss.backward()

    # L2 norm of gradient for each subword token position
    grad = embed_out.grad[0]  # [seq_len, hidden_dim]
    token_norms = grad.norm(dim=-1).detach().cpu().tolist()  # [seq_len]

    # Build word-level spans from get_words_and_spans
    words_spans = get_words_and_spans(text)

    # Aggregate subword gradient norms → word-level scores
    word_scores = [0.0] * len(words_spans)
    for tok_idx, (char_start, char_end) in enumerate(offset_mapping):
        if char_start == 0 and char_end == 0:
            continue  # skip special tokens ([CLS], [SEP], [PAD])
        # Find which word this subword belongs to
        for word_idx, (_word, w_start, w_end) in enumerate(words_spans):
            if char_start >= w_start and char_end <= w_end:
                word_scores[word_idx] += token_norms[tok_idx]
                break

    scores = [(i, s) for i, s in enumerate(word_scores)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def sentence_importance(model_wrapper, text: str) -> list[tuple[int, str, float]]:
    """Score each sentence's importance by removing it and measuring confidence drop.

    Part of the TextBugger black-box algorithm (Li et al., 2018): first rank
    sentences, then score words within the most important sentence.

    Args:
        model_wrapper: _TextModelWrapper with .predict() method
        text: original text (may contain multiple sentences)

    Returns:
        List of (sentence_index, sentence_text, importance_score) sorted by
        importance descending.  Returns a single-element list if the text
        contains only one sentence.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [(0, text, 1.0)]

    orig_label, orig_conf, _ = model_wrapper.predict(text)

    scores = []
    for i, sent in enumerate(sentences):
        reduced = " ".join(s for j, s in enumerate(sentences) if j != i).strip()
        if not reduced:
            scores.append((i, sent, orig_conf))
            continue
        _, conf_after, _ = model_wrapper.predict(reduced)
        importance = orig_conf - conf_after
        scores.append((i, sent, importance))

    scores.sort(key=lambda x: x[2], reverse=True)
    return scores

