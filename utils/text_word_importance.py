"""
Word importance scoring: determines which words to perturb first.

Two strategies:
  - delete_one_importance: black-box, measures confidence drop when word removed
  - gradient_importance: white-box, L1 norm of gradient w.r.t. each token embedding
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
        original_label, orig_conf, orig_label_idx = model_wrapper.predict(text)
    else:
        _, orig_conf, orig_label_idx = model_wrapper.predict(text)

    orig_probs = model_wrapper.predict_probs(text)

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
    # Also capture num_classes from the probability vector (the official code
    # receives this as a parameter; we derive it from the first query).
    prefix_confs = []
    num_classes = 0
    for i in range(num_words):
        prefix = " ".join(words[: i + 1])
        if not prefix:
            prefix_confs.append(0.0)
            continue
        probs = model_wrapper.predict_probs(prefix)
        if i == 0:
            num_classes = len(probs)
        prefix_confs.append(probs[orig_label_idx] if orig_label_idx < len(probs) else 0.0)

    # THS(0) = prefix_conf[0] - 1/num_classes (uniform baseline)
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
    # Also capture num_classes from the probability vector (the official code
    # receives this as a parameter; we derive it from the first query).
    suffix_confs = []
    num_classes = 0
    for i in range(num_words):
        suffix = " ".join(words[i:])
        if not suffix:
            suffix_confs.append(0.0)
            continue
        probs = model_wrapper.predict_probs(suffix)
        if i == 0:
            num_classes = len(probs)
        suffix_confs.append(probs[orig_label_idx] if orig_label_idx < len(probs) else 0.0)

    # TTS(last) = suffix_conf[last] - 1/num_classes
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
    """Score word importance using Jacobian of classifier confidence (white-box).

    Computes J_i = ∂F_y(x)/∂x_i (paper notation) where F_y is the softmax
    probability for the true class y.  Uses L1 norm of the gradient vector
    per token, with mean aggregation across subwords to produce word-level
    scores — matching the original TextBugger paper and TextAttack reference.

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

    tok_out = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512,
        return_offsets_mapping=True,
    )
    offset_mapping = tok_out.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(device) for k, v in tok_out.items()}

    embeddings = model.get_input_embeddings()
    input_ids = inputs["input_ids"]
    embed_out = embeddings(input_ids)
    embed_out.requires_grad_(True)
    embed_out.retain_grad()

    outputs = model(inputs_embeds=embed_out, attention_mask=inputs.get("attention_mask"))
    logits = outputs.logits

    if target_index is None:
        target_index = logits.argmax(dim=-1).item()

    probs = torch.softmax(logits, dim=-1)
    confidence = probs[0, target_index]
    confidence.backward()

    grad = embed_out.grad[0]  # [seq_len, hidden_dim]
    token_norms = grad.norm(p=1, dim=-1).detach().cpu().tolist()

    words_spans = get_words_and_spans(text)

    word_scores = [0.0] * len(words_spans)
    word_subword_counts = [0] * len(words_spans)
    for tok_idx, (char_start, char_end) in enumerate(offset_mapping):
        if char_start == 0 and char_end == 0:
            continue
        for word_idx, (_word, w_start, w_end) in enumerate(words_spans):
            if char_start >= w_start and char_end <= w_end:
                word_scores[word_idx] += token_norms[tok_idx]
                word_subword_counts[word_idx] += 1
                break

    scores = [
        (i, s / max(c, 1))
        for i, (s, c) in enumerate(zip(word_scores, word_subword_counts))
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def sentence_importance(model_wrapper, text: str) -> list[tuple[int, str, float]]:
    """Rank sentences by importance — TextBugger black-box Algorithm 3.

    Paper specification (Li et al., 2018, Section IV-B):
      1. Split document into sentences (spaCy).
      2. Filter: keep only sentences where F_label(s_i) == y (the sentence's
         own predicted label matches the document label).
      3. Rank remaining sentences by F_y(s_i) in descending order — higher
         confidence means the sentence contributes more to the prediction.

    Args:
        model_wrapper: _TextModelWrapper with .predict() method
        text: original text (may contain multiple sentences)

    Returns:
        List of (sentence_index, sentence_text, importance_score) sorted by
        importance descending.  Returns a single-element list if the text
        contains only one sentence.
    """
    sentences = _split_sentences(text)

    if len(sentences) <= 1:
        return [(0, text, 1.0)]

    orig_label, orig_conf, orig_label_idx = model_wrapper.predict(text)

    scores = []
    for i, sent in enumerate(sentences):
        sent_label, _, _ = model_wrapper.predict(sent)
        if sent_label != orig_label:
            continue

        sent_probs = model_wrapper.predict_probs(sent)
        f_y_si = sent_probs[orig_label_idx] if orig_label_idx < len(sent_probs) else 0.0
        scores.append((i, sent, f_y_si))

    if not scores:
        scores = [(i, sent, 0.0) for i, sent in enumerate(sentences)]

    scores.sort(key=lambda x: x[2], reverse=True)
    return scores


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using spaCy (preferred) or regex fallback."""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except OSError:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
        doc = nlp(text.strip())
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        if sentences:
            return sentences
    except ImportError:
        pass

    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]

