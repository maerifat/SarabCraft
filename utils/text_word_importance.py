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

        _, conf_after, _ = model_wrapper.predict(reduced)
        importance = orig_conf - conf_after
        scores.append((i, importance))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def gradient_importance(model, tokenizer, text: str, target_index: int = None) -> list[tuple[int, float]]:
    """Score token importance using gradient L2 norm (white-box).

    Args:
        model: HuggingFace model (requires gradient computation)
        tokenizer: HuggingFace tokenizer
        text: input text
        target_index: class index for gradient computation; if None, uses predicted class

    Returns:
        List of (token_position, importance_score) sorted by importance descending.
        Positions correspond to tokenizer output positions (not word positions).
    """
    from models.text_loader import device

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

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

    # L2 norm of gradient for each token position
    grad = embed_out.grad[0]  # [seq_len, hidden_dim]
    importance = grad.norm(dim=-1).detach().cpu().tolist()  # [seq_len]

    # Skip [CLS] and [SEP] tokens (first and last)
    scores = []
    for i in range(1, len(importance) - 1):
        scores.append((i, importance[i]))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
