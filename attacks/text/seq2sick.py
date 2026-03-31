"""
Seq2Sick Attack — Cheng et al., 2020 (ICLR 2020, arXiv:1803.01128)

Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models
with Adversarial Examples.

Projected gradient attack on encoder embeddings — the only attack in
SarabCraft that targets the EMBEDDING SPACE directly rather than
discrete token substitution. Adapted for classification models:

Algorithm:
  1. Compute continuous embedding for input tokens
  2. Apply PGD in continuous embedding space toward target class
  3. Project perturbed embeddings back to nearest discrete tokens
  4. Evaluate projected text for label flip
  5. Iterate with decreasing step size

Key innovation: operates in continuous space (like image PGD) then
projects back to discrete tokens. This enables gradient-based
optimization that's impossible with purely discrete methods.

For classification models, this reduces to: PGD on the embedding
layer → nearest-neighbour token projection → re-evaluation.
"""

import logging

logger = logging.getLogger("textattack.attacks.seq2sick")


def _get_token_embeddings(model):
    """Extract the token embedding weight matrix."""
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings().weight
    for name, param in model.named_parameters():
        if "word_embedding" in name or "embeddings.word_embeddings" in name:
            return param
    raise ValueError("Cannot find embedding matrix in model")


def _nearest_token(perturbed_emb, embedding_matrix, forbidden_ids: set = None):
    """Find nearest token in vocabulary by L2 distance."""
    import torch

    dists = torch.cdist(perturbed_emb.unsqueeze(0), embedding_matrix.unsqueeze(0))[0]
    # (seq_len, vocab_size)

    if forbidden_ids:
        for fid in forbidden_ids:
            if fid < dists.shape[1]:
                dists[:, fid] = float("inf")

    return torch.argmin(dists, dim=1)


def run_seq2sick(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    num_iterations: int = 30,
    step_size: float = 0.01,
    max_perturbation_ratio: float = 0.3,
    similarity_threshold: float = 0.7,
) -> str:
    """Seq2Sick-style projected gradient attack on embeddings.

    Applies PGD in continuous embedding space and projects back to
    discrete tokens. Adapted from the seq2seq formulation to work
    with classification models.

    Args:
        model_wrapper: wrapped model with .predict() / .model / .tokenizer.
        tokenizer: HuggingFace tokenizer.
        text: input text to attack.
        target_label: target class (None = untargeted).
        num_iterations: PGD iterations.
        step_size: gradient step size in embedding space.
        max_perturbation_ratio: max fraction of tokens to change.
        similarity_threshold: min semantic similarity for result.

    Returns: adversarial text (str).
    """
    import torch
    from models.text_loader import get_label_index
    from utils.text_constraints import compute_semantic_similarity

    logger.info(
        "Seq2Sick: starting (iters=%d, step=%.4f, max_pert=%.2f)",
        num_iterations, step_size, max_perturbation_ratio,
    )

    model = model_wrapper.model
    device = next(model.parameters()).device

    orig_label, orig_conf, orig_label_idx = model_wrapper.predict(text)

    if target_label is not None:
        target_idx = get_label_index(model, target_label)
        if target_idx is None:
            target_idx = 1 - orig_label_idx
    else:
        probs = model_wrapper.predict_probs(text)
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        target_idx = sorted_indices[1] if len(sorted_indices) > 1 else (1 - orig_label_idx)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    embedding_matrix = _get_token_embeddings(model)
    emb_layer = model.get_input_embeddings()

    special_ids = set()
    for attr in ("pad_token_id", "cls_token_id", "sep_token_id",
                 "unk_token_id", "mask_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)

    # Identify which token positions can be perturbed (skip special tokens)
    seq_len = input_ids.shape[1]
    mutable_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    for pos in range(seq_len):
        if input_ids[0, pos].item() in special_ids:
            mutable_mask[pos] = False
    max_changes = max(1, int(mutable_mask.sum().item() * max_perturbation_ratio))

    with torch.no_grad():
        orig_embeddings = emb_layer(input_ids).clone()

    perturbed = orig_embeddings.clone()
    best_adversarial = text
    best_score = 0.0

    target_tensor = torch.tensor([target_idx], device=device)

    for iteration in range(num_iterations):
        perturbed_var = perturbed.detach().clone().requires_grad_(True)

        outputs = model(inputs_embeds=perturbed_var, attention_mask=attention_mask)
        logits = outputs.logits

        # Maximize target class probability (minimize negative log prob)
        loss = torch.nn.functional.cross_entropy(logits, target_tensor)
        loss.backward()

        grad = perturbed_var.grad[0]  # (seq_len, hidden_dim)

        # PGD step: move in the direction that decreases loss (toward target)
        with torch.no_grad():
            update = -step_size * grad.sign()
            update[~mutable_mask] = 0
            perturbed = perturbed + update.unsqueeze(0)

        # Project to nearest tokens and evaluate
        with torch.no_grad():
            projected_ids = _nearest_token(
                perturbed[0], embedding_matrix, forbidden_ids=special_ids,
            )

            # Limit number of changed positions
            changed = (projected_ids != input_ids[0])
            changed[~mutable_mask] = False
            if changed.sum() > max_changes:
                change_indices = changed.nonzero(as_tuple=True)[0]
                diffs = (perturbed[0] - orig_embeddings[0]).norm(dim=1)
                diffs[~changed] = -1
                _, top_k = diffs.topk(max_changes)
                keep_mask = torch.zeros_like(changed)
                keep_mask[top_k] = True
                projected_ids[changed & ~keep_mask] = input_ids[0][changed & ~keep_mask]

            candidate_text = tokenizer.decode(projected_ids, skip_special_tokens=True)

        if not candidate_text.strip():
            continue

        label, conf, _ = model_wrapper.predict(candidate_text)

        # Check similarity
        sim = compute_semantic_similarity(text, candidate_text)
        if sim < similarity_threshold:
            # Reduce step size to stay closer
            step_size *= 0.8
            continue

        if target_label is not None:
            if label.lower() == target_label.lower():
                logger.info("Seq2Sick: success at iteration %d", iteration + 1)
                return candidate_text
        else:
            if label != orig_label:
                logger.info("Seq2Sick: success at iteration %d", iteration + 1)
                return candidate_text

        probs = model_wrapper.predict_probs(candidate_text)
        score = probs[target_idx] if target_idx < len(probs) else 0.0
        if score > best_score:
            best_score = score
            best_adversarial = candidate_text

    logger.info("Seq2Sick: finished (%d iterations)", num_iterations)
    return best_adversarial
