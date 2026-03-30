"""
Universal Adversarial Triggers — Wallace et al., 2019 (EMNLP 2019)

Universal Adversarial Triggers for Attacking and Analyzing NLP.
Gradient-based method that finds a short token sequence (trigger) that,
when prepended or appended to ANY input, causes misclassification.

Algorithm:
  1. Initialize trigger tokens (default: "the the the" for 3-token trigger)
  2. For each trigger position:
     a. Compute gradient of loss w.r.t. trigger token embedding
     b. Score all vocabulary tokens by dot product with gradient
     c. Evaluate top-k candidates via beam search
     d. Select token that maximises attack objective
  3. Repeat for multiple iterations until convergence

Unlike per-input attacks, UAT finds a UNIVERSAL perturbation that
transfers across inputs, making it a more powerful threat model.

Reference: https://github.com/Eric-Wallace/universal-triggers
"""

import logging
import random

logger = logging.getLogger("textattack.attacks.uat")


def _get_embedding_matrix(model, tokenizer):
    """Extract the token embedding matrix from a HuggingFace model."""
    import torch

    if hasattr(model, "get_input_embeddings"):
        emb_layer = model.get_input_embeddings()
        return emb_layer.weight.detach()

    for name, param in model.named_parameters():
        if "word_embedding" in name or "embeddings.word_embeddings" in name:
            return param.detach()

    raise ValueError("Cannot find embedding matrix in model")


def _get_trigger_grad(model, tokenizer, text: str, trigger_tokens: list[int],
                      target_label_idx: int, prepend: bool, device):
    """Compute gradient of loss w.r.t. trigger token embeddings.

    Forward pass with trigger + input, compute cross-entropy loss
    against target label, backprop to get embedding gradients.
    """
    import torch

    emb_layer = model.get_input_embeddings()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=480)
    input_ids = inputs["input_ids"].to(device)

    trigger_tensor = torch.tensor([trigger_tokens], device=device)
    if prepend:
        combined_ids = torch.cat([
            input_ids[:, :1],  # [CLS]
            trigger_tensor,
            input_ids[:, 1:],
        ], dim=1)
    else:
        # Find [SEP] position, insert before it
        sep_pos = (input_ids[0] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_pos) > 0:
            sep_idx = sep_pos[-1].item()
            combined_ids = torch.cat([
                input_ids[:, :sep_idx],
                trigger_tensor,
                input_ids[:, sep_idx:],
            ], dim=1)
        else:
            combined_ids = torch.cat([input_ids, trigger_tensor], dim=1)

    # Create attention mask for combined input
    attention_mask = torch.ones_like(combined_ids)

    embeddings = emb_layer(combined_ids)
    embeddings.requires_grad_(True)

    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    logits = outputs.logits

    target = torch.tensor([target_label_idx], device=device)
    loss = torch.nn.functional.cross_entropy(logits, target)

    loss.backward()

    grad = embeddings.grad[0]  # (seq_len, hidden_dim)

    if prepend:
        trigger_grad = grad[1:1 + len(trigger_tokens)]
    else:
        if len(sep_pos) > 0:
            trigger_grad = grad[sep_idx:sep_idx + len(trigger_tokens)]
        else:
            trigger_grad = grad[-len(trigger_tokens):]

    return trigger_grad


def _find_best_token(trigger_grad_pos, embedding_matrix, top_k: int = 20,
                     forbidden_ids: set = None):
    """Score all vocabulary tokens by negative gradient dot product.

    For targeted attack: minimize loss = maximize dot product with
    negative gradient direction.
    """
    import torch

    scores = torch.matmul(embedding_matrix, -trigger_grad_pos)

    if forbidden_ids:
        for fid in forbidden_ids:
            if fid < len(scores):
                scores[fid] = float("-inf")

    top_ids = torch.argsort(scores, descending=True)[:top_k]
    return top_ids.tolist()


def run_uat(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    trigger_length: int = 3,
    num_iterations: int = 20,
    beam_size: int = 5,
    position: str = "prepend",
) -> str:
    """Universal Adversarial Triggers attack (Wallace et al., 2019).

    Finds a short trigger sequence that, when prepended/appended to the
    input, causes misclassification. Uses first-order gradient approximation
    to search over the discrete vocabulary space.

    Note: In per-input mode (as used here), the trigger is optimised for
    this specific input. For true universal triggers, run across a batch
    and aggregate gradients (supported via batch job system).

    Args:
        model_wrapper: wrapped model with .predict() / .model / .tokenizer
        tokenizer: HuggingFace tokenizer
        text: input text to attack
        target_label: target class (None = untargeted, flips to any other)
        trigger_length: number of trigger tokens
        num_iterations: gradient update iterations
        beam_size: top-k candidates per position
        position: "prepend" or "append"

    Returns: adversarial text (str).
    """
    import torch
    from models.text_loader import get_label_index

    logger.info(
        "UAT: starting (trigger_len=%d, iters=%d, beam=%d, pos=%s)",
        trigger_length, num_iterations, beam_size, position,
    )

    model = model_wrapper.model
    device = next(model.parameters()).device
    prepend = position.lower() == "prepend"

    orig_label, orig_conf, orig_label_idx = model_wrapper.predict(text)

    # Determine target: for untargeted, pick the runner-up class
    if target_label is not None:
        target_idx = get_label_index(model, target_label)
        if target_idx is None:
            target_idx = 1 - orig_label_idx  # binary fallback
    else:
        probs = model_wrapper.predict_probs(text)
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        target_idx = sorted_indices[1] if len(sorted_indices) > 1 else (1 - orig_label_idx)

    embedding_matrix = _get_embedding_matrix(model, tokenizer)

    # Forbidden tokens: special tokens that shouldn't be used as triggers
    special_ids = set()
    for attr in ("pad_token_id", "cls_token_id", "sep_token_id",
                 "unk_token_id", "mask_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)

    # Initialize trigger with common neutral tokens
    init_token = tokenizer.encode("the", add_special_tokens=False)
    if init_token:
        trigger_tokens = init_token[:1] * trigger_length
    else:
        trigger_tokens = [random.randint(100, 1000) for _ in range(trigger_length)]

    best_trigger = list(trigger_tokens)
    best_adversarial = text

    for iteration in range(num_iterations):
        model.zero_grad()

        try:
            trigger_grad = _get_trigger_grad(
                model, tokenizer, text, trigger_tokens,
                target_idx, prepend, device,
            )
        except Exception as e:
            logger.warning("UAT: gradient computation failed at iter %d: %s", iteration, e)
            continue

        # Update each trigger position independently
        for pos in range(trigger_length):
            candidates = _find_best_token(
                trigger_grad[pos], embedding_matrix,
                top_k=beam_size, forbidden_ids=special_ids,
            )

            best_token = trigger_tokens[pos]
            best_score = -float("inf")

            for cand_id in candidates:
                test_trigger = list(trigger_tokens)
                test_trigger[pos] = cand_id

                trigger_text = tokenizer.decode(test_trigger, skip_special_tokens=True)
                if prepend:
                    candidate_text = f"{trigger_text} {text}"
                else:
                    candidate_text = f"{text} {trigger_text}"

                label, conf, label_idx = model_wrapper.predict(candidate_text)

                if target_label is not None:
                    if label.lower() == target_label.lower():
                        logger.info("UAT: success at iteration %d", iteration + 1)
                        return candidate_text
                else:
                    if label != orig_label:
                        logger.info("UAT: success at iteration %d", iteration + 1)
                        return candidate_text

                # Score: probability of target class
                probs = model_wrapper.predict_probs(candidate_text)
                score = probs[target_idx] if target_idx < len(probs) else 0.0

                if score > best_score:
                    best_score = score
                    best_token = cand_id
                    best_adversarial = candidate_text

            trigger_tokens[pos] = best_token

        best_trigger = list(trigger_tokens)

    trigger_text = tokenizer.decode(best_trigger, skip_special_tokens=True)
    if prepend:
        result = f"{trigger_text} {text}"
    else:
        result = f"{text} {trigger_text}"

    final_label, _, _ = model_wrapper.predict(result)
    if final_label != orig_label:
        logger.info("UAT: finished (trigger found: '%s')", trigger_text)
    else:
        logger.info("UAT: finished (best effort, trigger: '%s')", trigger_text)
        result = best_adversarial

    return result
