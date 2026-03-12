"""
HotFlip Attack — Ebrahimi et al., 2018 (arXiv:1712.06751)

White-box character/token-level attack. Uses gradient of loss w.r.t.
token embeddings to find the optimal token swap via first-order
Taylor approximation.
"""

import logging
import torch

logger = logging.getLogger("textattack.attacks.hotflip")


def run_hotflip(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_flips: int = 5,
    beam_width: int = 1,
) -> str:
    """HotFlip attack (white-box gradient-based token substitution).

    Uses gradient w.r.t. embedding matrix to find the token replacement
    that maximises the directional derivative of the loss.

    Returns: adversarial text (str).
    """
    from models.text_loader import device, get_label_index

    logger.info("HotFlip: starting (max_flips=%d, beam=%d)", max_flips, beam_width)

    model = model_wrapper.model
    model.eval()

    # Determine target index (target_label is already resolved by text_router)
    target_idx = None
    if target_label is not None:
        target_idx = get_label_index(model, target_label)
        if target_idx is None:
            logger.warning("HotFlip: target label '%s' not found, using untargeted", target_label)

    # Cache original prediction (doesn't change across flips)
    orig_label, _, _ = model_wrapper.predict(text)

    current_text = text
    embedding_layer = model.get_input_embeddings()
    embedding_weight = embedding_layer.weight  # [vocab_size, hidden_dim]

    for flip_num in range(max_flips):
        inputs = tokenizer(current_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        # Forward with gradients on embeddings
        embed_out = embedding_layer(input_ids)
        embed_out = embed_out.detach().requires_grad_(True)

        outputs = model(inputs_embeds=embed_out, attention_mask=inputs.get("attention_mask"))
        logits = outputs.logits

        if target_idx is not None:
            # Targeted: maximise target class
            loss = -logits[0, target_idx]
        else:
            # Untargeted: maximise loss on predicted class
            pred_idx = logits.argmax(dim=-1).item()
            loss = logits[0, pred_idx]

        loss.backward()
        grad = embed_out.grad[0]  # [seq_len, hidden_dim]

        # Find best flip: for each position (skip [CLS], [SEP]),
        # compute score = (e_new - e_old) · grad for each vocab token
        best_score = float("-inf")
        best_pos = -1
        best_token_id = -1

        for pos in range(1, seq_len - 1):
            old_embed = embed_out[0, pos].detach()  # [hidden_dim]
            pos_grad = grad[pos]  # [hidden_dim]

            # Score all vocab tokens: (e_new - e_old) dot grad
            # = e_new dot grad - e_old dot grad
            old_dot = torch.dot(old_embed, pos_grad)
            all_dots = embedding_weight @ pos_grad  # [vocab_size]
            scores = all_dots - old_dot

            # Exclude special tokens and the original token
            scores[tokenizer.all_special_ids] = float("-inf")
            scores[input_ids[0, pos]] = float("-inf")

            top_score, top_idx = scores.max(dim=0)
            if top_score.item() > best_score:
                best_score = top_score.item()
                best_pos = pos
                best_token_id = top_idx.item()

        if best_pos < 0:
            break

        # Apply the flip
        new_ids = input_ids.clone()
        new_ids[0, best_pos] = best_token_id
        current_text = tokenizer.decode(new_ids[0], skip_special_tokens=True)

        # Check if attack succeeded
        label, conf, _ = model_wrapper.predict(current_text)
        if target_label is not None:
            if label.lower() == target_label.lower():
                logger.info("HotFlip: success at flip %d", flip_num + 1)
                return current_text
        else:
            if label != orig_label:
                logger.info("HotFlip: success at flip %d", flip_num + 1)
                return current_text

        logger.debug("HotFlip: flip %d/%d, best_score=%.4f", flip_num + 1, max_flips, best_score)

    logger.info("HotFlip: finished (%d flips)", max_flips)
    return current_text
