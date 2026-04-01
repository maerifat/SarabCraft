"""
Seq2Sick Attack — Cheng et al., AAAI 2020 (arXiv:1803.01128)

Exact implementation of "Seq2Sick: Evaluating the Robustness of
Sequence-to-Sequence Models with Adversarial Examples."

Objective (Paper Eq. 9, Algorithm 1):
  min_δ  L(X+δ) + λ₁·Σᵢ‖δᵢ‖₂ + λ₂·Σᵢ min_{wⱼ∈W} ‖xᵢ+δᵢ−wⱼ‖²
  s.t.  xᵢ+δᵢ ∈ W  ∀i

Attack Modes (seq2seq encoder-decoder models):
  Non-overlapping (Eq. 3): output differs at every position from original.
  Targeted keyword (Eq. 7 + collision mask Eq. 6): specified keywords
  appear in the decoder output.

Key techniques from the paper:
  - Projected gradient descent in encoder embedding space
  - Group lasso (ℓ₂,₁) regularisation via proximal operator for sparse
    word-level changes (only ~2-3 words modified)
  - Gradient regularisation to keep perturbations near valid embeddings
  - Straight-through projection to discrete tokens each iteration
  - Multi-learning-rate search (paper: [0.1, 0.5] targeted, [2] untargeted)

For classification models the same optimisation framework is applied
with a CW-style hinge loss replacing the seq2seq-specific objectives.

Official reference: https://github.com/cmhcbb/Seq2Sick
"""

import logging

import torch

logger = logging.getLogger("textattack.attacks.seq2sick")


# ── Model helpers ────────────────────────────────────────────────────────────

def _is_seq2seq(model):
    """Detect encoder-decoder architecture via HuggingFace config."""
    return getattr(model.config, "is_encoder_decoder", False)


def _get_embedding_matrix(model):
    """Extract the (encoder) token embedding weight matrix."""
    if hasattr(model, "get_encoder"):
        enc = model.get_encoder()
        if hasattr(enc, "get_input_embeddings"):
            e = enc.get_input_embeddings()
            if e is not None:
                return e.weight
    if hasattr(model, "get_input_embeddings"):
        e = model.get_input_embeddings()
        if e is not None:
            return e.weight
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "word_embeddings" in name:
            return param
    raise ValueError("Cannot locate token embedding matrix in model")


def _get_embedding_layer(model):
    """Get the embedding layer callable for the encoder forward pass."""
    if hasattr(model, "get_encoder"):
        enc = model.get_encoder()
        if hasattr(enc, "get_input_embeddings"):
            e = enc.get_input_embeddings()
            if e is not None:
                return e
    if hasattr(model, "get_input_embeddings"):
        e = model.get_input_embeddings()
        if e is not None:
            return e
    raise ValueError("Cannot locate embedding layer in model")


# ── Loss functions (Paper Eq. 3, Eq. 6, Eq. 7) ─────────────────────────────

def _non_overlapping_loss(logits, original_ids, margin=0.0):
    """Non-overlapping attack loss — Paper Eq. 3.

    L = Σ_t max{−ε, z_t^(s_t) − max_{y≠s_t} z_t^(y)}

    Positive when original word still dominates at position t;
    ≤ 0 when the attack has succeeded at every position.
    """
    seq_len = min(logits.size(0), original_ids.size(0))
    if seq_len == 0:
        return torch.zeros(1, device=logits.device)

    vocab_size = logits.size(1)
    logits_t = logits[:seq_len]
    ids = original_ids[:seq_len]

    one_hot = torch.zeros(seq_len, vocab_size, device=logits.device)
    for t in range(seq_len):
        one_hot[t, ids[t]] = 1.0

    real = (logits_t * one_hot).sum(dim=1)
    other = (logits_t * (1 - one_hot) - one_hot * 1e4).max(dim=1).values

    return torch.clamp(real - other, min=-margin).sum()


def _keyword_loss(logits, keyword_ids, margin=0.0):
    """Targeted keyword attack loss — Paper Eq. 7 with collision mask (Eq. 6).

    L = Σᵢ min_t { m_t(max{−ε, max_{y≠kᵢ} z_t^(y) − z_t^(kᵢ)}) }

    The collision mask m_t (Eq. 6) removes positions already dominated
    by a keyword so that remaining keywords do not compete for the same
    decoder output slot.
    """
    vocab_size = logits.size(1)
    working = logits
    loss = torch.zeros(1, device=logits.device)

    for k_id in keyword_ids:
        if working.size(0) == 0:
            break

        k_one_hot = torch.zeros(working.size(0), vocab_size,
                                device=logits.device)
        k_one_hot[:, k_id] = 1.0

        real = (working * k_one_hot).sum(dim=1)
        other = (working * (1 - k_one_hot) - k_one_hot * 1e4).max(dim=1).values

        per_pos = torch.clamp(other - real, min=-margin)
        t_loss, t_pos = per_pos.min(dim=0)
        loss = loss + t_loss

        # Collision mask (Eq. 6): when keyword dominates (hinge ≤ 0),
        # remove that position so later keywords pick a different slot.
        if t_loss.item() <= 0:
            pos = t_pos.item()
            if working.size(0) > 1:
                keep = [i for i in range(working.size(0)) if i != pos]
                working = working[keep]
            else:
                working = working[:0]

    return loss


def _classification_loss(logits, target_idx, orig_idx, targeted, margin=0.0):
    """CW-style hinge loss for classifier adaptation.

    Targeted:   max{−ε, max_{y≠target} z(y) − z(target)}
    Untargeted: max{−ε, z(orig) − max_{y≠orig} z(y)}
    """
    num_classes = logits.size(-1)
    mask = torch.ones(num_classes, device=logits.device, dtype=torch.bool)

    if targeted:
        mask[target_idx] = False
        return torch.clamp(
            logits[mask].max() - logits[target_idx], min=-margin,
        )
    else:
        mask[orig_idx] = False
        return torch.clamp(
            logits[orig_idx] - logits[mask].max(), min=-margin,
        )


# ── Core optimisation (Paper Algorithm 1) ───────────────────────────────────

def _run_optimisation(
    model, embedding_matrix, input_embedding, attention_mask,
    compute_loss_fn, device, *,
    num_iterations, const, group_lasso, grad_reg,
    lr_list, is_seq2seq_model, decoder_input_ids=None,
):
    """Projected gradient descent with group lasso + gradient regularisation.

    Implements Algorithm 1 from the paper.  The modifier δ is optimised
    in continuous embedding space;  each iteration projects xᵢ+δᵢ to the
    nearest word in W (straight-through estimator for backward pass).
    """
    seq_len, hidden_dim = input_embedding.shape

    modifier = torch.zeros(seq_len, hidden_dim, device=device,
                           requires_grad=True)

    best_loss_val = float("inf")
    best_mod_norm = float("inf")
    best_words = None

    for lr in lr_list:
        found = False

        for k in range(num_iterations):
            # ── 1. New embeddings + project to nearest words ──
            new_emb = modifier + input_embedding

            with torch.no_grad():
                dists = torch.cdist(
                    new_emb.detach().unsqueeze(0),
                    embedding_matrix.unsqueeze(0),
                ).squeeze(0)                      # (seq_len, vocab_size)
                min_dists, nearest_ids = dists.min(dim=1)
                min_dist_sum = min_dists.sum()
                word_list = nearest_ids.tolist()
                projected = embedding_matrix[nearest_ids]

            # Straight-through: forward uses projected data,
            # backward flows through the original add graph.
            new_emb.data.copy_(projected.data)

            # ── 2. Forward pass ──
            if is_seq2seq_model:
                out = model(
                    inputs_embeds=new_emb.unsqueeze(0),
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )
            else:
                out = model(
                    inputs_embeds=new_emb.unsqueeze(0),
                    attention_mask=attention_mask,
                )

            logits = out.logits.squeeze(0)

            # ── 3. Adversarial loss ──
            loss1 = compute_loss_fn(logits)
            l1_val = loss1.item()

            # ── 4. Track best solution ──
            if l1_val <= 0:
                mn = modifier.data.norm().item()
                if mn < best_mod_norm:
                    best_mod_norm = mn
                    best_words = list(word_list)
                found = True

            if l1_val < best_loss_val:
                best_loss_val = l1_val
                if best_words is None:
                    best_words = list(word_list)

            # At the last iteration: stop (use tracked best).
            if k == num_iterations - 1:
                break

            # ── 5. Total loss (Paper Eq. 9) ──
            #   const·L  +  grad_reg_term  +  L∞ modifier penalty
            loss2 = modifier.max()
            if grad_reg:
                total = const * loss1 + min_dist_sum + loss2
            else:
                total = const * loss1 + loss2

            # ── 6. Back-propagation ──
            total.backward(retain_graph=True)

            if modifier.grad is None:
                logger.warning("Seq2Sick: no gradient at iter %d", k)
                break

            # ── 7. Group lasso proximal operator ──
            #   Per-word group: if ‖δᵢ‖₂ > γλ₁  → shrink
            #                   else             → zero out
            if group_lasso:
                with torch.no_grad():
                    gamma_const = lr * const
                    l2_norms = modifier.norm(2, dim=1)
                    for j in range(seq_len):
                        if l2_norms[j].item() > gamma_const:
                            modifier.data[j] -= (
                                gamma_const * modifier.data[j] / l2_norms[j]
                            )
                        else:
                            modifier.data[j].zero_()

            # ── 8. Gradient descent step ──
            with torch.no_grad():
                modifier.data -= lr * modifier.grad.data
                modifier.grad.zero_()

        # If the attack succeeded with this LR, stop trying more LRs.
        if found:
            logger.info("Seq2Sick: converged with lr=%.3f", lr)
            break
        else:
            logger.info(
                "Seq2Sick: lr=%.3f exhausted %d iters (best_loss=%.4f)",
                lr, num_iterations, best_loss_val,
            )

    return best_words


# ── Entry point ──────────────────────────────────────────────────────────────

def run_seq2sick(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    attack_mode: str = "non_overlapping",
    target_keywords: str = "",
    num_iterations: int = 200,
    const: float = 1.0,
    confidence_margin: float = 0.0,
    group_lasso: bool = True,
    grad_reg: bool = True,
) -> str:
    """Seq2Sick projected gradient attack — Cheng et al. AAAI 2020.

    For seq2seq models: non-overlapping (Eq. 3) or targeted keyword (Eq. 7)
    loss on decoder output logits — exact match to paper Algorithm 1.

    For classification models: CW-style hinge loss with the same projected
    gradient + group lasso + gradient regularisation framework.

    Args:
        model_wrapper:      wrapped model with .model / .predict() / .predict_probs().
        tokenizer:          HuggingFace tokenizer.
        text:               input text to attack.
        target_label:       target class (classifiers only; None = untargeted).
        attack_mode:        "non_overlapping" or "targeted_keyword" (seq2seq).
        target_keywords:    comma-separated keywords (targeted_keyword mode).
        num_iterations:     max optimisation iterations (paper: 200).
        const:              weight on adversarial loss λ (paper: 1.0).
        confidence_margin:  hinge loss margin ε (paper: ≥0).
        group_lasso:        enable group lasso proximal operator.
        grad_reg:           enable gradient regularisation term.

    Returns:
        Adversarial text (str).
    """
    model = model_wrapper.model
    device = next(model.parameters()).device
    seq2seq = _is_seq2seq(model)

    embedding_matrix = _get_embedding_matrix(model).detach()
    emb_layer = _get_embedding_layer(model)

    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        input_embedding = emb_layer(input_ids).squeeze(0)   # (N, d)

    # ── Seq2seq path (exact match to paper) ──────────────────────────────

    if seq2seq:
        logger.info(
            "Seq2Sick [seq2seq]: mode=%s iters=%d const=%.2f gl=%s gr=%s",
            attack_mode, num_iterations, const, group_lasso, grad_reg,
        )

        with torch.no_grad():
            original_output = model.generate(
                input_ids=input_ids, max_new_tokens=128, num_beams=5,
            )
        original_output_ids = original_output[0]

        # Teacher-forced decoder input: shift original output right
        ds = getattr(model.config, "decoder_start_token_id", None)
        if ds is None:
            ds = getattr(tokenizer, "pad_token_id", 0) or 0
        decoder_input_ids = torch.cat([
            torch.full((1, 1), ds, device=device, dtype=torch.long),
            original_output[:, :-1],
        ], dim=1)

        # Parse target keywords
        keyword_ids = []
        if attack_mode == "targeted_keyword" and target_keywords:
            for kw in target_keywords.split(","):
                kw = kw.strip()
                if kw:
                    keyword_ids.extend(
                        tokenizer.encode(kw, add_special_tokens=False),
                    )
            if not keyword_ids:
                logger.warning(
                    "No keywords parsed; falling back to non_overlapping",
                )
                attack_mode = "non_overlapping"

        # Paper: LR = [0.1, 0.5] for targeted, [2] for non-overlapping
        lr_list = ([0.1, 0.5] if attack_mode == "targeted_keyword"
                   else [2.0])

        def compute_loss(logits):
            if attack_mode == "targeted_keyword":
                return _keyword_loss(logits, keyword_ids, confidence_margin)
            return _non_overlapping_loss(
                logits, original_output_ids, confidence_margin,
            )

        best_words = _run_optimisation(
            model, embedding_matrix, input_embedding, attention_mask,
            compute_loss, device,
            num_iterations=num_iterations,
            const=const,
            group_lasso=group_lasso,
            grad_reg=grad_reg,
            lr_list=lr_list,
            is_seq2seq_model=True,
            decoder_input_ids=decoder_input_ids,
        )

    # ── Classification path (adapted framework) ─────────────────────────

    else:
        from models.text_loader import get_label_index

        orig_label, _, orig_idx = model_wrapper.predict(text)
        targeted = target_label is not None

        if targeted:
            target_idx = get_label_index(model, target_label)
            if target_idx is None:
                target_idx = 1 - orig_idx
        else:
            probs = model_wrapper.predict_probs(text)
            si = sorted(range(len(probs)),
                        key=lambda i: probs[i], reverse=True)
            target_idx = si[1] if len(si) > 1 else 1 - orig_idx

        logger.info(
            "Seq2Sick [classifier]: targeted=%s orig=%s(%d) tgt=%d "
            "iters=%d const=%.2f gl=%s gr=%s",
            targeted, orig_label, orig_idx, target_idx,
            num_iterations, const, group_lasso, grad_reg,
        )

        lr_list = [0.1, 0.5] if targeted else [2.0]

        def compute_loss(logits):
            return _classification_loss(
                logits, target_idx, orig_idx, targeted, confidence_margin,
            )

        best_words = _run_optimisation(
            model, embedding_matrix, input_embedding, attention_mask,
            compute_loss, device,
            num_iterations=num_iterations,
            const=const,
            group_lasso=group_lasso,
            grad_reg=grad_reg,
            lr_list=lr_list,
            is_seq2seq_model=False,
        )

    # ── Decode adversarial input ─────────────────────────────────────────

    if best_words is not None:
        adv_ids = torch.tensor(best_words, device=device)
        result = tokenizer.decode(adv_ids, skip_special_tokens=True)
        if result.strip():
            changed = sum(
                a != b
                for a, b in zip(best_words, input_ids[0].tolist())
            )
            logger.info(
                "Seq2Sick: done (changed %d / %d tokens)", changed,
                len(best_words),
            )
            return result

    logger.info("Seq2Sick: attack did not find valid adversarial input")
    return text
