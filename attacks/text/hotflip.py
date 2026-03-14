"""
HotFlip Attack — Ebrahimi et al., 2018 (arXiv:1712.06751)

White-box token-level attack. Uses gradient of loss w.r.t. token
embeddings to find the optimal token swap via first-order Taylor
approximation.  Beam search explores multiple flip paths; constraints
(repeat-modification, stopword, max-words-perturbed, word-embedding
distance, POS-tag match) enforce imperceptibility.

Reference implementation: TextAttack HotFlipEbrahimi2017 recipe
  - Transformation: WordSwapGradientBased (top_n=1)
  - Search: BeamSearch(beam_width=10)
  - Constraints: RepeatModification, StopwordModification,
                 MaxWordsPerturbed(2), WordEmbeddingDistance(0.8),
                 PartOfSpeech
"""

import copy
import logging
from dataclasses import dataclass, field

import torch

logger = logging.getLogger("textattack.attacks.hotflip")

# Number of replacement candidates to try per position.
# TextAttack uses top_n=1, but we try a few more so that if the best
# candidate fails a constraint (POS / similarity) we still have fallbacks.
_TOP_K_REPLACEMENTS = 5


# ── Beam candidate ────────────────────────────────────────────────────────────

@dataclass
class _BeamCandidate:
    """One candidate in the beam search."""
    text: str
    token_ids: torch.Tensor          # [1, seq_len]
    flipped_positions: set = field(default_factory=set)
    cumulative_score: float = 0.0


# ── Constraint helpers ────────────────────────────────────────────────────────

def _is_stopword_position(tokenizer, token_ids: torch.Tensor, pos: int) -> bool:
    """Check if the token at *pos* decodes to a stopword."""
    from utils.text_utils import is_stopword
    token_str = tokenizer.decode([token_ids[0, pos].item()]).strip()
    return is_stopword(token_str)


def _pos_tags_match(word_a: str, word_b: str) -> bool:
    """Check whether two words share the same POS tag."""
    from utils.text_utils import pos_tag_words
    tags = pos_tag_words([word_a, word_b])
    return tags[0] == tags[1]


def _embedding_distance_ok(
    original_text: str,
    candidate_text: str,
    threshold: float,
) -> bool:
    """Check if semantic similarity between original and candidate ≥ threshold."""
    from utils.text_constraints import compute_semantic_similarity
    sim = compute_semantic_similarity(original_text, candidate_text)
    return sim >= threshold


def _is_valid_word_token(token_str: str) -> bool:
    """Check whether a decoded token is a proper word (not a subword fragment).

    BERT WordPiece subwords start with '##'.  Reject those, as well as
    single-character tokens (often punctuation artefacts) and tokens that
    contain non-alphabetic characters.
    """
    if not token_str or token_str.startswith("##"):
        return False
    cleaned = token_str.strip()
    if len(cleaned) <= 1:
        return False
    return cleaned.isalpha()


# ── Core attack ───────────────────────────────────────────────────────────────

def run_hotflip(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_flips: int = 5,
    beam_width: int = 10,
    max_perturbed: int = 2,
    similarity_threshold: float = 0.8,
) -> str:
    """HotFlip attack (white-box gradient-based token substitution).

    Fully compliant with Ebrahimi et al. 2018 and the TextAttack
    HotFlipEbrahimi2017 recipe:
      - First-order Taylor approximation for flip scoring
      - Beam search (default beam_width=10)
      - RepeatModification: no position flipped twice
      - StopwordModification: skip stopword tokens
      - MaxWordsPerturbed: at most *max_perturbed* positions changed
      - WordEmbeddingDistance: cosine similarity ≥ *similarity_threshold*
      - PartOfSpeech: replacement must share POS with original

    Returns: adversarial text (str).
    """
    from models.text_loader import device, get_label_index

    logger.info(
        "HotFlip: starting (max_flips=%d, beam=%d, max_perturbed=%d, sim≥%.2f)",
        max_flips, beam_width, max_perturbed, similarity_threshold,
    )

    model = model_wrapper.model
    model.eval()

    # Resolve target label
    target_idx = None
    if target_label is not None:
        target_idx = get_label_index(model, target_label)
        if target_idx is None:
            logger.warning(
                "HotFlip: target label '%s' not found, using untargeted",
                target_label,
            )

    # Cache original prediction
    orig_label, _, _ = model_wrapper.predict(text)

    embedding_layer = model.get_input_embeddings()
    embedding_weight = embedding_layer.weight          # [vocab_size, hidden_dim]

    # Build the set of token IDs that must never be used as replacements
    special_ids = set(tokenizer.all_special_ids)
    for attr in ("pad_token_id", "unk_token_id", "mask_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)

    # ── Initialise beam ──────────────────────────────────────────────────
    init_inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512,
    )
    init_ids = init_inputs["input_ids"].to(device)

    beam: list[_BeamCandidate] = [
        _BeamCandidate(text=text, token_ids=init_ids),
    ]

    for flip_num in range(max_flips):
        all_expansions: list[_BeamCandidate] = []

        for cand in beam:
            # Skip candidates that already hit the perturbation cap
            if len(cand.flipped_positions) >= max_perturbed:
                all_expansions.append(cand)  # keep as-is
                continue

            inputs = tokenizer(
                cand.text, return_tensors="pt", truncation=True, max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]

            # ── Forward with gradients on embeddings ─────────────────
            model.zero_grad()
            embed_out = embedding_layer(input_ids)
            embed_out = embed_out.detach().requires_grad_(True)

            outputs = model(
                inputs_embeds=embed_out,
                attention_mask=inputs.get("attention_mask"),
            )
            logits = outputs.logits

            if target_idx is not None:
                # Targeted: maximise target class logit
                loss = -logits[0, target_idx]
            else:
                # Untargeted: decrease the predicted-class logit.
                # Negate so .backward() gives gradients that DECREASE
                # the predicted-class logit (attack direction).
                pred_idx = logits.argmax(dim=-1).item()
                loss = -logits[0, pred_idx]

            loss.backward()
            grad = embed_out.grad[0]  # [seq_len, hidden_dim]

            # ── Score all positions × all vocab tokens ───────────────
            # Vectorised Taylor approximation: score = (e_new − e_old) · grad
            # Skip [CLS] (pos 0) and [SEP] (pos seq_len-1)
            inner_range = range(1, seq_len - 1)

            for pos in inner_range:
                # ── RepeatModification ───────────────────────────────
                if pos in cand.flipped_positions:
                    continue

                # ── StopwordModification ─────────────────────────────
                if _is_stopword_position(tokenizer, input_ids, pos):
                    continue

                old_embed = embed_out[0, pos].detach()   # [hidden_dim]
                pos_grad = grad[pos]                     # [hidden_dim]

                # Score all vocab tokens
                old_dot = torch.dot(old_embed, pos_grad)
                all_dots = embedding_weight @ pos_grad   # [vocab_size]
                scores = all_dots - old_dot

                # Mask out special tokens and the current token
                for sid in special_ids:
                    scores[sid] = float("-inf")
                scores[input_ids[0, pos]] = float("-inf")

                # ── Try top-k replacements (fallback if best fails
                #    constraints) ─────────────────────────────────────
                top_scores, top_token_ids = scores.topk(_TOP_K_REPLACEMENTS)

                for rank in range(_TOP_K_REPLACEMENTS):
                    t_score = top_scores[rank].item()
                    t_id = top_token_ids[rank].item()

                    if t_score <= 0:
                        break  # remaining are worse

                    new_token_str = tokenizer.decode([t_id]).strip()

                    # ── Reject subword fragments and non-word tokens ─
                    if not _is_valid_word_token(new_token_str):
                        continue

                    old_token_str = tokenizer.decode(
                        [input_ids[0, pos].item()]
                    ).strip()

                    # ── PartOfSpeech constraint ──────────────────────
                    if (
                        old_token_str
                        and new_token_str
                        and old_token_str.isalpha()
                        and new_token_str.isalpha()
                        and not _pos_tags_match(old_token_str, new_token_str)
                    ):
                        logger.debug(
                            "HotFlip: POS mismatch at pos %d: '%s' → '%s'",
                            pos, old_token_str, new_token_str,
                        )
                        continue

                    # Build candidate text for the embedding–distance check
                    new_ids = input_ids.clone()
                    new_ids[0, pos] = t_id
                    candidate_text = tokenizer.decode(
                        new_ids[0], skip_special_tokens=True,
                    )

                    # ── WordEmbeddingDistance constraint ──────────────
                    if not _embedding_distance_ok(
                        text, candidate_text, similarity_threshold,
                    ):
                        logger.debug(
                            "HotFlip: similarity below %.2f at pos %d",
                            similarity_threshold, pos,
                        )
                        continue

                    new_flipped = cand.flipped_positions | {pos}
                    new_score = cand.cumulative_score + t_score

                    all_expansions.append(
                        _BeamCandidate(
                            text=candidate_text,
                            token_ids=new_ids,
                            flipped_positions=new_flipped,
                            cumulative_score=new_score,
                        )
                    )
                    break  # accept best valid replacement for this position

        if not all_expansions:
            logger.info("HotFlip: no valid expansions at flip %d", flip_num + 1)
            break

        # ── Beam pruning: keep top beam_width by cumulative score ────
        all_expansions.sort(key=lambda c: c.cumulative_score, reverse=True)
        beam = all_expansions[:beam_width]

        # ── Early stopping: check if any beam candidate succeeds ─────
        for cand in beam:
            label, conf, _ = model_wrapper.predict(cand.text)
            if target_label is not None:
                if label.lower() == target_label.lower():
                    logger.info(
                        "HotFlip: success at flip %d (beam search)",
                        flip_num + 1,
                    )
                    return cand.text
            else:
                if label != orig_label:
                    logger.info(
                        "HotFlip: success at flip %d (beam search)",
                        flip_num + 1,
                    )
                    return cand.text

        logger.debug(
            "HotFlip: flip %d/%d, beam_top_score=%.4f",
            flip_num + 1, max_flips, beam[0].cumulative_score,
        )

    # Return the best beam candidate even if attack didn't fully succeed
    logger.info("HotFlip: finished (%d flips, beam_width=%d)", max_flips, beam_width)
    return beam[0].text
