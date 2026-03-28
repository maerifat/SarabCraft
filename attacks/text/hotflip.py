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

import logging
from dataclasses import dataclass, field

import torch

logger = logging.getLogger("textattack.attacks.hotflip")


# ── Beam candidate ────────────────────────────────────────────────────────────

@dataclass
class _BeamCandidate:
    """One candidate in the beam search."""
    text: str
    token_ids: torch.Tensor          # [1, seq_len]
    flipped_positions: set = field(default_factory=set)


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


# ── Word Embedding Distance (counter-fitted GloVe, per-word cosine) ──────────
# Matches TextAttack WordEmbeddingDistance(min_cos_sim=0.8):
#   - Uses counter-fitted GloVe embeddings (paragramcf)
#   - Computes cosine similarity between INDIVIDUAL words (not sentence-level)
#   - include_unknown_words=True (TextAttack default): if either word is OOV,
#     the constraint passes

_wv_model = None
_wv_loaded = False


def _get_word_vectors():
    """Lazy-load counter-fitted word vectors (same source as text_word_substitution.py)."""
    global _wv_model, _wv_loaded
    if _wv_loaded:
        return _wv_model
    _wv_loaded = True

    try:
        from gensim.models import KeyedVectors
    except ImportError:
        logger.info("gensim not installed — WordEmbeddingDistance constraint will "
                     "allow all swaps (include_unknown_words=True)")
        return None

    import os
    candidate_paths = [
        os.path.expanduser("~/.textattack/embedding/paragramcf"),
        os.path.expanduser("~/.cache/textattack/paragramcf"),
        os.path.expanduser("~/.cache/glove/glove.840B.300d.txt"),
        os.path.expanduser("~/.cache/glove/glove.6B.300d.txt"),
    ]

    for path in candidate_paths:
        if os.path.isfile(path):
            try:
                logger.info("Loading word vectors from %s", path)
                _wv_model = KeyedVectors.load(path, mmap="r")
                return _wv_model
            except Exception:
                try:
                    _wv_model = KeyedVectors.load_word2vec_format(path, binary=False)
                    return _wv_model
                except Exception:
                    continue

    logger.info("No local word vectors found — WordEmbeddingDistance constraint "
                "will allow all swaps (include_unknown_words=True)")
    return None


def _word_embedding_cos_sim(word_a: str, word_b: str) -> float | None:
    """Per-word cosine similarity using counter-fitted GloVe.

    Returns cosine similarity float, or None if either word is OOV.
    Matches TextAttack WordEmbeddingDistance._check_constraint() logic.
    """
    vectors = _get_word_vectors()
    if vectors is None:
        return None  # OOV → include_unknown_words=True → pass

    a = word_a.lower()
    b = word_b.lower()

    try:
        return float(vectors.similarity(a, b))
    except KeyError:
        return None  # OOV → include_unknown_words=True → pass


def _word_embedding_distance_ok(
    old_word: str,
    new_word: str,
    min_cos_sim: float,
) -> bool:
    """Check WordEmbeddingDistance constraint (per-word, counter-fitted GloVe).

    Matches TextAttack defaults:
      - include_unknown_words=True → if either word is OOV, allow the swap
      - compare_against_original=True → always compare against original word
    """
    sim = _word_embedding_cos_sim(old_word, new_word)
    if sim is None:
        return True  # include_unknown_words=True
    return sim >= min_cos_sim


def _is_valid_word_token(token_str: str) -> bool:
    """Check whether a decoded token is a proper word (not a subword fragment).

    Matches TextAttack WordSwapGradientBased filter:
      - has_letter(word): must contain at least one letter
      - len(words_from_text(word)) == 1: must be a single word
    """
    if not token_str or token_str.startswith("##"):
        return False
    cleaned = token_str.strip()
    if not cleaned:
        return False
    # Must contain at least one letter (has_letter)
    if not any(c.isalpha() for c in cleaned):
        return False
    # Must be a single word (no spaces, no multi-word tokens)
    if " " in cleaned:
        return False
    return True


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

    Exact match with TextAttack HotFlipEbrahimi2017 recipe
    (Ebrahimi et al. 2018, arXiv:1712.06751):

    Transformation — WordSwapGradientBased(top_n=1):
      - Cross-entropy loss gradient w.r.t. input embeddings
        (matching TextAttack HuggingFaceModelWrapper.get_grad)
      - First-order Taylor approximation: diffs = E @ grad − (E @ grad)[cur]
      - lookup_table via model.get_input_embeddings().weight.data
      - Mask only pad_token_id
      - Global top-1 candidate: flatten [positions × vocab], argsort,
        filter has_letter + single-word, pick first valid

    Search — BeamSearch(beam_width=10):
      - Each beam candidate produces exactly 1 expansion (top-1)
      - All expansions scored via goal function (model re-query)
      - Top beam_width kept by goal function score
      - Early stopping when label flips

    Constraints:
      - Pre-transformation: RepeatModification, StopwordModification
      - Post-transformation: MaxWordsPerturbed(2),
        WordEmbeddingDistance(min_cos_sim=0.8, counter-fitted GloVe),
        PartOfSpeech

    Extension beyond TextAttack: targeted attack support via target_label.

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
    orig_label, orig_conf, orig_label_idx = model_wrapper.predict(text)

    embedding_layer = model.get_input_embeddings()
    embedding_weight = embedding_layer.weight.data     # [vocab_size, hidden_dim]

    # ── Initialise beam ──────────────────────────────────────────────────
    init_inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512,
    )
    init_ids = init_inputs["input_ids"].to(device)

    beam: list[_BeamCandidate] = [
        _BeamCandidate(text=text, token_ids=init_ids),
    ]

    best_result = beam[0]

    for flip_num in range(max_flips):
        potential_next_beam: list[_BeamCandidate] = []

        for cand in beam:
            # ── MaxWordsPerturbed (pre-check): skip candidates that
            #    already hit the perturbation cap ─────────────────────
            if len(cand.flipped_positions) >= max_perturbed:
                continue

            inputs = tokenizer(
                cand.text, return_tensors="pt", truncation=True, max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]

            # ── Pre-transformation constraints: determine eligible positions ─
            eligible_positions = []
            for pos in range(1, seq_len - 1):  # skip [CLS] and [SEP]
                # RepeatModification: no position flipped twice
                if pos in cand.flipped_positions:
                    continue
                # StopwordModification: skip stopword tokens
                if _is_stopword_position(tokenizer, input_ids, pos):
                    continue
                eligible_positions.append(pos)

            if not eligible_positions:
                continue

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
                # Targeted: minimise cross-entropy w.r.t. target → negate
                loss = -torch.nn.functional.cross_entropy(
                    logits,
                    torch.tensor([target_idx], device=logits.device),
                )
            else:
                # Untargeted: maximise cross-entropy w.r.t. predicted class
                # (matching TextAttack HuggingFaceModelWrapper.get_grad:
                #  labels = logits.argmax(dim=1); loss = model(..., labels=labels).loss)
                pred_idx = logits.argmax(dim=-1)
                loss = torch.nn.functional.cross_entropy(logits, pred_idx)

            loss.backward()
            grad = embed_out.grad[0]  # [seq_len, hidden_dim]

            # ── Score all eligible positions × all vocab tokens ──────
            # Vectorised Taylor approximation: score = (e_new − e_old) · grad
            # Build [num_eligible, vocab_size] score matrix, then flatten
            # and find global top-1 — matching TextAttack
            # WordSwapGradientBased._get_replacement_words_by_grad()

            num_eligible = len(eligible_positions)
            vocab_size = embedding_weight.shape[0]
            diffs = torch.zeros(num_eligible, vocab_size, device=device)

            for j, pos in enumerate(eligible_positions):
                pos_grad = grad[pos]                     # [hidden_dim]
                b_grads = embedding_weight @ pos_grad    # [vocab_size]
                a_grad = b_grads[input_ids[0, pos]]
                diffs[j] = b_grads - a_grad

            # Mask pad token (matching TextAttack WordSwapGradientBased:
            # diffs[:, self.tokenizer.pad_token_id] = float("-inf"))
            diffs[:, tokenizer.pad_token_id] = float("-inf")

            # ── Global top-1 selection (matching top_n=1) ────────────
            # Flatten, argsort, pick the single best valid (position, token)
            flat_sorted = (-diffs).flatten().argsort()

            found = False
            for flat_idx in flat_sorted.tolist():
                j_idx = flat_idx // vocab_size
                t_id = flat_idx % vocab_size

                pos = eligible_positions[j_idx]
                new_token_str = tokenizer.decode([t_id]).strip()

                # Valid word filter (matching TextAttack has_letter + single-word)
                if not _is_valid_word_token(new_token_str):
                    continue

                # Build the candidate
                new_ids = input_ids.clone()
                new_ids[0, pos] = t_id
                candidate_text = tokenizer.decode(
                    new_ids[0], skip_special_tokens=True,
                )

                new_flipped = cand.flipped_positions | {pos}

                potential_next_beam.append(
                    _BeamCandidate(
                        text=candidate_text,
                        token_ids=new_ids,
                        flipped_positions=new_flipped,
                    )
                )
                found = True
                break  # top_n=1: only one transformation per beam candidate

        if not potential_next_beam:
            logger.info("HotFlip: no valid expansions at flip %d", flip_num + 1)
            break

        # ── Post-transformation constraints ──────────────────────────
        # Apply WordEmbeddingDistance and PartOfSpeech AFTER candidate
        # generation, matching TextAttack's constraint pipeline.
        constrained_beam: list[_BeamCandidate] = []
        for cand in potential_next_beam:
            passes = True

            # Identify what changed: compare against original token_ids
            cand_ids = cand.token_ids
            orig_inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512,
            )
            orig_ids = orig_inputs["input_ids"].to(device)

            for pos in cand.flipped_positions:
                if pos >= orig_ids.shape[1] or pos >= cand_ids.shape[1]:
                    passes = False
                    break

                old_tid = orig_ids[0, pos].item()
                new_tid = cand_ids[0, pos].item()

                if old_tid == new_tid:
                    continue  # not actually modified

                old_word = tokenizer.decode([old_tid]).strip()
                new_word = tokenizer.decode([new_tid]).strip()

                # WordEmbeddingDistance: per-word counter-fitted GloVe cosine
                if not _word_embedding_distance_ok(
                    old_word, new_word, similarity_threshold,
                ):
                    logger.debug(
                        "HotFlip: WordEmbeddingDistance below %.2f at pos %d: '%s' → '%s'",
                        similarity_threshold, pos, old_word, new_word,
                    )
                    passes = False
                    break

                # PartOfSpeech: replacement must share POS with original
                if (
                    old_word
                    and new_word
                    and old_word.isalpha()
                    and new_word.isalpha()
                    and not _pos_tags_match(old_word, new_word)
                ):
                    logger.debug(
                        "HotFlip: POS mismatch at pos %d: '%s' → '%s'",
                        pos, old_word, new_word,
                    )
                    passes = False
                    break

            if passes:
                constrained_beam.append(cand)

        if not constrained_beam:
            logger.info("HotFlip: all expansions failed constraints at flip %d", flip_num + 1)
            break

        # ── Goal function scoring (matching TextAttack BeamSearch) ───
        # Score all candidates via model re-query, rank by goal function
        # score, keep top beam_width.
        scored: list[tuple[float, _BeamCandidate]] = []

        for cand in constrained_beam:
            probs = model_wrapper.predict_probs(cand.text)

            if target_idx is not None:
                # Targeted: score = P(target_class)
                goal_score = probs[target_idx]
            else:
                # Untargeted: score = 1 − P(original_class)
                goal_score = 1.0 - probs[orig_label_idx]

            scored.append((goal_score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_cand = scored[0]

        # ── Early stopping: check if best candidate succeeds ─────────
        best_label, _, _ = model_wrapper.predict(best_cand.text)
        if target_label is not None:
            if best_label.lower() == target_label.lower():
                logger.info(
                    "HotFlip: success at flip %d (beam search)",
                    flip_num + 1,
                )
                return best_cand.text
        else:
            if best_label != orig_label:
                logger.info(
                    "HotFlip: success at flip %d (beam search)",
                    flip_num + 1,
                )
                return best_cand.text

        # Update beam: top beam_width by goal function score
        beam = [cand for _, cand in scored[:beam_width]]
        best_result = beam[0]

        logger.debug(
            "HotFlip: flip %d/%d, beam_top_goal_score=%.4f",
            flip_num + 1, max_flips, best_score,
        )

    # Return the best beam candidate even if attack didn't fully succeed
    logger.info("HotFlip: finished (%d flips, beam_width=%d)", max_flips, beam_width)
    return best_result.text
