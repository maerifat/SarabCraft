"""
Clare Attack — Li et al., 2021 (arXiv:2009.07502)

Contextualized Perturbation for Textual Adversarial Attack.
Uses distilroberta-base MLM for three contextual operations:
  - Replace: mask word → fill  (BAE method, min_confidence=5e-4)
  - Insert:  insert [MASK] before word → fill  (min_confidence=0.0)
  - Merge:   merge two POS-eligible adjacent words → fill  (min_confidence=5e-3)

Global greedy search: at each step, generate ALL candidates across ALL
positions and ALL operations, select the globally best perturbation.
USE similarity constraint (threshold=0.7, window_size=15, compare_against_original).
RepeatModification + StopwordModification constraints.

Reference: TextAttack CLARE2020 recipe (textattack.attack_recipes.clare_li_2020)
"""

import logging
import string

logger = logging.getLogger("textattack.attacks.clare")

# ── CLARE-specific MLM: distilroberta-base (paper Section 4) ─────────────

_clare_mlm = None
_clare_tok = None
CLARE_MLM_MODEL = "distilroberta-base"


def _get_clare_mlm():
    """Lazy-load distilroberta-base for CLARE candidate generation."""
    global _clare_mlm, _clare_tok
    if _clare_mlm is not None:
        return _clare_mlm, _clare_tok

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    logger.info("Loading CLARE MLM: %s", CLARE_MLM_MODEL)
    _clare_tok = AutoTokenizer.from_pretrained(CLARE_MLM_MODEL, use_fast=True)
    _clare_mlm = AutoModelForMaskedLM.from_pretrained(CLARE_MLM_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _clare_mlm.to(device)
    _clare_mlm.eval()
    return _clare_mlm, _clare_tok


# ── Token utilities (model-type aware, matches TextAttack utils) ─────────

def _check_if_subword(token, model_type, is_first_word=False):
    """Check if token is a BPE/WordPiece subword continuation."""
    if model_type in ("roberta", "gpt2", "bart", "xlnet"):
        # RoBERTa BPE: word-initial tokens start with 'Ġ'; continuations don't
        if is_first_word:
            return False
        return not token.startswith("\u0120")
    # BERT WordPiece
    return token.startswith("##")


def _strip_bpe(token, model_type):
    """Remove BPE artifacts from a token."""
    if model_type in ("roberta", "gpt2", "bart", "xlnet"):
        return token.lstrip("\u0120")
    return token.lstrip("#")


def _is_one_word(w):
    return len(w.split()) == 1 and len(w) > 0


def _is_punct(w):
    return all(c in string.punctuation for c in w) if w else True


# ── MLM candidate generation per operation ───────────────────────────────

def _mlm_candidates(masked_text, max_candidates, min_confidence):
    """Run MLM on masked_text and return filtered candidate words.

    Mirrors TextAttack's BAE-style MLM prediction used by
    WordSwapMaskedLM, WordInsertionMaskedLM, and WordMergeMaskedLM.
    """
    import torch

    model, tokenizer = _get_clare_mlm()
    device = next(model.parameters()).device
    model_type = model.config.model_type

    encoding = tokenizer(
        masked_text, max_length=512, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        preds = model(**inputs)[0]

    ids = inputs["input_ids"][0].tolist()
    try:
        masked_index = ids.index(tokenizer.mask_token_id)
    except ValueError:
        return []

    logits = preds[0, masked_index]
    probs = torch.softmax(logits, dim=0)
    ranked = torch.argsort(probs, descending=True)

    candidates = []
    for _id in ranked:
        _id = _id.item()
        token = tokenizer.convert_ids_to_tokens(_id)
        word = token

        if _check_if_subword(word, model_type, (masked_index == 1)):
            word = _strip_bpe(word, model_type)

        if (
            probs[_id] >= min_confidence
            and _is_one_word(word)
            and not _is_punct(word)
        ):
            clean = word.strip("\u0120").strip()
            if clean:
                candidates.append(clean)

        if len(candidates) >= max_candidates or probs[_id] < min_confidence:
            break

    return candidates


def _replace_candidates(words, index):
    """Replace: mask word → MLM fill.  max_candidates=50, min_confidence=5e-4."""
    _, tokenizer = _get_clare_mlm()
    masked = list(words)
    masked[index] = tokenizer.mask_token
    return _mlm_candidates(" ".join(masked), 50, 5e-4)


def _insert_candidates(words, index):
    """Insert: insert [MASK] BEFORE word → MLM fill.  max_candidates=50, min_confidence=0.0."""
    _, tokenizer = _get_clare_mlm()
    masked = list(words[:index]) + [tokenizer.mask_token] + list(words[index:])
    return _mlm_candidates(" ".join(masked), 50, 0.0)


def _merge_candidates(words, index):
    """Merge: replace word[i] with [MASK], delete word[i+1] → MLM fill.
    max_candidates=50, min_confidence=5e-3."""
    if index + 1 >= len(words):
        return []
    _, tokenizer = _get_clare_mlm()
    merged = list(words)
    merged[index] = tokenizer.mask_token
    del merged[index + 1]
    return _mlm_candidates(" ".join(merged), 50, 5e-3)


# ── POS-based merge eligibility (matches TextAttack find_merge_index) ────

_PTB_TO_UPOS = {
    "NN": "NOUN", "NNS": "NOUN", "NNP": "NOUN", "NNPS": "NOUN",
    "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB",
    "VBP": "VERB", "VBZ": "VERB",
    "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
    "RB": "ADV", "RBR": "ADV", "RBS": "ADV",
    "DT": "DET", "PDT": "DET",
    "PRP": "PRON", "PRP$": "PRON", "WP": "PRON", "WP$": "PRON",
    "CD": "NUM",
    # heuristic fallback tags
    "noun": "NOUN", "verb": "VERB", "adj": "ADJ", "adv": "ADV",
}


def _find_merge_indices(words):
    """Return indices eligible for merge based on POS-tag patterns.

    Matches TextAttack ``find_merge_index()`` in WordMergeMaskedLM.
    """
    from utils.text_utils import pos_tag_words

    tags = pos_tag_words(words)
    upos = [_PTB_TO_UPOS.get(t, t.upper()) for t in tags]

    result = []
    for i in range(len(upos) - 1):
        cur, nxt = upos[i], upos[i + 1]
        if cur == "NOUN" and nxt == "NOUN":
            result.append(i)
        elif cur == "ADJ" and nxt in ("NOUN", "NUM", "ADJ", "ADV"):
            result.append(i)
        elif cur == "ADV" and nxt in ("ADJ", "VERB"):
            result.append(i)
        elif cur == "VERB" and nxt in ("ADV", "VERB", "NOUN", "ADJ"):
            result.append(i)
        elif cur == "DET" and nxt in ("NOUN", "ADJ"):
            result.append(i)
        elif cur == "PRON" and nxt in ("NOUN", "ADJ"):
            result.append(i)
        elif cur == "NUM" and nxt in ("NUM", "NOUN"):
            result.append(i)
    return result


# ── Main CLARE attack (global greedy search) ─────────────────────────────

def run_clare(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_perturbations: int = 5,
    similarity_threshold: float = 0.7,
) -> str:
    """CLARE attack: contextualized perturbation with Replace, Insert, Merge.

    Faithfully implements the CLARE algorithm (Li et al., 2021) as
    specified in the TextAttack CLARE2020 recipe:
      - distilroberta-base MLM
      - Global greedy search (all positions × all ops per step)
      - USE similarity (window=15, compare against original)
      - RepeatModification + StopwordModification

    Returns: adversarial text (str).
    """
    from utils.text_utils import get_words_and_spans, is_stopword
    from utils.text_constraints import compute_windowed_semantic_similarity

    logger.info("Clare: starting (max_pert=%d, sim=%.2f)",
                max_perturbations, similarity_threshold)

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    orig_label, orig_conf, _ = model_wrapper.predict(text)

    current_text = text
    modified_indices: set[int] = set()
    perturbations_made = 0

    for _step in range(max_perturbations):
        current_spans = get_words_and_spans(current_text)
        if not current_spans:
            break
        words = [w for w, _, _ in current_spans]
        n = len(words)

        # ── Eligible indices for Replace & Insert ────────────────────
        eligible = [
            i for i in range(n)
            if i not in modified_indices and not is_stopword(words[i])
        ]
        if not eligible:
            break

        # ── POS-eligible merge indices (independent of RepeatModification,
        #    matching TextAttack WordMergeMaskedLM behaviour) ──────────
        merge_indices = _find_merge_indices(words)

        # ── Generate ALL candidates ──────────────────────────────────
        all_cands: list[tuple[str, str, int]] = []   # (text, op, idx)

        for idx in eligible:
            orig_word = words[idx].lower()

            # Replace
            for cand in _replace_candidates(words, idx):
                if cand.lower() != orig_word:
                    w = list(words); w[idx] = cand
                    all_cands.append((" ".join(w), "replace", idx))

            # Insert (before)
            for cand in _insert_candidates(words, idx):
                if cand.lower() != words[idx].lower():
                    w = list(words[:idx]) + [cand] + list(words[idx:])
                    all_cands.append((" ".join(w), "insert", idx))

        for idx in merge_indices:
            for cand in _merge_candidates(words, idx):
                if cand.lower() != words[idx].lower():
                    w = list(words); w[idx] = cand; del w[idx + 1]
                    all_cands.append((" ".join(w), "merge", idx))

        if not all_cands:
            break

        # ── Evaluate candidates ──────────────────────────────────────
        best_text = None
        best_score = float("-inf")
        best_op = None
        best_idx = None

        for cand_text, op_type, mod_idx in all_cands:
            # USE similarity (windowed, compared against ORIGINAL text)
            passes, _ = compute_windowed_semantic_similarity(
                text, cand_text, mod_idx,
                threshold=similarity_threshold,
                window_size=15,
            )
            if not passes:
                continue

            label, conf, _ = model_wrapper.predict(cand_text)

            # Success check
            if target_label is not None:
                if label.lower() == target_label.lower():
                    logger.info("Clare: success at step %d (%s)",
                                perturbations_made + 1, op_type)
                    return cand_text
            else:
                if label != orig_label:
                    logger.info("Clare: success at step %d (%s)",
                                perturbations_made + 1, op_type)
                    return cand_text

            # Score = confidence drop on original label
            score = orig_conf - conf
            if score > best_score:
                best_score = score
                best_text = cand_text
                best_op = op_type
                best_idx = mod_idx

        if best_text is None:
            break

        # ── Apply best perturbation & update tracking ────────────────
        current_text = best_text
        perturbations_made += 1

        if best_op == "replace":
            modified_indices.add(best_idx)

        elif best_op == "insert":
            # Shift existing modified indices >= insert point
            shifted = {(mi + 1 if mi >= best_idx else mi)
                       for mi in modified_indices}
            shifted.add(best_idx)
            modified_indices = shifted

        elif best_op == "merge":
            # word[best_idx+1] deleted → shift indices > best_idx down
            shifted = set()
            for mi in modified_indices:
                if mi == best_idx + 1:
                    continue          # position removed
                elif mi > best_idx + 1:
                    shifted.add(mi - 1)
                else:
                    shifted.add(mi)
            shifted.add(best_idx)
            modified_indices = shifted

        logger.info("Clare: applied %s at idx %d (step %d)",
                    best_op, best_idx, perturbations_made)

    logger.info("Clare: finished (%d perturbations)", perturbations_made)
    return current_text
