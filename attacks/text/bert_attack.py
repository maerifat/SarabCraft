"""
BERT-Attack — Li et al., 2020 (arXiv:2004.09984)

Black-box word-level attack using BERT masked language model for
contextually appropriate word substitutions.  Sub-word aware:
handles WordPiece tokenization via Cartesian-product BPE combinations
ranked by perplexity.

Key innovation: feeds the ORIGINAL (unmasked) text into the MLM and reads
predictions at all token positions simultaneously, rather than masking and
predicting (which is the BAE approach).  This produces better contextual
substitutes because the model sees the actual word as context.
"""

import logging

logger = logging.getLogger("textattack.attacks.bert_attack")

# ---------------------------------------------------------------------------
# Official BERT-Attack filter_words list (~240 words)
# Taken from the official repo: LinyangLee/BERT-Attack/blob/master/bertattack.py
# ---------------------------------------------------------------------------
BERT_ATTACK_FILTER_WORDS = frozenset([
    "a", "about", "above", "after", "again", "against", "ain", "all", "am",
    "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn",
    "doesn't", "doing", "don", "don't", "down", "during", "each", "few",
    "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't",
    "have", "haven", "haven't", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma",
    "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my",
    "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's",
    "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t",
    "than", "that", "that'll", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we",
    "were", "weren", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't",
    "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
])


def run_bert_attack(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    max_candidates: int = 48,
    max_perturbation_ratio: float = 0.4,
    threshold_pred_score: float = 0.0,
    use_bpe: bool = True,
) -> str:
    """BERT-Attack: uses BERT MLM for contextual word replacement.

    Faithful to Li et al., 2020 — key algorithmic choices:
      1. Word importance: [UNK] replacement with combined formula
         (preserves sentence structure, matches official _get_masked())
      2. MLM candidates: feed ORIGINAL (unmasked) text, read predictions at
         all positions in one forward pass (core innovation of BERT-Attack)
      3. Sub-word BPE: Cartesian product of per-position top-k predictions,
         ranked by perplexity through MLM
      4. Stopword filter: official filter_words list (~240 words)
      5. Impact tracking: uses current_prob, updated after each substitution
      6. No inline semantic similarity — official uses post-hoc USE evaluation

    Args:
        model_wrapper: _TextModelWrapper with .predict() and .predict_probs()
        tokenizer: HuggingFace tokenizer (unused — MLM uses its own tokenizer)
        text: input text to attack
        target_label: target class name for targeted attack (None = untargeted)
        max_candidates: top-k MLM predictions per position (paper default: 48)
        max_perturbation_ratio: max fraction of words to perturb (paper: 0.4)
        threshold_pred_score: minimum MLM logit cutoff for single-token words
        use_bpe: handle multi-subword words via BPE combination

    Returns:
        Adversarial text (str).
    """
    from utils.text_word_importance import unk_importance
    from utils.text_utils import get_words_and_spans, replace_word_at, clean_word
    from utils.text_word_substitution import get_bert_attack_substitutions

    logger.info(
        "BERT-Attack: starting (cands=%d, pert_ratio=%.2f, threshold=%.2f, bpe=%s)",
        max_candidates, max_perturbation_ratio, threshold_pred_score, use_bpe,
    )

    words_spans = get_words_and_spans(text)
    if not words_spans:
        return text

    words = [w for w, _, _ in words_spans]
    max_perturbs = max(1, int(len(words) * max_perturbation_ratio))

    # ── Step 1: Word importance ranking via [UNK] replacement ──
    # Official uses [UNK] replacement (not deletion) with combined formula.
    orig_label, orig_conf, orig_label_idx = model_wrapper.predict(text)
    orig_probs = model_wrapper.predict_probs(text)
    importance = unk_importance(model_wrapper, text, original_label=orig_label)

    # ── Step 2: Get MLM predictions for ALL positions in one forward pass ──
    # Feeds original (unmasked) text — core BERT-Attack innovation.
    all_substitutions = get_bert_attack_substitutions(
        text,
        top_k=max_candidates,
        threshold_pred_score=threshold_pred_score,
        use_bpe=use_bpe,
    )

    current_text = text
    perturbations_made = 0
    # Track current probability for accurate impact calculation
    current_prob = orig_probs[orig_label_idx] if orig_label_idx < len(orig_probs) else orig_conf
    perturbed_indices = set()

    for word_idx, score in importance:
        if perturbations_made >= max_perturbs:
            break

        word = words[word_idx]
        cleaned = clean_word(word)

        # Official filter_words: extensive stopword + function word list
        if cleaned.lower() in BERT_ATTACK_FILTER_WORDS or len(cleaned) <= 1:
            continue

        # Prevent re-perturbing the same word position
        if word_idx in perturbed_indices:
            continue

        # Get pre-computed candidates for this word position
        if word_idx >= len(all_substitutions):
            continue
        candidates = all_substitutions[word_idx]

        if not candidates:
            continue

        # Official sub-word filter: reject candidates containing '##'
        # (single-subword candidates already filtered by get_bert_attack_substitutions;
        # apply to BPE combination outputs as well)
        filtered_candidates = []
        for cand in candidates:
            cand_clean = cand.strip()
            if not cand_clean:
                continue
            if "##" in cand_clean:
                continue
            if cand_clean.lower() == cleaned:
                continue
            filtered_candidates.append(cand_clean)

        best_text = None
        best_impact = -1.0

        for cand in filtered_candidates:
            candidate_text = replace_word_at(current_text, word_idx, cand)

            label, conf, label_idx = model_wrapper.predict(candidate_text)

            # Check for success (label flip)
            if target_label is not None:
                if label.lower() == target_label.lower():
                    logger.info("BERT-Attack: success at perturbation %d", perturbations_made + 1)
                    return candidate_text
            else:
                if label != orig_label:
                    logger.info("BERT-Attack: success at perturbation %d", perturbations_made + 1)
                    return candidate_text

            # Impact = drop in original-class probability (using CURRENT prob)
            cand_probs = model_wrapper.predict_probs(candidate_text)
            cand_orig_prob = cand_probs[orig_label_idx] if orig_label_idx < len(cand_probs) else conf
            impact = current_prob - cand_orig_prob
            if impact > best_impact:
                best_impact = impact
                best_text = candidate_text
                best_new_prob = cand_orig_prob

        if best_text is not None:
            current_text = best_text
            current_prob = best_new_prob  # Update tracked probability
            perturbations_made += 1
            perturbed_indices.add(word_idx)

    logger.info("BERT-Attack: finished (%d perturbations)", perturbations_made)
    return current_text
