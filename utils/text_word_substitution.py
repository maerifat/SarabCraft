"""
Word substitution sources for text adversarial attacks.

Three strategies:
  - WordNet synonyms (NLTK)
  - BERT MLM masked predictions (universal fallback)
  - Counter-fitted embedding neighbours (when available)
"""

import logging
from typing import Optional

from attacks.text.config import DEFAULT_MLM_MODEL

logger = logging.getLogger("textattack.substitution")

# Lazy-loaded MLM model for substitutions
_mlm_model = None
_mlm_tokenizer = None


def _get_mlm(model_name: str = DEFAULT_MLM_MODEL):
    """Lazy-load the MLM model for substitution generation."""
    global _mlm_model, _mlm_tokenizer
    if _mlm_model is not None:
        return _mlm_model, _mlm_tokenizer

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    logger.info("Loading MLM model for substitutions: %s", model_name)
    _mlm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _mlm_model.to(device)
    _mlm_model.eval()
    return _mlm_model, _mlm_tokenizer


def get_wordnet_synonyms(word: str, pos: Optional[str] = None, max_candidates: int = 20) -> list[str]:
    """Get synonyms from WordNet via NLTK.

    Falls back to empty list if NLTK/WordNet not available.
    """
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        logger.debug("NLTK not available, skipping WordNet synonyms")
        return []

    # Map simple POS to WordNet POS
    pos_map = {"noun": wn.NOUN, "verb": wn.VERB, "adj": wn.ADJ, "adv": wn.ADV}
    wn_pos = pos_map.get(pos) if pos else None

    synonyms = set()
    synsets = wn.synsets(word.lower(), pos=wn_pos) if wn_pos else wn.synsets(word.lower())

    for syn in synsets:
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                synonyms.add(name)
                if len(synonyms) >= max_candidates:
                    return list(synonyms)
    return list(synonyms)


def get_mlm_substitutions(
    text: str,
    position: int,
    top_k: int = 50,
    mlm_model_name: str = DEFAULT_MLM_MODEL,
) -> list[str]:
    """Get contextual word substitutions using BERT masked language model.

    Args:
        text: original text
        position: word index to mask (0-based, refers to whitespace-split words)
        top_k: number of candidates to return
        mlm_model_name: MLM model to use

    Returns:
        List of substitute words, sorted by MLM probability.
    """
    import torch
    from utils.text_utils import get_words_and_spans

    model, tokenizer = _get_mlm(mlm_model_name)
    device = next(model.parameters()).device

    words_spans = get_words_and_spans(text)
    if position < 0 or position >= len(words_spans):
        return []

    original_word = words_spans[position][0].lower().strip(".,!?;:'\"()[]{}")

    # Build masked text
    words = [w for w, _, _ in words_spans]
    words[position] = tokenizer.mask_token
    masked_text = " ".join(words)

    inputs = tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Find the [MASK] token position in tokenized input
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (inputs["input_ids"][0] == mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return []

    mask_pos = mask_positions[0].item()
    mask_logits = logits[0, mask_pos]
    top_tokens = torch.topk(mask_logits, top_k).indices.tolist()

    candidates = []
    for token_id in top_tokens:
        word = tokenizer.decode([token_id]).strip()
        # Filter out subword tokens, empty, punctuation, and the original word
        if (
            word
            and not word.startswith("##")
            and word.isalpha()
            and word.lower() != original_word
        ):
            candidates.append(word)

    return candidates


def get_bert_attack_substitutions(
    text: str,
    top_k: int = 48,
    threshold_pred_score: float = 0.0,
    use_bpe: bool = True,
    mlm_model_name: str = DEFAULT_MLM_MODEL,
) -> list[list[str]]:
    """BERT-Attack MLM substitutions: feed ORIGINAL (unmasked) text to MLM.

    Key innovation of Li et al., 2020: instead of masking a word and
    predicting what goes there (BAE approach), feed the original unmasked
    text and read MLM predictions at each token position.  This produces
    more contextually appropriate candidates because the model sees the
    actual word as context.

    For multi-subword words, generates the Cartesian product of per-position
    top-k predictions and ranks combinations by perplexity (cross-entropy
    loss through the MLM), matching the official implementation.

    Args:
        text: original input text
        top_k: number of candidate tokens per position (paper default: 48)
        threshold_pred_score: minimum MLM logit score cutoff for single-token
            words (official default: 0.0, paper Section 5.1)
        use_bpe: if True, handle multi-subword words via BPE combination;
            if False, skip multi-subword words (official --use_bpe flag)
        mlm_model_name: pretrained MLM to use

    Returns:
        List where result[word_idx] = list of candidate replacement strings,
        one entry per whitespace-split word in ``text``.
    """
    import torch
    import torch.nn as nn

    model, tokenizer = _get_mlm(mlm_model_name)
    device = next(model.parameters()).device

    # ── Tokenize: split into words, then sub-words, tracking alignment ──
    # Matches official _tokenize(): seq.lower().split(' ')
    words = text.replace("\n", "").lower().split(" ")
    words = [w for w in words if w]

    sub_words: list[str] = []
    keys: list[tuple[int, int]] = []  # keys[word_idx] = (start, end) in sub_words
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append((index, index + len(sub)))
        index += len(sub)

    if not sub_words:
        return [[] for _ in words]

    max_content = 510  # max_length(512) - 2 for [CLS]/[SEP]
    full_sub_words = ["[CLS]"] + sub_words[:max_content] + ["[SEP]"]
    input_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(full_sub_words)]
    ).to(device)

    # ── Feed ORIGINAL (unmasked) text to MLM — single forward pass ──
    with torch.no_grad():
        logits = model(input_ids)[0].squeeze()  # (seq_len, vocab)

    pred_scores_all, pred_ids_all = torch.topk(logits, top_k, dim=-1)

    # Shift: position 0 is [CLS], content starts at position 1.
    # keys are 0-indexed vs sub_words (before [CLS]/[SEP]), so
    # sub_word index i corresponds to tensor position i+1.
    # Slice [1:] so keys index directly into this tensor.
    pred_ids_content = pred_ids_all[1:]      # (content_len+1, k)
    pred_scores_content = pred_scores_all[1:]

    result: list[list[str]] = []

    for word_idx in range(len(words)):
        start, end = keys[word_idx]

        if start >= max_content:
            result.append([])
            continue

        sub_len = end - start
        if sub_len == 0:
            result.append([])
            continue

        word_preds = pred_ids_content[start:end]      # (sub_len, k)
        word_scores = pred_scores_content[start:end]

        if sub_len == 1:
            # ── Single sub-word: direct top-k with threshold cutoff ──
            candidates = []
            for token_id, score in zip(word_preds[0], word_scores[0]):
                if threshold_pred_score != 0 and score.item() < threshold_pred_score:
                    break
                token = tokenizer.convert_ids_to_tokens(int(token_id))
                candidates.append(token)
            result.append(candidates)

        else:
            # ── Multi sub-word: Cartesian product + perplexity ranking ──
            if not use_bpe:
                result.append([])
                continue

            # Official limits: max 12 sub-word positions, 4 candidates each
            max_sub_pos = min(sub_len, 12)
            max_sub_cands = 4
            limited = word_preds[:max_sub_pos, :max_sub_cands]

            # Build Cartesian product of token-id combinations
            all_combos: list[list[int]] = []
            for i in range(max_sub_pos):
                if not all_combos:
                    all_combos = [[int(c)] for c in limited[i]]
                else:
                    new = []
                    for combo in all_combos:
                        for j in limited[i]:
                            new.append(combo + [int(j)])
                    all_combos = new

            if not all_combos:
                result.append([])
                continue

            # Limit to 24 combinations (official cap)
            combo_tensor = torch.tensor(all_combos[:24]).to(device)
            N, L = combo_tensor.size()

            # Perplexity ranking via cross-entropy through MLM
            with torch.no_grad():
                combo_logits = model(combo_tensor)[0]  # (N, L, vocab)

            c_loss = nn.CrossEntropyLoss(reduction="none")
            ppl = c_loss(
                combo_logits.view(N * L, -1), combo_tensor.view(-1)
            )
            ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # (N,)
            _, sorted_idx = torch.sort(ppl)

            candidates = []
            for idx in sorted_idx:
                token_ids = combo_tensor[idx]
                tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in token_ids]
                word_text = tokenizer.convert_tokens_to_string(tokens)
                candidates.append(word_text)

            result.append(candidates)

    return result


# ── Embedding-based nearest neighbours (primary Sub-W method) ─────────────

_word_vectors = None
_word_vectors_loaded = False


def _load_word_vectors():
    """Try to load GloVe / counter-fitted word vectors via gensim.

    Search order:
      1. Counter-fitted vectors (preferred — same as TextAttack reference)
      2. GloVe vectors
      3. Any gensim KeyedVectors file at well-known paths

    Returns the KeyedVectors object or None.
    """
    global _word_vectors, _word_vectors_loaded
    if _word_vectors_loaded:
        return _word_vectors
    _word_vectors_loaded = True

    try:
        from gensim.models import KeyedVectors
    except ImportError:
        logger.info("gensim not installed — embedding neighbours will use MLM fallback")
        return None

    import os
    # Well-known paths for pre-trained vectors (user can symlink or download)
    candidate_paths = [
        os.path.expanduser("~/.textattack/embedding/paragramcf"),  # TextAttack counter-fitted
        os.path.expanduser("~/.cache/textattack/paragramcf"),
        os.path.expanduser("~/.cache/glove/glove.840B.300d.txt"),
        os.path.expanduser("~/.cache/glove/glove.6B.300d.txt"),
    ]

    for path in candidate_paths:
        if os.path.isfile(path):
            try:
                logger.info("Loading word vectors from %s", path)
                _word_vectors = KeyedVectors.load(path, mmap="r")
                return _word_vectors
            except Exception:
                try:
                    _word_vectors = KeyedVectors.load_word2vec_format(path, binary=False)
                    return _word_vectors
                except Exception:
                    continue

    # NOTE: We do NOT attempt gensim.downloader.api.load() here because
    # downloading GloVe vectors (~1.7 GB) at runtime would hang the attack.
    # Users who want embedding-based Sub-W should pre-download vectors to
    # one of the paths above.
    logger.info("No local word vectors found — Sub-W will use MLM fallback")
    return None


def get_embedding_neighbours(
    word: str, top_k: int = 5, context_text: Optional[str] = None, position: Optional[int] = None
) -> list[str]:
    """Get nearest neighbours from word embedding space (paper: GloVe).

    Implements the Sub-W perturbation from TextBugger (Li et al., 2018):
    replace a word with its top-k nearest neighbours in a pre-trained
    word vector space.

    Falls back to MLM substitutions when word vectors are unavailable.

    Args:
        word: target word
        top_k: number of neighbours to return (paper default: 5)

    Returns:
        List of substitute words sorted by cosine similarity.
    """
    vectors = _load_word_vectors()
    if vectors is not None:
        try:
            neighbours = vectors.most_similar(word.lower(), topn=top_k)
            # neighbours is list of (word, similarity)
            return [w for w, _sim in neighbours if w.lower() != word.lower()]
        except KeyError:
            logger.debug("'%s' not in word vectors — falling back to MLM", word)

    # Fallback: MLM-based contextual substitutions
    logger.debug("Using MLM fallback for embedding neighbours of '%s'", word)
    if context_text and position is not None:
        return get_mlm_substitutions(context_text, position=position, top_k=top_k)
    text = f"The {word} is important."
    return get_mlm_substitutions(text, position=1, top_k=top_k)


def get_embedding_neighbours_with_scores(
    word: str, top_k: int = 50, context_text: Optional[str] = None, position: Optional[int] = None
) -> list[tuple[str, float]]:
    """Get nearest neighbours with cosine similarity scores.

    Same as get_embedding_neighbours() but returns (word, cosine_similarity)
    tuples so callers can apply a minimum-similarity pre-filter (e.g. δ ≥ 0.5
    for TextFooler — Jin et al., 2020).

    Falls back to MLM substitutions (with similarity = 1.0) when word vectors
    are unavailable.

    Args:
        word: target word
        top_k: number of neighbours to return

    Returns:
        List of (substitute_word, cosine_similarity) sorted by similarity descending.
    """
    vectors = _load_word_vectors()
    if vectors is not None:
        try:
            neighbours = vectors.most_similar(word.lower(), topn=top_k)
            return [(w, sim) for w, sim in neighbours if w.lower() != word.lower()]
        except KeyError:
            logger.debug("'%s' not in word vectors — falling back to MLM", word)

    # Fallback: MLM-based contextual substitutions (no real cosine score available)
    logger.debug("Using MLM fallback for embedding neighbours of '%s'", word)
    if context_text and position is not None:
        mlm_cands = get_mlm_substitutions(context_text, position=position, top_k=top_k)
    else:
        text = f"The {word} is important."
        mlm_cands = get_mlm_substitutions(text, position=1, top_k=top_k)
    # Assign similarity=1.0 so MLM fallback candidates aren't filtered by the
    # word-embedding cosine threshold (the caller can still apply sentence-level
    # semantic similarity as a gate).
    return [(w, 1.0) for w in mlm_cands]


