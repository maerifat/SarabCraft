"""
Word substitution sources for text adversarial attacks.

Three strategies:
  - WordNet synonyms (NLTK)
  - BERT MLM masked predictions (universal fallback)
  - Counter-fitted embedding neighbours (when available)
"""

import logging
from typing import Optional

from attacks.text_config import DEFAULT_MLM_MODEL

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


def get_embedding_neighbours(word: str, top_k: int = 5) -> list[str]:
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
    text = f"The {word} is important."
    return get_mlm_substitutions(text, position=1, top_k=top_k)

