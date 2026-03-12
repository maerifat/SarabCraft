"""
Back-Translation Attack — Paraphrase via translation round-trip.

English → pivot language → English using HuggingFace MarianMT models.
Produces semantically equivalent paraphrases and checks if the classifier
label flips. If MarianMT is unavailable, falls back to word-level shuffling.
"""

import logging
import random

logger = logging.getLogger("textattack.attacks.back_translation")

# MarianMT model pairs for paraphrase generation
_PIVOT_LANGUAGES = [
    ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"),   # French
    ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"),   # German
    ("Helsinki-NLP/opus-mt-en-es", "Helsinki-NLP/opus-mt-es-en"),   # Spanish
]

_translation_cache = {}


def _load_translation_model(model_name: str):
    """Lazy-load MarianMT translation model."""
    if model_name in _translation_cache:
        return _translation_cache[model_name]

    try:
        from transformers import MarianMTModel, MarianTokenizer
        import torch

        logger.info("Loading translation model: %s", model_name)
        tok = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        _translation_cache[model_name] = (model, tok, device)
        return model, tok, device
    except Exception as e:
        logger.warning("Failed to load translation model %s: %s", model_name, e)
        return None


def _translate(text: str, model_name: str) -> str:
    """Translate text using a MarianMT model."""
    import torch

    result = _load_translation_model(model_name)
    if result is None:
        return text

    model, tok, device = result
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        translated = model.generate(**inputs, max_length=512)

    return tok.decode(translated[0], skip_special_tokens=True)


def _paraphrase_via_translation(text: str, pivot_idx: int = 0) -> str:
    """Translate text to pivot language and back."""
    if pivot_idx >= len(_PIVOT_LANGUAGES):
        return text

    en_to_pivot, pivot_to_en = _PIVOT_LANGUAGES[pivot_idx]
    intermediate = _translate(text, en_to_pivot)
    return _translate(intermediate, pivot_to_en)


def _paraphrase_fallback(text: str) -> str:
    """Fallback paraphrase: word-level synonym substitution + reordering."""
    from utils.text_utils import get_words_and_spans, replace_word_at, is_stopword
    from utils.text_word_substitution import get_mlm_substitutions

    words_spans = get_words_and_spans(text)
    if len(words_spans) < 3:
        return text

    # Randomly substitute 1-2 words via MLM
    current = text
    positions = [i for i, (w, _, _) in enumerate(words_spans) if not is_stopword(w) and len(w) > 2]

    if not positions:
        return text

    n_subs = min(2, len(positions))
    for pos in random.sample(positions, n_subs):
        candidates = get_mlm_substitutions(current, pos, top_k=5)
        if candidates:
            current = replace_word_at(current, pos, random.choice(candidates))

    return current


def run_back_translation(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    num_paraphrases: int = 5,
    similarity_threshold: float = 0.6,
) -> str:
    """Back-translation attack: generate paraphrases and check for label flip.

    Returns: adversarial text (str).
    """
    from utils.text_constraints import compute_semantic_similarity

    logger.info("Back-Translation: starting (num_paraphrases=%d, sim=%.2f)", num_paraphrases, similarity_threshold)

    orig_label, orig_conf, _ = model_wrapper.predict(text)

    # Try translation-based paraphrases first
    best_text = text
    best_impact = 0.0

    for pivot_idx in range(len(_PIVOT_LANGUAGES)):
        try:
            paraphrase = _paraphrase_via_translation(text, pivot_idx)
        except Exception:
            paraphrase = _paraphrase_fallback(text)

        if paraphrase == text:
            continue

        sim = compute_semantic_similarity(text, paraphrase)
        if sim < similarity_threshold:
            continue

        label, conf, _ = model_wrapper.predict(paraphrase)

        if target_label is not None:
            if label.lower() == target_label.lower():
                logger.info("Back-Translation: success with pivot %d", pivot_idx)
                return paraphrase
        else:
            if label != orig_label:
                logger.info("Back-Translation: success with pivot %d", pivot_idx)
                return paraphrase

        impact = orig_conf - conf
        if impact > best_impact:
            best_impact = impact
            best_text = paraphrase

    # Try additional MLM-based paraphrases
    for i in range(num_paraphrases):
        paraphrase = _paraphrase_fallback(text)
        if paraphrase == text:
            continue

        sim = compute_semantic_similarity(text, paraphrase)
        if sim < similarity_threshold:
            continue

        label, conf, _ = model_wrapper.predict(paraphrase)

        if target_label is not None:
            if label.lower() == target_label.lower():
                logger.info("Back-Translation: success with MLM paraphrase %d", i + 1)
                return paraphrase
        else:
            if label != orig_label:
                logger.info("Back-Translation: success with MLM paraphrase %d", i + 1)
                return paraphrase

        impact = orig_conf - conf
        if impact > best_impact:
            best_impact = impact
            best_text = paraphrase

    logger.info("Back-Translation: finished")
    return best_text
