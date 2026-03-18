"""
Back-Translation Attack — Paraphrase via translation round-trip.

English → pivot language → English using HuggingFace MarianMT ROMANCE models.
Produces semantically equivalent paraphrases and checks if the classifier
label flips. Matches TextAttack BackTranslation transformation.
"""

import logging
import random

logger = logging.getLogger("textattack.attacks.back_translation")

# ROMANCE multilingual models matching TextAttack BackTranslation defaults
_DEFAULT_TARGET_MODEL = "Helsinki-NLP/opus-mt-en-ROMANCE"   # en → ROMANCE
_DEFAULT_SRC_MODEL = "Helsinki-NLP/opus-mt-ROMANCE-en"       # ROMANCE → en

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


def _translate(text: str, model_name: str, lang: str = "en") -> str:
    """Translate text using a MarianMT model.

    Matches TextAttack BackTranslation.translate():
    - For English target (lang="en"): no language tag prefix.
    - For other targets: prepend >>lang<< to input text for ROMANCE models.
    """
    import torch

    result = _load_translation_model(model_name)
    if result is None:
        return text

    model, tok, device = result

    # Language tag handling matching TextAttack BackTranslation.translate()
    if lang == "en":
        src_text = text
    else:
        # Matching TextAttack: if ">>" and "<<" not in lang
        # (prepend >>lang<< tag for ROMANCE models)
        if ">>" and "<<" not in lang:
            lang = ">>" + lang + "<< "
        src_text = lang + text

    inputs = tok(src_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        translated = model.generate(**inputs, max_length=512)

    return tok.decode(translated[0], skip_special_tokens=True)


def _get_supported_language_codes(model_name: str = _DEFAULT_TARGET_MODEL) -> list:
    """Get supported language codes from the target model's tokenizer.

    Used by chained back-translation to randomly sample pivot languages.
    """
    result = _load_translation_model(model_name)
    if result is None:
        return []
    _, tok, _ = result
    return list(getattr(tok, "supported_language_codes", []))


def _paraphrase_via_translation(
    text: str,
    target_lang: str = "es",
    chained_back_translation: int = 0,
    target_model: str = _DEFAULT_TARGET_MODEL,
    src_model: str = _DEFAULT_SRC_MODEL,
    src_lang: str = "en",
) -> str:
    """Generate a paraphrase via back-translation.

    Matches TextAttack BackTranslation._get_transformations():
    - chained_back_translation > 0: chain through N random pivot languages
      (en → lang1 → en → lang2 → en → ... → langN → en)
    - chained_back_translation == 0: single translation via target_lang
      (en → target_lang → en)
    """
    current = text

    if chained_back_translation > 0:
        # Chained back-translation: randomly sample N pivot languages
        # and chain translations through them sequentially.
        # Matches TextAttack: random.sample(tokenizer.supported_language_codes, N)
        supported = _get_supported_language_codes(target_model)
        if not supported:
            return text
        n = min(chained_back_translation, len(supported))
        pivot_langs = random.sample(supported, n)
        for pivot in pivot_langs:
            # Forward: src → pivot
            intermediate = _translate(current, target_model, lang=pivot)
            # Backward: pivot → src
            current = _translate(intermediate, src_model, lang=src_lang)
        return current

    # Single back-translation (default)
    intermediate = _translate(text, target_model, lang=target_lang)
    return _translate(intermediate, src_model, lang=src_lang)


def run_back_translation(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    num_paraphrases: int = 5,
    similarity_threshold: float = 0.6,
    chained_back_translation: int = 0,
    target_lang: str = "es",
) -> str:
    """Back-translation attack: generate paraphrases and check for label flip.

    Generates paraphrases via MarianMT ROMANCE model back-translation,
    filters by semantic similarity, and returns the first paraphrase
    that flips the model prediction (or the highest-impact candidate).

    Args:
        model_wrapper: wrapped target model with predict() interface.
        tokenizer: HuggingFace tokenizer (passed by router, not used directly).
        text: input text to attack.
        target_label: target class name for targeted attack (None = untargeted).
        num_paraphrases: number of paraphrase candidates to generate.
        similarity_threshold: minimum semantic similarity for candidates.
        chained_back_translation: number of chained pivot languages (0 = single).
        target_lang: pivot language code for single back-translation.

    Returns: adversarial text (str).
    """
    from utils.text_constraints import compute_semantic_similarity

    logger.info(
        "Back-Translation: starting (num_paraphrases=%d, sim=%.2f, chain=%d, target_lang=%s)",
        num_paraphrases, similarity_threshold, chained_back_translation, target_lang,
    )

    orig_label, orig_conf, _ = model_wrapper.predict(text)

    best_text = text
    best_impact = 0.0
    seen = set()

    for i in range(num_paraphrases):
        try:
            paraphrase = _paraphrase_via_translation(
                text,
                target_lang=target_lang,
                chained_back_translation=chained_back_translation,
            )
        except Exception as e:
            logger.warning("Back-Translation: paraphrase %d failed: %s", i + 1, e)
            continue

        if paraphrase == text or paraphrase in seen:
            continue
        seen.add(paraphrase)

        sim = compute_semantic_similarity(text, paraphrase)
        if sim < similarity_threshold:
            continue

        label, conf, _ = model_wrapper.predict(paraphrase)

        if target_label is not None:
            if label.lower() == target_label.lower():
                logger.info("Back-Translation: success at attempt %d", i + 1)
                return paraphrase
        else:
            if label != orig_label:
                logger.info("Back-Translation: success at attempt %d", i + 1)
                return paraphrase

        impact = orig_conf - conf
        if impact > best_impact:
            best_impact = impact
            best_text = paraphrase

    logger.info("Back-Translation: finished")
    return best_text
