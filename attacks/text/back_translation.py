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

# Core ROMANCE pivot languages for single-pivot diversity.
# These are the high-resource languages in the opus-mt-en-ROMANCE model;
# each produces a distinct paraphrase due to different translation paths.
_ROMANCE_PIVOT_LANGS = ["es", "fr", "it", "pt", "ro", "ca", "gl"]

_translation_cache = {}


def _load_translation_model(model_name: str):
    """Lazy-load MarianMT translation model.

    Raises on failure so callers can surface meaningful diagnostics
    instead of silently returning the original text.
    """
    if model_name in _translation_cache:
        return _translation_cache[model_name]

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


def _translate(text: str, model_name: str, lang: str = "en") -> str:
    """Translate text using a MarianMT model.

    Matches TextAttack BackTranslation.translate():
    - For English target (lang="en"): no language tag prefix.
    - For other targets: prepend >>lang<< to input text for ROMANCE models.
    """
    import torch

    model, tok, device = _load_translation_model(model_name)

    # Language tag handling matching TextAttack BackTranslation.translate()
    if lang == "en":
        src_text = text
    else:
        if ">>" not in lang and "<<" not in lang:
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
    _, tok, _ = _load_translation_model(model_name)
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

    In single-pivot mode (chained_back_translation=0), each attempt uses
    a different ROMANCE pivot language to produce diverse candidates —
    matching the diversity that TextAttack's search methods achieve by
    calling _get_transformations repeatedly across the transformation
    space.

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

    # Build the pivot language schedule for single-pivot mode.
    # The user-specified target_lang is tried first, then we rotate through
    # other ROMANCE languages so each attempt produces a distinct paraphrase
    # (MarianMT greedy decoding is deterministic for a given pivot).
    pivot_schedule: list[str] = []
    if chained_back_translation == 0:
        pivot_schedule = [target_lang]
        for lang in _ROMANCE_PIVOT_LANGS:
            if lang != target_lang:
                pivot_schedule.append(lang)

    best_text = text
    best_impact = 0.0
    seen: set[str] = set()
    consecutive_load_failures = 0

    for i in range(num_paraphrases):
        try:
            if chained_back_translation > 0:
                paraphrase = _paraphrase_via_translation(
                    text,
                    target_lang=target_lang,
                    chained_back_translation=chained_back_translation,
                )
            else:
                pivot = pivot_schedule[i % len(pivot_schedule)]
                paraphrase = _paraphrase_via_translation(
                    text, target_lang=pivot,
                )
            consecutive_load_failures = 0
        except (ImportError, OSError) as e:
            consecutive_load_failures += 1
            logger.error("Back-Translation: translation model failed to load: %s", e)
            if consecutive_load_failures >= 2:
                raise RuntimeError(
                    f"Back-Translation failed: cannot load MarianMT translation models. "
                    f"Ensure 'sentencepiece' is installed (pip install sentencepiece) "
                    f"and the model can be downloaded. Original error: {e}"
                ) from e
            continue
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
                logger.info("Back-Translation: success at attempt %d (%s)",
                            i + 1, pivot_schedule[i % len(pivot_schedule)] if pivot_schedule else "chained")
                return paraphrase
        else:
            if label != orig_label:
                logger.info("Back-Translation: success at attempt %d (%s)",
                            i + 1, pivot_schedule[i % len(pivot_schedule)] if pivot_schedule else "chained")
                return paraphrase

        impact = orig_conf - conf
        if impact > best_impact:
            best_impact = impact
            best_text = paraphrase

    logger.info("Back-Translation: finished (%d unique candidates evaluated)", len(seen))
    return best_text
