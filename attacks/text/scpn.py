"""
SCPN — Syntactically Controlled Paraphrase Network

Iyyer et al., 2018 (arXiv:1804.06516)

Adversarial Example Generation with Syntactically Controlled Paraphrase
Networks. Generates paraphrases that conform to a target syntactic
structure (constituency parse template).

Original SCPN requires a specific pretrained model that is difficult to
obtain. This implementation uses Pegasus-paraphrase as a modern drop-in
replacement, generating diverse paraphrases via:
  - Beam search with diverse beam groups
  - Nucleus (top-p) sampling for syntactic variety
  - Temperature scaling for diversity control

The attack generates multiple paraphrase candidates and selects the one
that flips the classifier while maintaining semantic similarity.

Reference: TextAttack uses similar T5/Pegasus paraphrase models as
SCPN-equivalent transformation in its generic paraphrase pipeline.
"""

import logging

logger = logging.getLogger("textattack.attacks.scpn")

_PARAPHRASE_MODEL_NAME = "tuner007/pegasus_paraphrase"
_paraphrase_model = None
_paraphrase_tok = None


def _load_paraphrase_model():
    """Lazy-load Pegasus paraphrase model."""
    global _paraphrase_model, _paraphrase_tok
    if _paraphrase_model is not None:
        return _paraphrase_model, _paraphrase_tok

    import torch
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer

    logger.info("Loading paraphrase model: %s", _PARAPHRASE_MODEL_NAME)
    _paraphrase_tok = PegasusTokenizer.from_pretrained(_PARAPHRASE_MODEL_NAME)
    _paraphrase_model = PegasusForConditionalGeneration.from_pretrained(_PARAPHRASE_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _paraphrase_model.to(device)
    _paraphrase_model.eval()
    return _paraphrase_model, _paraphrase_tok


def _generate_paraphrases(
    text: str,
    num_paraphrases: int = 10,
    temperature: float = 1.5,
    top_p: float = 0.95,
    max_length: int = 256,
) -> list[str]:
    """Generate diverse paraphrases using Pegasus.

    Uses nucleus sampling with high temperature for syntactic diversity,
    approximating SCPN's controlled generation by sampling broadly from
    the paraphrase model's output distribution.
    """
    import torch

    model, tokenizer = _load_paraphrase_model()
    device = next(model.parameters()).device

    inputs = tokenizer(
        text, truncation=True, max_length=512,
        padding="longest", return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    paraphrases = set()

    # Strategy 1: Diverse beam search (structured diversity)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=max(num_paraphrases, 6),
                num_beam_groups=min(3, max(num_paraphrases, 6)),
                diversity_penalty=2.0,
                num_return_sequences=min(num_paraphrases, 6),
                temperature=temperature,
            )
        for out in outputs:
            decoded = tokenizer.decode(out, skip_special_tokens=True).strip()
            if decoded and decoded.lower() != text.lower():
                paraphrases.add(decoded)
    except Exception as e:
        logger.debug("Diverse beam search failed: %s", e)

    # Strategy 2: Nucleus sampling (free-form diversity)
    remaining = num_paraphrases - len(paraphrases)
    if remaining > 0:
        for _ in range(remaining * 2):
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=max_length,
                        do_sample=True,
                        top_p=top_p,
                        temperature=temperature,
                        num_return_sequences=1,
                    )
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                if decoded and decoded.lower() != text.lower():
                    paraphrases.add(decoded)
                if len(paraphrases) >= num_paraphrases:
                    break
            except Exception:
                continue

    return list(paraphrases)[:num_paraphrases]


def run_scpn(
    model_wrapper,
    tokenizer,
    text: str,
    target_label: str = None,
    num_paraphrases: int = 10,
    similarity_threshold: float = 0.7,
    temperature: float = 1.5,
    top_p: float = 0.95,
) -> str:
    """SCPN-style syntactic paraphrase attack.

    Generates diverse paraphrases via Pegasus and selects the one that
    flips the classifier while preserving semantic similarity.

    Args:
        model_wrapper: wrapped model with .predict() interface.
        tokenizer: HuggingFace tokenizer (unused, API compat).
        text: input text to attack.
        target_label: target class (None = untargeted).
        num_paraphrases: number of paraphrase candidates.
        similarity_threshold: minimum semantic similarity.
        temperature: sampling temperature (higher = more diverse).
        top_p: nucleus sampling probability mass.

    Returns: adversarial text (str).
    """
    from utils.text_constraints import compute_semantic_similarity

    logger.info(
        "SCPN: starting (num=%d, sim=%.2f, temp=%.2f, top_p=%.2f)",
        num_paraphrases, similarity_threshold, temperature, top_p,
    )

    orig_label, orig_conf, _ = model_wrapper.predict(text)

    try:
        paraphrases = _generate_paraphrases(
            text,
            num_paraphrases=num_paraphrases,
            temperature=temperature,
            top_p=top_p,
        )
    except (ImportError, OSError) as e:
        raise RuntimeError(
            f"SCPN failed: cannot load Pegasus paraphrase model. "
            f"Ensure 'transformers' is installed and the model can be downloaded. "
            f"Original error: {e}"
        ) from e

    if not paraphrases:
        logger.info("SCPN: no paraphrases generated")
        return text

    best_text = text
    best_impact = 0.0
    evaluated = 0

    for paraphrase in paraphrases:
        sim = compute_semantic_similarity(text, paraphrase)
        if sim < similarity_threshold:
            continue

        evaluated += 1
        label, conf, _ = model_wrapper.predict(paraphrase)

        if target_label is not None:
            if label.lower() == target_label.lower():
                logger.info("SCPN: success at candidate %d", evaluated)
                return paraphrase
        else:
            if label != orig_label:
                logger.info("SCPN: success at candidate %d", evaluated)
                return paraphrase

        impact = orig_conf - conf
        if impact > best_impact:
            best_impact = impact
            best_text = paraphrase

    logger.info("SCPN: finished (%d candidates evaluated)", evaluated)
    return best_text
