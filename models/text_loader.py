"""
Text model loader with in-memory cache.

Mirrors models/loader.py — loads HuggingFace sequence classification models,
caches them, and provides a get_predictions() helper.
"""

import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("textattack.models")

_model_cache: dict[str, tuple] = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_text_model(model_name: str, progress=None):
    """Load a HuggingFace text classification model + tokenizer.

    Returns (model, tokenizer). Cached after first load.
    """
    if model_name in _model_cache:
        logger.debug("Cache hit: %s", model_name)
        return _model_cache[model_name]

    logger.info("Loading text model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    _model_cache[model_name] = (model, tokenizer)
    return model, tokenizer


def get_predictions(model, tokenizer, text: str, top_k: int = 5) -> list[dict]:
    """Classify text, return top-k predictions sorted by confidence.

    Returns: [{"label": str, "confidence": float, "index": int}, ...]
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    top_probs, top_indices = probs.topk(min(top_k, len(probs)))
    id2label = getattr(model.config, "id2label", {})

    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        label = id2label.get(idx, f"LABEL_{idx}")
        results.append({"label": label, "confidence": prob, "index": idx})
    return results


def get_label_and_confidence(model, tokenizer, text: str) -> tuple[str, float, int]:
    """Quick helper: returns (top_label, top_confidence, top_index)."""
    preds = get_predictions(model, tokenizer, text, top_k=1)
    if not preds:
        return "UNKNOWN", 0.0, -1
    return preds[0]["label"], preds[0]["confidence"], preds[0]["index"]


def get_label_index(model, label_name: str) -> int | None:
    """Resolve a label name like 'POSITIVE' to its index."""
    label2id = getattr(model.config, "label2id", {})
    # Try exact match first
    if label_name in label2id:
        return label2id[label_name]
    # Try case-insensitive
    for k, v in label2id.items():
        if k.lower() == label_name.lower():
            return v
    return None


# Common aliases users might type → canonical sentiment names
_LABEL_ALIASES = {
    "positive": ["positive", "pos", "1", "label_1"],
    "negative": ["negative", "neg", "0", "label_0"],
    "neutral":  ["neutral", "neu", "2", "label_2"],
}


def resolve_target_label(model, target_label: str) -> str | None:
    """Resolve a user-friendly target label to the model's actual label name.

    Handles cases like:
      - User types 'POSITIVE' but model labels are 'LABEL_0' / 'LABEL_1'
      - User types 'LABEL_1' but model labels are 'POSITIVE' / 'NEGATIVE'
      - User types the exact model label name

    Returns the model's label string, or None if unresolvable.
    """
    if not target_label:
        return None

    id2label = getattr(model.config, "id2label", {})
    label2id = getattr(model.config, "label2id", {})
    model_labels = list(id2label.values()) if id2label else list(label2id.keys())

    target_lower = target_label.strip().lower()

    # 1. Exact match (case-insensitive) against model labels
    for ml in model_labels:
        if ml.lower() == target_lower:
            return ml

    # 2. Try alias resolution: user typed 'POSITIVE' → find which model label
    #    corresponds to the 'positive' concept
    for concept, aliases in _LABEL_ALIASES.items():
        if target_lower in aliases:
            # The user wants this concept — find the model label for it
            for ml in model_labels:
                ml_lower = ml.lower()
                if ml_lower in aliases or ml_lower == concept:
                    return ml
            # If model uses generic LABEL_N, map by convention
            # positive→LABEL_1, negative→LABEL_0 (standard HF sentiment convention)
            if concept == "positive" and "LABEL_1" in model_labels:
                return "LABEL_1"
            if concept == "negative" and "LABEL_0" in model_labels:
                return "LABEL_0"
            if concept == "positive" and 1 in id2label:
                return id2label[1]
            if concept == "negative" and 0 in id2label:
                return id2label[0]
            break

    # 3. Try numeric index: user typed '1' → id2label[1]
    try:
        idx = int(target_lower)
        if idx in id2label:
            return id2label[idx]
    except ValueError:
        pass

    # 4. Fallback: return as-is (let the attack try its best)
    logger.warning("Could not resolve target label '%s' to model labels %s", target_label, model_labels)
    return target_label
