"""
Example local image plugin — random classifier (for testing).

CONTRACT:
  - PLUGIN_NAME (str): Display name in the UI
  - PLUGIN_TYPE (str): "image" | "audio" | "both"
  - classify(adversarial_image, *, original_image=None, config={}) -> list[dict]
    Each dict: {"label": str, "confidence": float 0-1}
    config: dict of variables set in Settings > Plugins > Variables
"""
import random

PLUGIN_NAME = "Random Classifier (Example)"
PLUGIN_TYPE = "image"
PLUGIN_DESCRIPTION = "Returns random predictions — use as a starter template."


def classify(adversarial_image, *, original_image=None, config={}):
    # Example: read an API key from config (set in UI under Variables)
    # api_key = config.get("API_KEY", "")

    labels = ["cat", "dog", "airplane", "ship", "automobile"]
    n = random.randint(2, 4)
    picked = random.sample(labels, n)
    confs = sorted([random.random() for _ in picked], reverse=True)
    total = sum(confs)
    return [{"label": lbl, "confidence": round(c / total, 4)} for lbl, c in zip(picked, confs)]
