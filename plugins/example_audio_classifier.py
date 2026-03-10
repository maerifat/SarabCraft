"""
Example local audio plugin — random transcriber (for testing).

CONTRACT:
  - PLUGIN_NAME (str): Display name in the UI
  - PLUGIN_TYPE (str): "image" | "audio" | "both"
  - classify(adversarial_audio, *, original_audio=None, sample_rate=16000, config={}) -> list[dict]
    Each dict: {"label": str, "confidence": float 0-1}
    config: dict of variables set in Settings > Plugins > Variables
"""
import random

PLUGIN_NAME = "Random Transcriber (Example)"
PLUGIN_TYPE = "audio"
PLUGIN_DESCRIPTION = "Returns random transcriptions — use as a starter template."


def classify(adversarial_audio, *, original_audio=None, sample_rate=16000, config={}):
    phrases = [
        "hello world", "open the door", "play music",
        "set a timer", "turn off lights",
    ]
    return [
        {"label": random.choice(phrases), "confidence": round(random.uniform(0.5, 0.99), 4)},
    ]
