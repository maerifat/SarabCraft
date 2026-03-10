"""
Local HuggingFace model verification.

Tests adversarial images against locally-loaded HuggingFace
ImageClassification models to measure transfer success.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from config import device
from backend.models.registry import TASK_IMAGE_CLASSIFICATION, list_local_verification_models, resolve_source_model
from models.loader import load_model, get_model_input_size
from verification.base import Verifier, Prediction, ConfigField
from verification.registry import register


@register
class LocalModelVerifier(Verifier):
    """Classify using any local registry-backed image source model."""

    def __init__(self):
        self._model_names = None
        self._exact_preprocess = True

    @property
    def name(self):
        return "Local Models"

    @property
    def service_type(self):
        return "local"

    def is_available(self):
        return True

    def status_message(self):
        return "Ready (runs locally)"

    def set_model_names(self, names: list):
        """Configure which models to test against."""
        self._model_names = names

    def set_exact_preprocess(self, exact: bool):
        """Toggle exact preprocessing (bypass resize/crop) vs standard (full processor pipeline)."""
        self._exact_preprocess = exact

    @staticmethod
    def list_available_models() -> list:
        return [item["display_name"] for item in list_local_verification_models()]

    def classify(self, image: Image.Image) -> list:
        defaults = list_local_verification_models()[:1]
        names = self._model_names or [item["id"] for item in defaults]
        all_preds = []
        for requested in names:
            entry = resolve_source_model(requested, domain="image", task=TASK_IMAGE_CLASSIFICATION)
            hf_id = (entry or {}).get("model_ref")
            display_name = (entry or {}).get("display_name") or requested
            if hf_id is None:
                continue
            try:
                mdl, proc = load_model(hf_id)
                w, h = image.size

                if self._exact_preprocess:
                    expected_h, expected_w = get_model_input_size(proc)
                    use_exact = (w == expected_w and h == expected_h)
                else:
                    use_exact = False

                if use_exact:
                    img_array = np.array(image).astype(np.float32) / 255.0
                    mean = np.array(getattr(proc, 'image_mean', [0.485, 0.456, 0.406]))
                    std = np.array(getattr(proc, 'image_std', [0.229, 0.224, 0.225]))
                    img_normalized = (img_array - mean) / std
                    pixel_values = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)
                else:
                    inputs = proc(images=image, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(device)
                with torch.no_grad():
                    logits = mdl(pixel_values).logits
                    probs = F.softmax(logits, dim=1)
                    top_probs, top_idxs = torch.topk(probs, 5)
                for i in range(5):
                    idx = int(top_idxs[0][i].item())
                    conf = float(top_probs[0][i].item())
                    label_name = mdl.config.id2label.get(idx, f"Class {idx}")
                    all_preds.append(Prediction(
                        label=label_name,
                        confidence=conf,
                        raw={"model": display_name, "rank": i + 1},
                    ))
            except Exception as e:
                all_preds.append(Prediction(
                    label=f"[Error: {display_name}]",
                    confidence=0.0,
                    raw={"error": str(e)},
                ))
        all_preds.sort(key=lambda p: p.confidence, reverse=True)
        return all_preds
