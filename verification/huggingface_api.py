"""
HuggingFace Inference API verification.

Tests adversarial images against the HuggingFace serverless Inference API
(router.huggingface.co). Auto-detects the model's pipeline type and routes to
the correct API method:
  - image-classification (ViT, ResNet, DeiT, ...)
  - zero-shot-image-classification (CLIP, SigLIP, ...)
  - object-detection (DETR, YOLO, ...)
  - image-segmentation (SegFormer, Mask2Former, ...)
  - image-to-text (BLIP, GIT, ...)
  - visual-question-answering (ViLT, BLIP-VQA, ...)

Uses raw HTTP requests with explicit Content-Type headers because the
huggingface_hub InferenceClient (<=0.30) omits Content-Type for binary
payloads, causing 400 errors on the router.
"""

import io
import json
import os
import urllib.request
import urllib.error
from functools import lru_cache

from PIL import Image

from backend.models.hf_presets import (
    HF_DEFAULT_MODEL,
    HF_IMAGE_CLASSIFICATION_MODELS,
    HF_PRESET_MODELS,
)
from verification.base import Verifier, Prediction, ConfigField
from verification.registry import register

_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"

SUPPORTED_TASKS = {
    "image-classification",
    "zero-shot-image-classification",
    "object-detection",
    "image-segmentation",
    "image-to-text",
    "visual-question-answering",
}

_COMMON_LABELS = [
    "dog", "cat", "car", "bird", "fish", "person", "building",
    "airplane", "horse", "boat", "flower", "tree", "food", "insect",
]



@lru_cache(maxsize=64)
def _resolve_pipeline_tag(model_id: str, token: str = None) -> str:
    """Query HuggingFace Hub for a model's pipeline_tag. Cached per model."""
    try:
        from huggingface_hub import HfApi
        info = HfApi(token=token).model_info(model_id)
        tag = getattr(info, "pipeline_tag", None) or ""
        print(f"[HuggingFace API] Model '{model_id}' pipeline_tag: {tag}", flush=True)
        return tag
    except Exception as e:
        print(f"[HuggingFace API] Could not resolve pipeline_tag for '{model_id}': {e}", flush=True)
        return ""


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _hf_post(model_id: str, data: bytes, content_type: str, token: str, timeout: float = 60.0) -> dict | list:
    """POST to the HF router and return parsed JSON. Raises RuntimeError on failure."""
    url = f"{_ROUTER_BASE}/{model_id}"
    headers = {"Content-Type": content_type}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        code = e.code
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:300].strip()
        except Exception:
            pass
        print(f"[HuggingFace API] HTTP {code} for '{model_id}': {body}", flush=True)
        if code == 404:
            raise RuntimeError(
                f"Model '{model_id}' is not available on the HuggingFace serverless Inference API. "
                "It may need a dedicated Inference Endpoint or local inference."
            ) from e
        if code == 503 or "loading" in body.lower():
            raise RuntimeError(f"Model '{model_id}' is loading on HF servers — retry in ~30s") from e
        if code == 429:
            raise RuntimeError(f"HF API rate limit hit for '{model_id}' — wait a moment and retry") from e
        if code in {401, 403}:
            raise RuntimeError("HuggingFace API authentication failed. Check the configured token.") from e
        raise RuntimeError(f"HF API request failed for '{model_id}' (HTTP {code})") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"HF API connection failed for '{model_id}'") from e


def _post_binary(model_id: str, img_bytes: bytes, token: str) -> dict | list:
    return _hf_post(model_id, img_bytes, "image/png", token)


def _post_json(model_id: str, payload: dict, token: str) -> dict | list:
    return _hf_post(model_id, json.dumps(payload).encode("utf-8"), "application/json", token)


@register
class HuggingFaceAPIVerifier(Verifier):

    def __init__(self):
        self._model_id = HF_DEFAULT_MODEL
        self._target_label = None
        self._original_label = None

    @property
    def name(self):
        return "HuggingFace API"

    @property
    def service_type(self):
        return "api"

    def set_model(self, model_id: str) -> None:
        self._model_id = model_id or HF_DEFAULT_MODEL

    def set_labels(self, target_label: str = None, original_label: str = None) -> None:
        self._target_label = target_label
        self._original_label = original_label

    def get_config_schema(self):
        return [
            ConfigField(
                name="API Token",
                env_var="HF_API_TOKEN",
                description="HuggingFace API token (free tier works)",
                required=False,
                secret=True,
            ),
        ]

    def is_available(self):
        try:
            from huggingface_hub import HfApi  # noqa: F401
            return True
        except ImportError:
            return False

    def status_message(self):
        if not self.is_available():
            return "Install: pip install huggingface_hub"
        token = os.environ.get("HF_API_TOKEN")
        return "Ready" if token else "Set HF_API_TOKEN"

    def heartbeat(self) -> dict:
        if not self.is_available():
            return {
                "ok": False,
                "message": self.status_message(),
            }

        token = (os.environ.get("HF_API_TOKEN") or "").strip()
        if not token:
            return {
                "ok": False,
                "message": "Set HF_API_TOKEN",
            }

        try:
            from huggingface_hub import HfApi

            HfApi(token=token).whoami()
            return {
                "ok": True,
                "message": "Ready",
            }
        except Exception as exc:
            detail = str(exc).lower()
            if "401" in detail or "403" in detail or "unauthorized" in detail or "invalid user token" in detail:
                return {
                    "ok": False,
                    "message": "HuggingFace API token is invalid or expired",
                }
            return {
                "ok": False,
                "message": "HuggingFace API connection failed",
            }

    def _build_candidate_labels(self) -> list[str]:
        labels = set()
        if self._target_label:
            labels.add(self._target_label.strip())
        if self._original_label:
            labels.add(self._original_label.strip())
        for lbl in _COMMON_LABELS:
            labels.add(lbl)
            if len(labels) >= 16:
                break
        return sorted(labels)

    def classify(self, image: Image.Image) -> list:
        if image is None:
            return []

        token = (os.environ.get("HF_API_TOKEN") or "").strip() or None
        model_id = self._model_id or HF_DEFAULT_MODEL
        task = _resolve_pipeline_tag(model_id, token=token)

        if task and task not in SUPPORTED_TASKS:
            raise RuntimeError(
                f"Model '{model_id}' has pipeline_tag '{task}' which is not a vision task. "
                f"Supported: {', '.join(sorted(SUPPORTED_TASKS))}"
            )

        img_bytes = _image_to_png_bytes(image)

        try:
            if task == "zero-shot-image-classification":
                return self._run_zero_shot(model_id, img_bytes, token)
            elif task == "object-detection":
                return self._run_object_detection(model_id, img_bytes, token)
            elif task == "image-segmentation":
                return self._run_segmentation(model_id, img_bytes, token)
            elif task == "image-to-text":
                return self._run_image_to_text(model_id, img_bytes, token)
            elif task == "visual-question-answering":
                return self._run_vqa(model_id, img_bytes, token)
            else:
                return self._run_image_classification(model_id, img_bytes, token)
        except RuntimeError:
            raise
        except Exception as e:
            raise self._wrap_error(model_id, e) from e

    # -- Task-specific runners using raw HTTP --

    def _run_image_classification(self, model_id, img_bytes, token) -> list:
        results = _post_binary(model_id, img_bytes, token)

        if not results:
            raise RuntimeError(f"Model '{model_id}' returned no predictions")

        return [
            Prediction(
                label=str(r.get("label", "")),
                confidence=float(r.get("score", 0.0)),
                raw={"model": model_id, "task": "image-classification"},
            )
            for r in results
        ]

    def _run_zero_shot(self, model_id, img_bytes, token) -> list:
        import base64
        labels = self._build_candidate_labels()
        if len(labels) < 2:
            raise RuntimeError(
                f"Zero-shot model '{model_id}' needs at least 2 candidate labels. "
                "Provide target and/or original labels in the Transfer Test."
            )
        print(f"[HuggingFace API] zero-shot with {len(labels)} candidates: {labels[:5]}...", flush=True)

        img_b64 = base64.b64encode(img_bytes).decode("ascii")
        payload = {
            "inputs": {"image": img_b64},
            "parameters": {"candidate_labels": labels},
        }
        results = _post_json(model_id, payload, token)

        if isinstance(results, list):
            return [
                Prediction(
                    label=str(r.get("label", "")),
                    confidence=float(r.get("score", 0.0)),
                    raw={"model": model_id, "task": "zero-shot-image-classification"},
                )
                for r in results
            ]
        return [Prediction(label=str(results), confidence=1.0, raw={"model": model_id})]

    def _run_object_detection(self, model_id, img_bytes, token) -> list:
        results = _post_binary(model_id, img_bytes, token)

        if not results:
            return [Prediction(label="(no objects detected)", confidence=0.0, raw={"model": model_id})]

        label_scores: dict[str, float] = {}
        label_counts: dict[str, int] = {}
        for r in results:
            lbl = str(r.get("label", ""))
            scr = float(r.get("score", 0.0))
            if lbl:
                label_scores[lbl] = max(label_scores.get(lbl, 0.0), scr)
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

        return sorted(
            [
                Prediction(
                    label=f"{lbl} (x{label_counts[lbl]})" if label_counts[lbl] > 1 else lbl,
                    confidence=scr,
                    raw={"model": model_id, "task": "object-detection", "count": label_counts[lbl]},
                )
                for lbl, scr in label_scores.items()
            ],
            key=lambda p: p.confidence, reverse=True,
        )

    def _run_segmentation(self, model_id, img_bytes, token) -> list:
        results = _post_binary(model_id, img_bytes, token)

        if not results:
            return [Prediction(label="(no segments)", confidence=0.0, raw={"model": model_id})]

        return sorted(
            [
                Prediction(
                    label=str(r.get("label", "")),
                    confidence=float(r.get("score", 0.0)),
                    raw={"model": model_id, "task": "image-segmentation"},
                )
                for r in results
                if r.get("label")
            ],
            key=lambda p: p.confidence, reverse=True,
        )

    def _run_image_to_text(self, model_id, img_bytes, token) -> list:
        data = _post_binary(model_id, img_bytes, token)

        if isinstance(data, list) and data:
            text = data[0].get("generated_text", "") or str(data[0])
        elif isinstance(data, dict):
            text = data.get("generated_text", "") or str(data)
        else:
            text = str(data) or "(empty caption)"

        return [
            Prediction(
                label=text.strip(),
                confidence=1.0,
                raw={"model": model_id, "task": "image-to-text"},
            )
        ]

    def _run_vqa(self, model_id, img_bytes, token) -> list:
        import base64
        img_b64 = base64.b64encode(img_bytes).decode("ascii")
        payload = {
            "inputs": {
                "image": img_b64,
                "question": "What is in this image?",
            },
        }
        results = _post_json(model_id, payload, token)

        if not results:
            return [Prediction(label="(no answer)", confidence=0.0, raw={"model": model_id})]

        if isinstance(results, list):
            return [
                Prediction(
                    label=str(r.get("answer", "") or r.get("label", "")),
                    confidence=float(r.get("score", 0.0)),
                    raw={"model": model_id, "task": "visual-question-answering"},
                )
                for r in results
            ]
        return [Prediction(label=str(results), confidence=1.0, raw={"model": model_id})]

    @staticmethod
    def _wrap_error(model_id: str, exc: Exception) -> RuntimeError:
        err_msg = str(exc).strip() or repr(exc)
        print(f"[HuggingFace API] Error with {model_id}: {err_msg}", flush=True)
        if "loading" in err_msg.lower() or "503" in err_msg:
            return RuntimeError(f"Model '{model_id}' is loading on HF servers — retry in ~30s")
        if "404" in err_msg or "not found" in err_msg.lower():
            return RuntimeError(
                f"Model '{model_id}' is not available on the HuggingFace serverless Inference API. "
                "It may need a dedicated Inference Endpoint or local inference."
            )
        return RuntimeError(f"HF API error for '{model_id}': {err_msg}")
