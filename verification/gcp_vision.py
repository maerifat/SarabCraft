"""
Google Cloud Vision verification.

Sends adversarial images to GCP Cloud Vision label detection
and returns classification results.
Credentials via GOOGLE_APPLICATION_CREDENTIALS env var (JSON key file).
"""

import io
import os
from PIL import Image

from verification.base import Verifier, Prediction, ConfigField
from verification.registry import register


@register
class GCPVisionVerifier(Verifier):

    @property
    def name(self):
        return "Google Cloud Vision"

    @property
    def service_type(self):
        return "cloud"

    def get_config_schema(self):
        return [
            ConfigField("Credentials", "GOOGLE_APPLICATION_CREDENTIALS",
                         "Path to GCP service account JSON key file",
                         required=True, secret=False),
        ]

    def is_available(self):
        try:
            from google.cloud import vision  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

    def status_message(self):
        try:
            from google.cloud import vision  # noqa: F401
        except ImportError:
            return "Install: pip install google-cloud-vision"
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            return "Set GOOGLE_APPLICATION_CREDENTIALS or GCP_SERVICE_ACCOUNT_JSON"
        return "Ready"

    def heartbeat(self) -> dict:
        if not self.is_available():
            return {
                "ok": False,
                "message": self.status_message(),
            }

        try:
            from google.cloud import vision

            client = vision.ImageAnnotatorClient()
            probe = io.BytesIO()
            Image.new("RGB", (1, 1), (128, 128, 128)).save(probe, format="PNG")
            response = client.label_detection(
                image=vision.Image(content=probe.getvalue()),
                max_results=1,
            )
            if response.error.message:
                raise RuntimeError(response.error.message)
            return {
                "ok": True,
                "message": "Ready",
            }
        except Exception as exc:
            detail = str(exc).lower()
            if "permission" in detail or "403" in detail or "unauth" in detail or "credential" in detail:
                return {
                    "ok": False,
                    "message": "Google Cloud Vision credentials are invalid or unauthorized",
                }
            if "not found" in detail and "credentials" in detail:
                return {
                    "ok": False,
                    "message": "Google Cloud Vision credentials file was not found",
                }
            return {
                "ok": False,
                "message": "Google Cloud Vision readiness check failed",
            }

    def classify(self, image: Image.Image) -> list:
        from google.cloud import vision

        client = vision.ImageAnnotatorClient()

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        content = buf.getvalue()

        gcp_image = vision.Image(content=content)
        response = client.label_detection(image=gcp_image, max_results=10)

        if response.error.message:
            print(f"[GCP Vision] Error: {response.error.message}", flush=True)
            raise RuntimeError("GCP Vision request failed")

        predictions = []
        for label in response.label_annotations:
            predictions.append(Prediction(
                label=label.description,
                confidence=label.score,
                raw={"mid": label.mid, "topicality": label.topicality},
            ))
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions
