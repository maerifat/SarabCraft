"""
Azure Computer Vision verification.

Sends adversarial images to Azure's Analyze Image endpoint
and returns tag/category predictions.
Credentials are read from environment variables.
"""

import io
import os
from PIL import Image

from verification.base import Verifier, Prediction, ConfigField
from verification.registry import register


@register
class AzureVisionVerifier(Verifier):

    @property
    def name(self):
        return "Azure Computer Vision"

    @property
    def service_type(self):
        return "cloud"

    def get_config_schema(self):
        return [
            ConfigField("Endpoint", "AZURE_VISION_ENDPOINT",
                         "Azure Computer Vision endpoint URL", required=True, secret=False),
            ConfigField("API Key", "AZURE_VISION_KEY",
                         "Azure Computer Vision subscription key", required=True, secret=True),
        ]

    def is_available(self):
        try:
            from azure.cognitiveservices.vision.computervision import (
                ComputerVisionClient,  # noqa: F401
            )
            from msrest.authentication import CognitiveServicesCredentials  # noqa: F401
        except ImportError:
            return False
        return bool(
            os.environ.get("AZURE_VISION_ENDPOINT")
            and os.environ.get("AZURE_VISION_KEY")
        )

    def status_message(self):
        try:
            from azure.cognitiveservices.vision.computervision import (
                ComputerVisionClient,  # noqa: F401
            )
        except ImportError:
            return "Install: pip install azure-cognitiveservices-vision-computervision msrest"
        if not os.environ.get("AZURE_VISION_ENDPOINT"):
            return "Set AZURE_VISION_ENDPOINT and AZURE_VISION_KEY"
        return "Ready"

    def heartbeat(self) -> dict:
        if not self.is_available():
            return {
                "ok": False,
                "message": self.status_message(),
            }

        try:
            from azure.cognitiveservices.vision.computervision import ComputerVisionClient
            from msrest.authentication import CognitiveServicesCredentials

            endpoint = os.environ["AZURE_VISION_ENDPOINT"]
            key = os.environ["AZURE_VISION_KEY"]
            client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

            probe = io.BytesIO()
            Image.new("RGB", (1, 1), (128, 128, 128)).save(probe, format="PNG")
            probe.seek(0)
            client.tag_image_in_stream(probe)
            return {
                "ok": True,
                "message": "Ready",
            }
        except Exception as exc:
            detail = str(exc).lower()
            if "401" in detail or "403" in detail or "access denied" in detail or "permission" in detail:
                return {
                    "ok": False,
                    "message": "Azure Vision credentials are invalid or unauthorized",
                }
            if "endpoint" in detail or "dns" in detail or "name or service not known" in detail:
                return {
                    "ok": False,
                    "message": "Azure Vision endpoint is invalid or unreachable",
                }
            return {
                "ok": False,
                "message": "Azure Vision readiness check failed",
            }

    def classify(self, image: Image.Image) -> list:
        from azure.cognitiveservices.vision.computervision import ComputerVisionClient
        from msrest.authentication import CognitiveServicesCredentials

        endpoint = os.environ["AZURE_VISION_ENDPOINT"]
        key = os.environ["AZURE_VISION_KEY"]
        client = ComputerVisionClient(
            endpoint, CognitiveServicesCredentials(key)
        )

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        result = client.tag_image_in_stream(buf)

        predictions = []
        for tag in result.tags:
            predictions.append(Prediction(
                label=tag.name,
                confidence=tag.confidence,
                raw={"hint": tag.hint} if tag.hint else {},
            ))
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions
