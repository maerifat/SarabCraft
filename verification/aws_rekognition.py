"""
AWS Rekognition verification.

Sends adversarial images to AWS Rekognition DetectLabels
and returns the classification results as normalised Predictions.
Credentials are read from environment variables.
"""

import io
import os
from PIL import Image

from verification.base import Verifier, Prediction, ConfigField
from verification.registry import register


def _aws_error_message(exc: Exception, service_name: str) -> str:
    code = ""
    response = getattr(exc, "response", None) or {}
    if isinstance(response, dict):
        code = str(((response.get("Error") or {}).get("Code") or "")).strip()
    if code in {"UnrecognizedClientException", "InvalidClientTokenId", "InvalidSignatureException", "ExpiredTokenException", "AuthFailure"}:
        return "AWS credentials are invalid or expired"
    if code == "AccessDeniedException":
        return f"AWS credentials are missing {service_name} permissions"
    if code:
        return f"{service_name} readiness check failed ({code})"
    return f"{service_name} readiness check failed"


@register
class AWSRekognitionVerifier(Verifier):

    @property
    def name(self):
        return "AWS Rekognition"

    @property
    def service_type(self):
        return "cloud"

    def get_config_schema(self):
        return [
            ConfigField("Access Key ID", "AWS_ACCESS_KEY_ID",
                         "AWS access key", required=True, secret=True),
            ConfigField("Secret Access Key", "AWS_SECRET_ACCESS_KEY",
                         "AWS secret key", required=True, secret=True),
            ConfigField("Region", "AWS_DEFAULT_REGION",
                         "AWS region (e.g. us-east-1)", required=False, secret=False),
            ConfigField("Image format", "AWS_REKOGNITION_IMAGE_FORMAT",
                         "How adversarial images are sent to Rekognition: "
                         "'png' = lossless/pixel-perfect (best fidelity, may fail if >5MB), "
                         "'jpeg' = compressed, "
                         "'auto' = try PNG first and only fall back to JPEG if it exceeds AWS's 5MB limit.",
                         required=False, secret=False),
        ]

    def is_available(self):
        try:
            import boto3  # noqa: F401
        except ImportError:
            return False
        return bool(
            os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
        )

    def status_message(self):
        try:
            import boto3  # noqa: F401
        except ImportError:
            return "Install: pip install boto3"
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            return "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
        return "Ready"

    def _make_client(self):
        import boto3

        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
        token = os.environ.get("AWS_SESSION_TOKEN") or None
        return boto3.client(
            "rekognition",
            region_name=region,
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            **({"aws_session_token": token} if token else {}),
        )

    def heartbeat(self) -> dict:
        if not self.is_available():
            return {
                "ok": False,
                "message": self.status_message(),
            }

        try:
            client = self._make_client()
            client.list_collections(MaxResults=1)
            return {
                "ok": True,
                "message": "Ready",
            }
        except Exception as exc:
            return {
                "ok": False,
                "message": _aws_error_message(exc, "AWS Rekognition"),
            }

    _REKOGNITION_BYTE_LIMIT = 5 * 1024 * 1024 - 1024  # safety margin under 5 MiB

    @classmethod
    def _encode_png(cls, image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    @classmethod
    def _encode_jpeg_within_limit(cls, image: Image.Image) -> bytes:
        """JPEG-encode, lowering quality then downscaling until under the limit."""
        rgb = image.convert("RGB")
        for quality in (95, 90, 80, 70, 60):
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=quality)
            data = buf.getvalue()
            if len(data) <= cls._REKOGNITION_BYTE_LIMIT:
                return data

        work = rgb
        for _ in range(6):
            w, h = work.size
            work = work.resize((max(w // 2, 64), max(h // 2, 64)), Image.LANCZOS)
            buf = io.BytesIO()
            work.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
            if len(data) <= cls._REKOGNITION_BYTE_LIMIT:
                return data
        return data

    @classmethod
    def _encode_for_rekognition(cls, image: Image.Image) -> bytes:
        """Encode an image for Rekognition honouring the configured format.

        Rekognition rejects raw ``Image.Bytes`` payloads larger than 5 MB.
        Modes (env ``AWS_REKOGNITION_IMAGE_FORMAT``):
          - ``png``  : always lossless PNG (pixel-perfect). Raises if AWS later
                       rejects it for size — fidelity is preserved by request.
          - ``jpeg`` : always JPEG, shrinking to fit the 5 MB limit.
          - ``auto`` (default): send true PNG when it fits the limit; otherwise
                       fall back to JPEG/downscale so the call still succeeds.

        Re-encoding only affects the transport copy sent to the cloud service;
        local-model verification is unaffected and always uses the exact
        adversarial image.
        """
        mode = (os.environ.get("AWS_REKOGNITION_IMAGE_FORMAT") or "auto").strip().lower()

        if mode == "png":
            return cls._encode_png(image)
        if mode in {"jpg", "jpeg"}:
            return cls._encode_jpeg_within_limit(image)

        # auto: prefer lossless PNG, fall back only if it would exceed the limit.
        png = cls._encode_png(image)
        if len(png) <= cls._REKOGNITION_BYTE_LIMIT:
            return png
        return cls._encode_jpeg_within_limit(image)

    def classify(self, image: Image.Image) -> list:
        client = self._make_client()

        img_bytes = self._encode_for_rekognition(image)

        resp = client.detect_labels(
            Image={"Bytes": img_bytes},
            MaxLabels=10,
            MinConfidence=1.0,
        )

        predictions = []
        for lab in resp.get("Labels", []):
            predictions.append(Prediction(
                label=lab["Name"],
                confidence=lab["Confidence"] / 100.0,
                raw={
                    "parents": [p["Name"] for p in lab.get("Parents", [])],
                    "instances": len(lab.get("Instances", [])),
                },
            ))
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions
