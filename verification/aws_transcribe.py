"""
AWS Transcribe audio verification.

Sends adversarial audio to AWS Transcribe and returns the transcription.
Uses a temporary S3 upload because the Transcribe batch API requires an
S3 URI.  The bucket is configurable via AWS_TRANSCRIBE_BUCKET (defaults
to 'mlsec-transcribe-temp') and is auto-created if it doesn't exist.

Credentials are reused from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
(same as AWS Rekognition).
"""

import io
import json
import os
import time
import uuid
import struct

from verification.base import ConfigField
from verification.audio_base import AudioVerifier
from verification.audio_registry import register_audio


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


def _pcm16_to_wav(pcm_bytes: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    """Wrap raw PCM-16 bytes in a WAV header."""
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_bytes)

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate,
                          byte_rate, block_align, bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_bytes)
    return buf.getvalue()


@register_audio
class AWSTranscribeVerifier(AudioVerifier):

    _DEFAULT_BUCKET = "mlsec-transcribe-temp"

    @property
    def name(self):
        return "AWS Transcribe"

    @property
    def service_type(self):
        return "cloud"

    def get_config_schema(self):
        return [
            ConfigField("Access Key ID", "AWS_ACCESS_KEY_ID",
                        "AWS access key (shared with Rekognition)", required=True, secret=True),
            ConfigField("Secret Access Key", "AWS_SECRET_ACCESS_KEY",
                        "AWS secret key", required=True, secret=True),
            ConfigField("Region", "AWS_DEFAULT_REGION",
                        "AWS region (e.g. us-east-1)", required=False, secret=False),
            ConfigField("S3 Bucket", "AWS_TRANSCRIBE_BUCKET",
                        f"Temp bucket for audio uploads (default: {_DEFAULT_BUCKET})",
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

    def heartbeat(self) -> dict:
        if not self.is_available():
            return {
                "ok": False,
                "message": self.status_message(),
            }

        try:
            import boto3

            _, boto_kwargs = self._make_boto_kwargs()
            client = boto3.client("transcribe", **boto_kwargs)
            client.list_transcription_jobs(MaxResults=1)
            return {
                "ok": True,
                "message": "Ready",
            }
        except Exception as exc:
            return {
                "ok": False,
                "message": _aws_error_message(exc, "AWS Transcribe"),
            }

    def _get_bucket(self, s3, region):
        bucket = os.environ.get("AWS_TRANSCRIBE_BUCKET", self._DEFAULT_BUCKET)
        try:
            s3.head_bucket(Bucket=bucket)
        except Exception:
            print(f"[AWS Transcribe] Creating bucket '{bucket}' in {region}...", flush=True)
            create_args = {"Bucket": bucket}
            if region and region != "us-east-1":
                create_args["CreateBucketConfiguration"] = {"LocationConstraint": region}
            s3.create_bucket(**create_args)
        return bucket

    def _make_boto_kwargs(self):
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        kw = {
            "region_name": region,
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        }
        token = os.environ.get("AWS_SESSION_TOKEN")
        if token:
            kw["aws_session_token"] = token
        return region, kw

    def transcribe(self, audio_bytes: bytes, sample_rate: int, language: str = "en-US") -> str:
        import boto3
        import urllib.request

        region, bkw = self._make_boto_kwargs()
        s3 = boto3.client("s3", **bkw)
        transcribe_client = boto3.client("transcribe", **bkw)

        bucket = self._get_bucket(s3, region)
        job_id = uuid.uuid4().hex[:12]
        s3_key = f"mlsec-temp/{job_id}.wav"
        job_name = f"mlsec-{job_id}"

        if not audio_bytes[:4] == b"RIFF":
            audio_bytes = _pcm16_to_wav(audio_bytes, sample_rate)

        try:
            print(f"[AWS Transcribe] Uploading {len(audio_bytes)} bytes to s3://{bucket}/{s3_key}", flush=True)
            s3.put_object(Bucket=bucket, Key=s3_key, Body=audio_bytes, ContentType="audio/wav")

            print(f"[AWS Transcribe] Starting job '{job_name}'...", flush=True)
            transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={"MediaFileUri": f"s3://{bucket}/{s3_key}"},
                MediaFormat="wav",
                MediaSampleRateHertz=sample_rate,
                LanguageCode=language,
            )

            text = self._poll_job(transcribe_client, job_name)
            return text

        finally:
            self._cleanup(s3, transcribe_client, bucket, s3_key, job_name)

    def _poll_job(self, client, job_name, timeout=120):
        """Poll transcription job until complete or timeout."""
        start = time.time()
        delay = 2
        while time.time() - start < timeout:
            resp = client.get_transcription_job(TranscriptionJobName=job_name)
            status = resp["TranscriptionJob"]["TranscriptionJobStatus"]
            if status == "COMPLETED":
                uri = resp["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
                return self._fetch_transcript(uri)
            if status == "FAILED":
                reason = resp["TranscriptionJob"].get("FailureReason", "unknown")
                raise RuntimeError(f"Transcription job failed: {reason}")
            time.sleep(delay)
            delay = min(delay * 1.5, 10)

        raise RuntimeError(f"Transcription job '{job_name}' timed out after {timeout}s")

    @staticmethod
    def _fetch_transcript(uri):
        """Download the JSON transcript from the result URI."""
        import urllib.request
        with urllib.request.urlopen(uri, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        transcripts = data.get("results", {}).get("transcripts", [])
        if transcripts:
            return transcripts[0].get("transcript", "")
        return ""

    @staticmethod
    def _cleanup(s3, transcribe_client, bucket, key, job_name):
        """Best-effort cleanup of temp S3 object and transcription job."""
        try:
            s3.delete_object(Bucket=bucket, Key=key)
        except Exception as e:
            print(f"[AWS Transcribe] Cleanup S3 warning: {e}", flush=True)
        try:
            transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
        except Exception as e:
            print(f"[AWS Transcribe] Cleanup job warning: {e}", flush=True)
