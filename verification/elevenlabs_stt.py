"""
ElevenLabs Speech-to-Text verification.

Sends adversarial audio to ElevenLabs' Scribe v2 model for transcription.
Simple synchronous POST — no S3, no polling.  Just send the WAV and get text back.

API: POST https://api.elevenlabs.io/v1/speech-to-text
Header: xi-api-key
Body: multipart/form-data  (file + model_id)
Response: { "text": "...", "language_code": "en", ... }
"""

import io
import json
import os
import struct
import urllib.request
import urllib.error

from verification.base import ConfigField
from verification.audio_base import AudioVerifier
from verification.audio_registry import register_audio

_API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
_MODELS_URL = "https://api.elevenlabs.io/v1/models"


def _ensure_wav(audio_bytes: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM-16 in a WAV header if not already WAV."""
    if audio_bytes[:4] == b"RIFF":
        return audio_bytes
    bits = 16
    ch = 1
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(audio_bytes)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, ch, sample_rate,
                          sample_rate * ch * bits // 8, ch * bits // 8, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(audio_bytes)))
    buf.write(audio_bytes)
    return buf.getvalue()


@register_audio
class ElevenLabsSTTVerifier(AudioVerifier):

    @property
    def name(self):
        return "ElevenLabs STT"

    @property
    def service_type(self):
        return "api"

    def get_config_schema(self):
        return [
            ConfigField(
                "ElevenLabs API Key", "ELEVENLABS_API_KEY",
                "API key from elevenlabs.io/app/settings/api-keys",
                required=True, secret=True,
            ),
        ]

    def is_available(self):
        return bool(os.environ.get("ELEVENLABS_API_KEY"))

    def status_message(self):
        if not os.environ.get("ELEVENLABS_API_KEY"):
            return "Set ELEVENLABS_API_KEY"
        return "Ready"

    def heartbeat(self) -> dict:
        api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not api_key:
            return {
                "ok": False,
                "message": "Set ELEVENLABS_API_KEY",
            }

        req = urllib.request.Request(
            _MODELS_URL,
            headers={"xi-api-key": api_key},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=30):
                return {
                    "ok": True,
                    "message": "Ready",
                }
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                return {
                    "ok": False,
                    "message": "ElevenLabs API key is invalid or expired",
                }
            if exc.code == 429:
                return {
                    "ok": False,
                    "message": "ElevenLabs rate limit hit - retry shortly",
                }
            return {
                "ok": False,
                "message": f"ElevenLabs readiness check failed (HTTP {exc.code})",
            }
        except urllib.error.URLError:
            return {
                "ok": False,
                "message": "ElevenLabs connection failed",
            }

    def transcribe(self, audio_bytes: bytes, sample_rate: int, language: str = "en-US") -> str:
        api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set")

        wav_data = _ensure_wav(audio_bytes, sample_rate)

        lang_code = language.split("-")[0] if language else "en"

        boundary = "----ElevenLabsBoundary9876543210"
        body = io.BytesIO()

        def write_field(name, value):
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            body.write(f"{value}\r\n".encode())

        def write_file(name, filename, content_type, data):
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode())
            body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
            body.write(data)
            body.write(b"\r\n")

        write_file("file", "audio.wav", "audio/wav", wav_data)
        write_field("model_id", "scribe_v2")
        write_field("language_code", lang_code)
        write_field("tag_audio_events", "false")
        body.write(f"--{boundary}--\r\n".encode())

        payload = body.getvalue()

        headers = {
            "xi-api-key": api_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }

        req = urllib.request.Request(_API_URL, data=payload, headers=headers, method="POST")

        try:
            print(f"[ElevenLabs STT] Sending {len(wav_data)} bytes, lang={lang_code}...", flush=True)
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            text = result.get("text", "").strip()
            print(f"[ElevenLabs STT] Transcription: \"{text[:100]}\"", flush=True)
            return text

        except urllib.error.HTTPError as e:
            code = e.code
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            print(f"[ElevenLabs STT] HTTP {code}: {err_body}", flush=True)
            if code == 401:
                if "unusual_activity" in err_body.lower() or "free tier" in err_body.lower():
                    raise RuntimeError(
                        "ElevenLabs Free Tier disabled for this account "
                        "(unusual activity detected — upgrade to a paid plan or use a different account)"
                    ) from e
                raise RuntimeError("ElevenLabs API key is invalid or expired") from e
            if code == 429:
                raise RuntimeError("ElevenLabs rate limit — wait and retry") from e
            raise RuntimeError(f"ElevenLabs API request failed (HTTP {code})") from e
        except urllib.error.URLError as e:
            raise RuntimeError("ElevenLabs connection failed") from e
