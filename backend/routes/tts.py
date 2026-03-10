"""
Text-to-speech API.
"""

import asyncio
import os
import shutil
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

TTS_VOICES = {
    "en-US-JennyNeural": "Jenny (US Female)",
    "en-US-GuyNeural": "Guy (US Male)",
    "en-US-AriaNeural": "Aria (US Female)",
    "en-GB-RyanNeural": "Ryan (UK Male)",
    "en-GB-SoniaNeural": "Sonia (UK Female)",
    "en-AU-NatashaNeural": "Natasha (AU Female)",
}

ALLOWED_VOICES = set(TTS_VOICES.keys())


class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-JennyNeural"


@router.get("/voices")
def list_voices():
    return {"voices": TTS_VOICES}


@router.post("/generate")
async def generate_tts(req: TTSRequest):
    """Generate WAV from text using edge-tts."""
    if not req.text or not req.text.strip():
        raise HTTPException(400, "Text required")
    if req.voice not in ALLOWED_VOICES:
        raise HTTPException(400, f"Invalid voice. Choose from: {sorted(ALLOWED_VOICES)}")

    mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(mp3_fd)
    os.close(wav_fd)

    try:
        import edge_tts
        import soundfile as sf

        communicate = edge_tts.Communicate(req.text.strip(), req.voice)
        await communicate.save(mp3_path)

        try:
            data, sr = sf.read(mp3_path, dtype="float32")
        except Exception:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise HTTPException(500, "ffmpeg not found and soundfile cannot read mp3")
            subprocess.run(
                [ffmpeg, "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True, timeout=30,
            )
            data, sr = sf.read(wav_path, dtype="float32")

        if data.ndim > 1:
            data = data.mean(axis=1)

        import base64
        import io
        out_buf = io.BytesIO()
        sf.write(out_buf, data, int(sr), format="WAV")
        return {"wav_b64": base64.b64encode(out_buf.getvalue()).decode(), "sample_rate": int(sr)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"TTS failed: {str(e)}")
    finally:
        for p in (mp3_path, wav_path):
            try:
                os.unlink(p)
            except OSError:
                pass
