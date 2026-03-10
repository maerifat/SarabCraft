"""
Plugin management API routes (local Python plugins only).
"""

import base64
import io
import os
import sys
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from plugins._base import (
    get_all_plugins, get_enabled_plugins,
    save_plugin_code, update_plugin_code, read_plugin_code,
    delete_local_plugin, upload_plugin_file, upload_plugin_zip,
    _validate_plugin_code, run_local_plugin, run_playground,
    set_plugin_enabled, update_plugin_type,
)

router = APIRouter()


class SaveCodeRequest(BaseModel):
    filename: str
    code: str


class UpdateCodeRequest(BaseModel):
    code: str


class ToggleRequest(BaseModel):
    enabled: bool


class TypeChangeRequest(BaseModel):
    type: str


@router.get("/list")
def list_plugins():
    """List all local plugins."""
    return {"plugins": get_all_plugins()}


@router.get("/enabled")
def list_enabled_plugins(type: str = "image"):
    """List enabled plugins for a given type."""
    return {"plugins": get_enabled_plugins(type)}


# ── Plugin File Management ────────────────────────────────────────────────────

@router.post("/upload")
async def upload_plugin(file: UploadFile = File(...)):
    """Upload a .py file or .zip archive as a plugin."""
    content = await file.read()
    filename = file.filename or "plugin.py"

    if filename.endswith(".zip"):
        result = upload_plugin_zip(content)
    elif filename.endswith(".py"):
        result = upload_plugin_file(filename, content)
    else:
        raise HTTPException(400, "Only .py and .zip files are accepted")

    if "error" in result and result.get("error"):
        raise HTTPException(400, result["error"])
    return result


@router.post("/code/save")
def save_code(req: SaveCodeRequest):
    """Create a new local plugin from inline code."""
    if not req.code.strip():
        raise HTTPException(400, "Code is required")
    result = save_plugin_code(req.filename, req.code)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/code/validate")
def validate_code(req: UpdateCodeRequest):
    """Validate plugin code without saving."""
    return _validate_plugin_code(req.code)


@router.get("/{plugin_id}/code")
def get_code(plugin_id: str):
    """Read the source code of a local plugin."""
    if not plugin_id.startswith("local_"):
        raise HTTPException(400, "Only local plugins have viewable code")
    result = read_plugin_code(plugin_id)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@router.put("/{plugin_id}/code")
def edit_code(plugin_id: str, req: UpdateCodeRequest):
    """Update the code of an existing local plugin."""
    if not plugin_id.startswith("local_"):
        raise HTTPException(400, "Only local plugins can be edited")
    result = update_plugin_code(plugin_id, req.code)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.delete("/{plugin_id}")
def remove_plugin(plugin_id: str):
    """Delete a local plugin."""
    result = delete_local_plugin(plugin_id)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return {"ok": True}


@router.post("/{plugin_id}/toggle")
def toggle_plugin(plugin_id: str, req: ToggleRequest):
    """Enable or disable a plugin."""
    return set_plugin_enabled(plugin_id, req.enabled)


@router.post("/{plugin_id}/type")
def change_plugin_type(plugin_id: str, req: TypeChangeRequest):
    """Change the PLUGIN_TYPE of a local plugin."""
    if not plugin_id.startswith("local_"):
        raise HTTPException(400, "Only local plugins can be modified")
    result = update_plugin_type(plugin_id, req.type)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/{plugin_id}/test")
def test_plugin(plugin_id: str):
    """Test a plugin with dummy input appropriate for its type."""
    import numpy as np
    all_plugins = get_all_plugins()
    plugin = next((p for p in all_plugins if p["id"] == plugin_id), None)
    ptype = plugin["type"] if plugin else "image"

    kwargs = {}
    if ptype in ("image", "both"):
        kwargs["adversarial_image"] = Image.new("RGB", (224, 224), color="gray")
    if ptype in ("audio", "both"):
        kwargs["adversarial_audio"] = np.zeros(16000, dtype=np.float32)
        kwargs["sample_rate"] = 16000

    if not kwargs:
        kwargs["adversarial_image"] = Image.new("RGB", (224, 224), color="gray")

    result = run_local_plugin(plugin_id, **kwargs)
    return {
        "ok": "error" not in result or not result["error"],
        "elapsed_ms": result.get("elapsed_ms", 0),
        "predictions_count": len(result.get("predictions", [])),
        "error": result.get("error"),
    }


# ── Playground ────────────────────────────────────────────────────────────────

class PlaygroundRequest(BaseModel):
    code: str
    image_b64: Optional[str] = None
    image_url: Optional[str] = None
    audio_b64: Optional[str] = None
    sample_rate: int = 16000


@router.post("/playground/run")
def playground_run(req: PlaygroundRequest):
    """Execute plugin code in-memory against provided input."""
    import numpy as np

    adv_img = None
    adv_audio = None

    if req.image_b64:
        try:
            raw = base64.b64decode(req.image_b64)
            adv_img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"Invalid image data: {e}")

    elif req.image_url:
        import urllib.request
        import urllib.parse
        parsed = urllib.parse.urlparse(req.image_url)
        if parsed.scheme not in ("http", "https"):
            raise HTTPException(400, "Only http/https URLs are allowed")
        hostname = parsed.hostname or ""
        if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or hostname.startswith("169.254.") or hostname.startswith("10.") or hostname.startswith("192.168.") or hostname.startswith("172."):
            raise HTTPException(400, "Internal/private URLs are not allowed")
        try:
            url_req = urllib.request.Request(req.image_url, headers={"User-Agent": "SarabCraft/1.0"})
            with urllib.request.urlopen(url_req, timeout=15) as resp:
                adv_img = Image.open(io.BytesIO(resp.read())).convert("RGB")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Failed to fetch image URL: {e}")

    if req.audio_b64:
        try:
            raw = base64.b64decode(req.audio_b64)
            import soundfile as sf
            adv_audio, _ = sf.read(io.BytesIO(raw))
            adv_audio = adv_audio.astype(np.float32)
        except Exception as e:
            raise HTTPException(400, f"Invalid audio data: {e}")

    if adv_img is None and adv_audio is None:
        ptype = "image"
        if "PLUGIN_TYPE" in req.code:
            for t in ("audio", "both", "image"):
                if f'"{t}"' in req.code or f"'{t}'" in req.code:
                    ptype = t
                    break
        if ptype in ("image", "both"):
            adv_img = Image.new("RGB", (224, 224), color="gray")
        if ptype in ("audio", "both"):
            adv_audio = np.zeros(16000, dtype=np.float32)

    result = run_playground(
        req.code,
        adversarial_image=adv_img,
        adversarial_audio=adv_audio,
        sample_rate=req.sample_rate,
    )
    return result
