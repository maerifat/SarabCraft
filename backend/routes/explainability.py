"""
GradCAM attention overlays — highlights which image regions drive the
model's prediction, comparing original vs adversarial.
"""

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import device
from models.loader import load_model
from utils.image import preprocess_image

router = APIRouter()


class GradCAMRequest(BaseModel):
    image_b64: str
    model: str = "microsoft/resnet-50"
    target_class: int | None = None


class GradCAMCompareRequest(BaseModel):
    model: str = "microsoft/resnet-50"
    original_b64: str | None = None
    adversarial_b64: str | None = None


def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _find_last_conv(model) -> torch.nn.Module | None:
    last = None
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d,)):
            last = m
    return last


def _generate_gradcam(model, tensor, target_idx=None):
    model.eval()
    activations = []
    gradients = []

    target_layer = _find_last_conv(model)
    if target_layer is None:
        return None

    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    try:
        outputs = model(tensor)
        logits = outputs.logits
        if target_idx is None:
            target_idx = logits.argmax(dim=1).item()

        model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=False)

        if not activations or not gradients:
            return None

        act = activations[0]
        grad = gradients[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze(0).squeeze(0)

        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.cpu().numpy()
        return cam
    finally:
        fh.remove()
        bh.remove()


def _overlay_heatmap_fast(image: Image.Image, cam: np.ndarray, alpha=0.5) -> Image.Image:
    w, h = image.size
    cam_u8 = (cam * 255).astype(np.uint8)
    cam_pil = Image.fromarray(cam_u8).resize((w, h), Image.BILINEAR)
    cam_arr = np.array(cam_pil).astype(np.float32) / 255.0

    lut = np.zeros((256, 3), dtype=np.float32)
    for i in range(256):
        v = i / 255.0
        hue = max(0, (1.0 - v) * 0.66)
        s, bri = 1.0, 1.0
        c = bri * s
        x = c * (1 - abs((hue * 6) % 2 - 1))
        m = bri - c
        if hue < 1/6:   r, g, b = c, x, 0
        elif hue < 2/6: r, g, b = x, c, 0
        elif hue < 3/6: r, g, b = 0, c, x
        elif hue < 4/6: r, g, b = 0, x, c
        elif hue < 5/6: r, g, b = x, 0, c
        else:            r, g, b = c, 0, x
        lut[i] = [r + m, g + m, b + m]

    idx = (cam_arr * 255).astype(np.uint8)
    heatmap = lut[idx]

    img_arr = np.array(image).astype(np.float32) / 255.0
    blended = img_arr * (1 - alpha) + heatmap * alpha
    blended = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(blended)


@router.post("/gradcam")
def run_gradcam(req: GradCAMRequest):
    try:
        pil_img = _b64_to_pil(req.image_b64)
        model_name = req.model if "/" in req.model else "microsoft/resnet-50"
        mdl, _ = load_model(model_name, progress=None)

        tensor = preprocess_image(pil_img, model_name)
        tensor.requires_grad_(True)

        cam = _generate_gradcam(mdl, tensor, req.target_class)
        if cam is None:
            raise HTTPException(400, "Could not generate GradCAM for this model architecture")

        overlay = _overlay_heatmap_fast(pil_img, cam)
        return {
            "overlay_b64": _pil_to_b64(overlay),
            "heatmap_b64": _pil_to_b64(Image.fromarray(
                ((cam / cam.max()) * 255).astype(np.uint8) if cam.max() > 0
                else np.zeros_like(cam, dtype=np.uint8)
            )),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"GradCAM failed: {str(e)}")


@router.post("/gradcam/compare")
def compare_gradcam(req: GradCAMCompareRequest):
    """Generate GradCAM for original and adversarial side by side."""
    try:
        model_name = req.model
        if "/" not in model_name:
            model_name = "microsoft/resnet-50"
        mdl, _ = load_model(model_name, progress=None)

        results = {}
        for key, b64 in [("original", req.original_b64), ("adversarial", req.adversarial_b64)]:
            if not b64:
                continue
            pil_img = _b64_to_pil(b64)
            tensor = preprocess_image(pil_img, model_name)
            tensor.requires_grad_(True)
            cam = _generate_gradcam(mdl, tensor)
            if cam is not None:
                overlay = _overlay_heatmap_fast(pil_img, cam)
                results[f"{key}_overlay"] = _pil_to_b64(overlay)

        return results
    except Exception as e:
        raise HTTPException(500, f"GradCAM compare failed: {str(e)}")
