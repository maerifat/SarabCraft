"""
Image preprocessing, tensor conversion, and prediction utilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from config import device
from models.loader import load_model


def preprocess_image(image, model_name="microsoft/resnet-50", exact_mode=False):
    """Preprocess image using HuggingFace processor.
    Note: load_model internally caches models+processors, so repeated calls
    with the same model_name are cheap (dict lookup).
    """
    _, proc = load_model(model_name)

    if exact_mode:
        inputs = proc(images=image, return_tensors="pt", do_resize=False, do_center_crop=False)
    else:
        inputs = proc(images=image, return_tensors="pt")

    return inputs["pixel_values"].to(device)


def tensor_to_pil(tensor):
    """Convert normalized tensor back to PIL image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)

    tensor = tensor.squeeze(0)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor.cpu().permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor)


def get_predictions(model, img_tensor, top_k=5):
    """Get top-k predictions with probabilities."""
    model.eval()

    with torch.no_grad():
        outputs = model(img_tensor)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        top_prob, top_idx = torch.topk(probs, top_k)

    results = {}
    for i in range(top_k):
        idx = int(top_idx[0][i].item())
        prob = float(top_prob[0][i].item())
        name = model.config.id2label[idx]
        results[name] = prob

    top_class = model.config.id2label[int(top_idx[0][0].item())]
    top_idx_val = int(top_idx[0][0].item())

    return results, top_class, top_idx_val
