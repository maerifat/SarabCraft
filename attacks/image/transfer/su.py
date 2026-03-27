"""
SU (Self-Universality) — Wei et al., CVPR 2023
"Enhancing the Self-Universality for Transferable Targeted Attacks"

Optimizes perturbation to be universal across different regions of a single
image. Feature similarity loss between global and random crops ensures the
perturbation doesn't overfit to the full-image structure. +12% improvement.
"""

import torch
import torch.nn.functional as F
from config import device


def _get_mid_layer(model):
    hf = model.hf_model if hasattr(model, 'hf_model') else model
    base = getattr(hf, next((n for n in dir(hf) if n in
        ('vit', 'swin', 'convnext', 'resnet', 'model', 'base_model')), 'model'), hf)
    if hasattr(base, 'encoder') and hasattr(base.encoder, 'layers'):
        layers = base.encoder.layers
        return layers[len(layers) // 2]
    if hasattr(base, 'encoder') and hasattr(base.encoder, 'layer'):
        layers = base.encoder.layer
        return layers[len(layers) // 2]
    if hasattr(base, 'stages'):
        return base.stages[len(base.stages) // 2]
    if hasattr(base, 'features'):
        return base.features[len(base.features) // 2]
    return None


def _random_crop(x, crop_ratio=0.7):
    """Take a random crop and resize back to original dimensions."""
    B, C, H, W = x.shape
    ch, cw = int(H * crop_ratio), int(W * crop_ratio)
    top = torch.randint(0, H - ch + 1, (1,)).item()
    left = torch.randint(0, W - cw + 1, (1,)).item()
    crop = x[:, :, top:top+ch, left:left+cw]
    return F.interpolate(crop, size=(H, W), mode='bilinear', align_corners=False)


def targeted_su(model, img_tensor, target_class, epsilon, iterations,
                decay=1.0, n_crops=3, crop_ratio=0.7, su_weight=0.5):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    mid_layer = _get_mid_layer(model)
    feats = {}

    if mid_layer is not None:
        handle = mid_layer.register_forward_hook(
            lambda m, i, o: feats.update({'out': o[0] if isinstance(o, tuple) else o})
        )
    else:
        handle = None

    try:
        for _ in range(int(iterations)):
            adv_var = adv.detach().requires_grad_(True)
            logits_full = model(adv_var).logits
            loss_ce = F.cross_entropy(logits_full, target)

            loss_su = torch.tensor(0., device=device)
            if handle is not None:
                feat_full = feats['out'].detach()
                for _ in range(n_crops):
                    crop_in = _random_crop(adv_var, crop_ratio)
                    _ = model(crop_in)
                    feat_crop = feats['out']
                    f1 = feat_full.reshape(1, -1)
                    f2 = feat_crop.reshape(1, -1)
                    cos_sim = F.cosine_similarity(f1, f2, dim=-1)
                    loss_su = loss_su + (1 - cos_sim).mean()
                loss_su = loss_su / n_crops

            loss = loss_ce + su_weight * loss_su
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad = adv_var.grad.detach()
                grad_norm = grad / (grad.abs().mean() + 1e-12)
                momentum = decay * momentum + grad_norm
                adv = adv - alpha * momentum.sign()
                adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
                adv = adv.clamp(0., 1.)
    finally:
        if handle is not None:
            handle.remove()

    return adv.detach()
