"""
ILA (Intermediate Level Attack) — Huang et al., ICCV 2019
"Enhancing Adversarial Example Transferability with an Intermediate Level Attack"

Two-stage attack:
  1. Standard attack (MI-FGSM) to get initial adversarial direction
  2. Fine-tune by maximizing intermediate-layer perturbation magnitude
     in the direction found in stage 1 — amplifies transferable features.
"""

import torch
import torch.nn.functional as F
from config import device


def _get_mid_layer(model):
    """Hook into an intermediate layer of a HF model."""
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


def targeted_ila(model, img_tensor, target_class, epsilon, iterations,
                 decay=1.0, ila_ratio=0.5):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    stage1_iters = max(int(iterations * (1.0 - ila_ratio)), 1)
    stage2_iters = max(int(iterations * ila_ratio), 1)

    # Stage 1: standard MI-FGSM
    for _ in range(stage1_iters):
        adv_var = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv_var).logits, target)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            grad = adv_var.grad.detach()
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    mid_layer = _get_mid_layer(model)
    if mid_layer is None:
        for _ in range(stage2_iters):
            adv_var = adv.detach().requires_grad_(True)
            loss = F.cross_entropy(model(adv_var).logits, target)
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                grad = adv_var.grad.detach()
                grad_norm = grad / (grad.abs().mean() + 1e-12)
                momentum = decay * momentum + grad_norm
                adv = adv - alpha * momentum.sign()
                adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
                adv = adv.clamp(0., 1.)
        return adv.detach()

    # Stage 2: ILA — maximize intermediate perturbation in the found direction
    feats = {}
    def hook_fn(module, inp, out):
        feats['out'] = out
    handle = mid_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(img_tensor)
        feat_clean = feats['out']
        if isinstance(feat_clean, tuple):
            feat_clean = feat_clean[0]
        feat_clean = feat_clean.detach()

        _ = model(adv)
        feat_adv_ref = feats['out']
        if isinstance(feat_adv_ref, tuple):
            feat_adv_ref = feat_adv_ref[0]
        feat_dir = (feat_adv_ref - feat_clean).detach()
        feat_dir_norm = feat_dir / (feat_dir.norm() + 1e-12)

    for _ in range(stage2_iters):
        adv_var = adv.detach().requires_grad_(True)
        _ = model(adv_var)
        feat_cur = feats['out']
        if isinstance(feat_cur, tuple):
            feat_cur = feat_cur[0]

        delta_feat = feat_cur - feat_clean
        projection = (delta_feat * feat_dir_norm).sum()
        loss = -projection

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = adv_var.grad.detach()
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    handle.remove()
    return adv.detach()
