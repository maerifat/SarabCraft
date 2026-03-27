"""
ILPD (Intermediate-Level Perturbation Decay) — Li et al., NeurIPS 2023
"Improving Adversarial Transferability via Intermediate-level Perturbation Decay"

Single-stage alternative to ILA's two-stage approach. Applies perturbation
decay that balances adversarial direction effectiveness and intermediate-layer
magnitude. +10.07% average improvement over ILA.
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


def targeted_ilpd(model, img_tensor, target_class, epsilon, iterations,
                  decay=1.0, ilpd_gamma=0.3):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    mid_layer = _get_mid_layer(model)
    if mid_layer is None:
        return _fallback(model, img_tensor, target_class, epsilon, iterations, decay)

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

    for i in range(int(iterations)):
        adv_var = adv.detach().requires_grad_(True)
        logits = model(adv_var).logits
        feat_cur = feats['out']
        if isinstance(feat_cur, tuple):
            feat_cur = feat_cur[0]

        loss_ce = F.cross_entropy(logits, target)

        feat_delta = feat_cur - feat_clean
        progress = i / max(int(iterations) - 1, 1)
        decay_weight = ilpd_gamma * (1 - progress)
        loss_decay = decay_weight * feat_delta.pow(2).mean()

        loss = loss_ce + loss_decay

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


def _fallback(model, img_tensor, target_class, epsilon, iterations, decay):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)
    for _ in range(int(iterations)):
        adv_var = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv_var).logits, target)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            grad = adv_var.grad / (adv_var.grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)
    return adv.detach()
