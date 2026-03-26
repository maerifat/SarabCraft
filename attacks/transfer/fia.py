"""
FIA (Feature Importance-aware Attack) — Wang et al., ICCV 2021
"Feature Importance-aware Transferable Adversarial Attacks"

Computes aggregate gradient from random drop masks on intermediate feature
maps to identify object-aware, important features. Attacks only these
important features, reducing overfitting to surrogate-specific details.
"""

import torch
import torch.nn.functional as F
from config import device


def _get_mid_layer(model):
    """Try to hook into an intermediate layer of a HF model."""
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


def targeted_fia(model, img_tensor, target_class, epsilon, iterations,
                 decay=1.0, n_drop=30, drop_rate=0.3):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    mid_layer = _get_mid_layer(model)
    if mid_layer is None:
        return _fallback_mi_fgsm(model, img_tensor, target_class, epsilon, iterations, decay)

    # Phase 1: compute aggregate feature importance via random drop
    agg_grad = torch.zeros_like(img_tensor, device=device)
    feats = {}

    def hook_fn(module, inp, out):
        feats['out'] = out

    handle = mid_layer.register_forward_hook(hook_fn)

    for _ in range(n_drop):
        x_in = adv.detach().requires_grad_(True)
        _ = model(x_in)
        feat = feats.get('out')
        if feat is None:
            break
        if isinstance(feat, tuple):
            feat = feat[0]

        mask = (torch.rand_like(feat.float()) > drop_rate).float()
        masked_feat = (feat * mask).sum()
        model.zero_grad()
        masked_feat.backward()
        if x_in.grad is not None:
            agg_grad = agg_grad + x_in.grad.detach()

    handle.remove()
    importance = agg_grad / max(n_drop, 1)

    # Phase 2: guided attack using feature importance
    for i in range(int(iterations)):
        adv_var = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv_var).logits, target)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = adv_var.grad.detach()
            grad = grad * importance.sign()
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()


def _fallback_mi_fgsm(model, img_tensor, target_class, epsilon, iterations, decay):
    """Fallback if no intermediate layer found."""
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
