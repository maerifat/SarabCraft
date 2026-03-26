"""
DA (Dilated Attention Attack) — Wei et al., IJCV 2025
"Boosting Targeted Transferability via Dilated Attention and Dynamic Linear Augmentation"

Targeted-specific: maximizes attention maps of the target class from multiple
intermediate layers with dynamic linear augmentation. Works on both CNNs
and ViTs by leveraging class-specific attention amplification.
"""

import torch
import torch.nn.functional as F
from config import device


def _get_layer_hooks(model, n_layers=3):
    """Get multiple intermediate layers at different depths."""
    hf = model.hf_model if hasattr(model, 'hf_model') else model
    base = getattr(hf, next((n for n in dir(hf) if n in
        ('vit', 'swin', 'convnext', 'resnet', 'model', 'base_model')), 'model'), hf)

    candidates = []
    if hasattr(base, 'encoder') and hasattr(base.encoder, 'layers'):
        candidates = list(base.encoder.layers)
    elif hasattr(base, 'encoder') and hasattr(base.encoder, 'layer'):
        candidates = list(base.encoder.layer)
    elif hasattr(base, 'stages'):
        candidates = list(base.stages)
    elif hasattr(base, 'features'):
        candidates = list(base.features)

    if len(candidates) < n_layers:
        return candidates if candidates else [None]

    indices = torch.linspace(0, len(candidates) - 1, n_layers).long().tolist()
    return [candidates[i] for i in indices]


def _dynamic_linear_aug(x, strength=0.3):
    """Dynamic linear augmentation: linear interpolation with noise."""
    noise = torch.randn_like(x) * strength
    alpha = 0.7 + torch.rand(1, device=x.device).item() * 0.3
    return (alpha * x + (1 - alpha) * (x + noise)).clamp(0., 1.)


def targeted_da(model, img_tensor, target_class, epsilon, iterations,
                decay=1.0, n_aug=3, da_strength=0.3):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    layers = _get_layer_hooks(model)
    if layers[0] is None:
        return _fallback(model, img_tensor, target_class, epsilon, iterations, decay)

    feats = {}
    handles = []
    for idx, layer in enumerate(layers):
        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                feats[layer_idx] = out[0] if isinstance(out, tuple) else out
            return hook_fn
        handles.append(layer.register_forward_hook(make_hook(idx)))

    try:
        for _ in range(int(iterations)):
            g_sum = torch.zeros_like(img_tensor, device=device)

            for _ in range(n_aug):
                x_aug = _dynamic_linear_aug(adv.detach(), da_strength)
                x_aug = x_aug.requires_grad_(True)
                logits = model(x_aug).logits

                loss_ce = F.cross_entropy(logits, target)
                loss_attn = torch.tensor(0., device=device)
                for feat in feats.values():
                    if feat.ndim == 3:
                        attn = feat.mean(dim=1)
                    elif feat.ndim == 4:
                        attn = feat.mean(dim=1).flatten(start_dim=1)
                    else:
                        continue
                    loss_attn = loss_attn - attn.mean()

                loss = loss_ce + 0.1 * loss_attn
                model.zero_grad()
                loss.backward()
                g_sum = g_sum + x_aug.grad.detach()

            with torch.no_grad():
                grad = g_sum / n_aug
                grad_norm = grad / (grad.abs().mean() + 1e-12)
                momentum = decay * momentum + grad_norm
                adv = adv - alpha * momentum.sign()
                adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
                adv = adv.clamp(0., 1.)
    finally:
        for h in handles:
            h.remove()

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
