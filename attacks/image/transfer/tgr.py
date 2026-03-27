"""
TGR (Token Gradient Regularization) — Zhang et al., CVPR 2023
"Transferable Adversarial Attacks on Vision Transformers with Token
Gradient Regularization"

Regularizes per-token gradients during backprop to prevent extreme tokens
from dominating. Reduces surrogate-specific gradient spikes, improving
transfer to other ViT and CNN models.
"""

import torch
import torch.nn.functional as F
from config import device


def _register_tgr_hooks(model, hooks_list):
    """Register backward hooks on ViT attention layers to regularize token gradients."""
    hf = model.hf_model if hasattr(model, 'hf_model') else model
    base = getattr(hf, next((n for n in dir(hf) if n in
        ('vit', 'swin', 'deit', 'beit', 'model', 'base_model')), 'model'), hf)

    layers = []
    if hasattr(base, 'encoder') and hasattr(base.encoder, 'layer'):
        layers = list(base.encoder.layer)
    elif hasattr(base, 'encoder') and hasattr(base.encoder, 'layers'):
        layers = list(base.encoder.layers)

    if not layers:
        return False

    for layer in layers:
        attn = getattr(layer, 'attention', getattr(layer, 'self_attn', None))
        if attn is None:
            continue

        def _backward_hook(module, grad_in, grad_out):
            g = grad_out[0]
            if g is None or g.dim() < 3:
                return None
            token_norms = g.norm(dim=-1, keepdim=True) + 1e-12
            mean_norm = token_norms.mean(dim=1, keepdim=True)
            scale = mean_norm / token_norms
            scale = scale.clamp(0.1, 3.0)
            return (g * scale,)

        h = attn.register_full_backward_hook(_backward_hook)
        hooks_list.append(h)

    return len(hooks_list) > 0


def targeted_tgr(model, img_tensor, target_class, epsilon, iterations,
                 decay=1.0):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    hooks = []
    is_vit = _register_tgr_hooks(model, hooks)

    try:
        for i in range(int(iterations)):
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
    finally:
        for h in hooks:
            h.remove()

    return adv.detach()
