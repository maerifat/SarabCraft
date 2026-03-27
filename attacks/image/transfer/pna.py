"""
PNA (Pay No Attention) — Wei et al., AAAI 2022
"Towards Transferable Adversarial Attacks on Vision Transformers"

Skips attention gradients during backpropagation through ViT self-attention
layers. By bypassing the model-specific attention gradient paths, the
resulting gradient is less overfitted to the surrogate attention pattern.
Also applies PatchOut (random token dropping) for input diversity.
"""

import torch
import torch.nn.functional as F
from config import device


def _register_pna_hooks(model, hooks_list, skip_attn_grad=True):
    """Replace attention backward with identity (skip attention gradients)."""
    hooked = 0
    for name, module in model.named_modules():
        is_attn = any(k in name.lower() for k in ('attention', 'self_attn', 'attn'))
        is_qkv = any(k in name.lower() for k in ('query', 'key', 'value', 'qkv', 'in_proj'))

        if is_attn and not is_qkv and hasattr(module, 'weight' if hasattr(module, 'weight') else '_parameters'):
            def _skip_hook(module, grad_in, grad_out):
                return grad_out

            h = module.register_full_backward_hook(_skip_hook)
            hooks_list.append(h)
            hooked += 1

    return hooked > 0


def _patch_out(x, keep_prob=0.7):
    """PatchOut: randomly drop image patches for input diversity."""
    B, C, H, W = x.shape
    patch_size = 16
    nH, nW = H // patch_size, W // patch_size
    if nH < 1 or nW < 1:
        return x

    mask = (torch.rand(B, 1, nH, nW, device=x.device) < keep_prob).float()
    mask = F.interpolate(mask, size=(H, W), mode='nearest')
    return x * mask


def targeted_pna(model, img_tensor, target_class, epsilon, iterations,
                 decay=1.0, pna_patchout=0.7):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    hooks = []
    _register_pna_hooks(model, hooks)

    try:
        for i in range(int(iterations)):
            x_aug = _patch_out(adv.detach(), keep_prob=pna_patchout)
            x_aug = x_aug.requires_grad_(True)

            loss = F.cross_entropy(model(x_aug).logits, target)
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad = x_aug.grad.detach()
                grad_norm = grad / (grad.abs().mean() + 1e-12)
                momentum = decay * momentum + grad_norm
                adv = adv - alpha * momentum.sign()
                adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
                adv = adv.clamp(0., 1.)
    finally:
        for h in hooks:
            h.remove()

    return adv.detach()
