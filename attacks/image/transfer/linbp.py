"""
LinBP (Linear Backpropagation) — Guo et al., NeurIPS 2020
"Backpropagating Linearly Improves Transferability of Adversarial Examples"

Linearizes ReLU/GELU during backpropagation by removing the zero-clipping,
letting gradients flow through regardless of activation sign. Prevents
gradient information loss specific to the surrogate's activation patterns.
"""

import torch
import torch.nn.functional as F
from config import device


def _register_linbp_hooks(model, hooks_list):
    """Replace ReLU/GELU backward pass with linear (identity) pass."""
    modules_hooked = 0

    def _linear_backward(module, grad_in, grad_out):
        return grad_out

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU)):
            h = module.register_full_backward_hook(_linear_backward)
            hooks_list.append(h)
            modules_hooked += 1

    return modules_hooked > 0


def targeted_linbp(model, img_tensor, target_class, epsilon, iterations,
                   decay=1.0):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    hooks = []
    _register_linbp_hooks(model, hooks)

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
