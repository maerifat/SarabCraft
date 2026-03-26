"""
SGM (Skip Gradient Method) — Wu et al., ICLR 2020
"Skip Connections Matter: On the Transferability of Adversarial Examples
Generated with ResNets"

Amplifies gradients through skip (residual) connections by applying a decay
factor to gradients flowing through residual blocks. This makes the gradient
more skip-connection-dominated and less overfitted to residual branch specifics.
"""

import torch
import torch.nn.functional as F
from config import device


def _register_sgm_hooks(model, hooks_list, gamma=0.5):
    """Apply decay to residual-branch gradients in ResNet-like architectures."""
    modules_hooked = 0

    for name, module in model.named_modules():
        is_residual = any(k in name.lower() for k in ('layer', 'block', 'stage'))
        has_downsample = hasattr(module, 'downsample') or hasattr(module, 'shortcut')
        has_conv = any(hasattr(module, f'conv{i}') for i in range(1, 4))

        if is_residual and has_conv:
            def _make_hook(mod, g):
                def _backward_hook(module, grad_in, grad_out):
                    if grad_in[0] is not None:
                        return (grad_in[0] * g,) + grad_in[1:]
                    return None
                return _backward_hook

            for cname, child in module.named_children():
                if 'conv' in cname.lower() and 'shortcut' not in cname.lower() and 'downsample' not in cname.lower():
                    h = child.register_full_backward_hook(_make_hook(child, gamma))
                    hooks_list.append(h)
                    modules_hooked += 1

    return modules_hooked > 0


def targeted_sgm(model, img_tensor, target_class, epsilon, iterations,
                 decay=1.0, sgm_gamma=0.5):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    hooks = []
    _register_sgm_hooks(model, hooks, sgm_gamma)

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
