"""
Ghost Networks — Li et al., AAAI 2020
"Learning Transferable Adversarial Examples via Ghost Networks"

Creates virtual model variants from a single network via dense dropout
erosion and skip-connection erosion with random scalars. Longitudinal
ensemble across iterations simulates attacking multiple models from one.
"""

import torch
import torch.nn.functional as F
from config import device


def _apply_ghost_dropout(model, hooks_list, dropout_rate=0.1):
    """Apply random dropout to intermediate features via forward hooks."""
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if module.weight.ndim >= 2:
                def _drop_hook(module, inp, out, rate=dropout_rate):
                    if isinstance(out, tuple):
                        masked = out[0] * (torch.rand_like(out[0].float()) > rate).float() / (1.0 - rate)
                        return (masked,) + out[1:]
                    return out * (torch.rand_like(out.float()) > rate).float() / (1.0 - rate)

                h = module.register_forward_hook(_drop_hook)
                hooks_list.append(h)


def targeted_ghost(model, img_tensor, target_class, epsilon, iterations,
                   decay=1.0, ghost_dropout=0.1, n_ghost=5):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for _ in range(int(iterations)):
        g_sum = torch.zeros_like(img_tensor, device=device)

        for _ in range(n_ghost):
            hooks = []
            _apply_ghost_dropout(model, hooks, dropout_rate=ghost_dropout)

            x_var = adv.detach().requires_grad_(True)
            loss = F.cross_entropy(model(x_var).logits, target)
            model.zero_grad()
            loss.backward()
            g_sum = g_sum + x_var.grad.detach()

            for h in hooks:
                h.remove()

        with torch.no_grad():
            grad = g_sum / n_ghost
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
