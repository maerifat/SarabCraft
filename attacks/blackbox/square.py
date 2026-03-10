"""
Square Attack — Andriushchenko et al., ECCV 2020
"Square Attack: a query-efficient black-box adversarial attack via random search"
https://arxiv.org/abs/1912.00049

Score-based black-box attack. Needs NO gradients — only model output.
Uses random square-shaped colour perturbations with adaptive size.
"""

import torch
import torch.nn.functional as F
from config import device


def _margin_loss(logits, target):
    """Targeted margin: z_target - max(z_other)."""
    target_logit = logits[:, target]
    logits_m = logits.clone()
    logits_m[:, target] = -float('inf')
    max_other = logits_m.max(dim=1).values
    return (target_logit - max_other).item()


def _p_schedule(t, T, p_init=0.8):
    """Decreasing schedule for square size."""
    if t < 0.1 * T:
        return p_init
    elif t < 0.5 * T:
        return p_init * 0.5
    elif t < 0.8 * T:
        return p_init * 0.2
    else:
        return p_init * 0.1


def targeted_square(model, img_tensor, target_class, epsilon,
                    n_queries=5000, p_init=0.8):
    """
    Square Attack (L-inf): random search with square-shaped perturbations.
    n_queries: max forward passes (budget).
    p_init: initial square size as fraction of image side.
    """
    x0 = img_tensor.clone().detach().to(device)
    B, C, H, W = x0.shape

    delta = torch.empty_like(x0).uniform_(-epsilon, epsilon)
    best_adv = (x0 + delta).clamp(0., 1.)

    with torch.no_grad():
        logits = model(best_adv).logits
    best_margin = _margin_loss(logits, target_class)

    if logits.argmax(dim=1).item() == target_class:
        return best_adv.detach()

    for q in range(int(n_queries)):
        p = _p_schedule(q, n_queries, p_init)
        s = max(int(p * min(H, W)), 1)

        yc = torch.randint(0, max(H - s, 1), (1,)).item()
        xc = torch.randint(0, max(W - s, 1), (1,)).item()

        new_delta = (best_adv - x0).clone()
        patch_val = torch.empty(1, C, 1, 1, device=device).uniform_(-epsilon, epsilon)
        new_delta[:, :, yc:yc+s, xc:xc+s] = patch_val.expand(B, C, s, s)

        candidate = (x0 + new_delta).clamp(0., 1.)
        candidate = x0 + torch.clamp(candidate - x0, -epsilon, epsilon)

        with torch.no_grad():
            logits = model(candidate).logits
        margin = _margin_loss(logits, target_class)

        if margin > best_margin:
            best_margin = margin
            best_adv = candidate

        if logits.argmax(dim=1).item() == target_class:
            break

    return best_adv.detach()
