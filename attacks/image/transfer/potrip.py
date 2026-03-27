"""
Po+Trip (Poincaré Distance + Triplet Loss) — Li et al., CVPR 2020
"Towards Transferable Targeted Attack"

Uses Poincaré distance for self-adaptive gradient magnitude combined with
a triplet loss that pushes away from the source class while pulling toward
the target class in the model's logit/feature space.
"""

import torch
import torch.nn.functional as F
from config import device


def _poincare_distance(u, v, eps=1e-5):
    """Poincaré ball distance between two vectors (projected to unit ball)."""
    u_norm = u.norm(dim=-1, keepdim=True).clamp(max=1.0 - eps)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(max=1.0 - eps)
    u_safe = u / (u_norm + eps) * u_norm
    v_safe = v / (v_norm + eps) * v_norm

    diff_sq = (u_safe - v_safe).pow(2).sum(dim=-1)
    denom = (1 - u_safe.pow(2).sum(dim=-1)) * (1 - v_safe.pow(2).sum(dim=-1))
    return torch.acosh(1 + 2 * diff_sq / (denom + eps))


def targeted_potrip(model, img_tensor, target_class, epsilon, iterations,
                    decay=1.0, margin=0.3):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    momentum = torch.zeros_like(img_tensor, device=device)

    with torch.no_grad():
        clean_logits = model(img_tensor).logits
        source_class = clean_logits.argmax(dim=-1).item()

    one_hot_target = torch.zeros(1, clean_logits.shape[1], device=device)
    one_hot_target[0, target_class] = 1.0
    one_hot_source = torch.zeros(1, clean_logits.shape[1], device=device)
    one_hot_source[0, source_class] = 1.0

    for _ in range(int(iterations)):
        adv_var = adv.detach().requires_grad_(True)
        logits = model(adv_var).logits

        logits_norm = logits / (logits.norm(dim=-1, keepdim=True) + 1e-8)
        d_target = _poincare_distance(logits_norm, one_hot_target)
        d_source = _poincare_distance(logits_norm, one_hot_source)

        loss_pull = d_target.mean()
        loss_push = F.relu(margin - d_source).mean()
        loss = loss_pull + loss_push

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
