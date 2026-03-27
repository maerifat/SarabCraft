"""
Boundary Attack — Brendel, Rauber & Bethge, 2018
"Decision-Based Adversarial Attacks: Reliable Attacks Against
Black-Box Machine Learning Models"
https://arxiv.org/abs/1712.04248

Decision-based black-box: starts from an adversarial image and
progressively walks along the decision boundary toward the original.
Only needs hard-label model output (argmax).
"""

import torch
from config import device


def targeted_boundary(model, img_tensor, target_class, epsilon=None,
                      iterations=5000, delta_init=0.01, step_init=0.01):
    """
    Boundary Attack: starts from adversarial noise, walks toward clean image.
    delta_init: initial step size for orthogonal perturbation.
    step_init: initial step size toward original image.
    """
    x0 = img_tensor.clone().detach().to(device)
    B, C, H, W = x0.shape

    adv = torch.rand_like(x0)
    with torch.no_grad():
        for trial in range(1000):
            logits = model(adv).logits
            if logits.argmax(dim=1).item() == target_class:
                break
            adv = torch.rand_like(x0)
        else:
            return x0.clone()

    delta = delta_init
    step = step_init

    for t in range(int(iterations)):
        pert = torch.randn_like(x0)
        direction = adv - x0
        pert = pert - (pert * direction).sum() / (direction.norm() ** 2 + 1e-12) * direction
        pert = pert / (pert.norm() + 1e-12) * direction.norm() * delta

        candidate = adv + pert
        candidate = candidate / (candidate.norm() + 1e-12) * adv.norm()

        candidate = (1 - step) * candidate + step * x0
        candidate = candidate.clamp(0., 1.)

        if epsilon is not None:
            candidate = x0 + torch.clamp(candidate - x0, -epsilon, epsilon)

        with torch.no_grad():
            logits = model(candidate).logits

        if logits.argmax(dim=1).item() == target_class:
            dist_new = torch.norm(candidate - x0)
            dist_old = torch.norm(adv - x0)
            if dist_new < dist_old:
                adv = candidate
                delta = min(delta * 1.02, 1.0)
                step = min(step * 1.02, 1.0)
            else:
                delta = max(delta * 0.98, 1e-6)
        else:
            delta = max(delta * 0.95, 1e-6)
            step = max(step * 0.95, 1e-6)

    return adv.detach()
