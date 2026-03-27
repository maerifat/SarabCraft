"""
FAB (Fast Adaptive Boundary) — Croce & Hein, 2020
"Minimally distorted Adversarial Examples with a Fast Adaptive Boundary Attack"
https://arxiv.org/abs/1907.02044

Minimum-norm attack that finds the closest adversarial example by
iteratively projecting onto the decision boundary.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_fab(model, img_tensor, target_class, epsilon, iterations=100,
                 alpha_max=0.1, eta=1.05, beta=0.9, n_restarts=1):
    """
    FAB: finds minimal perturbation by linearising the boundary and projecting.
    alpha_max: max step size fraction.
    eta: overshoot factor (>1 to cross boundary).
    beta: backward step factor.
    """
    x0 = img_tensor.clone().detach().to(device)
    target_t = torch.tensor([target_class], dtype=torch.long, device=device)
    best_adv = x0.clone()
    best_norm = float('inf')

    for restart in range(int(n_restarts)):
        if restart == 0:
            adv = x0.clone()
        else:
            adv = x0 + torch.empty_like(x0).uniform_(-epsilon, epsilon)
            adv = adv.clamp(0., 1.)

        for t in range(int(iterations)):
            adv_r = adv.detach().requires_grad_(True)
            logits = model(adv_r).logits
            pred = logits.argmax(dim=1).item()

            if pred == target_class:
                pert_norm = torch.norm(adv - x0).item()
                if pert_norm < best_norm:
                    best_norm = pert_norm
                    best_adv = adv.clone()

                with torch.no_grad():
                    adv = x0 + beta * (adv - x0)
                continue

            z_target = logits[0, target_class]
            z_pred = logits[0, pred]
            diff = z_pred - z_target

            model.zero_grad()
            diff.backward()
            grad = adv_r.grad.detach()
            grad_flat = grad.view(-1)
            grad_norm_sq = (grad_flat ** 2).sum() + 1e-12

            step_size = min(diff.abs().item() / grad_norm_sq.item(), alpha_max * epsilon)
            direction = -eta * step_size * grad

            with torch.no_grad():
                adv = adv + direction
                perturbation = adv - x0
                pert_norm = torch.norm(perturbation)
                if pert_norm > epsilon:
                    perturbation = perturbation * (epsilon / pert_norm)
                adv = (x0 + perturbation).clamp(0., 1.)

    if best_norm < float('inf'):
        return best_adv.detach()
    return adv.detach()
