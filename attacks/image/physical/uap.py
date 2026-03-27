"""
UAP (Universal Adversarial Perturbation) — Moosavi-Dezfooli et al., 2017
"Universal adversarial perturbations"
https://arxiv.org/abs/1610.08401

Computes a single image-agnostic perturbation that fools the model
on most inputs. Adapted for targeted single-image setting.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_uap(model, img_tensor, target_class, epsilon,
                 iterations=100, overshoot=0.02):
    """
    UAP (single-image targeted): iterative DeepFool-like perturbation
    projected to epsilon-ball. For full universal training, supply
    multiple images; this single-image version builds a targeted
    perturbation using iterative boundary projection.
    """
    x0 = img_tensor.clone().detach().to(device)
    v = torch.zeros_like(x0, device=device)
    target_t = torch.tensor([target_class], dtype=torch.long, device=device)

    for i in range(int(iterations)):
        x_pert = (x0 + v).clamp(0., 1.)
        x_pert = x_pert.detach().requires_grad_(True)
        logits = model(x_pert).logits
        pred = logits.argmax(dim=1).item()

        if pred == target_class:
            v = v * 0.95
            continue

        loss = logits[0, pred] - logits[0, target_class]
        model.zero_grad()
        loss.backward()

        grad = x_pert.grad.detach()
        grad_norm_sq = (grad ** 2).sum() + 1e-12

        with torch.no_grad():
            r = (loss.abs() / grad_norm_sq) * grad
            v = v - (1 + overshoot) * r

            v_norm = torch.norm(v)
            if v_norm > epsilon:
                v = v * (epsilon / v_norm)

    return (x0 + v).clamp(0., 1.).detach()
