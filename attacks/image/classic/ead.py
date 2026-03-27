"""
EAD (Elastic-Net Attack) — Chen et al., 2018
"EAD: Elastic-Net Attacks to DNNs via Feature-Level Adversarial Training"
https://arxiv.org/abs/1709.04114

L1 + L2 attack using ISTA (Iterative Shrinkage-Thresholding Algorithm).
Produces sparser perturbations than C&W L2.
"""

import torch
import torch.nn.functional as F
from config import device


def _shrinkage(z, beta):
    """Proximal operator for L1: soft thresholding."""
    return torch.sign(z) * torch.clamp(z.abs() - beta, min=0.)


def targeted_ead(model, img_tensor, target_class, epsilon=None,
                 iterations=200, lr=0.01, c=1.0, kappa=0.0, beta=0.001,
                 decision_rule='EN'):
    """
    EAD: Elastic-Net attack using ISTA for L1 sparsity.
    c: loss trade-off constant.
    kappa: confidence margin (higher = stronger adversarial).
    beta: L1 regularisation weight.
    decision_rule: 'EN' (elastic net) or 'L1' (pure L1).
    """
    x0 = img_tensor.clone().detach().to(device)
    y = torch.zeros_like(x0, requires_grad=True, device=device)
    target_t = torch.tensor([target_class], dtype=torch.long, device=device)

    optimizer = torch.optim.Adam([y], lr=lr)

    best_adv = x0.clone()
    best_dist = float('inf')

    for i in range(int(iterations)):
        optimizer.zero_grad()

        adv = x0 + y
        adv_clamped = adv.clamp(0., 1.)
        logits = model(adv_clamped).logits

        target_logit = logits[0, target_class]
        logits_masked = logits.clone()
        logits_masked[0, target_class] = -float('inf')
        max_other = logits_masked.max()

        f_loss = torch.clamp(max_other - target_logit + kappa, min=0.)

        l2_dist = torch.norm(y)
        loss = l2_dist + c * f_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y.data = _shrinkage(y.data, beta * lr)

            delta = y.data
            if epsilon is not None:
                delta = torch.clamp(delta, -epsilon, epsilon)
            y.data = delta

        with torch.no_grad():
            if f_loss.item() == 0:
                if decision_rule == 'L1':
                    dist = y.abs().sum().item()
                else:
                    dist = y.abs().sum().item() + l2_dist.item()
                if dist < best_dist:
                    best_dist = dist
                    best_adv = (x0 + y).clamp(0., 1.).clone()

    if best_dist < float('inf'):
        return best_adv.detach()
    return (x0 + y).clamp(0., 1.).detach()
