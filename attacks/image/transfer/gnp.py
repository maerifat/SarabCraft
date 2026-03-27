"""
GNP (Gradient Norm Penalty) — Wu et al., ICME 2023
"Boosting Adversarial Transferability via Gradient Norm Penalty"

Penalizes the gradient norm to drive optimization toward flat loss regions.
A plug-in enhancement for any gradient-based method. Flat regions in the
loss landscape generalise better across models.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_gnp(model, img_tensor, target_class, epsilon, iterations,
                 decay=1.0, gnp_lambda=1.0):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for _ in range(int(iterations)):
        adv_var = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv_var).logits, target)
        model.zero_grad()
        loss.backward(create_graph=True)

        grad = adv_var.grad
        grad_norm = grad.pow(2).sum()
        penalty = gnp_lambda * grad_norm

        model.zero_grad()
        penalty.backward()
        gnp_grad = adv_var.grad.detach()

        with torch.no_grad():
            combined = grad.detach() + gnp_grad
            combined_norm = combined / (combined.abs().mean() + 1e-12)
            momentum = decay * momentum + combined_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
