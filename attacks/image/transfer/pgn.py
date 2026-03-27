"""
PGN (Penalizing Gradient Norm) — Ge et al., NeurIPS 2023
"Boosting Adversarial Transferability by Achieving Flat Local Maxima"

Adds a gradient-norm penalty to the loss to seek flat local maxima in the
loss landscape. Flat maxima generalise better across models, significantly
boosting black-box transfer rates.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_pgn(model, img_tensor, target_class, epsilon, iterations,
                 decay=1.0, pgn_lambda=1.0):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for i in range(int(iterations)):
        adv_var = adv.detach().requires_grad_(True)
        loss_ce = F.cross_entropy(model(adv_var).logits, target)
        model.zero_grad()
        grad_ce = torch.autograd.grad(loss_ce, adv_var, create_graph=True)[0]

        grad_norm = torch.norm(grad_ce)
        combined_loss = loss_ce + pgn_lambda * grad_norm
        combined_loss.backward()

        with torch.no_grad():
            grad = adv_var.grad.detach()
            grad_norm_val = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm_val
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
