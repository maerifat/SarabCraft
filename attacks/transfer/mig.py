"""
MIG (Momentum Integrated Gradients) — Ma et al., ICLR 2022
"Transferable Adversarial Attack based on Integrated Gradients"

Combines path-integrated gradients (from scaled baselines) with momentum.
Instead of computing gradient at just the current point, averages over the
integration path from zero to current adversarial example, providing a more
generalised gradient signal that transfers better.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_mig(model, img_tensor, target_class, epsilon, iterations,
                 decay=1.0, n_ig=10):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for i in range(int(iterations)):
        g_sum = torch.zeros_like(img_tensor, device=device)

        for k in range(1, n_ig + 1):
            scale = k / n_ig
            x_scaled = (adv.detach() * scale).requires_grad_(True)
            loss = F.cross_entropy(model(x_scaled).logits, target)
            model.zero_grad()
            loss.backward()
            g_sum = g_sum + x_scaled.grad.detach()

        with torch.no_grad():
            grad = (g_sum / n_ig) * adv.detach()
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
