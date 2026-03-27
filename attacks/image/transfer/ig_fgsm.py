"""
IG-FGSM (Integrated Gradients FGSM) — Qi et al., 2021
"Improving the Transferability of Adversarial Attacks through Integrated Gradients"

Replaces single-point gradients with path-integrated gradients from a
baseline (black image) to the current point. Produces smoother, less
model-specific gradients. Simpler predecessor to MIG.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_ig_fgsm(model, img_tensor, target_class, epsilon, iterations,
                     decay=1.0, n_ig=10):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)
    baseline = torch.zeros_like(img_tensor, device=device)

    for _ in range(int(iterations)):
        ig_grad = torch.zeros_like(img_tensor, device=device)

        for k in range(n_ig):
            lam = k / max(n_ig - 1, 1)
            x_interp = baseline + lam * (adv.detach() - baseline)
            x_interp = x_interp.requires_grad_(True)

            loss = F.cross_entropy(model(x_interp).logits, target)
            model.zero_grad()
            loss.backward()
            ig_grad = ig_grad + x_interp.grad.detach()

        with torch.no_grad():
            grad = ig_grad / n_ig * (adv - baseline)
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
