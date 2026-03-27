"""
Logit Loss Attack — Zhao et al., NeurIPS 2021
"On Success and Simplicity: A Second Look at Transferable Targeted Attacks"

Simply maximizes the target logit: L = -Z(x')_t, avoiding the gradient
vanishing problem of cross-entropy loss at high confidence (logit saturation).
Surprisingly effective with enough iterations, outperforming CE loss.
"""

import torch
from config import device


def targeted_logit(model, img_tensor, target_class, epsilon, iterations,
                   decay=1.0):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    momentum = torch.zeros_like(img_tensor, device=device)

    for _ in range(int(iterations)):
        adv_var = adv.detach().requires_grad_(True)
        logits = model(adv_var).logits
        loss = -logits[0, target_class]
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
