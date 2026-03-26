"""
RAP (Reverse Adversarial Perturbation) — Qin et al., NeurIPS 2022
"Boosting the Transferability of Adversarial Attacks with Reverse
Adversarial Perturbation"

Bi-level min-max: inner loop seeks a "worst-case" reverse perturbation that
*reduces* the adversarial loss, outer loop must overcome it. Forces the
adversarial perturbation into flat loss regions that transfer better.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_rap(model, img_tensor, target_class, epsilon, iterations,
                 decay=1.0, rap_epsilon=0.02, rap_steps=5, late_start=0.0):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)
    rap_alpha = rap_epsilon / max(rap_steps, 1)
    late_start_iter = int(late_start * iterations)

    for i in range(int(iterations)):
        use_rap = (i >= late_start_iter)

        if use_rap:
            # Inner loop: find reverse perturbation that hurts the attack
            r = torch.zeros_like(adv, device=device)
            for _ in range(rap_steps):
                x_inner = (adv.detach() + r).clamp(0., 1.).requires_grad_(True)
                loss_inner = F.cross_entropy(model(x_inner).logits, target)
                model.zero_grad()
                loss_inner.backward()
                with torch.no_grad():
                    # Maximize loss = move *away* from target (reverse direction)
                    r = r + rap_alpha * x_inner.grad.sign()
                    r = torch.clamp(r, -rap_epsilon, rap_epsilon)

            x_outer = (adv.detach() + r.detach()).clamp(0., 1.).requires_grad_(True)
        else:
            x_outer = adv.detach().requires_grad_(True)

        loss = F.cross_entropy(model(x_outer).logits, target)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = x_outer.grad.detach()
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
