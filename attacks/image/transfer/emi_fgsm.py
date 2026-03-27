"""
EMI-FGSM (Enhanced Momentum Iterative FGSM) — Wang et al., BMVC 2021
"Boosting Adversarial Transferability through Enhanced Momentum"

Averages gradients along the path between previous and current perturbation
(linear interpolation sampling) to stabilize momentum and reduce overfitting
to the surrogate model.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_emi_fgsm(model, img_tensor, target_class, epsilon, iterations,
                      decay=1.0, n_sample=11, sampling_range=0.3):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)
    prev_adv = adv.clone()

    for i in range(int(iterations)):
        g_sum = torch.zeros_like(img_tensor, device=device)

        for si in range(n_sample):
            lam = si / max(n_sample - 1, 1)
            x_path = prev_adv + lam * (adv.detach() - prev_adv)
            noise = torch.randn_like(x_path) * sampling_range * epsilon
            x_sample = (x_path + noise).clamp(0., 1.).requires_grad_(True)

            loss = F.cross_entropy(model(x_sample).logits, target)
            model.zero_grad()
            loss.backward()
            g_sum = g_sum + x_sample.grad.detach()

        with torch.no_grad():
            grad = g_sum / n_sample
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            prev_adv = adv.clone()
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
