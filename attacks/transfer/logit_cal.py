"""
Logit Calibration — He et al., TIFS 2023
"Logit Calibration for Transferable Targeted Attacks"

Temperature scaling: L = -Z_t/T + log(sum(exp(Z_i/T))). Also supports
margin-based variant with adaptive margin to prevent logit saturation.
Addresses gradient vanishing in CE loss through calibrated temperature.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_logit_cal(model, img_tensor, target_class, epsilon, iterations,
                       decay=1.0, temperature=3.0, margin=0.0):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for _ in range(int(iterations)):
        adv_var = adv.detach().requires_grad_(True)
        logits = model(adv_var).logits

        scaled_logits = logits / temperature
        if margin > 0:
            scaled_logits = scaled_logits.clone()
            scaled_logits[0, target_class] += margin / temperature

        loss = F.cross_entropy(scaled_logits, target)
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
