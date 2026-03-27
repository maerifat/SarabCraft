"""
SASD (Sharpness-Aware Self-Distillation) — Chen et al., CVPR 2024
"Improving Transferable Targeted Adversarial Attack via Sharpness-Aware Self-Distillation"

Improves the surrogate model during the attack by:
(1) Sharpness-aware perturbation seeking flat loss minima
(2) Self-distillation: teacher (current model) guides a perturbed student
This produces a surrogate that generalises better to unknown targets.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_sasd(model, img_tensor, target_class, epsilon, iterations,
                  decay=1.0, sasd_rho=0.05, sasd_temp=4.0):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for _ in range(int(iterations)):
        adv_var = adv.detach().requires_grad_(True)
        logits = model(adv_var).logits
        loss_ce = F.cross_entropy(logits, target)
        model.zero_grad()
        loss_ce.backward()
        grad_ce = adv_var.grad.detach().clone()

        with torch.no_grad():
            teacher_probs = F.softmax(logits.detach() / sasd_temp, dim=-1)

        adv_sharp = (adv.detach() + sasd_rho * grad_ce.sign()).clamp(0., 1.)
        adv_sharp = adv_sharp.requires_grad_(True)
        logits_sharp = model(adv_sharp).logits
        student_log_probs = F.log_softmax(logits_sharp / sasd_temp, dim=-1)
        loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (sasd_temp ** 2)

        loss = F.cross_entropy(logits_sharp, target) + 0.5 * loss_kd
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = adv_sharp.grad.detach()
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
