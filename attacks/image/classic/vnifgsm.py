"""
VNI-FGSM (Variance-tuned Nesterov Iterative FGSM) — Wang & He, CVPR 2021
Combines VMI-FGSM's variance tuning with NI-FGSM's Nesterov momentum.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_vnifgsm(model, img_tensor, target_class, epsilon, iterations,
                     decay=1.0, n_var=20, beta_var=1.5):
    """VNI-FGSM: Nesterov lookahead + neighbourhood variance tuning."""
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)
    variance = torch.zeros_like(img_tensor, device=device)

    for i in range(int(iterations)):
        with torch.no_grad():
            nes = adv - alpha * decay * momentum.sign()
            nes = img_tensor + torch.clamp(nes - img_tensor, -epsilon, epsilon)

        nes = nes.detach().requires_grad_(True)
        loss = F.cross_entropy(model(nes).logits, target)
        model.zero_grad()
        loss.backward()
        cur_grad = nes.grad.detach().clone()

        if i > 0:
            g_nb = torch.zeros_like(img_tensor, device=device)
            for _ in range(n_var):
                xn = (nes.detach() + torch.randn_like(img_tensor) * beta_var * epsilon)
                xn = xn.requires_grad_(True)
                loss_nb = F.cross_entropy(model(xn).logits, target)
                model.zero_grad()
                loss_nb.backward()
                g_nb = g_nb + xn.grad.detach()
            variance = g_nb / n_var - cur_grad

        with torch.no_grad():
            grad = cur_grad + variance
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)

    return adv.detach()
