"""
CWA (Common Weakness Attack) — Chen et al., ICLR 2024
"Rethinking Model Ensemble in Transfer-based Adversarial Attacks"

Targets "common weakness" defined by (1) flatness of loss landscape +
(2) closeness to local optima of each model. Both strongly correlate with
transferability. When used with a single model, simulates ensemble via
model augmentation (dropout + noise injection).
"""

import torch
import torch.nn.functional as F
from config import device


def _model_augment(model, x, n_aug=3, noise_std=0.01, dropout_rate=0.05):
    """Simulate ensemble by augmenting model predictions with noise."""
    logits_list = []
    for _ in range(n_aug):
        noise = torch.randn_like(x) * noise_std
        x_noisy = (x + noise).clamp(0., 1.)
        logits_list.append(model(x_noisy).logits)
    return torch.stack(logits_list, dim=0).mean(dim=0)


def targeted_cwa(model, img_tensor, target_class, epsilon, iterations,
                 decay=1.0, n_model_aug=3, cwa_flat_weight=0.5):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for _ in range(int(iterations)):
        adv_var = adv.detach().requires_grad_(True)

        logits_main = model(adv_var).logits
        loss_ce = F.cross_entropy(logits_main, target)

        loss_flat = torch.tensor(0., device=device)
        for _ in range(n_model_aug):
            noise = torch.randn_like(adv_var) * 0.01
            x_n = (adv_var + noise).clamp(0., 1.)
            logits_n = model(x_n).logits
            loss_n = F.cross_entropy(logits_n, target)
            loss_flat = loss_flat + (loss_n - loss_ce).pow(2)
        loss_flat = loss_flat / n_model_aug

        loss = loss_ce + cwa_flat_weight * loss_flat

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
