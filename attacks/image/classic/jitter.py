"""
Jitter Attack — Schwinn et al., 2021
PGD variant that samples from a random neighbourhood before computing
gradients, improving escape from sharp local minima.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_jitter(model, img_tensor, target_class, epsilon, iterations,
                    alpha_mult=1.0, random_start=True, jitter_ratio=0.1):
    """
    Jitter: PGD + random neighbourhood perturbation before each gradient step.
    jitter_ratio: fraction of epsilon used for random jitter.
    """
    adv = img_tensor.clone().detach().to(device)

    if random_start:
        adv = adv + torch.empty_like(adv).uniform_(-epsilon, epsilon)
        adv = adv.clamp(0., 1.)

    alpha = (epsilon / max(int(iterations) // 4, 1)) * alpha_mult
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    jitter_mag = epsilon * jitter_ratio

    for i in range(int(iterations)):
        with torch.no_grad():
            x_jit = adv + torch.empty_like(adv).uniform_(-jitter_mag, jitter_mag)
            x_jit = torch.clamp(x_jit, 0., 1.)

        x_jit = x_jit.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_jit).logits, target)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            adv = adv - alpha * x_jit.grad.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
