"""
Admix — Wang et al., ICCV 2021
"Admix: Enhancing the Transferability of Adversarial Attacks"
https://arxiv.org/abs/2102.00436

Mixes portions of randomly-sampled images into the input during attack
to prevent over-fitting to the source model.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_admix(model, img_tensor, target_class, epsilon, iterations,
                   decay=1.0, n_mix=5, mix_ratio=0.2):
    """
    Admix: average gradients over images mixed with random patches.
    n_mix: number of admixed copies per iteration.
    mix_ratio: how much of the random image to mix in (eta in paper, default 0.2).
    """
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for i in range(int(iterations)):
        g_sum = torch.zeros_like(img_tensor, device=device)

        for _ in range(n_mix):
            rand_img = torch.rand_like(img_tensor)
            x_mix = ((1 - mix_ratio) * adv + mix_ratio * rand_img).clamp(0., 1.)

            for s in range(3):
                x_s = (x_mix / (2 ** s)).detach().requires_grad_(True)
                loss = F.cross_entropy(model(x_s).logits, target)
                model.zero_grad()
                loss.backward()
                g_sum = g_sum + x_s.grad.detach()

        with torch.no_grad():
            grad = g_sum / (n_mix * 3)
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
