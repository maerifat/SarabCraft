"""
SMI-FGSM (Spatial Momentum Iterative FGSM) — Wang et al., 2022
"Boosting Transferability of Adversarial Examples via Spatial Momentum"

Dual momentum: temporal (across iterations) + spatial (across image regions).
Spatial momentum prevents gradient cancellation across different spatial
positions, yielding ~10% average improvement in transfer rate.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_smi_fgsm(model, img_tensor, target_class, epsilon, iterations,
                      decay=1.0, spatial_decay=0.6, n_block=3):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)
    spatial_momentum = torch.zeros_like(img_tensor, device=device)

    for _ in range(int(iterations)):
        adv_var = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv_var).logits, target)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = adv_var.grad.detach()

            B, C, H, W = grad.shape
            bh, bw = max(H // n_block, 1), max(W // n_block, 1)
            for bi in range(0, H, bh):
                for bj in range(0, W, bw):
                    patch = grad[:, :, bi:bi+bh, bj:bj+bw]
                    sp = spatial_momentum[:, :, bi:bi+bh, bj:bj+bw]
                    norm_patch = patch / (patch.abs().mean() + 1e-12)
                    spatial_momentum[:, :, bi:bi+bh, bj:bj+bw] = spatial_decay * sp + norm_patch

            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            combined = momentum + spatial_momentum

            adv = adv - alpha * combined.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
