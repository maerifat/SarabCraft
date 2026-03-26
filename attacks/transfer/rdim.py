"""
RDIM (Resized-Diverse-Inputs Method) — Zou et al., 2021
"Improving the Transferability of Adversarial Examples with Resized-Diverse-Inputs"

Combines resized diverse inputs with diversity-ensemble and region fitting.
Random resize to various dimensions + padding, improving over standard DIM
by using a wider resize range for more aggressive input diversity.
"""

import torch
import torch.nn.functional as F
from config import device


def _resized_diverse_input(x, resize_range=(0.5, 1.3)):
    """Random resize + zero-pad back to original size."""
    B, C, H, W = x.shape
    ratio = resize_range[0] + torch.rand(1).item() * (resize_range[1] - resize_range[0])
    new_h, new_w = int(H * ratio), int(W * ratio)
    new_h, new_w = max(new_h, 8), max(new_w, 8)

    resized = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)

    if new_h >= H and new_w >= W:
        top = torch.randint(0, max(new_h - H, 1) + 1, (1,)).item()
        left = torch.randint(0, max(new_w - W, 1) + 1, (1,)).item()
        return resized[:, :, top:top+H, left:left+W]

    pad_h, pad_w = max(H - new_h, 0), max(W - new_w, 0)
    top = torch.randint(0, pad_h + 1, (1,)).item()
    left = torch.randint(0, pad_w + 1, (1,)).item()
    return F.pad(resized, [left, pad_w - left, top, pad_h - top])


def targeted_rdim(model, img_tensor, target_class, epsilon, iterations,
                  decay=1.0, n_diverse=5, resize_lo=0.5, resize_hi=1.3):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for _ in range(int(iterations)):
        g_sum = torch.zeros_like(img_tensor, device=device)

        for _ in range(n_diverse):
            x_aug = _resized_diverse_input(adv.detach(), (resize_lo, resize_hi))
            x_aug = x_aug.requires_grad_(True)
            loss = F.cross_entropy(model(x_aug).logits, target)
            model.zero_grad()
            loss.backward()
            g_sum = g_sum + x_aug.grad.detach()

        with torch.no_grad():
            grad = g_sum / n_diverse
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
