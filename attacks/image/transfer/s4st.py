"""
S⁴ST — Liu et al., 2024
"Improving Targeted Transferability via Scale-Aware and Structure-Preserving Attack"

Dimensionally consistent scaling + complementary low-redundancy transforms
+ block-wise operations. Achieves SOTA targeted transfer (77.7% ASR).
Combines strong scale transforms with block-level structure preservation.
"""

import torch
import torch.nn.functional as F
from config import device


def _scale_transform(x, scale_range=(0.5, 1.5)):
    """Random consistent scaling per image."""
    s = scale_range[0] + torch.rand(1).item() * (scale_range[1] - scale_range[0])
    B, C, H, W = x.shape
    new_h, new_w = int(H * s), int(W * s)
    new_h, new_w = max(new_h, 8), max(new_w, 8)
    scaled = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return F.interpolate(scaled, size=(H, W), mode='bilinear', align_corners=False)


def _block_shift(x, n_blocks=2, max_shift=4):
    """Block-wise random shift for structure preservation."""
    B, C, H, W = x.shape
    out = x.clone()
    bh, bw = H // n_blocks, W // n_blocks

    for bi in range(n_blocks):
        for bj in range(n_blocks):
            dy = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            dx = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            y0, x0 = bi * bh, bj * bw
            y1, x1 = y0 + bh, x0 + bw

            sy0 = max(y0, y0 - dy)
            sy1 = min(y1, y1 - dy)
            sx0 = max(x0, x0 - dx)
            sx1 = min(x1, x1 - dx)
            dy0 = max(y0, y0 + dy)
            dx0 = max(x0, x0 + dx)
            h_s = sy1 - sy0
            w_s = sx1 - sx0
            if h_s > 0 and w_s > 0:
                out[:, :, dy0:dy0+h_s, dx0:dx0+w_s] = x[:, :, sy0:sy0+h_s, sx0:sx0+w_s]

    return out


def _s4st_augment(x, scale_range=(0.5, 1.5), n_blocks=2, max_shift=4):
    """Combined S4ST augmentation pipeline."""
    x = _scale_transform(x, scale_range)
    x = _block_shift(x, n_blocks, max_shift)
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, dims=[3])
    return x.clamp(0., 1.)


def targeted_s4st(model, img_tensor, target_class, epsilon, iterations,
                  decay=1.0, n_aug=5, scale_lo=0.5, scale_hi=1.5):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for _ in range(int(iterations)):
        g_sum = torch.zeros_like(img_tensor, device=device)

        for _ in range(n_aug):
            x_aug = _s4st_augment(adv.detach(), (scale_lo, scale_hi))
            x_aug = x_aug.requires_grad_(True)
            loss = F.cross_entropy(model(x_aug).logits, target)
            model.zero_grad()
            loss.backward()
            g_sum = g_sum + x_aug.grad.detach()

        with torch.no_grad():
            grad = g_sum / n_aug
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
