"""
SIA (Structure Invariant Attack) — Wang et al., ICCV 2023
"Structure Invariant Transformation for better Adversarial Transferability"

Splits image into s×s blocks, applies independent random affine + colour
transforms to each block while preserving relative spatial structure.
This forces the attack to rely on structural features rather than
model-specific spatial details.
"""

import torch
import torch.nn.functional as F
from config import device


def _block_transform(x, n_blocks=3):
    """Apply independent random transforms to each block."""
    B, C, H, W = x.shape
    bh, bw = H // n_blocks, W // n_blocks
    out = x.clone()

    for bi in range(n_blocks):
        for bj in range(n_blocks):
            y0, x0 = bi * bh, bj * bw
            y1, x1 = y0 + bh, x0 + bw
            patch = x[:, :, y0:y1, x0:x1]

            angle = (torch.rand(1).item() - 0.5) * 20.0
            scale = 0.85 + torch.rand(1).item() * 0.3
            theta = torch.tensor([
                [scale * torch.cos(torch.tensor(angle * 3.14159 / 180)),
                 -scale * torch.sin(torch.tensor(angle * 3.14159 / 180)), 0],
                [scale * torch.sin(torch.tensor(angle * 3.14159 / 180)),
                 scale * torch.cos(torch.tensor(angle * 3.14159 / 180)), 0]
            ], dtype=x.dtype, device=x.device).unsqueeze(0).expand(B, -1, -1)

            grid = F.affine_grid(theta, patch.size(), align_corners=False)
            patch_t = F.grid_sample(patch, grid, align_corners=False, padding_mode='reflection')

            brightness = 0.9 + torch.rand(1, device=x.device).item() * 0.2
            patch_t = (patch_t * brightness).clamp(0., 1.)
            out[:, :, y0:y1, x0:x1] = patch_t

    return out


def targeted_sia(model, img_tensor, target_class, epsilon, iterations,
                 decay=1.0, n_blocks=3):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for _ in range(int(iterations)):
        x_aug = _block_transform(adv.detach(), n_blocks=n_blocks)
        x_aug = x_aug.requires_grad_(True)

        loss = F.cross_entropy(model(x_aug).logits, target)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = x_aug.grad.detach()
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
