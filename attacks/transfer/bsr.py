"""
BSR (Block Shuffle & Rotation) — Wang et al., 2024
Input transformation that randomly shuffles and rotates image blocks
during attack iterations to improve adversarial transferability.
"""

import torch
import torch.nn.functional as F
import math
from config import device


def _block_shuffle_rotate(x, n_block=3, rotation_range=10):
    """Divide image into n_block×n_block grid, shuffle and rotate blocks."""
    B, C, H, W = x.shape
    bh, bw = H // n_block, W // n_block
    blocks = []
    for i in range(n_block):
        for j in range(n_block):
            block = x[:, :, i*bh:(i+1)*bh, j*bw:(j+1)*bw].clone()
            if torch.rand(1).item() < 0.5:
                angle = (torch.rand(1).item() - 0.5) * 2 * rotation_range
                theta = torch.tensor([
                    [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
                    [math.sin(math.radians(angle)),  math.cos(math.radians(angle)), 0],
                ], dtype=x.dtype, device=x.device).unsqueeze(0).expand(B, -1, -1)
                grid = F.affine_grid(theta, block.size(), align_corners=False)
                block = F.grid_sample(block, grid, align_corners=False)
            blocks.append(block)

    perm = torch.randperm(len(blocks))
    shuffled = [blocks[p] for p in perm]

    rows = []
    for i in range(n_block):
        row = torch.cat(shuffled[i*n_block:(i+1)*n_block], dim=3)
        rows.append(row)
    out = torch.cat(rows, dim=2)

    if out.shape[2:] != (H, W):
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
    return out


def targeted_bsr(model, img_tensor, target_class, epsilon, iterations,
                 decay=1.0, n_block=3, rotation_range=10):
    """
    BSR: momentum FGSM with block-shuffle-rotate augmentation.
    n_block: number of blocks per axis.
    rotation_range: max rotation angle in degrees per block.
    """
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for i in range(int(iterations)):
        x_bsr = _block_shuffle_rotate(adv.detach(), n_block, rotation_range)
        x_bsr = x_bsr.clamp(0., 1.).requires_grad_(True)

        loss = F.cross_entropy(model(x_bsr).logits, target)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = x_bsr.grad
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
