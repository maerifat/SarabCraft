"""
Perturbation quality metrics — L0, L1, L2, Linf, SSIM, PSNR.
Computed on pixel-space tensors in [0,1] range.
"""

import torch
import torch.nn.functional as F
import math


def compute_metrics(original: torch.Tensor, adversarial: torch.Tensor) -> dict:
    """Compute perturbation quality metrics between two tensors.
    Both should be [1, C, H, W] in normalized space. They'll be
    denormalized internally for pixel-space metrics.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=original.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=original.device).view(1, 3, 1, 1)

    orig_px = (original * std + mean).clamp(0, 1)
    adv_px = (adversarial * std + mean).clamp(0, 1)

    diff = (adv_px - orig_px).float()

    # L0: count pixels where ANY channel changed (pixel-level, not channel-level)
    l0 = int((diff.abs() > 1e-6).any(dim=1).sum().item())

    flat = diff.view(-1)
    l1 = float(flat.abs().mean().item())
    l2 = float(flat.norm(2).item())
    linf = float(flat.abs().max().item())

    mse = F.mse_loss(adv_px, orig_px).item()
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else float("inf")

    ssim_val = _ssim(orig_px, adv_px)

    return {
        "l0_pixels": l0,
        "l1": round(l1, 6),
        "l2": round(l2, 6),
        "linf": round(linf, 6),
        "linf_255": round(linf * 255, 2),
        "psnr": round(psnr, 2),
        "ssim": round(ssim_val, 4),
        "mse": round(mse, 8),
    }


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size=11) -> float:
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    img1 = img1.float()
    img2 = img2.float()

    channels = img1.size(1)
    window = _gaussian_window(window_size, 1.5).to(img1.device)
    window = window.expand(channels, 1, window_size, window_size)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


def _gaussian_window(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
