"""
TI-FGSM Attack (Translation-Invariant FGSM) — Dong et al., CVPR 2019
Convolves gradient with Gaussian kernel for translation invariance.
"""

import torch
import torch.nn.functional as F
from config import device

_kernel_cache = {}


def _gaussian_kernel(size, sigma, channels):
    """Create depthwise Gaussian kernel."""
    key = (size, sigma, channels)
    if key in _kernel_cache:
        return _kernel_cache[key]
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    g = g / g.sum()
    k2d = g.unsqueeze(0) * g.unsqueeze(1)
    k2d = k2d / k2d.sum()
    kernel = k2d.view(1, 1, size, size).repeat(channels, 1, 1, 1).to(device)
    _kernel_cache[key] = kernel
    return kernel


def targeted_tifgsm(model, img_tensor, target_class, epsilon, iterations,
                    decay=1.0, kernel_size=5):
    """Translation-Invariant FGSM — Gaussian-smoothed gradient + momentum."""
    print(f"[DEBUG] TI-FGSM: starting, target={target_class}, eps={epsilon:.4f}, "
          f"iter={iterations}, kernel={kernel_size}", flush=True)

    C = img_tensor.shape[1]
    kern = _gaussian_kernel(kernel_size, sigma=kernel_size / 3, channels=C)
    pad = kernel_size // 2

    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(iterations, 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for i in range(int(iterations)):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv).logits, target)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = F.conv2d(adv.grad, kern, padding=pad, groups=C)
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)

        if i % 5 == 0:
            print(f"[DEBUG] TI-FGSM: iter {i+1}/{iterations}", flush=True)

    print("[DEBUG] TI-FGSM: complete", flush=True)
    return adv.detach()
