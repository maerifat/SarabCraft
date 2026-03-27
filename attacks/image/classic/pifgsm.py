"""
PI-FGSM (Patch-wise Iterative FGSM) — Gao et al., 2020
"Patch-wise Attack for Fooling Deep Neural Network"
https://arxiv.org/abs/2007.06765

Amplifies perturbation through a project kernel for patch-wise updates,
improving targeted transfer attack success.
"""

import torch
import torch.nn.functional as F
from config import device


def _project_kern(kern_size=3):
    """Create average-pooling-like project kernel."""
    kern = torch.ones(3, 1, kern_size, kern_size, device=device) / (kern_size ** 2)
    return kern


def _project_noise(delta, kern):
    """Project perturbation through average kernel."""
    pad = kern.shape[-1] // 2
    return F.conv2d(delta, kern, padding=pad, groups=3)


def targeted_pifgsm(model, img_tensor, target_class, epsilon, iterations,
                    amplification=10.0, prob=0.7, kern_size=3):
    """
    PI-FGSM: patch-wise perturbation with amplification factor.
    amplification: controls perturbation amplification (paper default: 10.0).
    prob: DI transform probability.
    kern_size: project kernel size.
    """
    iterations = int(iterations)
    alpha = epsilon / max(iterations, 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    kern = _project_kern(kern_size)
    amp_factor = alpha * amplification

    delta = torch.zeros_like(img_tensor, device=device)
    amp = torch.zeros_like(img_tensor, device=device)

    for i in range(iterations):
        x = (img_tensor + delta).detach().requires_grad_(True)

        if torch.rand(1).item() < prob:
            _, _, H, W = x.shape
            rnd = int(H * (0.9 + torch.rand(1).item() * 0.1))
            x_in = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
            pad_h, pad_w = H - rnd, W - rnd
            top = torch.randint(0, max(pad_h, 0) + 1, (1,)).item()
            left = torch.randint(0, max(pad_w, 0) + 1, (1,)).item()
            x_in = F.pad(x_in, (left, pad_w - left, top, pad_h - top))
        else:
            x_in = x

        loss = F.cross_entropy(model(x_in).logits, target)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = x.grad
            amp = amp + alpha * grad.sign()
            cut = torch.clamp(abs(amp) - epsilon, min=0.0) * amp.sign()
            projection = alpha * cut.sign()
            amp = amp - projection

            delta = delta - amp_factor * grad.sign() - projection
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = torch.clamp(img_tensor + delta, 0., 1.) - img_tensor

    return (img_tensor + delta).detach()
