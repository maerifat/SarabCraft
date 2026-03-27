"""
SSA (Spectrum Simulation Attack) — Long et al., 2022
"Frequency Domain Model Augmentation for Adversarial Attack"
https://arxiv.org/abs/2207.05382

Augments input in DCT frequency domain during attack to improve transfer.
"""

import torch
import torch.nn.functional as F
from config import device


def _dct_2d(x):
    """Simple 2D DCT via FFT."""
    return torch.fft.fft2(x, norm='ortho').real


def _idct_2d(x):
    """Inverse 2D DCT via IFFT."""
    return torch.fft.ifft2(x, norm='ortho').real


def _spectrum_transform(x, rho=0.5):
    """Randomly transform spectrum: mix original and shuffled frequency components."""
    x_dct = _dct_2d(x)
    mask = (torch.rand_like(x_dct) < rho).float()
    x_shuffled = x_dct.flip(dims=[-1])
    mixed = mask * x_dct + (1 - mask) * x_shuffled
    return _idct_2d(mixed)


def targeted_ssa(model, img_tensor, target_class, epsilon, iterations,
                 decay=1.0, n_spectrum=20, rho_spectrum=0.5):
    """
    SSA: spectrum-augmented momentum FGSM for improved transfer.
    n_spectrum: number of spectrum samples to average gradient over.
    rho_spectrum: probability of keeping original frequency component.
    """
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for i in range(int(iterations)):
        g_sum = torch.zeros_like(img_tensor, device=device)

        for _ in range(n_spectrum):
            x_spec = _spectrum_transform(adv.detach(), rho_spectrum)
            x_spec = x_spec.clamp(0., 1.).requires_grad_(True)
            loss = F.cross_entropy(model(x_spec).logits, target)
            model.zero_grad()
            loss.backward()
            g_sum = g_sum + x_spec.grad.detach()

        with torch.no_grad():
            grad = g_sum / n_spectrum
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
