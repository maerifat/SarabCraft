"""
NewBackend: UN-DP-DI2-TI-PI-FGSM (TA-Bench NeurIPS 2023)
=========================================================

The strongest composite baseline from the Transfer-Attack Benchmark:

  UN  — Uniform Noise: adds random uniform noise ∈ [-ε, ε] before forward
  PI  — Patch Interaction: grid-based patch sampling on the perturbation delta
  DI  — Diverse Input: random resize-pad data augmentation (DI-FGSM, Xie 2019)
  TI  — Translation Invariance: random affine translation (Dong 2019)
  NI  — Nesterov look-ahead on momentum (Lin 2020)
  MI  — Momentum Iterative gradient accumulation (Dong 2018)

All operations run in [0,1] pixel space.  The model passed here must
accept [0,1] inputs and return raw logit tensors (not HuggingFace
output objects).  The router handles the space conversion.

Reference: github.com/Trustworthy-AI-Group/TransferAttack (NeurIPS 2023)
"""

import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T


def _di_transform(x, resize_rate=0.9, prob=0.5):
    """Diverse Input transformation (Xie et al., 2019)."""
    if torch.rand(1).item() >= prob:
        return x
    B, C, H, W = x.shape
    img_resize = int(H * resize_rate)
    img_size, img_resize = img_resize, H
    rnd = torch.randint(img_size, img_resize, (1,)).item()
    x = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
    rem = img_resize - rnd
    pt = torch.randint(0, rem, (1,)).item()
    pl = torch.randint(0, rem, (1,)).item()
    x = F.pad(x, (pl, rem - pl, pt, rem - pt), value=0)
    return x


def _patch_sample(delta, npatch=128, grid_scale=16, img_size=224):
    """Patch-wise interaction: grid-based sampling of perturbation delta."""
    grid_size = img_size // grid_scale
    mask = torch.zeros_like(delta)
    ids = np.random.randint(0, grid_scale ** 2, size=npatch)
    rows, cols = ids // grid_scale, ids % grid_scale
    for r, c in zip(rows, cols):
        mask[:, :, r * grid_size:(r + 1) * grid_size,
             c * grid_size:(c + 1) * grid_size] = 1
    return delta * mask


def _update_and_clip(ori, adv, grad, eps, ss):
    """PGD-style update with Linf clipping."""
    adv = adv.data + ss * grad.sign()
    adv = torch.where(adv > ori + eps, ori + eps, adv)
    adv = torch.where(adv < ori - eps, ori - eps, adv)
    return torch.clamp(adv, 0.0, 1.0)


def targeted_newbackend(model, img_tensor, target_class, epsilon, iterations,
                        decay=1.0, step_size=None,
                        di_prob=0.5, di_resize_rate=0.9,
                        ti_len=3, npatch=128, grid_scale=16,
                        enable_un=True, enable_pi=True,
                        enable_di=True, enable_ti=True,
                        enable_ni=True,
                        quiet=False):
    """
    NewBackend: UN + PI + DI + TI + NI + MI — targeted version.

    Operates in [0,1] pixel space.
    Model must accept [0,1] inputs and return logit tensor.

    Parameters
    ----------
    model        : pixel-space model returning raw logits
    img_tensor   : [B, 3, H, W] clean image in [0, 1]
    target_class : int, target class index
    epsilon      : float, Linf budget in [0, 1] scale
    iterations   : int, number of PGD steps
    decay        : float, momentum decay factor (mu)
    step_size    : float or None, per-step size; defaults to epsilon / iterations
    di_prob      : float, probability of applying DI transform
    di_resize_rate : float, DI resize ratio
    ti_len       : int, max translation in pixels for TI; 0 disables
    npatch       : int, number of patches for PI
    grid_scale   : int, grid scale for PI (img_size / grid_scale = grid_cell_size)
    enable_un    : bool, enable Uniform Noise
    enable_pi    : bool, enable Patch Interaction
    enable_di    : bool, enable Diverse Input
    enable_ti    : bool, enable Translation Invariance
    enable_ni    : bool, enable Nesterov look-ahead
    quiet        : bool, suppress logging
    """
    iterations = int(iterations)
    dev = img_tensor.device
    B, C, H, W = img_tensor.shape
    target_t = torch.tensor([target_class], device=dev).expand(B)
    mu = decay
    ss = step_size if step_size is not None else epsilon / max(iterations, 1)
    img_size = H

    if not quiet:
        components = []
        if enable_un:
            components.append('UN')
        if enable_pi:
            components.append(f'PI(n={npatch},g={grid_scale})')
        if enable_di:
            components.append(f'DI(p={di_prob:.1f},r={di_resize_rate:.2f})')
        if enable_ti:
            components.append(f'TI(len={ti_len})')
        if enable_ni:
            components.append('NI')
        components.append(f'MI(mu={mu:.1f})')
        print(f"[NewBackend] target={target_class}, eps={epsilon:.4f}, "
              f"iter={iterations}, ss={ss:.5f}, "
              f"[{', '.join(components)}]", flush=True)

    adv = img_tensor.clone()
    mom = torch.zeros_like(img_tensor)

    for t in range(iterations):
        adv.requires_grad_(True)

        x = adv
        if enable_ni:
            x = x + mu * ss * mom

        if enable_pi:
            delta = adv - img_tensor
            x = img_tensor + _patch_sample(delta, npatch, grid_scale, img_size)
            if enable_ni:
                x = x + mu * ss * mom

        if enable_un:
            x = x + x.new(x.size()).uniform_(-epsilon, epsilon)
            x = torch.clamp(x, 0.0, 1.0)

        if enable_di:
            x = _di_transform(x, resize_rate=di_resize_rate, prob=di_prob)

        if enable_ti and ti_len > 0:
            tx = (ti_len + 1) / max(x.shape[-1], 1)
            ty = (ti_len + 1) / max(x.shape[-2], 1)
            x = T.RandomAffine(0, translate=(tx, ty))(x)

        logits = model(x)
        loss = -logits.gather(1, target_t.unsqueeze(1)).sum()
        g = torch.autograd.grad(loss, adv)[0].data

        pre_g = F.normalize(g, p=1, dim=(1, 2, 3))
        mom = mu * mom + pre_g
        adv = _update_and_clip(img_tensor, adv, mom, epsilon, ss)

        if not quiet and (t % 20 == 0 or t == iterations - 1):
            with torch.no_grad():
                pred = model(img_tensor + (adv - img_tensor)).argmax(dim=1).item()
                prob = F.softmax(model(adv), dim=1)[0, target_class].item()
            print(f"[NewBackend] iter {t+1}/{iterations} | pred={pred} | "
                  f"p_target={prob:.3f}", flush=True)

    if not quiet:
        with torch.no_grad():
            fp = model(adv).argmax(dim=1).item()
            tag = "SUCCESS" if fp == target_class else "NOT REACHED"
            print(f"[NewBackend] {tag} (pred={fp})", flush=True)

    return adv.detach()
