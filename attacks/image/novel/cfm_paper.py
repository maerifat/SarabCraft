"""
CFM Paper Recipe: CFM + RDI + MI + TI + Logit Loss (Byun et al., CVPR 2023)
============================================================================

Exact replication of Config 578 from the CFM paper:
  CFM — Clean Feature Mixup (channelwise, skip last FC, p=0.1, upper=0.75)
  RDI — Resized Diverse Input (resize 340/299, pad, resize back, NEAREST)
  MI  — Momentum Iterative (decay=1.0)
  TI  — Translation-Invariant Gaussian kernel
  Logit Loss — maximise target logit directly

This is the strongest single-model transfer baseline from:
  "Introducing Competition to Boost the Transferability of Targeted
   Adversarial Examples through Clean Feature Mixup"
  Byun et al., CVPR 2023 — https://github.com/dreamflake/CFM

All operations in [0,1] pixel space. Model must accept [0,1] inputs
and return raw logit tensors (not HuggingFace output objects).
The router handles the space conversion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .sarabcraft_r1 import _CFMWrapper, _make_ti_kernel


def _rdi_transform(x):
    """Resized Diverse Input from CFM paper (attacks.py line 430).

    Resize up to 340/299 * img_width, random pad, resize back.
    Uses NEAREST interpolation to match original code exactly.
    """
    img_width = x.size(-1)
    enlarged_img_width = int(img_width * 340.0 / 299.0)
    di_pad_amount = enlarged_img_width - img_width
    ori_size = img_width

    rnd = int(torch.rand(1).item() * di_pad_amount) + ori_size
    x = F.interpolate(x, size=(rnd, rnd), mode='nearest')

    pad_max = ori_size + di_pad_amount - rnd
    pad_left = int(torch.rand(1).item() * pad_max)
    pad_right = pad_max - pad_left
    pad_top = int(torch.rand(1).item() * pad_max)
    pad_bottom = pad_max - pad_top
    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    x = F.interpolate(x, size=(ori_size, ori_size), mode='nearest')
    return x


def targeted_cfm_paper(model, img_tensor, target_class, epsilon, iterations,
                       decay=1.0, kernel_size=5, mix_prob=0.1,
                       mix_upper=0.75, quiet=False,
                       ensemble_models=None,
                       ensemble_mode='simultaneous'):
    """
    CFM paper recipe (Config 578): CFM + RDI + MI + TI + Logit Loss.

    Parameters
    ----------
    model         : pixel-space model returning raw logits
    img_tensor    : [B, 3, H, W] clean image in [0, 1]
    target_class  : int, target class index
    epsilon       : float, Linf budget in [0, 1] scale
    iterations    : int, number of PGD steps
    decay         : float, momentum decay (μ)
    kernel_size   : int, TI Gaussian kernel size (odd)
    mix_prob      : float, per-layer CFM mixing probability
    mix_upper     : float, max blending alpha for CFM
    quiet         : bool, suppress logging
    ensemble_models : list of additional models for ensemble (optional)
    ensemble_mode : 'simultaneous' or 'alternating'
    """
    iterations = int(iterations)
    kernel_size = int(kernel_size) | 1
    dev = img_tensor.device
    B, C, H, W = img_tensor.shape
    target_t = torch.tensor([target_class], device=dev).expand(B)
    alpha = 2.0 / 255
    sigma = 1.0

    all_models = [model] + (ensemble_models or [])
    n_models = len(all_models)
    alternating = ensemble_mode == 'alternating'

    if not quiet:
        ens_tag = (f", ensemble={n_models} ({ensemble_mode})"
                   if n_models > 1 else "")
        print(f"[CFM-Paper] target={target_class}, eps={epsilon:.4f}, "
              f"iter={iterations}, decay={decay}, ti_k={kernel_size}, "
              f"mix_p={mix_prob}, mix_u={mix_upper}{ens_tag}, "
              f"[CFM,RDI,MI,TI,LogitLoss]", flush=True)

    cfm_wrappers = [_CFMWrapper(m, mix_prob=mix_prob, mix_upper=mix_upper,
                                input_size=H)
                    for m in all_models]
    try:
        ti_kernel = _make_ti_kernel(kernel_size, device=dev)

        for cfm in cfm_wrappers:
            cfm.record_clean(img_tensor)

        delta = torch.zeros_like(img_tensor)
        momentum = 0

        for t in range(iterations):
            x = (img_tensor + delta).detach().requires_grad_(True)
            x_aug = _rdi_transform(x)

            if alternating:
                logits = cfm_wrappers[t % n_models](x_aug)
            else:
                logits = sum(cfm(x_aug) for cfm in cfm_wrappers) / n_models

            loss = logits.gather(1, target_t.unsqueeze(1)).sum()
            g = torch.autograd.grad(loss, x)[0].detach()

            g = F.conv2d(g, ti_kernel, stride=1, padding='same', groups=3)
            g_norm = g / (g.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
            momentum = decay * momentum + g_norm

            delta = delta + alpha * momentum.sign()
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = torch.clamp(img_tensor + delta, 0., 1.) - img_tensor

            if not quiet and (t % 20 == 0 or t == iterations - 1):
                with torch.no_grad():
                    pred = all_models[0](
                        img_tensor + delta).argmax(dim=1).item()
                print(f"[CFM-Paper] iter {t+1}/{iterations} | pred={pred}",
                      flush=True)

        if not quiet:
            with torch.no_grad():
                fp = all_models[0](
                    img_tensor + delta).argmax(dim=1).item()
                tag = "SUCCESS" if fp == target_class else "NOT REACHED"
                print(f"[CFM-Paper] {tag} (pred={fp})", flush=True)

    finally:
        for cfm in cfm_wrappers:
            cfm.cleanup()

    return (img_tensor + delta).detach()
