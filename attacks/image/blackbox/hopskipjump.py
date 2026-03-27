"""
HopSkipJumpAttack — Chen et al., 2020
"HopSkipJumpAttack: A Query-Efficient Decision-Based Attack"
https://arxiv.org/abs/1904.02144

Improved boundary attack with gradient-direction estimation at boundary
using binary search + Monte Carlo sampling. Decision-based black-box.
"""

import torch
import numpy as np
from config import device


def _binary_search_boundary(model, x0, adv, target_class, tol=1e-3, max_steps=20):
    """Binary search to find point on decision boundary."""
    lo, hi = 0.0, 1.0
    for _ in range(max_steps):
        mid = (lo + hi) / 2
        candidate = (1 - mid) * adv + mid * x0
        with torch.no_grad():
            pred = model(candidate.clamp(0., 1.)).logits.argmax(dim=1).item()
        if pred == target_class:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    boundary = (1 - lo) * adv + lo * x0
    return boundary.clamp(0., 1.)


def _estimate_gradient(model, boundary_pt, target_class, n_samples=100, sigma=0.01):
    """Estimate gradient direction at boundary via Monte Carlo."""
    shape = boundary_pt.shape
    n_dim = boundary_pt.numel()
    grad_est = torch.zeros_like(boundary_pt)

    for i in range(0, n_samples, 10):
        bs = min(10, n_samples - i)
        noise = torch.randn(bs, *shape[1:], device=device) * sigma
        x_plus = (boundary_pt + noise).clamp(0., 1.)

        with torch.no_grad():
            logits = model(x_plus).logits
            preds = logits.argmax(dim=1)

        for j in range(bs):
            if preds[j].item() == target_class:
                grad_est = grad_est + noise[j:j+1]
            else:
                grad_est = grad_est - noise[j:j+1]

    return grad_est / (grad_est.norm() + 1e-12)


def targeted_hopskipjump(model, img_tensor, target_class, epsilon=None,
                         iterations=50, initial_num_evals=100,
                         max_num_evals=1000, gamma=1.0):
    """
    HopSkipJump: boundary + gradient estimation for query-efficient attack.
    initial_num_evals: samples for initial gradient estimation.
    max_num_evals: max samples per gradient estimation step.
    gamma: step size scaling.
    """
    x0 = img_tensor.clone().detach().to(device)

    adv = torch.rand_like(x0)
    with torch.no_grad():
        for trial in range(1000):
            logits = model(adv).logits
            if logits.argmax(dim=1).item() == target_class:
                break
            adv = torch.rand_like(x0)
        else:
            return x0.clone()

    for t in range(int(iterations)):
        boundary_pt = _binary_search_boundary(model, x0, adv, target_class)

        n_evals = min(int(initial_num_evals * (t + 1) ** 0.5), max_num_evals)
        dist = torch.norm(boundary_pt - x0)
        sigma = dist.item() * 0.01 / max(t + 1, 1) ** 0.5

        grad_dir = _estimate_gradient(model, boundary_pt, target_class,
                                      n_samples=n_evals, sigma=max(sigma, 1e-4))

        step_size = gamma * dist.item() / max(t + 1, 1) ** 0.5
        candidate = boundary_pt + step_size * grad_dir
        candidate = candidate.clamp(0., 1.)

        if epsilon is not None:
            candidate = x0 + torch.clamp(candidate - x0, -epsilon, epsilon)

        with torch.no_grad():
            pred = model(candidate).logits.argmax(dim=1).item()

        if pred == target_class:
            adv = candidate
        else:
            adv = boundary_pt

    adv = _binary_search_boundary(model, x0, adv, target_class)
    return adv.detach()
