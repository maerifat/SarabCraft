"""
Auto-PGD (APGD) — Croce & Hein, ICML 2020
"Reliable evaluation of adversarial robustness with an ensemble of diverse
parameter-free attacks"
https://arxiv.org/abs/2003.01690

Key: automatic step-size schedule + DLR loss. Strictly stronger than PGD.
"""

import torch
import torch.nn.functional as F
from config import device


def _dlr_loss_targeted(logits, target):
    """Difference of Logits Ratio (targeted): minimise z_target - z_max_other."""
    n_cls = logits.shape[1]
    sorted_logits, _ = logits.sort(dim=1, descending=True)
    z1 = sorted_logits[:, 0]
    z3 = sorted_logits[:, min(2, n_cls - 1)]
    denom = (z1 - z3 + 1e-12)
    z_target = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, target.unsqueeze(1), False)
    z_other_max = logits[mask].view(logits.shape[0], -1).max(dim=1).values
    return -((z_other_max - z_target) / denom).sum()


def targeted_apgd(model, img_tensor, target_class, epsilon, iterations,
                  loss_type='dlr', n_restarts=1, rho=0.75):
    """
    Auto-PGD with adaptive step size. Halves step when stalled.
    loss_type: 'ce' (cross-entropy) or 'dlr' (Difference of Logits Ratio).
    """
    best_adv = img_tensor.clone()
    best_loss = float('inf')
    target_t = torch.tensor([target_class], dtype=torch.long, device=device)
    iterations = int(iterations)

    for restart in range(int(n_restarts)):
        delta = torch.empty_like(img_tensor).uniform_(-epsilon, epsilon)
        delta = torch.clamp(img_tensor + delta, 0., 1.) - img_tensor

        eta = 2.0 * epsilon / max(iterations, 1)
        momentum = torch.zeros_like(img_tensor)

        checkpoints = [int(iterations * p) for p in [0.22, 0.5, 0.75]]
        loss_history = []

        for t in range(iterations):
            x = (img_tensor + delta).detach().requires_grad_(True)
            logits = model(x).logits

            if loss_type == 'dlr':
                loss = _dlr_loss_targeted(logits, target_t.expand(x.shape[0]))
            else:
                loss = -F.cross_entropy(logits, target_t)

            grad = torch.autograd.grad(loss, x)[0].detach()
            loss_val = loss.item()
            loss_history.append(loss_val)

            grad_norm = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
            momentum = 0.75 * momentum + grad_norm

            with torch.no_grad():
                delta = delta - eta * momentum.sign()
                delta = torch.clamp(delta, -epsilon, epsilon)
                delta = torch.clamp(img_tensor + delta, 0., 1.) - img_tensor

            if t in checkpoints and len(loss_history) > 5:
                recent = loss_history[-5:]
                earlier = loss_history[-10:-5] if len(loss_history) >= 10 else loss_history[:5]
                if min(recent) >= min(earlier) * rho:
                    eta *= 0.5
                    delta = (best_adv - img_tensor).clone()

            curr_loss = loss_val
            if curr_loss < best_loss:
                best_loss = curr_loss
                best_adv = (img_tensor + delta).clone()

    return best_adv.detach()
