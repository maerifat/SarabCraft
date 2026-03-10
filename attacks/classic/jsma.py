"""
JSMA (Jacobian-based Saliency Map Attack) — Papernot et al., 2016
"The Limitations of Deep Learning in Adversarial Settings"
https://arxiv.org/abs/1511.07528

L0 attack: modifies the fewest pixels by using the Jacobian matrix to
identify the most influential pixel pairs.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_jsma(model, img_tensor, target_class, epsilon=None,
                  iterations=None, theta=1.0, gamma=0.1):
    """
    JSMA: iteratively perturb highest-saliency pixel pairs.
    theta: perturbation magnitude per pixel step (+1 or -1 scaled).
    gamma: max fraction of pixels to modify (e.g. 0.1 = 10%).
    """
    x = img_tensor.clone().detach().to(device)
    n_features = x[0].numel()
    max_iters = int(n_features * gamma)

    search_domain = torch.ones(n_features, dtype=torch.bool, device=device)

    for it in range(max_iters):
        x_var = x.detach().requires_grad_(True)
        logits = model(x_var).logits

        pred = logits.argmax(dim=1).item()
        if pred == target_class:
            break

        n_cls = logits.shape[1]
        jacobian = torch.zeros(n_cls, n_features, device=device)
        for c in range(n_cls):
            if x_var.grad is not None:
                x_var.grad.zero_()
            logits[0, c].backward(retain_graph=(c < n_cls - 1))
            jacobian[c] = x_var.grad.detach().view(-1).clone()

        dt = jacobian[target_class]
        do = jacobian.sum(dim=0) - dt

        valid = search_domain.clone()
        if theta > 0:
            valid &= (x.view(-1) < 1.0 - 1e-6)
        else:
            valid &= (x.view(-1) > 1e-6)

        alpha = dt * valid.float()
        beta = do * valid.float()

        if theta > 0:
            saliency = (alpha > 0).float() * (beta < 0).float() * alpha.abs() * beta.abs()
        else:
            saliency = (alpha < 0).float() * (beta > 0).float() * alpha.abs() * beta.abs()

        if saliency.max() == 0:
            break

        best_idx = saliency.argmax().item()

        with torch.no_grad():
            flat = x.view(-1)
            flat[best_idx] = torch.clamp(flat[best_idx] + theta, 0., 1.)
            x = flat.view(img_tensor.shape)

        search_domain[best_idx] = False

    return x.detach()
