"""
SparseFool — Modas et al., CVPR 2019
"SparseFool: a few pixels make a big difference"
https://arxiv.org/abs/1811.02248

Finds sparse L1 adversarial perturbations by iteratively computing
DeepFool direction and projecting to L1 ball.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_sparsefool(model, img_tensor, target_class, epsilon=None,
                        iterations=100, overshoot=0.02, lambda_=1.0):
    """
    SparseFool: DeepFool direction + L1 projection for sparse perturbations.
    lambda_: controls sparsity (higher = sparser).
    """
    adv = img_tensor.clone().detach().to(device)
    target_t = torch.tensor([target_class], dtype=torch.long, device=device)

    for i in range(int(iterations)):
        adv_r = adv.detach().requires_grad_(True)
        logits = model(adv_r).logits
        pred = logits.argmax(dim=1).item()

        if pred == target_class:
            break

        loss = logits[0, pred] - logits[0, target_class]
        model.zero_grad()
        loss.backward()

        grad = adv_r.grad.detach()
        grad_norm = torch.norm(grad.flatten())
        if grad_norm < 1e-8:
            break

        r = (loss.abs() / (grad_norm ** 2 + 1e-8)) * grad

        with torch.no_grad():
            r_flat = r.view(-1)
            if lambda_ > 0:
                topk = max(1, int(r_flat.numel() * (1.0 / (1.0 + lambda_))))
                _, indices = r_flat.abs().topk(topk)
                mask = torch.zeros_like(r_flat)
                mask[indices] = 1.0
                r_flat = r_flat * mask
                r = r_flat.view(img_tensor.shape)

            adv = adv - (1.0 + overshoot) * r

            if epsilon is not None:
                perturbation = torch.clamp(adv - img_tensor, -epsilon, epsilon)
                adv = img_tensor + perturbation

            adv = adv.clamp(0., 1.)

    return adv.detach()
