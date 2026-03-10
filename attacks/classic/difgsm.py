"""
DI-FGSM Attack (Diverse Input FGSM) — Xie et al., CVPR 2019
Applies random resize + pad at each iteration to improve transferability.
"""

import torch
import torch.nn.functional as F
from config import device


def _di_transform(x, resize_lo=0.9):
    """Random resize + pad (Diverse Input transform)."""
    _, _, H, W = x.shape
    rnd = int(H * (resize_lo + torch.rand(1).item() * (1.0 - resize_lo)))
    rescaled = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
    pad_h, pad_w = H - rnd, W - rnd
    top  = torch.randint(0, pad_h + 1, (1,)).item()
    left = torch.randint(0, pad_w + 1, (1,)).item()
    return F.pad(rescaled, (left, pad_w - left, top, pad_h - top))


def targeted_difgsm(model, img_tensor, target_class, epsilon, iterations,
                    decay=1.0, p_di=0.7):
    """Diverse Input FGSM — momentum + random input transform."""
    print(f"[DEBUG] DI-FGSM: starting, target={target_class}, eps={epsilon:.4f}, "
          f"iter={iterations}, p_di={p_di}", flush=True)

    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(iterations, 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for i in range(int(iterations)):
        adv = adv.detach().requires_grad_(True)

        # Apply DI transform with probability p_di
        x_in = _di_transform(adv) if torch.rand(1).item() < p_di else adv
        loss = F.cross_entropy(model(x_in).logits, target)
        model.zero_grad()
        loss.backward()

        grad = adv.grad
        grad_norm = grad / (grad.abs().mean() + 1e-12)
        momentum = decay * momentum + grad_norm

        with torch.no_grad():
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)

        if i % 5 == 0:
            print(f"[DEBUG] DI-FGSM: iter {i+1}/{iterations}", flush=True)

    print("[DEBUG] DI-FGSM: complete", flush=True)
    return adv.detach()
