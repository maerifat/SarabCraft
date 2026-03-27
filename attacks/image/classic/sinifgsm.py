"""
SI-NI-FGSM Attack (Scale-Invariant Nesterov FGSM) — Lin et al., ICLR 2020
Averages gradients at multiple scales + Nesterov momentum.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_sinifgsm(model, img_tensor, target_class, epsilon, iterations,
                      decay=1.0, n_scale=5):
    """Scale-Invariant NI-FGSM — average gradient across n_scale copies."""
    print(f"[DEBUG] SI-NI-FGSM: starting, target={target_class}, eps={epsilon:.4f}, "
          f"iter={iterations}, scales={n_scale}", flush=True)

    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(iterations, 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for i in range(int(iterations)):
        # Nesterov lookahead
        with torch.no_grad():
            nes = adv - alpha * decay * momentum.sign()
            nes = img_tensor + torch.clamp(nes - img_tensor, -epsilon, epsilon)

        # Average gradients over n_scale copies: x / 2^s, s=0..n_scale-1
        g_sum = torch.zeros_like(img_tensor, device=device)
        for s in range(n_scale):
            x_s = (nes / (2 ** s)).detach().requires_grad_(True)
            loss = F.cross_entropy(model(x_s).logits, target)
            model.zero_grad()
            loss.backward()
            g_sum = g_sum + x_s.grad.detach()

        with torch.no_grad():
            grad = g_sum / n_scale
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)

        if i % 5 == 0:
            print(f"[DEBUG] SI-NI-FGSM: iter {i+1}/{iterations}", flush=True)

    print("[DEBUG] SI-NI-FGSM: complete", flush=True)
    return adv.detach()
