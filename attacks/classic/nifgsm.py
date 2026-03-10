"""
NI-FGSM Attack (Nesterov Iterative FGSM) — Lin et al., ICLR 2020
Computes gradient at Nesterov lookahead position for faster convergence.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_nifgsm(model, img_tensor, target_class, epsilon, iterations,
                    decay=1.0):
    """Nesterov Iterative FGSM — gradient at lookahead position."""
    print(f"[DEBUG] NI-FGSM: starting, target={target_class}, eps={epsilon:.4f}, "
          f"iter={iterations}", flush=True)

    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(iterations, 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for i in range(int(iterations)):
        # Nesterov lookahead
        with torch.no_grad():
            nes = adv - alpha * decay * momentum.sign()
            nes_pert = torch.clamp(nes - img_tensor, -epsilon, epsilon)
            nes = img_tensor + nes_pert

        nes = nes.detach().requires_grad_(True)
        loss = F.cross_entropy(model(nes).logits, target)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = nes.grad
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)

        if i % 5 == 0:
            print(f"[DEBUG] NI-FGSM: iter {i+1}/{iterations}", flush=True)

    print("[DEBUG] NI-FGSM: complete", flush=True)
    return adv.detach()
