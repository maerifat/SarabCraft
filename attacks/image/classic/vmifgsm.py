"""
VMI-FGSM Attack (Variance-Tuned MI-FGSM) — Wang & He, CVPR 2021
Adds gradient variance from neighbourhood samples to momentum.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_vmifgsm(model, img_tensor, target_class, epsilon, iterations,
                     decay=1.0, n_var=20, beta_var=1.5):
    """Variance-Tuned MI-FGSM — neighbourhood variance improves gradient estimation."""
    print(f"[DEBUG] VMI-FGSM: starting, target={target_class}, eps={epsilon:.4f}, "
          f"iter={iterations}, n_var={n_var}", flush=True)

    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(iterations, 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)
    variance = torch.zeros_like(img_tensor, device=device)

    for i in range(int(iterations)):
        adv = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv).logits, target)
        model.zero_grad()
        loss.backward()
        cur_grad = adv.grad.detach().clone()

        # Neighbourhood variance estimation (needs grad — outside no_grad)
        if i > 0:
            g_nb = torch.zeros_like(img_tensor, device=device)
            for _ in range(n_var):
                xn = (adv.detach() + torch.randn_like(img_tensor) * beta_var * epsilon)
                xn = xn.requires_grad_(True)
                loss_nb = F.cross_entropy(model(xn).logits, target)
                model.zero_grad()
                loss_nb.backward()
                g_nb = g_nb + xn.grad.detach()
            variance = g_nb / n_var - cur_grad

        with torch.no_grad():
            grad = cur_grad + variance
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)

        if i % 5 == 0:
            print(f"[DEBUG] VMI-FGSM: iter {i+1}/{iterations}", flush=True)

    print("[DEBUG] VMI-FGSM: complete", flush=True)
    return adv.detach()
