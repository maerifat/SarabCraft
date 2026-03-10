"""
MI-FGSM Attack (Momentum Iterative FGSM)
Adds momentum to improve transferability across models.
"""

import torch
import torch.nn.functional as F

from config import device


def targeted_mi_fgsm(model, img_tensor, target_class, epsilon, iterations, decay=1.0, alpha_mult=1.0):
    """
    Momentum Iterative FGSM - adds momentum to improve transferability.
    Better for black-box transfer attacks.
    """
    print(f"[DEBUG] MI-FGSM: starting, target={target_class}, eps={epsilon}, iter={iterations}, decay={decay}", flush=True)

    adv_img = img_tensor.clone().detach().to(device)
    alpha = (epsilon / max(iterations, 1)) * alpha_mult
    target = torch.tensor([target_class], dtype=torch.long).to(device)
    momentum = torch.zeros_like(img_tensor).to(device)

    for i in range(int(iterations)):
        adv_img = adv_img.detach().requires_grad_(True)

        outputs = model(adv_img)
        logits = outputs.logits
        loss = F.cross_entropy(logits, target)

        model.zero_grad()
        loss.backward()

        # Normalize gradient
        grad = adv_img.grad
        grad_norm = grad / (torch.norm(grad, p=1) + 1e-12)

        # Accumulate momentum
        momentum = decay * momentum + grad_norm

        with torch.no_grad():
            adv_img = adv_img - alpha * momentum.sign()
            perturbation = torch.clamp(adv_img - img_tensor, -epsilon, epsilon)
            adv_img = img_tensor + perturbation

        if i % 5 == 0:
            print(f"[DEBUG] MI-FGSM: iteration {i+1}/{iterations}", flush=True)

    print("[DEBUG] MI-FGSM: complete", flush=True)
    return adv_img.detach()
