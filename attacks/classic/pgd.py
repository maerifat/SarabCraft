"""
PGD Attack (Projected Gradient Descent)
Strongest first-order attack method with random initialization.
"""

import torch
import torch.nn.functional as F

from config import device


def targeted_pgd(model, img_tensor, target_class, epsilon, iterations, alpha_mult=1.0, random_start=True):
    """
    Projected Gradient Descent attack with random initialization.
    Strongest first-order attack method.
    """
    print(f"[DEBUG] PGD: starting, target={target_class}, eps={epsilon}, iter={iterations}, random_start={random_start}", flush=True)

    adv_img = img_tensor.clone().detach().to(device)

    if random_start:
        random_noise = torch.empty_like(adv_img).uniform_(-epsilon, epsilon)
        adv_img = (adv_img + random_noise).clamp(0., 1.)

    alpha = (epsilon / max(iterations // 4, 1)) * alpha_mult
    target = torch.tensor([target_class], dtype=torch.long).to(device)

    for i in range(int(iterations)):
        adv_img = adv_img.detach().requires_grad_(True)

        outputs = model(adv_img)
        logits = outputs.logits
        loss = F.cross_entropy(logits, target)

        model.zero_grad()
        loss.backward()

        grad_sign = adv_img.grad.sign()

        with torch.no_grad():
            adv_img = adv_img - alpha * grad_sign
            perturbation = torch.clamp(adv_img - img_tensor, -epsilon, epsilon)
            adv_img = img_tensor + perturbation

        if i % 5 == 0:
            print(f"[DEBUG] PGD: iteration {i+1}/{iterations}", flush=True)

    print("[DEBUG] PGD: complete", flush=True)
    return adv_img.detach()
