"""
I-FGSM / BIM Attack (Basic Iterative Method)
Iterative version of FGSM for stronger attacks.
"""

import torch
import torch.nn.functional as F

from config import device


def targeted_bim(model, img_tensor, target_class, epsilon, iterations, alpha_mult=1.0):
    """
    Basic Iterative Method (I-FGSM) - iterative version of FGSM.
    Applies small perturbations iteratively for stronger attack.
    """
    print(f"[DEBUG] BIM: starting, target={target_class}, eps={epsilon}, iter={iterations}, alpha_mult={alpha_mult}", flush=True)

    adv_img = img_tensor.clone().detach().to(device)
    alpha = (epsilon / max(iterations, 1)) * alpha_mult
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
            adv_img = torch.max(torch.min(adv_img, img_tensor + epsilon), img_tensor - epsilon)

        if i % 5 == 0:
            print(f"[DEBUG] BIM: iteration {i+1}/{iterations}", flush=True)

    print("[DEBUG] BIM: complete", flush=True)
    return adv_img.detach()
