"""
ACA (Attention-Concentrated Attack) - Novel Attack (Maerifat)
Concentrate perturbation where the model looks.
"""

import torch
import torch.nn.functional as F

from config import device


def targeted_aca(model, img_tensor, target_class, epsilon, iterations, concentration=3.0):
    """
    ACA - Attention-Concentrated Attack.

    Novel technique: uses the model's own gradient-based saliency
    to determine WHERE the model is looking, then concentrates
    the perturbation budget in those regions for maximum impact.

    Instead of uniform epsilon everywhere, attention regions get
    up to concentration× more budget.

    concentration: multiplier for attended regions (1.0-5.0)
                   higher = more focused perturbation
    """
    print(f"[DEBUG] ACA: target={target_class}, eps={epsilon:.4f}, iter={iterations}, concentration={concentration}", flush=True)

    # Step 1: Compute attention/saliency map (where model looks)
    saliency_img = img_tensor.clone().detach().requires_grad_(True)
    outputs = model(saliency_img)
    logits = outputs.logits

    pred_class = logits.argmax(dim=1).item()
    logits[0, pred_class].backward()

    with torch.no_grad():
        saliency = saliency_img.grad.abs()
        saliency_spatial = saliency.mean(dim=1, keepdim=True)

        s_min = saliency_spatial.min()
        s_max = saliency_spatial.max()
        if s_max - s_min > 1e-8:
            attention_map = 1.0 + (concentration - 1.0) * (saliency_spatial - s_min) / (s_max - s_min)
        else:
            attention_map = torch.ones_like(saliency_spatial)

        attention_map = attention_map.expand_as(img_tensor)
        epsilon_map = attention_map * epsilon

        print(f"[DEBUG] ACA: attention range [{attention_map.min():.2f}, {attention_map.max():.2f}]", flush=True)

    # Step 2: Attack with concentrated perturbation
    adv_img = img_tensor.clone().detach().to(device)
    target_tensor = torch.tensor([target_class], dtype=torch.long).to(device)

    for i in range(int(iterations)):
        adv_img = adv_img.detach().requires_grad_(True)

        outputs = model(adv_img)
        logits = outputs.logits
        loss = F.cross_entropy(logits, target_tensor)

        model.zero_grad()
        loss.backward()

        grad_sign = adv_img.grad.sign()

        with torch.no_grad():
            alpha_map = epsilon_map / max(iterations, 1)
            adv_img = adv_img - alpha_map * grad_sign

            perturbation = adv_img - img_tensor
            perturbation = torch.max(torch.min(perturbation, epsilon_map), -epsilon_map)
            adv_img = img_tensor + perturbation

        if i % 5 == 0:
            print(f"[DEBUG] ACA: iteration {i+1}/{iterations}", flush=True)

    print("[DEBUG] ACA: complete", flush=True)
    return adv_img.detach()
