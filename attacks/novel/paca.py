"""
PACA (Perceptually-Adaptive Concentrated Attack) - Novel Attack (Maerifat)
COMBINES: Perceptual Budget + Attention Concentration + Dual Objective
The ultimate novel attack technique.
"""

import torch
import torch.nn.functional as F

from config import device


def targeted_paca(model, img_tensor, target_class, epsilon, iterations,
                  lambda_weight=0.5, texture_scale=2.0, concentration=3.0):
    """
    PACA - Perceptually-Adaptive Concentrated Attack.

    Novel combined technique that fuses three concepts:
    1. PERCEPTUAL BUDGET: Allocate more epsilon to textured areas (invisible)
    2. ATTENTION CONCENTRATION: Focus perturbation where model looks
    3. DUAL OBJECTIVE: Push from original + pull toward target

    The epsilon map is: ε_pixel = base_ε × texture_weight × attention_weight
    The gradient uses dual-objective: ∇(L_target - λ × L_original)

    Parameters:
        lambda_weight: dual-objective balance (0.0-1.0)
        texture_scale: perceptual budget multiplier (1.0-5.0)
        concentration: attention concentration factor (1.0-5.0)
    """
    print(f"[DEBUG] PACA: target={target_class}, eps={epsilon:.4f}, iter={iterations}", flush=True)
    print(f"[DEBUG] PACA: lambda={lambda_weight}, texture={texture_scale}, conc={concentration}", flush=True)

    # ===== COMPONENT 1: Perceptual Sensitivity Map =====
    with torch.no_grad():
        img_gray = img_tensor.mean(dim=1, keepdim=True)
        kernel_size = 7
        padding = kernel_size // 2
        local_mean = F.avg_pool2d(img_gray, kernel_size, stride=1, padding=padding)
        local_sq_mean = F.avg_pool2d(img_gray ** 2, kernel_size, stride=1, padding=padding)
        local_var = (local_sq_mean - local_mean ** 2).clamp(min=0)

        var_min, var_max = local_var.min(), local_var.max()
        if var_max - var_min > 1e-8:
            texture_map = 0.3 + (texture_scale - 0.3) * (local_var - var_min) / (var_max - var_min)
        else:
            texture_map = torch.ones_like(local_var)

    # ===== COMPONENT 2: Attention/Saliency Map =====
    saliency_img = img_tensor.clone().detach().requires_grad_(True)
    outputs = model(saliency_img)
    logits = outputs.logits
    pred_class = logits.argmax(dim=1).item()
    logits[0, pred_class].backward()

    with torch.no_grad():
        saliency = saliency_img.grad.abs().mean(dim=1, keepdim=True)
        s_min, s_max = saliency.min(), saliency.max()
        if s_max - s_min > 1e-8:
            attention_map = 1.0 + (concentration - 1.0) * (saliency - s_min) / (s_max - s_min)
        else:
            attention_map = torch.ones_like(saliency)

    # ===== COMBINE: Adaptive epsilon map =====
    with torch.no_grad():
        combined_map = texture_map * attention_map
        combined_map = combined_map / combined_map.mean()
        epsilon_map = combined_map.expand_as(img_tensor) * epsilon

        print(f"[DEBUG] PACA: adaptive eps range [{epsilon_map.min():.4f}, {epsilon_map.max():.4f}]", flush=True)

    # ===== ATTACK LOOP with Dual Objective =====
    adv_img = img_tensor.clone().detach().to(device)
    target_tensor = torch.tensor([target_class], dtype=torch.long).to(device)
    orig_tensor = torch.tensor([pred_class], dtype=torch.long).to(device)

    for i in range(int(iterations)):
        adv_img = adv_img.detach().requires_grad_(True)

        outputs = model(adv_img)
        logits = outputs.logits

        # DUAL OBJECTIVE: pull to target + push from original
        loss_target = F.cross_entropy(logits, target_tensor)
        loss_original = F.cross_entropy(logits, orig_tensor)
        combined_loss = loss_target - lambda_weight * loss_original

        model.zero_grad()
        combined_loss.backward()

        grad_sign = adv_img.grad.sign()

        with torch.no_grad():
            alpha_map = epsilon_map / max(iterations, 1)
            adv_img = adv_img - alpha_map * grad_sign

            perturbation = adv_img - img_tensor
            perturbation = torch.max(torch.min(perturbation, epsilon_map), -epsilon_map)
            adv_img = img_tensor + perturbation

        if i % 5 == 0:
            with torch.no_grad():
                current_pred = model(adv_img).logits.argmax(dim=1).item()
                print(f"[DEBUG] PACA: iter {i+1}/{iterations}, pred={current_pred}", flush=True)

    print("[DEBUG] PACA: complete", flush=True)
    return adv_img.detach()
