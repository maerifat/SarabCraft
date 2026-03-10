"""
PBA (Perceptual Budget Allocation) - Novel Attack (Maerifat)
Adaptive per-pixel epsilon based on texture/edge complexity.
"""

import torch
import torch.nn.functional as F

from config import device


def targeted_pba(model, img_tensor, target_class, epsilon, iterations, texture_scale=2.0):
    """
    PBA - Perceptual Budget Allocation Attack.

    Novel technique: allocates perturbation budget per-pixel based on
    local image texture/complexity. Textured areas get MORE budget
    (changes are invisible there), flat areas get LESS budget.

    Uses local variance as perceptual sensitivity measure.

    texture_scale: multiplier for texture-based budget (1.0-5.0)
                   higher = more aggressive in textured regions
    """
    print(f"[DEBUG] PBA: target={target_class}, eps={epsilon:.4f}, iter={iterations}, texture_scale={texture_scale}", flush=True)

    # Step 1: Compute perceptual sensitivity map from original image
    with torch.no_grad():
        img_gray = img_tensor.mean(dim=1, keepdim=True)

        kernel_size = 7
        padding = kernel_size // 2
        local_mean = F.avg_pool2d(img_gray, kernel_size, stride=1, padding=padding)
        local_sq_mean = F.avg_pool2d(img_gray ** 2, kernel_size, stride=1, padding=padding)
        local_var = (local_sq_mean - local_mean ** 2).clamp(min=0)

        var_min = local_var.min()
        var_max = local_var.max()
        if var_max - var_min > 1e-8:
            texture_map = 0.3 + (texture_scale - 0.3) * (local_var - var_min) / (var_max - var_min)
        else:
            texture_map = torch.ones_like(local_var)

        epsilon_map = texture_map.expand_as(img_tensor) * epsilon

        print(f"[DEBUG] PBA: epsilon range [{epsilon_map.min():.4f}, {epsilon_map.max():.4f}]", flush=True)

    # Step 2: Iterative attack with adaptive epsilon
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
            print(f"[DEBUG] PBA: iteration {i+1}/{iterations}", flush=True)

    print("[DEBUG] PBA: complete", flush=True)
    return adv_img.detach()
