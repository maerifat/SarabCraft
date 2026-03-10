"""
DOSS (Dual-Objective Single Step) - Novel Attack (Maerifat)
Push from original class + Pull toward target class simultaneously.
"""

import torch
import torch.nn.functional as F

from config import device


def targeted_doss(model, img_tensor, target_class, epsilon, lambda_weight=0.5):
    """
    DOSS - Dual-Objective Single Step Attack.

    Novel technique: combines two gradient objectives in a single step:
    1. MINIMIZE loss for target class (pull toward target)
    2. MAXIMIZE loss for original class (push away from original)

    Combined gradient covers more of the perturbation space.

    Formula: x_adv = x - ε × sign(∇_x [L(x, y_target) - λ × L(x, y_original)])

    lambda_weight: balance between pull (toward target) and push (from original)
                   0.0 = pure targeted FGSM, 1.0 = equal push+pull
    """
    print(f"[DEBUG] DOSS: target={target_class}, eps={epsilon:.4f}, lambda={lambda_weight}", flush=True)

    adv_img = img_tensor.clone().detach().requires_grad_(True)

    # Forward pass
    outputs = model(adv_img)
    logits = outputs.logits

    # Get original class
    with torch.no_grad():
        orig_class = logits.argmax(dim=1).item()

    target_tensor = torch.tensor([target_class], dtype=torch.long).to(device)
    orig_tensor = torch.tensor([orig_class], dtype=torch.long).to(device)

    # Dual objective:
    loss_target = F.cross_entropy(logits, target_tensor)
    loss_original = F.cross_entropy(logits, orig_tensor)

    # Combined: minimize target loss - λ × original loss
    combined_loss = loss_target - lambda_weight * loss_original

    model.zero_grad()
    combined_loss.backward()

    grad_sign = adv_img.grad.data.sign()

    with torch.no_grad():
        adv_img = img_tensor - epsilon * grad_sign
        adv_img = torch.clamp(adv_img, img_tensor - epsilon, img_tensor + epsilon)

    # Debug
    with torch.no_grad():
        final_out = model(adv_img)
        final_pred = final_out.logits.argmax(dim=1).item()
        print(f"[DEBUG] DOSS: orig={orig_class} → pred={final_pred} (target={target_class})", flush=True)

    return adv_img.detach()
