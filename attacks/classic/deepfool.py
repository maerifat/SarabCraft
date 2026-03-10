"""
DeepFool Attack (Minimal Perturbation)
Finds minimal perturbation to cross the decision boundary.
"""

import torch
import torch.nn.functional as F

from config import device


def targeted_deepfool(model, img_tensor, target_class, epsilon, iterations, overshoot=0.02):
    """
    DeepFool attack - finds minimal perturbation to cross decision boundary.
    Epsilon is used as max perturbation limit.
    """
    print(f"[DEBUG] DeepFool: starting, target={target_class}, max_eps={epsilon}, iter={iterations}, overshoot={overshoot}", flush=True)

    adv_img = img_tensor.clone().detach().to(device)
    target = torch.tensor([target_class], dtype=torch.long).to(device)

    for i in range(int(iterations)):
        adv_img = adv_img.detach().requires_grad_(True)

        outputs = model(adv_img)
        logits = outputs.logits

        # Check if already classified as target
        pred = logits.argmax(dim=1)
        if pred.item() == target_class:
            print(f"[DEBUG] DeepFool: success at iteration {i+1}", flush=True)
            break

        # Get current class logit and target class logit
        current_class = pred.item()

        # Compute gradient of difference
        loss = logits[0, current_class] - logits[0, target_class]
        model.zero_grad()
        loss.backward()

        grad = adv_img.grad
        grad_norm = torch.norm(grad.flatten())

        if grad_norm > 1e-8:
            # Compute perturbation
            r = (loss.abs() / (grad_norm ** 2 + 1e-8)) * grad

            with torch.no_grad():
                adv_img = adv_img - (1.0 + overshoot) * r
                # Clip to epsilon ball
                perturbation = torch.clamp(adv_img - img_tensor, -epsilon, epsilon)
                adv_img = img_tensor + perturbation

        if i % 10 == 0:
            print(f"[DEBUG] DeepFool: iteration {i+1}/{iterations}", flush=True)

    print("[DEBUG] DeepFool: complete", flush=True)
    return adv_img.detach()
