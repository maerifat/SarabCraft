"""
C&W Attack (Carlini-Wagner L2)
Optimization-based attack using Adam optimizer.
"""

import torch
import torch.nn.functional as F

from config import device


def targeted_cw(model, img_tensor, target_class, epsilon, iterations, c=1.0, lr=0.01, kappa=0.0):
    """
    Carlini-Wagner L2 attack - optimization-based attack.
    Highly effective but slower. Uses Adam optimizer.
    kappa (confidence): how confident the adversarial example should be
    """
    print(f"[DEBUG] C&W: starting, target={target_class}, c={c}, lr={lr}, kappa={kappa}, iter={iterations}", flush=True)

    # Initialize perturbation in tanh space for box constraints
    w = torch.zeros_like(img_tensor, requires_grad=True, device=device)
    target = torch.tensor([target_class], dtype=torch.long).to(device)

    optimizer = torch.optim.Adam([w], lr=lr)

    best_adv = img_tensor.clone()
    best_l2 = float('inf')

    for i in range(int(iterations)):
        optimizer.zero_grad()

        # Transform w to valid image space using tanh
        adv_img = img_tensor + torch.tanh(w) * epsilon

        outputs = model(adv_img)
        logits = outputs.logits

        # Get target class logit and max other logit
        target_logit = logits[0, target_class]

        # Mask out target class to find max other
        logits_masked = logits.clone()
        logits_masked[0, target_class] = -float('inf')
        max_other_logit = logits_masked.max()

        # f(x) = max(max_other - target + kappa, 0)
        f_loss = torch.clamp(max_other_logit - target_logit + kappa, min=0)

        # L2 distance
        l2_dist = torch.norm(adv_img - img_tensor)

        # Combined loss
        loss = l2_dist + c * f_loss

        loss.backward()
        optimizer.step()

        # Track best adversarial example
        with torch.no_grad():
            if f_loss.item() == 0 and l2_dist.item() < best_l2:
                best_l2 = l2_dist.item()
                best_adv = adv_img.clone()

        if i % 20 == 0:
            print(f"[DEBUG] C&W: iter {i+1}/{iterations}, loss={loss.item():.4f}, f={f_loss.item():.4f}", flush=True)

    # Return best or final
    if best_l2 < float('inf'):
        print(f"[DEBUG] C&W: complete, best L2={best_l2:.4f}", flush=True)
        return best_adv.detach()
    else:
        print("[DEBUG] C&W: complete (no successful example found)", flush=True)
        return (img_tensor + torch.tanh(w) * epsilon).detach()
