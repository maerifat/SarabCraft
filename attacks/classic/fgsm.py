"""
Pure FGSM Attack (Single Step)
Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
https://arxiv.org/abs/1412.6572
"""

import torch
import torch.nn.functional as F

from config import device


def targeted_fgsm(model, img_tensor, target_class, epsilon):
    """
    Pure Targeted FGSM - Fast Gradient Sign Method (SINGLE STEP).

    Original FGSM (untargeted): x_adv = x + ε * sign(∇_x L(x, y_true))
    Targeted FGSM:              x_adv = x - ε * sign(∇_x L(x, y_target))

    The targeted version MINIMIZES loss w.r.t. target class,
    pushing the model's prediction TOWARD the target.
    """
    print(f"[DEBUG] Pure FGSM (Targeted): target_class={target_class}, epsilon={epsilon:.4f}", flush=True)

    # Step 1: Prepare input with gradient tracking
    adv_img = img_tensor.clone().detach().to(device)
    adv_img.requires_grad = True

    target = torch.tensor([target_class], dtype=torch.long).to(device)

    # Step 2: Forward pass - get model predictions
    outputs = model(adv_img)
    logits = outputs.logits

    # Get current prediction for debugging
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        orig_pred = logits.argmax(dim=1).item()
        orig_conf = probs[0, orig_pred].item()
        target_conf = probs[0, target_class].item()
        print(f"[DEBUG] Before attack: pred={orig_pred} ({orig_conf:.2%}), target_conf={target_conf:.2%}", flush=True)

    # Step 3: Compute loss w.r.t. TARGET class
    loss = F.cross_entropy(logits, target)
    print(f"[DEBUG] Loss (lower=better for target): {loss.item():.4f}", flush=True)

    # Step 4: Backward pass - compute gradients
    model.zero_grad()
    loss.backward()

    # Step 5: Get the sign of gradients
    data_grad = adv_img.grad.data
    grad_sign = data_grad.sign()

    # Step 6: Create adversarial image
    # SUBTRACT gradient to DECREASE loss (move toward target)
    with torch.no_grad():
        perturbation = epsilon * grad_sign
        adv_img = img_tensor - perturbation

        # Clamp to epsilon-ball around original image
        adv_img = torch.clamp(adv_img, img_tensor - epsilon, img_tensor + epsilon)

    # Debug: Check result
    with torch.no_grad():
        final_outputs = model(adv_img)
        final_probs = F.softmax(final_outputs.logits, dim=1)
        final_pred = final_outputs.logits.argmax(dim=1).item()
        final_conf = final_probs[0, final_pred].item()
        target_conf_after = final_probs[0, target_class].item()
        print(f"[DEBUG] After attack: pred={final_pred} ({final_conf:.2%}), target_conf={target_conf_after:.2%}", flush=True)

        if final_pred == target_class:
            print("[DEBUG] ✅ SUCCESS: Model now predicts target class!", flush=True)
        else:
            print("[DEBUG] ⚠️ Attack changed prediction but not to target. Try higher epsilon.", flush=True)

    return adv_img.detach()
