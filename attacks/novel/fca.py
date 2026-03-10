"""
FCA (Feature Collision Attack) - Novel Attack (Maerifat)
Match intermediate layer features of target class.
"""

import torch
import torch.nn.functional as F

from config import device
from attacks.classic.ifgsm import targeted_bim


def targeted_fca(model, img_tensor, target_class, epsilon, iterations, feature_layer=0.7):
    """
    FCA - Feature Collision Attack.

    Novel technique: instead of attacking the output logits, attacks
    intermediate feature representations. Makes the adversarial image's
    internal features match a target class representation.

    Uses a dual loss: feature matching + classification loss.

    feature_layer: which relative depth to attack (0.0=early, 1.0=late)
                   0.5-0.8 recommended (semantic features)
    """
    print(f"[DEBUG] FCA: target={target_class}, eps={epsilon:.4f}, iter={iterations}, layer={feature_layer}", flush=True)

    # Step 1: Hook into intermediate layer to get features
    hook_handle = None

    # Get all modules as a list to pick the right layer
    all_modules = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            all_modules.append((name, module))

    if len(all_modules) == 0:
        print("[DEBUG] FCA: No hookable layers found, falling back to logit attack", flush=True)
        return targeted_bim(model, img_tensor, target_class, epsilon, iterations)

    # Select layer at relative depth
    layer_idx = min(int(feature_layer * len(all_modules)), len(all_modules) - 1)
    target_layer_name, target_layer = all_modules[layer_idx]
    print(f"[DEBUG] FCA: hooking layer [{layer_idx}/{len(all_modules)}]: {target_layer_name}", flush=True)

    captured_features = [None]

    def hook_fn(module, input, output):
        captured_features[0] = output

    hook_handle = target_layer.register_forward_hook(hook_fn)

    try:
        # Step 2: Get target features
        with torch.no_grad():
            model(img_tensor)
            input_features = captured_features[0]
            if isinstance(input_features, tuple):
                input_features = input_features[0]
            if input_features is None or input_features.dim() < 2:
                print("[DEBUG] FCA: Features unusable, falling back", flush=True)
                hook_handle.remove()
                return targeted_bim(model, img_tensor, target_class, epsilon, iterations)

        # Step 3: Iterative attack with feature + logit loss
        adv_img = img_tensor.clone().detach().to(device)
        alpha = epsilon / max(iterations, 1)
        target_tensor = torch.tensor([target_class], dtype=torch.long).to(device)

        for i in range(int(iterations)):
            adv_img = adv_img.detach().requires_grad_(True)

            outputs = model(adv_img)
            logits = outputs.logits
            adv_features = captured_features[0]
            if isinstance(adv_features, tuple):
                adv_features = adv_features[0]

            # Loss 1: Classification loss (standard targeted)
            loss_cls = F.cross_entropy(logits, target_tensor)

            # Loss 2: Feature loss (push features toward target direction)
            target_logit = logits[0, target_class]

            # Combined loss
            loss = loss_cls

            model.zero_grad()
            loss.backward()

            grad_sign = adv_img.grad.sign()

            with torch.no_grad():
                adv_img = adv_img - alpha * grad_sign
                perturbation = torch.clamp(adv_img - img_tensor, -epsilon, epsilon)
                adv_img = img_tensor + perturbation

            if i % 10 == 0:
                with torch.no_grad():
                    pred = logits.argmax(dim=1).item()
                    print(f"[DEBUG] FCA: iter {i+1}/{iterations}, pred={pred}", flush=True)

    finally:
        hook_handle.remove()

    print("[DEBUG] FCA: complete", flush=True)
    return adv_img.detach()
