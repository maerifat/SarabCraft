"""
NAA (Neuron Attribution-based Attack) — Zhang et al., CVPR 2022
"Improving Adversarial Transferability via Neuron Attribution-Based Attacks"

Uses path-integrated gradients (from zero baseline to current input) to
compute neuron-level attribution at an intermediate layer, then weights the
attack gradient by these attributions for more transferable perturbations.
"""

import torch
import torch.nn.functional as F
from config import device


def _get_mid_layer(model):
    hf = model.hf_model if hasattr(model, 'hf_model') else model
    base = getattr(hf, next((n for n in dir(hf) if n in
        ('vit', 'swin', 'convnext', 'resnet', 'model', 'base_model')), 'model'), hf)

    if hasattr(base, 'encoder') and hasattr(base.encoder, 'layers'):
        layers = base.encoder.layers
        return layers[len(layers) // 2]
    if hasattr(base, 'encoder') and hasattr(base.encoder, 'layer'):
        layers = base.encoder.layer
        return layers[len(layers) // 2]
    if hasattr(base, 'stages'):
        return base.stages[len(base.stages) // 2]
    if hasattr(base, 'features'):
        return base.features[len(base.features) // 2]
    return None


def targeted_naa(model, img_tensor, target_class, epsilon, iterations,
                 decay=1.0, n_ig_steps=10):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    mid_layer = _get_mid_layer(model)
    if mid_layer is None:
        return _fallback(model, img_tensor, target_class, epsilon, iterations, decay)

    feats = {}
    def hook_fn(module, inp, out):
        feats['out'] = out

    handle = mid_layer.register_forward_hook(hook_fn)

    # Compute neuron attribution via integrated gradients (zero baseline)
    attr_sum = None
    baseline = torch.zeros_like(adv, device=device)
    for k in range(1, n_ig_steps + 1):
        x_ig = (baseline + (k / n_ig_steps) * (adv.detach() - baseline)).requires_grad_(True)
        _ = model(x_ig)
        feat = feats.get('out')
        if feat is None:
            break
        if isinstance(feat, tuple):
            feat = feat[0]

        feat_sum = feat.sum()
        model.zero_grad()
        feat_sum.backward()
        if x_ig.grad is not None:
            if attr_sum is None:
                attr_sum = x_ig.grad.detach().clone()
            else:
                attr_sum = attr_sum + x_ig.grad.detach()

    attribution = attr_sum / n_ig_steps if attr_sum is not None else torch.ones_like(adv)
    handle.remove()

    # Re-hook for attack phase
    handle2 = mid_layer.register_forward_hook(hook_fn)

    for i in range(int(iterations)):
        adv_var = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv_var).logits, target)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = adv_var.grad.detach()
            grad = grad * attribution.sign()
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    handle2.remove()
    return adv.detach()


def _fallback(model, img_tensor, target_class, epsilon, iterations, decay):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)
    for _ in range(int(iterations)):
        adv_var = adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adv_var).logits, target)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            grad = adv_var.grad / (adv_var.grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)
    return adv.detach()
