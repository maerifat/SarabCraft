"""
DRA (Direction-Aggregated Attack) — Huang et al., 2023
"Boosting Adversarial Transferability by Aggregating Gradient Directions"

Aggregates gradient directions from multiple augmented views to prevent
white-box overfitting. Uses cosine-similarity-weighted aggregation to
retain only the transferable gradient components. 94.6% ASR on
adversarially trained models.
"""

import torch
import torch.nn.functional as F
from config import device


def _augment_view(x):
    """Generate a random augmented view for direction aggregation."""
    B, C, H, W = x.shape
    transforms = []
    if torch.rand(1).item() > 0.5:
        ratio = 0.9 + torch.rand(1).item() * 0.2
        new_h = int(H * ratio)
        new_w = int(W * ratio)
        x_r = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        x_r = F.interpolate(x_r, size=(H, W), mode='bilinear', align_corners=False)
        transforms.append(x_r)
    if torch.rand(1).item() > 0.5:
        noise = torch.randn_like(x) * 0.02
        transforms.append((x + noise).clamp(0., 1.))
    if torch.rand(1).item() > 0.5:
        transforms.append(torch.flip(x, dims=[3]))

    return transforms[0] if transforms else x


def targeted_dra(model, img_tensor, target_class, epsilon, iterations,
                 decay=1.0, n_views=5):
    adv = img_tensor.clone().detach().to(device)
    alpha = epsilon / max(int(iterations), 1)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    momentum = torch.zeros_like(img_tensor, device=device)

    for _ in range(int(iterations)):
        directions = []

        for _ in range(n_views):
            x_aug = _augment_view(adv.detach())
            x_aug = x_aug.requires_grad_(True)
            loss = F.cross_entropy(model(x_aug).logits, target)
            model.zero_grad()
            loss.backward()
            g = x_aug.grad.detach()
            g_flat = g.reshape(1, -1)
            directions.append(g_flat)

        with torch.no_grad():
            d_stack = torch.cat(directions, dim=0)
            d_norm = d_stack / (d_stack.norm(dim=-1, keepdim=True) + 1e-12)
            cos_matrix = d_norm @ d_norm.T
            weights = cos_matrix.mean(dim=1)
            weights = F.softmax(weights, dim=0)

            agg = (weights.unsqueeze(-1) * d_stack).sum(dim=0)
            grad = agg.reshape_as(img_tensor)
            grad_norm = grad / (grad.abs().mean() + 1e-12)
            momentum = decay * momentum + grad_norm
            adv = adv - alpha * momentum.sign()
            adv = img_tensor + torch.clamp(adv - img_tensor, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
