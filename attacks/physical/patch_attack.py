"""
Adversarial Patch — Brown et al., 2017
"Adversarial Patch"
https://arxiv.org/abs/1712.09665

Optimises a universal image patch that causes targeted misclassification
when placed anywhere on an image. Works in the physical world.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_patch(model, img_tensor, target_class, epsilon=None,
                   iterations=500, lr=0.01, patch_ratio=0.1):
    """
    Adversarial Patch: optimise a small patch placed on the image.
    patch_ratio: patch side length as fraction of image side (0.1 = 10%).
    lr: learning rate for patch optimisation.
    """
    x0 = img_tensor.clone().detach().to(device)
    B, C, H, W = x0.shape
    target_t = torch.tensor([target_class], dtype=torch.long, device=device)

    ph = max(int(H * patch_ratio), 4)
    pw = max(int(W * patch_ratio), 4)
    patch = torch.rand(1, C, ph, pw, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([patch], lr=lr)

    best_adv = x0.clone()
    best_loss = float('inf')

    for i in range(int(iterations)):
        optimizer.zero_grad()

        top = torch.randint(0, max(H - ph, 1), (1,)).item()
        left = torch.randint(0, max(W - pw, 1), (1,)).item()

        patched = x0.clone()
        patched[:, :, top:top+ph, left:left+pw] = patch.clamp(0., 1.)

        logits = model(patched).logits
        loss = F.cross_entropy(logits, target_t)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            patch.data.clamp_(0., 1.)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_adv = patched.clone()

            if logits.argmax(dim=1).item() == target_class:
                return patched.detach()

    return best_adv.detach()
