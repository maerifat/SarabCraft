"""
StAdv (Spatially Transformed Adversarial) — Xiao et al., ICLR 2018
"Spatially Transformed Adversarial Examples"

Uses spatial/geometric deformations (flow field) instead of additive pixel
perturbations. Generates adversarial examples by optimising a per-pixel
displacement field. Harder to defend with Lp-based defenses.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_stadv(model, img_tensor, target_class, epsilon, iterations,
                   flow_lr=0.005, flow_reg=0.05):
    B, C, H, W = img_tensor.shape
    target = torch.tensor([target_class], dtype=torch.long, device=device)

    flow = torch.zeros(B, 2, H, W, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([flow], lr=flow_lr)

    base_grid = F.affine_grid(
        torch.eye(2, 3, device=device).unsqueeze(0).expand(B, -1, -1),
        img_tensor.size(), align_corners=True
    )

    for _ in range(int(iterations)):
        optimizer.zero_grad()
        model.zero_grad()

        disp = flow.permute(0, 2, 3, 1)
        max_disp = epsilon * 2
        disp_clamped = disp.clamp(-max_disp, max_disp)
        grid = base_grid + disp_clamped

        adv = F.grid_sample(img_tensor, grid, align_corners=True,
                           padding_mode='reflection', mode='bilinear')
        adv = adv.clamp(0., 1.)

        loss_ce = F.cross_entropy(model(adv).logits, target)

        flow_diff_h = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).pow(2).mean()
        flow_diff_w = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).pow(2).mean()
        loss_smooth = flow_reg * (flow_diff_h + flow_diff_w)

        loss = loss_ce + loss_smooth
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        disp = flow.permute(0, 2, 3, 1).clamp(-max_disp, max_disp)
        grid = base_grid + disp
        adv = F.grid_sample(img_tensor, grid, align_corners=True,
                           padding_mode='reflection', mode='bilinear')
        adv = adv.clamp(0., 1.)

    return adv.detach()
