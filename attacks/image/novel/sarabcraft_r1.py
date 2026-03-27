"""
SarabCraft R1 core transfer attack.
===================================

This module implements the standard SarabCraft R1 transfer recipe:

  CFM + CSA + MI + TI + logit maximisation

It operates in [0,1] pixel space. The router handles model-space
normalisation before and after the attack.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CFMWrapper(nn.Module):
    """
    Records clean features and randomly mixes them during adversarial
    optimisation to reduce overfitting to adversarial-only features.
    """

    def __init__(self, model, mix_prob=0.1, mix_upper=0.75, input_size=224):
        super().__init__()
        self.model = model
        self.mix_prob = mix_prob
        self.mix_upper = mix_upper
        self.divisor = 4
        self.input_size = input_size
        self.recording = False
        self.stored = {}
        self.hooks = []

        layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        self.n = len(layers)
        for i, layer in enumerate(layers):
            self.hooks.append(layer.register_forward_hook(self._hook(i)))

    def _hook(self, idx):
        def fn(mod, inp, out):
            deep = (
                isinstance(mod, nn.Linear)
                or out.dim() < 4
                or out.size(-1) <= self.input_size // self.divisor
            )
            if not deep:
                return
            if idx == self.n - 1 and isinstance(mod, nn.Linear):
                return
            if self.recording:
                self.stored[idx] = out.clone().detach()
                return
            if idx not in self.stored or torch.rand(1).item() > self.mix_prob:
                return
            clean = self.stored[idx].detach()
            shuffled = clean[torch.randperm(out.shape[0])].view(clean.size())
            if out.dim() == 4 and shuffled.shape[2:] != out.shape[2:]:
                shuffled = F.interpolate(shuffled, size=out.shape[2:], mode="bilinear", align_corners=False)
            if out.dim() == 4:
                alpha = (torch.rand(out.shape[0], out.shape[1]) * self.mix_upper).view(out.shape[0], out.shape[1], 1, 1).to(out.device)
            elif out.dim() == 3:
                alpha = (torch.rand(out.shape[0], out.shape[1]) * self.mix_upper).view(out.shape[0], out.shape[1], 1).to(out.device)
            else:
                alpha = (torch.rand(out.shape[0], out.shape[1]) * self.mix_upper).to(out.device)
            return (1 - alpha) * out + alpha * shuffled

        return fn

    def record_clean(self, x):
        self.recording = True
        with torch.no_grad():
            self.model(x)
        self.recording = False

    def forward(self, x):
        return self.model(x)

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.stored.clear()


def _csa_transform(x, sigma):
    """Random resize-pad + translate + scale + flip."""
    _, _, height, width = x.shape

    if torch.rand(1).item() < 0.5 + 0.5 * sigma:
        min_ratio = max(1.0 - 0.3 * sigma, 0.7)
        rnd_ratio = min_ratio + torch.rand(1).item() * (1.0 - min_ratio)
        new_h = max(int(height * rnd_ratio), 1)
        new_w = max(int(width * rnd_ratio), 1)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_h, pad_w = height - new_h, width - new_w
        top = torch.randint(0, max(pad_h, 0) + 1, (1,)).item()
        left = torch.randint(0, max(pad_w, 0) + 1, (1,)).item()
        x = F.pad(x, (left, pad_w - left, top, pad_h - top))

    if torch.rand(1).item() < 0.4 * sigma:
        max_shift = max(int(4 * sigma), 1)
        shift_y = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        shift_x = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        x = torch.roll(x, shifts=(shift_y, shift_x), dims=(2, 3))

    if torch.rand(1).item() < 0.3 * sigma:
        scale_exp = torch.randint(0, 3, (1,)).item()
        if scale_exp > 0:
            x = x / (2.0 ** scale_exp)

    if sigma > 0.5 and torch.rand(1).item() < 0.3 * (sigma - 0.5):
        x = x.flip(dims=[3])

    return x


def _make_ti_kernel(kernel_size=5, nsig=3, device="cpu"):
    """Gaussian TI kernel without scipy."""
    x = np.linspace(-nsig, nsig, kernel_size)
    kernel_1d = np.exp(-0.5 * x ** 2)
    kernel_raw = np.outer(kernel_1d, kernel_1d)
    kernel = kernel_raw / kernel_raw.sum()
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    return torch.from_numpy(stack_kernel.astype(np.float32)).to(device)


def targeted_sarabcraft_r1(
    model,
    img_tensor,
    target_class,
    epsilon,
    iterations,
    decay=1.0,
    kernel_size=5,
    mix_prob=0.1,
    mix_upper=0.75,
    quiet=False,
    ensemble_models=None,
    ensemble_mode="simultaneous",
):
    """
    Standard SarabCraft R1 transfer attack with optional multi-model ensemble.
    """
    iterations = int(iterations)
    kernel_size = int(kernel_size) | 1
    dev = img_tensor.device
    batch_size, _, input_size, _ = img_tensor.shape
    target_t = torch.tensor([target_class], device=dev).expand(batch_size)
    alpha = 2.0 / 255
    sigma = 1.0

    all_models = [model] + (ensemble_models or [])
    n_models = len(all_models)
    alternating = ensemble_mode == "alternating"

    if not quiet:
        ensemble_tag = f", ensemble={n_models} models ({ensemble_mode})" if n_models > 1 else ""
        print(
            f"[SarabCraft-R1] target={target_class}, eps={epsilon:.4f}, "
            f"iter={iterations}, decay={decay}, ti_k={kernel_size}, "
            f"mix_p={mix_prob}, mix_u={mix_upper}{ensemble_tag}, "
            f"[CFM,CSA,MI,TI,LogitLoss]",
            flush=True,
        )

    cfm_wrappers = [
        _CFMWrapper(m, mix_prob=mix_prob, mix_upper=mix_upper, input_size=input_size)
        for m in all_models
    ]
    try:
        ti_kernel = _make_ti_kernel(kernel_size, device=dev)

        for cfm in cfm_wrappers:
            cfm.record_clean(img_tensor)

        delta = torch.zeros_like(img_tensor)
        momentum = 0

        for t in range(iterations):
            x = (img_tensor + delta).detach().requires_grad_(True)
            x_aug = _csa_transform(x, sigma)

            if alternating:
                logits = cfm_wrappers[t % n_models](x_aug)
            else:
                logits = sum(cfm(x_aug) for cfm in cfm_wrappers) / n_models

            loss = logits.gather(1, target_t.unsqueeze(1)).sum()
            g = torch.autograd.grad(loss, x)[0].detach()

            g = F.conv2d(g, ti_kernel, stride=1, padding="same", groups=3)
            g_norm = g / (g.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
            momentum = decay * momentum + g_norm

            delta = delta + alpha * momentum.sign()
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = torch.clamp(img_tensor + delta, 0.0, 1.0) - img_tensor

            if not quiet and (t % 20 == 0 or t == iterations - 1):
                with torch.no_grad():
                    pred = all_models[0](img_tensor + delta).argmax(dim=1).item()
                print(f"[SarabCraft-R1] iter {t+1}/{iterations} | pred={pred}", flush=True)

        if not quiet:
            with torch.no_grad():
                final_pred = all_models[0](img_tensor + delta).argmax(dim=1).item()
                tag = "SUCCESS" if final_pred == target_class else "NOT REACHED"
                print(f"[SarabCraft-R1] {tag} (pred={final_pred})", flush=True)

    finally:
        for cfm in cfm_wrappers:
            cfm.cleanup()

    return (img_tensor + delta).detach()
