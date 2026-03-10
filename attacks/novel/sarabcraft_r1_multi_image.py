"""
SarabCraft R1 multi-image transfer mode.
=======================================

This module extends the standard SarabCraft R1 recipe with selectable
reference-bank strategies for harder single-image transfer studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sarabcraft_r1 import _CFMWrapper, _csa_transform, _make_ti_kernel


class _CFMBankWrapper(nn.Module):
    def __init__(self, model, mix_prob=0.1, mix_upper=0.75, input_size=224):
        super().__init__()
        self.model = model
        self.mix_prob = mix_prob
        self.mix_upper = mix_upper
        self.divisor = 4
        self.input_size = input_size
        self.recording = False
        self.bank = {}
        self.hooks = []

        layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        self.n = len(layers)
        for i, layer in enumerate(layers):
            self.hooks.append(layer.register_forward_hook(self._hook(i)))

    def _hook(self, idx):
        def fn(mod, inp, out):
            deep = isinstance(mod, nn.Linear) or out.dim() < 4 or out.size(-1) <= self.input_size // self.divisor
            if not deep:
                return
            if idx == self.n - 1 and isinstance(mod, nn.Linear):
                return
            if self.recording:
                if idx not in self.bank:
                    self.bank[idx] = []
                self.bank[idx].append(out.clone().detach())
                return
            if idx not in self.bank or torch.rand(1).item() > self.mix_prob:
                return
            all_feats = self.bank[idx]
            rand_idx = torch.randint(0, all_feats.shape[0], (out.shape[0],))
            sampled = all_feats[rand_idx]
            if out.dim() == 4 and sampled.shape[2:] != out.shape[2:]:
                sampled = F.interpolate(sampled, size=out.shape[2:], mode="bilinear", align_corners=False)
            alpha = (torch.rand(out.shape[0], out.shape[1]) * self.mix_upper).to(out.device)
            if out.dim() == 4:
                alpha = alpha.view(out.shape[0], out.shape[1], 1, 1)
            elif out.dim() == 3:
                alpha = alpha.view(out.shape[0], out.shape[1], 1)
            return (1 - alpha) * out + alpha * sampled

        return fn

    def build_bank(self, reference_images):
        self.bank = {}
        self.recording = True
        with torch.no_grad():
            self.model(reference_images)
        self.recording = False
        for key in self.bank:
            self.bank[key] = torch.cat(self.bank[key], dim=0)

    def forward(self, x):
        return self.model(x)

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.bank.clear()


class _ProgressiveBankWrapper(nn.Module):
    def __init__(self, model, mix_prob=0.1, mix_upper=0.75, input_size=224, total_iters=300):
        super().__init__()
        self.model = model
        self.mix_prob = mix_prob
        self.mix_upper = mix_upper
        self.divisor = 4
        self.input_size = input_size
        self.total_iters = total_iters
        self.cur_iter = 0
        self.recording = False
        self.bank = {}
        self.self_feats = {}
        self.hooks = []

        layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        self.n = len(layers)
        for i, layer in enumerate(layers):
            self.hooks.append(layer.register_forward_hook(self._hook(i)))

    def _hook(self, idx):
        def fn(mod, inp, out):
            deep = isinstance(mod, nn.Linear) or out.dim() < 4 or out.size(-1) <= self.input_size // self.divisor
            if not deep:
                return
            if idx == self.n - 1 and isinstance(mod, nn.Linear):
                return
            if self.recording == "bank":
                if idx not in self.bank:
                    self.bank[idx] = []
                self.bank[idx].append(out.clone().detach())
                return
            if self.recording == "self":
                self.self_feats[idx] = out.clone().detach()
                return
            if idx not in self.bank or torch.rand(1).item() > self.mix_prob:
                return

            progress = self.cur_iter / max(self.total_iters - 1, 1)
            bank_weight = 1.0 - progress

            sampled = None
            if bank_weight > 0.01 and idx in self.bank:
                bank_features = self.bank[idx]
                rand_idx = torch.randint(0, bank_features.shape[0], (out.shape[0],))
                sampled = bank_features[rand_idx]
                if out.dim() == 4 and sampled.shape[2:] != out.shape[2:]:
                    sampled = F.interpolate(sampled, size=out.shape[2:], mode="bilinear", align_corners=False)

            self_features = self.self_feats.get(idx)
            if self_features is not None and self_features.dim() == 4 and self_features.shape[2:] != out.shape[2:]:
                self_features = F.interpolate(self_features, size=out.shape[2:], mode="bilinear", align_corners=False)

            if sampled is not None and self_features is not None:
                mixed_ref = bank_weight * sampled + (1 - bank_weight) * self_features
            elif sampled is not None:
                mixed_ref = sampled
            elif self_features is not None:
                mixed_ref = self_features
            else:
                return

            alpha = (torch.rand(out.shape[0], out.shape[1]) * self.mix_upper).to(out.device)
            if out.dim() == 4:
                alpha = alpha.view(out.shape[0], out.shape[1], 1, 1)
            elif out.dim() == 3:
                alpha = alpha.view(out.shape[0], out.shape[1], 1)
            return (1 - alpha) * out + alpha * mixed_ref

        return fn

    def build_bank(self, ref_imgs):
        self.bank = {}
        self.recording = "bank"
        with torch.no_grad():
            self.model(ref_imgs)
        self.recording = False
        for key in self.bank:
            self.bank[key] = torch.cat(self.bank[key], dim=0)

    def record_self(self, x):
        self.recording = "self"
        with torch.no_grad():
            self.model(x)
        self.recording = False

    def forward(self, x):
        return self.model(x)

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.bank.clear()
        self.self_feats.clear()


def _make_self_aug_bank(src_img, n_copies, input_size):
    """Build a bank from augmented copies of the source image."""
    copies = []
    for i in range(n_copies):
        x = src_img.clone().unsqueeze(0)
        if i % 2 == 1:
            x = x.flip(dims=[3])
        if i >= 2:
            scale = 0.7 + 0.3 * (i / n_copies)
            _, _, height, width = x.shape
            new_h = max(int(height * scale), 1)
            new_w = max(int(width * scale), 1)
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            pad_h, pad_w = height - new_h, width - new_w
            top = torch.randint(0, max(pad_h, 0) + 1, (1,)).item()
            left = torch.randint(0, max(pad_w, 0) + 1, (1,)).item()
            x = F.pad(x, (left, pad_w - left, top, pad_h - top))
        if i >= n_copies // 2:
            noise = torch.randn_like(x) * 0.05
            x = (x + noise).clamp(0, 1)
        copies.append(x.squeeze(0))
    return torch.stack(copies)


def _make_random_ref_images(n_images, input_size, device):
    """Deterministic random images for bank diversity."""
    rng = torch.Generator(device="cpu").manual_seed(42)
    return torch.rand(n_images, 3, input_size, input_size, generator=rng).to(device)


def targeted_sarabcraft_r1_multi_image(
    model,
    img_tensor,
    target_class,
    epsilon,
    iterations,
    decay=1.0,
    kernel_size=5,
    mix_prob=0.1,
    mix_upper=0.75,
    cfm_strategy="tile_shuffle",
    n_images=10,
    quiet=False,
    ensemble_models=None,
    ensemble_mode="simultaneous",
):
    """
    SarabCraft R1 with selectable multi-image transfer strategies.
    """
    iterations = int(iterations)
    kernel_size = int(kernel_size) | 1
    n_images = max(int(n_images), 2)
    dev = img_tensor.device
    batch_size, _, height, _ = img_tensor.shape
    target_t = torch.tensor([target_class], device=dev).expand(batch_size)
    alpha = 2.0 / 255
    sigma = 1.0

    all_models = [model] + (ensemble_models or [])
    n_models = len(all_models)
    alternating = ensemble_mode == "alternating"

    if not quiet:
        ensemble_tag = f", ensemble={n_models} ({ensemble_mode})" if n_models > 1 else ""
        print(
            f"[SarabCraft-R1-MI] target={target_class}, eps={epsilon:.4f}, "
            f"iter={iterations}, strategy={cfm_strategy}, n_images={n_images}, "
            f"mix_p={mix_prob}, mix_u={mix_upper}{ensemble_tag}",
            flush=True,
        )

    if cfm_strategy == "tile_shuffle":
        return _run_tile_shuffle(
            all_models,
            img_tensor,
            target_class,
            epsilon,
            iterations,
            decay,
            kernel_size,
            mix_prob,
            mix_upper,
            n_images,
            alternating,
            quiet,
        )

    wrappers = _build_wrappers(
        all_models,
        img_tensor,
        cfm_strategy,
        n_images,
        mix_prob,
        mix_upper,
        height,
        iterations,
        dev,
    )

    try:
        ti_kernel = _make_ti_kernel(kernel_size, device=dev)
        delta = torch.zeros_like(img_tensor)
        momentum = 0

        for t in range(iterations):
            for wrapper in wrappers:
                if hasattr(wrapper, "cur_iter"):
                    wrapper.cur_iter = t

            x = (img_tensor + delta).detach().requires_grad_(True)
            x_aug = _csa_transform(x, sigma)

            if alternating:
                logits = wrappers[t % n_models](x_aug)
            else:
                logits = sum(wrapper(x_aug) for wrapper in wrappers) / n_models

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
                    pred = all_models[0](img_tensor + delta).argmax(1).item()
                print(f"[SarabCraft-R1-MI] iter {t+1}/{iterations} | pred={pred}", flush=True)

        if not quiet:
            with torch.no_grad():
                final_pred = all_models[0](img_tensor + delta).argmax(1).item()
                tag = "SUCCESS" if final_pred == target_class else "NOT REACHED"
                print(f"[SarabCraft-R1-MI] {tag} (pred={final_pred})", flush=True)

    finally:
        for wrapper in wrappers:
            wrapper.cleanup()

    return (img_tensor + delta).detach()


def _build_wrappers(all_models, img_tensor, strategy, n_images, mix_prob, mix_upper, input_size, iterations, device):
    wrappers = []

    if strategy == "self_mix":
        for model in all_models:
            wrapper = _CFMWrapper(model, mix_prob=mix_prob, mix_upper=mix_upper, input_size=input_size)
            wrapper.record_clean(img_tensor)
            wrappers.append(wrapper)

    elif strategy == "self_aug_bank":
        aug_bank = _make_self_aug_bank(img_tensor[0], n_images, input_size).to(device)
        for model in all_models:
            wrapper = _CFMBankWrapper(model, mix_prob=mix_prob, mix_upper=mix_upper, input_size=input_size)
            wrapper.build_bank(aug_bank)
            wrappers.append(wrapper)

    elif strategy == "progressive":
        rand_ref = _make_random_ref_images(n_images, input_size, device)
        for model in all_models:
            wrapper = _ProgressiveBankWrapper(
                model,
                mix_prob=mix_prob,
                mix_upper=mix_upper,
                input_size=input_size,
                total_iters=iterations,
            )
            wrapper.build_bank(rand_ref)
            wrapper.record_self(img_tensor)
            wrappers.append(wrapper)

    else:
        raise ValueError(f"Unknown CFM strategy: {strategy}")

    return wrappers


def _run_tile_shuffle(
    all_models,
    img_tensor,
    target_class,
    epsilon,
    iterations,
    decay,
    kernel_size,
    mix_prob,
    mix_upper,
    n_tiles,
    alternating,
    quiet,
):
    """Tile the source image into a pseudo-batch and run CFM across tiles."""
    dev = img_tensor.device
    n_models = len(all_models)
    alpha = 2.0 / 255
    sigma = 1.0

    batch = img_tensor.repeat(n_tiles, 1, 1, 1)
    for i in range(1, n_tiles):
        batch[i] = _csa_transform(batch[i:i + 1], sigma).squeeze(0)

    tile_targets = torch.full((n_tiles,), target_class, dtype=torch.long, device=dev)

    wrappers = []
    for model in all_models:
        wrapper = _CFMWrapper(model, mix_prob=mix_prob, mix_upper=mix_upper, input_size=img_tensor.shape[-1])
        wrapper.record_clean(batch)
        wrappers.append(wrapper)

    try:
        ti_kernel = _make_ti_kernel(kernel_size, device=dev)
        delta = torch.zeros_like(batch)
        momentum = 0

        for t in range(iterations):
            x = (batch + delta).detach().requires_grad_(True)
            x_aug = _csa_transform(x, sigma)

            if alternating:
                logits = wrappers[t % n_models](x_aug)
            else:
                logits = sum(wrapper(x_aug) for wrapper in wrappers) / n_models

            loss = logits.gather(1, tile_targets.unsqueeze(1)).sum()
            g = torch.autograd.grad(loss, x)[0].detach()

            g = F.conv2d(g, ti_kernel, stride=1, padding="same", groups=3)
            g_norm = g / (g.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
            momentum = decay * momentum + g_norm

            delta = delta + alpha * momentum.sign()
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = torch.clamp(batch + delta, 0.0, 1.0) - batch

            if not quiet and (t % 20 == 0 or t == iterations - 1):
                with torch.no_grad():
                    pred = all_models[0](batch + delta).argmax(1)[0].item()
                print(f"[SarabCraft-R1-MI|tile] iter {t+1}/{iterations} | pred={pred}", flush=True)

        if not quiet:
            with torch.no_grad():
                final_pred = all_models[0](batch + delta).argmax(1)[0].item()
                tag = "SUCCESS" if final_pred == target_class else "NOT REACHED"
                print(f"[SarabCraft-R1-MI|tile] {tag} (pred={final_pred})", flush=True)

    finally:
        for wrapper in wrappers:
            wrapper.cleanup()

    return (batch[0:1] + delta[0:1]).detach()
