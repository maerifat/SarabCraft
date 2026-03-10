"""
Attack router: dispatches to the correct attack function.

ALL attacks are automatically converted to operate in [0,1] pixel space.
The router handles:
  1. Denormalising HuggingFace input -> [0,1] pixels
  2. Wrapping the model so it normalises internally
  3. Running the attack in pixel space
  4. Re-normalising the result for the HuggingFace display pipeline
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

from attacks.classic.fgsm import targeted_fgsm
from attacks.classic.ifgsm import targeted_bim
from attacks.classic.pgd import targeted_pgd
from attacks.classic.apgd import targeted_apgd
from attacks.classic.mi_fgsm import targeted_mi_fgsm
from attacks.classic.difgsm import targeted_difgsm
from attacks.classic.tifgsm import targeted_tifgsm
from attacks.classic.nifgsm import targeted_nifgsm
from attacks.classic.sinifgsm import targeted_sinifgsm
from attacks.classic.vmifgsm import targeted_vmifgsm
from attacks.classic.vnifgsm import targeted_vnifgsm
from attacks.classic.pifgsm import targeted_pifgsm
from attacks.classic.jitter import targeted_jitter
from attacks.classic.deepfool import targeted_deepfool
from attacks.classic.cw import targeted_cw
from attacks.classic.fab import targeted_fab
from attacks.classic.jsma import targeted_jsma
from attacks.classic.ead import targeted_ead
from attacks.classic.sparsefool import targeted_sparsefool
from attacks.transfer.ssa import targeted_ssa
from attacks.transfer.admix import targeted_admix
from attacks.transfer.bsr import targeted_bsr
from attacks.blackbox.square import targeted_square
from attacks.blackbox.spsa import targeted_spsa
from attacks.blackbox.onepixel import targeted_onepixel
from attacks.blackbox.boundary import targeted_boundary
from attacks.blackbox.hopskipjump import targeted_hopskipjump
from attacks.physical.patch_attack import targeted_patch
from attacks.physical.uap import targeted_uap
from attacks.meta.autoattack import targeted_autoattack
from attacks.novel.sarabcraft_r1 import targeted_sarabcraft_r1
from attacks.novel.sarabcraft_r1_multi_image import targeted_sarabcraft_r1_multi_image
from attacks.novel.cfm_paper import targeted_cfm_paper
from attacks.novel.newbackend import targeted_newbackend
from utils.attack_names import SARABCRAFT_R1_NAME


# ── Normalisation constants ──────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Model wrappers (pixel-space → normalised-space) ─────────────────────────

class _PixelModelWrapper(nn.Module):
    """[0,1] pixel input → raw logit tensor. Used by SarabCraft R1."""

    def __init__(self, hf_model, mean, std):
        super().__init__()
        self.hf_model = hf_model
        self._mean_vals = mean
        self._std_vals = std

    def forward(self, x):
        mean = torch.tensor(self._mean_vals, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self._std_vals, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return self.hf_model((x - mean) / std).logits


class _PixelModelWrapperHF(nn.Module):
    """[0,1] pixel input → HuggingFace output (with .logits attribute)."""

    def __init__(self, hf_model, mean, std):
        super().__init__()
        self.hf_model = hf_model
        self._mean_vals = mean
        self._std_vals = std

    def forward(self, x):
        mean = torch.tensor(self._mean_vals, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self._std_vals, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return self.hf_model((x - mean) / std)


class _EnsembleLogitOutput:
    __slots__ = ('logits',)
    def __init__(self, logits):
        self.logits = logits


class _EnsemblePixelModelWrapperHF(nn.Module):
    def __init__(self, hf_models, mean, std):
        super().__init__()
        self.wrappers = nn.ModuleList(
            [_PixelModelWrapperHF(m, mean, std) for m in hf_models]
        )

    def forward(self, x):
        all_logits = [w(x).logits for w in self.wrappers]
        return _EnsembleLogitOutput(sum(all_logits) / len(all_logits))


class AttackCancelledError(RuntimeError):
    """Raised when a running attack should stop due to job cancellation."""


class _CancelableModelWrapper(nn.Module):
    def __init__(self, wrapped_model, should_cancel):
        super().__init__()
        self.wrapped_model = wrapped_model
        self.should_cancel = should_cancel

    def forward(self, *args, **kwargs):
        if self.should_cancel and self.should_cancel():
            raise AttackCancelledError("Attack cancelled")
        return self.wrapped_model(*args, **kwargs)


def _wrap_cancelable(model, should_cancel):
    if not should_cancel:
        return model
    return _CancelableModelWrapper(model, should_cancel)


# ── Normalisation helpers ────────────────────────────────────────────────────

def _denormalize(tensor, mean, std):
    m = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor * s + m).clamp(0., 1.)


def _normalize(tensor, mean, std):
    m = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - m) / s


def _get_norm_params(processor):
    if processor is not None:
        mean = list(getattr(processor, 'image_mean', IMAGENET_MEAN))
        std = list(getattr(processor, 'image_std', IMAGENET_STD))
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    return mean, std


# ── Attack dispatch registry ─────────────────────────────────────────────────
# Each entry: attack_name → lambda(model, input, target, epsilon, iterations, params)
# The lambda extracts only the params that attack needs, with correct types/defaults.

def _p(params, key, default, cast=float):
    return cast(params.get(key, default))

ATTACK_DISPATCH = {
    "FGSM": lambda m, x, t, e, i, p:
        targeted_fgsm(m, x, t, e),

    "I-FGSM (BIM)": lambda m, x, t, e, i, p:
        targeted_bim(m, x, t, e, i, _p(p, 'alpha', 1.0)),

    "PGD": lambda m, x, t, e, i, p:
        targeted_pgd(m, x, t, e, i, _p(p, 'alpha', 1.0), p.get('random_start', True)),

    "APGD": lambda m, x, t, e, i, p:
        targeted_apgd(m, x, t, e, i, p.get('apgd_loss', 'dlr'), _p(p, 'n_restarts', 1, int), _p(p, 'rho', 0.75)),

    "MI-FGSM": lambda m, x, t, e, i, p:
        targeted_mi_fgsm(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'alpha', 1.0)),

    "DI-FGSM": lambda m, x, t, e, i, p:
        targeted_difgsm(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'p_di', 0.7)),

    "TI-FGSM": lambda m, x, t, e, i, p:
        targeted_tifgsm(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'kernel_size', 5, int)),

    "NI-FGSM": lambda m, x, t, e, i, p:
        targeted_nifgsm(m, x, t, e, i, _p(p, 'momentum_decay', 1.0)),

    "SI-NI-FGSM": lambda m, x, t, e, i, p:
        targeted_sinifgsm(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'n_scale', 5, int)),

    "VMI-FGSM": lambda m, x, t, e, i, p:
        targeted_vmifgsm(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'n_var', 20, int), _p(p, 'beta_var', 1.5)),

    "VNI-FGSM": lambda m, x, t, e, i, p:
        targeted_vnifgsm(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'n_var', 20, int), _p(p, 'beta_var', 1.5)),

    "PI-FGSM": lambda m, x, t, e, i, p:
        targeted_pifgsm(m, x, t, e, i, _p(p, 'amplification', 10.0), _p(p, 'pi_prob', 0.7), _p(p, 'pi_kern_size', 3, int)),

    "Jitter": lambda m, x, t, e, i, p:
        targeted_jitter(m, x, t, e, i, _p(p, 'alpha', 1.0), p.get('random_start', True), _p(p, 'jitter_ratio', 0.1)),

    "DeepFool": lambda m, x, t, e, i, p:
        targeted_deepfool(m, x, t, e, i, _p(p, 'overshoot', 0.02)),

    "C&W (L2)": lambda m, x, t, e, i, p:
        targeted_cw(m, x, t, e, i, _p(p, 'cw_c', 1.0), _p(p, 'cw_lr', 0.01), _p(p, 'cw_confidence', 0.0)),

    "FAB": lambda m, x, t, e, i, p:
        targeted_fab(m, x, t, e, i, _p(p, 'fab_alpha_max', 0.1), _p(p, 'fab_eta', 1.05), _p(p, 'fab_beta', 0.9), _p(p, 'n_restarts', 1, int)),

    "JSMA": lambda m, x, t, e, i, p:
        targeted_jsma(m, x, t, e, i, _p(p, 'theta', 1.0), _p(p, 'gamma', 0.1)),

    "EAD (Elastic Net)": lambda m, x, t, e, i, p:
        targeted_ead(m, x, t, e, i, _p(p, 'cw_lr', 0.01), _p(p, 'cw_c', 1.0), _p(p, 'cw_confidence', 0.0), _p(p, 'ead_beta', 0.001), p.get('ead_rule', 'EN')),

    "SparseFool": lambda m, x, t, e, i, p:
        targeted_sparsefool(m, x, t, e, i, _p(p, 'overshoot', 0.02), _p(p, 'sf_lambda', 1.0)),

    "SSA": lambda m, x, t, e, i, p:
        targeted_ssa(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'n_spectrum', 20, int), _p(p, 'rho_spectrum', 0.5)),

    "Admix": lambda m, x, t, e, i, p:
        targeted_admix(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'n_mix', 5, int), _p(p, 'mix_ratio', 0.2)),

    "BSR": lambda m, x, t, e, i, p:
        targeted_bsr(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'n_block', 3, int), _p(p, 'rotation_range', 10.0)),

    "Square Attack": lambda m, x, t, e, i, p:
        targeted_square(m, x, t, e, _p(p, 'n_queries', 5000, int), _p(p, 'p_init', 0.8)),

    "SPSA": lambda m, x, t, e, i, p:
        targeted_spsa(m, x, t, e, i, _p(p, 'spsa_delta', 0.01), _p(p, 'spsa_lr', 0.01), _p(p, 'nb_sample', 128, int)),

    "One Pixel": lambda m, x, t, e, i, p:
        targeted_onepixel(m, x, t, e, _p(p, 'pixels', 5, int), _p(p, 'popsize', 400, int), i),

    "Boundary Attack": lambda m, x, t, e, i, p:
        targeted_boundary(m, x, t, e, i, _p(p, 'boundary_delta', 0.01), _p(p, 'boundary_step', 0.01)),

    "HopSkipJump": lambda m, x, t, e, i, p:
        targeted_hopskipjump(m, x, t, e, i, _p(p, 'hsj_init_evals', 100, int), _p(p, 'hsj_max_evals', 1000, int), _p(p, 'hsj_gamma', 1.0)),

    "Adversarial Patch": lambda m, x, t, e, i, p:
        targeted_patch(m, x, t, e, i, _p(p, 'patch_lr', 0.01), _p(p, 'patch_ratio', 0.1)),

    "UAP": lambda m, x, t, e, i, p:
        targeted_uap(m, x, t, e, i, _p(p, 'overshoot', 0.02)),

    "AutoAttack": lambda m, x, t, e, i, p:
        targeted_autoattack(m, x, t, e, i, p.get('aa_version', 'standard')),
}


# ── Main entry point ─────────────────────────────────────────────────────────

def run_attack_method(attack_name, model, img_tensor, target_class,
                      epsilon, iterations, params=None, processor=None,
                      ensemble_models=None, should_cancel=None):
    """Route to appropriate attack function.

    All attacks run in [0,1] pixel space regardless of model source.
    """
    if params is None:
        params = {}

    mean, std = _get_norm_params(processor)
    pixel_input = _denormalize(img_tensor, mean, std)

    # SarabCraft R1 and NewBackend have their own ensemble handling and return raw logits
    if attack_name == SARABCRAFT_R1_NAME:
        return _run_sarabcraft_r1(model, pixel_input, target_class, epsilon, iterations, params, mean, std, ensemble_models, should_cancel)

    if attack_name == "NewBackend (TA-Bench)":
        return _run_newbackend(model, pixel_input, target_class, epsilon, iterations, params, mean, std, should_cancel)

    if attack_name == "CFM Paper (CVPR 2023)":
        return _run_cfm_paper(model, pixel_input, target_class, epsilon, iterations, params, mean, std, ensemble_models, should_cancel)

    # All other attacks use HF output wrapper
    pixel_model = _build_pixel_model(model, mean, std, ensemble_models, should_cancel)

    dispatch_fn = ATTACK_DISPATCH.get(attack_name)
    if dispatch_fn is None:
        logger.warning(f"Unknown attack '{attack_name}', falling back to FGSM")
        dispatch_fn = ATTACK_DISPATCH["FGSM"]

    pixel_adv = dispatch_fn(pixel_model, pixel_input, target_class, epsilon, iterations, params)
    return _normalize(pixel_adv.clamp(0., 1.), mean, std)


def _build_pixel_model(model, mean, std, ensemble_models, should_cancel=None):
    if ensemble_models:
        return _wrap_cancelable(_EnsemblePixelModelWrapperHF([model] + list(ensemble_models), mean, std), should_cancel)
    return _wrap_cancelable(_PixelModelWrapperHF(model, mean, std), should_cancel)


def _run_sarabcraft_r1_standard(model, pixel_input, target_class, epsilon, iterations, params, mean, std, ensemble_models, should_cancel=None):
    pixel_model = _wrap_cancelable(_PixelModelWrapper(model, mean, std), should_cancel)
    ens_pixel = [_wrap_cancelable(_PixelModelWrapper(em, mean, std), should_cancel) for em in ensemble_models] if ensemble_models else None
    pixel_adv = targeted_sarabcraft_r1(
        pixel_model, pixel_input, target_class, epsilon, iterations,
        decay=params.get('momentum_decay', 1.0),
        kernel_size=int(params.get('kernel_size', 5)),
        mix_prob=params.get('cfm_mix_prob', 0.1),
        mix_upper=params.get('cfm_mix_upper', 0.75),
        ensemble_models=ens_pixel,
        ensemble_mode=params.get('ensemble_mode', 'simultaneous'),
    )
    return _normalize(pixel_adv, mean, std)


def _run_sarabcraft_r1(model, pixel_input, target_class, epsilon, iterations, params, mean, std, ensemble_models, should_cancel=None):
    if params.get("r1_multi_image"):
        return _run_sarabcraft_r1_multi_image(model, pixel_input, target_class, epsilon, iterations, params, mean, std, ensemble_models, should_cancel)
    return _run_sarabcraft_r1_standard(model, pixel_input, target_class, epsilon, iterations, params, mean, std, ensemble_models, should_cancel)


def _run_newbackend(model, pixel_input, target_class, epsilon, iterations, params, mean, std, should_cancel=None):
    pixel_model = _wrap_cancelable(_PixelModelWrapper(model, mean, std), should_cancel)
    pixel_adv = targeted_newbackend(
        pixel_model, pixel_input, target_class, epsilon, iterations,
        decay=params.get('momentum_decay', 1.0),
        step_size=params.get('nb_step_size', None),
        di_prob=params.get('nb_di_prob', 0.5),
        di_resize_rate=params.get('nb_di_resize_rate', 0.9),
        ti_len=int(params.get('nb_ti_len', 3)),
        npatch=int(params.get('nb_npatch', 128)),
        grid_scale=int(params.get('nb_grid_scale', 16)),
        enable_un=params.get('nb_enable_un', True),
        enable_pi=params.get('nb_enable_pi', True),
        enable_di=params.get('nb_enable_di', True),
        enable_ti=params.get('nb_enable_ti', True),
        enable_ni=params.get('nb_enable_ni', True),
    )
    return _normalize(pixel_adv, mean, std)


def _run_sarabcraft_r1_multi_image(model, pixel_input, target_class, epsilon, iterations, params, mean, std, ensemble_models, should_cancel=None):
    pixel_model = _wrap_cancelable(_PixelModelWrapper(model, mean, std), should_cancel)
    ens_pixel = [_wrap_cancelable(_PixelModelWrapper(em, mean, std), should_cancel) for em in ensemble_models] if ensemble_models else None
    pixel_adv = targeted_sarabcraft_r1_multi_image(
        pixel_model, pixel_input, target_class, epsilon, iterations,
        decay=params.get('momentum_decay', 1.0),
        kernel_size=int(params.get('kernel_size', 5)),
        mix_prob=params.get('cfm_mix_prob', 0.1),
        mix_upper=params.get('cfm_mix_upper', 0.75),
        cfm_strategy=params.get('r1_multi_image_strategy', 'tile_shuffle'),
        n_images=int(params.get('r1_multi_image_count', 10)),
        ensemble_models=ens_pixel,
        ensemble_mode=params.get('ensemble_mode', 'simultaneous'),
    )
    return _normalize(pixel_adv, mean, std)


def _run_cfm_paper(model, pixel_input, target_class, epsilon, iterations, params, mean, std, ensemble_models, should_cancel=None):
    pixel_model = _wrap_cancelable(_PixelModelWrapper(model, mean, std), should_cancel)
    ens_pixel = [_wrap_cancelable(_PixelModelWrapper(em, mean, std), should_cancel) for em in ensemble_models] if ensemble_models else None
    pixel_adv = targeted_cfm_paper(
        pixel_model, pixel_input, target_class, epsilon, iterations,
        decay=params.get('momentum_decay', 1.0),
        kernel_size=int(params.get('kernel_size', 5)),
        mix_prob=params.get('cfm_mix_prob', 0.1),
        mix_upper=params.get('cfm_mix_upper', 0.75),
        ensemble_models=ens_pixel,
        ensemble_mode=params.get('ensemble_mode', 'simultaneous'),
    )
    return _normalize(pixel_adv, mean, std)
