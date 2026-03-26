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
from attacks.transfer.emi_fgsm import targeted_emi_fgsm
from attacks.transfer.pgn import targeted_pgn
from attacks.transfer.rap import targeted_rap
from attacks.transfer.fia import targeted_fia
from attacks.transfer.naa import targeted_naa
from attacks.transfer.tgr import targeted_tgr
from attacks.transfer.linbp import targeted_linbp
from attacks.transfer.sgm import targeted_sgm
from attacks.transfer.mig import targeted_mig
from attacks.transfer.pna import targeted_pna
from attacks.transfer.smi_fgsm import targeted_smi_fgsm
from attacks.transfer.sia import targeted_sia
from attacks.transfer.s4st import targeted_s4st
from attacks.transfer.rdim import targeted_rdim
from attacks.transfer.stadv import targeted_stadv
from attacks.transfer.logit_loss import targeted_logit
from attacks.transfer.potrip import targeted_potrip
from attacks.transfer.ila import targeted_ila
from attacks.transfer.ilpd import targeted_ilpd
from attacks.transfer.ghost_network import targeted_ghost
from attacks.transfer.su import targeted_su
from attacks.transfer.da import targeted_da
from attacks.transfer.ig_fgsm import targeted_ig_fgsm
from attacks.transfer.gnp import targeted_gnp
from attacks.transfer.dra import targeted_dra
from attacks.transfer.cwa import targeted_cwa
from attacks.transfer.logit_cal import targeted_logit_cal
from attacks.transfer.sasd import targeted_sasd
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
    """[0,1] pixel input → raw logit tensor (no .logits attribute)."""

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


# ── Attack dispatch registries ────────────────────────────────────────────────
#
# Two registries, same interface but different model wrappers:
#   HF_DISPATCH  — model receives _PixelModelWrapperHF  (output has .logits)
#   RAW_DISPATCH — model receives _PixelModelWrapper    (output is raw tensor)
#                  + gets ensemble_models list for internal handling
#
# Signature for HF_DISPATCH:
#   lambda(model, x, target, eps, iters, params) -> pixel_adv
#
# Signature for RAW_DISPATCH:
#   lambda(model, x, target, eps, iters, params, ensemble_models) -> pixel_adv

def _p(params, key, default, cast=float):
    return cast(params.get(key, default))

HF_DISPATCH = {
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

    "EMI-FGSM": lambda m, x, t, e, i, p:
        targeted_emi_fgsm(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'n_sample', 11, int), _p(p, 'sampling_range', 0.3)),

    "PGN": lambda m, x, t, e, i, p:
        targeted_pgn(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'pgn_lambda', 1.0)),

    "RAP": lambda m, x, t, e, i, p:
        targeted_rap(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'rap_epsilon', 0.02), _p(p, 'rap_steps', 5, int), _p(p, 'late_start', 0.0)),

    "FIA": lambda m, x, t, e, i, p:
        targeted_fia(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'n_drop', 30, int), _p(p, 'drop_rate', 0.3)),

    "NAA": lambda m, x, t, e, i, p:
        targeted_naa(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'n_ig_steps', 10, int)),

    "TGR": lambda m, x, t, e, i, p:
        targeted_tgr(m, x, t, e, i, _p(p, 'momentum_decay', 1.0)),

    "LinBP": lambda m, x, t, e, i, p:
        targeted_linbp(m, x, t, e, i, _p(p, 'momentum_decay', 1.0)),

    "SGM": lambda m, x, t, e, i, p:
        targeted_sgm(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'sgm_gamma', 0.5)),

    "MIG": lambda m, x, t, e, i, p:
        targeted_mig(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'n_ig', 10, int)),

    "PNA": lambda m, x, t, e, i, p:
        targeted_pna(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'pna_patchout', 0.7)),

    "SMI-FGSM": lambda m, x, t, e, i, p:
        targeted_smi_fgsm(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'spatial_decay', 0.6), _p(p, 'smi_blocks', 3, int)),

    "SIA": lambda m, x, t, e, i, p:
        targeted_sia(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'sia_blocks', 3, int)),

    "S4ST": lambda m, x, t, e, i, p:
        targeted_s4st(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'n_aug', 5, int), _p(p, 'scale_lo', 0.5), _p(p, 'scale_hi', 1.5)),

    "RDIM": lambda m, x, t, e, i, p:
        targeted_rdim(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'n_diverse', 5, int), _p(p, 'resize_lo', 0.5), _p(p, 'resize_hi', 1.3)),

    "StAdv": lambda m, x, t, e, i, p:
        targeted_stadv(m, x, t, e, i, _p(p, 'flow_lr', 0.005), _p(p, 'flow_reg', 0.05)),

    "Logit Loss": lambda m, x, t, e, i, p:
        targeted_logit(m, x, t, e, i, _p(p, 'momentum_decay', 1.0)),

    "Po+Trip": lambda m, x, t, e, i, p:
        targeted_potrip(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'potrip_margin', 0.3)),

    "ILA": lambda m, x, t, e, i, p:
        targeted_ila(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'ila_ratio', 0.5)),

    "ILPD": lambda m, x, t, e, i, p:
        targeted_ilpd(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'ilpd_gamma', 0.3)),

    "Ghost Networks": lambda m, x, t, e, i, p:
        targeted_ghost(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'ghost_dropout', 0.1), _p(p, 'n_ghost', 5, int)),

    "SU": lambda m, x, t, e, i, p:
        targeted_su(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'su_crops', 3, int), _p(p, 'su_crop_ratio', 0.7), _p(p, 'su_weight', 0.5)),

    "DA": lambda m, x, t, e, i, p:
        targeted_da(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'da_n_aug', 3, int), _p(p, 'da_strength', 0.3)),

    "IG-FGSM": lambda m, x, t, e, i, p:
        targeted_ig_fgsm(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'n_ig', 10, int)),

    "GNP": lambda m, x, t, e, i, p:
        targeted_gnp(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'gnp_lambda', 1.0)),

    "DRA": lambda m, x, t, e, i, p:
        targeted_dra(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'dra_views', 5, int)),

    "CWA": lambda m, x, t, e, i, p:
        targeted_cwa(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'cwa_n_aug', 3, int), _p(p, 'cwa_flat_weight', 0.5)),

    "Logit Calibration": lambda m, x, t, e, i, p:
        targeted_logit_cal(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'lc_temperature', 3.0), _p(p, 'lc_margin', 0.0)),

    "SASD": lambda m, x, t, e, i, p:
        targeted_sasd(m, x, t, e, i, _p(p, 'momentum_decay', 1.0), _p(p, 'sasd_rho', 0.05), _p(p, 'sasd_temp', 4.0)),

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

RAW_DISPATCH = {
    SARABCRAFT_R1_NAME: lambda m, x, t, e, i, p, ens:
        targeted_sarabcraft_r1_multi_image(
            m, x, t, e, i,
            decay=p.get('momentum_decay', 1.0),
            kernel_size=int(p.get('kernel_size', 5)),
            mix_prob=p.get('cfm_mix_prob', 0.1),
            mix_upper=p.get('cfm_mix_upper', 0.75),
            cfm_strategy=p.get('r1_multi_image_strategy', 'tile_shuffle'),
            n_images=int(p.get('r1_multi_image_count', 10)),
            ensemble_models=ens,
            ensemble_mode=p.get('ensemble_mode', 'simultaneous'),
        ) if p.get("r1_multi_image") else targeted_sarabcraft_r1(
            m, x, t, e, i,
            decay=p.get('momentum_decay', 1.0),
            kernel_size=int(p.get('kernel_size', 5)),
            mix_prob=p.get('cfm_mix_prob', 0.1),
            mix_upper=p.get('cfm_mix_upper', 0.75),
            ensemble_models=ens,
            ensemble_mode=p.get('ensemble_mode', 'simultaneous'),
        ),

    "CFM Paper (CVPR 2023)": lambda m, x, t, e, i, p, ens:
        targeted_cfm_paper(
            m, x, t, e, i,
            decay=p.get('momentum_decay', 1.0),
            kernel_size=int(p.get('kernel_size', 5)),
            mix_prob=p.get('cfm_mix_prob', 0.1),
            mix_upper=p.get('cfm_mix_upper', 0.75),
            ensemble_models=ens,
            ensemble_mode=p.get('ensemble_mode', 'simultaneous'),
        ),

    "NewBackend (TA-Bench)": lambda m, x, t, e, i, p, ens:
        targeted_newbackend(
            m, x, t, e, i,
            decay=p.get('momentum_decay', 1.0),
            step_size=p.get('nb_step_size', None),
            di_prob=p.get('nb_di_prob', 0.5),
            di_resize_rate=p.get('nb_di_resize_rate', 0.9),
            ti_len=int(p.get('nb_ti_len', 3)),
            npatch=int(p.get('nb_npatch', 128)),
            grid_scale=int(p.get('nb_grid_scale', 16)),
            enable_un=p.get('nb_enable_un', True),
            enable_pi=p.get('nb_enable_pi', True),
            enable_di=p.get('nb_enable_di', True),
            enable_ti=p.get('nb_enable_ti', True),
            enable_ni=p.get('nb_enable_ni', True),
        ),
}

# Combined lookup for external code that needs the full list
ATTACK_DISPATCH = {**HF_DISPATCH, **RAW_DISPATCH}


# ── Model builder helpers ────────────────────────────────────────────────────

def _build_hf_model(model, mean, std, ensemble_models, should_cancel):
    if ensemble_models:
        return _wrap_cancelable(_EnsemblePixelModelWrapperHF([model] + list(ensemble_models), mean, std), should_cancel)
    return _wrap_cancelable(_PixelModelWrapperHF(model, mean, std), should_cancel)


def _build_raw_models(model, mean, std, ensemble_models, should_cancel):
    """Returns (primary_model, ensemble_list) both using raw-logit wrapper."""
    primary = _wrap_cancelable(_PixelModelWrapper(model, mean, std), should_cancel)
    ens = [_wrap_cancelable(_PixelModelWrapper(em, mean, std), should_cancel)
           for em in ensemble_models] if ensemble_models else None
    return primary, ens


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

    # Raw-logit attacks (handle ensembles internally, expect raw tensor output)
    if attack_name in RAW_DISPATCH:
        primary, ens = _build_raw_models(model, mean, std, ensemble_models, should_cancel)
        pixel_adv = RAW_DISPATCH[attack_name](primary, pixel_input, target_class, epsilon, iterations, params, ens)
        return _normalize(pixel_adv.clamp(0., 1.), mean, std)

    # HF-wrapped attacks (use .logits, ensemble handled by wrapper)
    pixel_model = _build_hf_model(model, mean, std, ensemble_models, should_cancel)
    dispatch_fn = HF_DISPATCH.get(attack_name)
    if dispatch_fn is None:
        logger.warning(f"Unknown attack '{attack_name}', falling back to FGSM")
        dispatch_fn = HF_DISPATCH["FGSM"]

    pixel_adv = dispatch_fn(pixel_model, pixel_input, target_class, epsilon, iterations, params)
    return _normalize(pixel_adv.clamp(0., 1.), mean, std)
