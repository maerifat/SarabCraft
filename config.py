"""
Configuration for Adversarial Attack Demo.
Constants, model registry, attack registry, and environment setup.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model that is pre-downloaded in Docker image
PRELOADED_MODEL = "microsoft/resnet-50"

# Available models organized by category
AVAILABLE_MODELS = {
    # ===== CNN (Convolutional Neural Networks) =====
    "[CNN] ResNet-50 (Microsoft) ⚡": "microsoft/resnet-50",
    "[CNN] ResNet-18 (Microsoft)": "microsoft/resnet-18",
    "[CNN] ResNet-101 (Microsoft)": "microsoft/resnet-101",
    "[CNN] ConvNeXt-Tiny (Facebook)": "facebook/convnext-tiny-224",
    "[CNN] ConvNeXt-Base (Facebook)": "facebook/convnext-base-224",
    "[CNN] ConvNeXt-Large (Facebook)": "facebook/convnext-large-224",
    "[CNN] ConvNeXt-XLarge (Facebook)": "facebook/convnext-xlarge-224-22k",
    "[CNN] ConvNeXt-V2-Tiny (Facebook)": "facebook/convnextv2-tiny-1k-224",
    "[CNN] MobileNetV2 (Google)": "google/mobilenet_v2_1.0_224",

    # ===== Vision Transformers =====
    "[ViT] ViT-Base (Google)": "google/vit-base-patch16-224",
    "[ViT] ViT-Large (Google)": "google/vit-large-patch16-224",
    "[ViT] ViT-Huge (Google)": "google/vit-huge-patch14-224-in21k",
    "[ViT] DeiT-Small (Facebook)": "facebook/deit-small-patch16-224",
    "[ViT] DeiT-Base (Facebook)": "facebook/deit-base-patch16-224",
    "[ViT] BEiT-Base (Microsoft)": "microsoft/beit-base-patch16-224",
    "[ViT] Swin-Tiny (Microsoft)": "microsoft/swin-tiny-patch4-window7-224",
    "[ViT] Swin-Base (Microsoft)": "microsoft/swin-base-patch4-window7-224",
    "[ViT] Swin-V2-Base (Microsoft)": "microsoft/swinv2-base-patch4-window8-256",
    "[ViT] Swin-V2-Large (Microsoft)": "microsoft/swinv2-large-patch4-window12-192-22k",
    "[ViT] DINOv2-Base (Facebook)": "facebook/dinov2-base-imagenet1k-1-layer",
    "[ViT] Data2Vec-Base (Facebook)": "facebook/data2vec-vision-base-ft1k",
    "[ViT] PoolFormer-S12 (Sea AI)": "sail/poolformer_s12",

    # ===== Hybrid (CNN + Transformer) =====
    "[Hybrid] MobileViT-Small (Apple)": "apple/mobilevit-small",
    "[Hybrid] MobileViT-XSmall (Apple)": "apple/mobilevit-x-small",
    "[Hybrid] LeViT-256 (Facebook)": "facebook/levit-256",

    # ===== NAS-designed (Neural Architecture Search) =====
    "[NAS] EfficientNet-B0 (Google)": "google/efficientnet-b0",
    "[NAS] EfficientNet-B4 (Google)": "google/efficientnet-b4",
    "[NAS] RegNet-Y-040 (Facebook)": "facebook/regnet-y-040",

    # ===== Large models (1K fine-tuned) =====
    "[ViT] Swin-Large 224 (Microsoft)": "microsoft/swin-large-patch4-window7-224",
    "[ViT] Swin-Base 384 (Microsoft)": "microsoft/swin-base-patch4-window12-384",
    "[ViT] BEiT-Large (Microsoft)": "microsoft/beit-large-patch16-224-pt22k-ft22k",
    "[ViT] DeiT-Base Distilled (Facebook)": "facebook/deit-base-distilled-patch16-224",
}


# ============================================================================
# ATTACK REGISTRY — full metadata for frontend
# ============================================================================
# Each attack has: category, threat model, year, authors, description, norm,
# and parameter schema with paper-proven defaults.
# ============================================================================

ATTACK_REGISTRY = {
    # ── Gradient-Based (L∞) ────────────────────────────────────────────
    "FGSM": {
        "cat": "Gradient (L∞)", "threat": "whitebox", "year": 2015, "norm": "L∞",
        "authors": "Goodfellow et al.",
        "desc": "Single-step gradient sign perturbation. Fastest but weakest attack.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
        },
    },
    "I-FGSM (BIM)": {
        "cat": "Gradient (L∞)", "threat": "whitebox", "year": 2017, "norm": "L∞",
        "authors": "Kurakin et al.",
        "desc": "Iterative FGSM with small steps. Stronger than single-step FGSM.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 2000, "step": 1, "default": 40},
            "alpha": {"label": "Step multiplier (α)", "min": 0.1, "max": 10, "step": 0.1, "default": 1.0},
        },
    },
    "PGD": {
        "cat": "Gradient (L∞)", "threat": "whitebox", "year": 2018, "norm": "L∞",
        "authors": "Madry et al.",
        "desc": "Projected Gradient Descent with random start. Standard strong white-box baseline.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 2000, "step": 1, "default": 50},
            "alpha": {"label": "Step multiplier (α)", "min": 0.1, "max": 10, "step": 0.1, "default": 1.0},
            "random_start": {"label": "Random start", "type": "bool", "default": True},
        },
    },
    "APGD": {
        "cat": "Gradient (L∞)", "threat": "whitebox", "year": 2020, "norm": "L∞/L2",
        "authors": "Croce & Hein",
        "desc": "Auto-PGD: adaptive step-size schedule + DLR loss. Parameter-free, strictly stronger than PGD. Used in AutoAttack.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 10, "max": 2000, "step": 10, "default": 100},
            "apgd_loss": {"label": "Loss function", "type": "select", "options": ["dlr", "ce"], "default": "dlr"},
            "n_restarts": {"label": "Restarts", "min": 1, "max": 10, "step": 1, "default": 1},
            "rho": {"label": "Step decay (ρ)", "min": 0.1, "max": 1, "step": 0.05, "default": 0.75},
        },
    },
    "MI-FGSM": {
        "cat": "Gradient (L∞)", "threat": "whitebox", "year": 2018, "norm": "L∞",
        "authors": "Dong et al.",
        "desc": "Momentum Iterative FGSM. Accumulates gradient momentum for better transferability.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 2000, "step": 1, "default": 20},
            "alpha": {"label": "Step multiplier (α)", "min": 0.1, "max": 10, "step": 0.1, "default": 1.0},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
        },
    },
    "DI-FGSM": {
        "cat": "Gradient (L∞)", "threat": "whitebox", "year": 2019, "norm": "L∞",
        "authors": "Xie et al.",
        "desc": "Diverse Input FGSM. Random resize-pad transform at each step improves transfer.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 2000, "step": 1, "default": 20},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "p_di": {"label": "DI probability", "min": 0.1, "max": 1, "step": 0.1, "default": 0.7},
        },
    },
    "TI-FGSM": {
        "cat": "Gradient (L∞)", "threat": "whitebox", "year": 2019, "norm": "L∞",
        "authors": "Dong et al.",
        "desc": "Translation-Invariant FGSM. Gaussian kernel smooths gradients for shift invariance.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 2000, "step": 1, "default": 20},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "kernel_size": {"label": "TI kernel size", "min": 3, "max": 15, "step": 2, "default": 5},
        },
    },
    "NI-FGSM": {
        "cat": "Gradient (L∞)", "threat": "whitebox", "year": 2020, "norm": "L∞",
        "authors": "Lin et al.",
        "desc": "Nesterov Iterative FGSM. Computes gradient at lookahead position for faster convergence.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 2000, "step": 1, "default": 20},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
        },
    },
    "SI-NI-FGSM": {
        "cat": "Gradient (L∞)", "threat": "whitebox", "year": 2020, "norm": "L∞",
        "authors": "Lin et al.",
        "desc": "Scale-Invariant Nesterov FGSM. Averages gradients at multiple image scales.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 2000, "step": 1, "default": 20},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "n_scale": {"label": "# Scales", "min": 2, "max": 10, "step": 1, "default": 5},
        },
    },
    "VMI-FGSM": {
        "cat": "Gradient (L∞)", "threat": "whitebox", "year": 2021, "norm": "L∞",
        "authors": "Wang & He",
        "desc": "Variance-Tuned MI-FGSM. Adds neighbourhood gradient variance to reduce overfitting.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 2000, "step": 1, "default": 20},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "n_var": {"label": "Variance samples (N)", "min": 5, "max": 30, "step": 5, "default": 20},
            "beta_var": {"label": "Variance scale (β)", "min": 0.5, "max": 5, "step": 0.1, "default": 1.5},
        },
    },
    "VNI-FGSM": {
        "cat": "Gradient (L∞)", "threat": "whitebox", "year": 2021, "norm": "L∞",
        "authors": "Wang & He",
        "desc": "Variance-tuned Nesterov FGSM. Combines VMI variance with Nesterov lookahead momentum.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 2000, "step": 1, "default": 20},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "n_var": {"label": "Variance samples (N)", "min": 5, "max": 30, "step": 5, "default": 20},
            "beta_var": {"label": "Variance scale (β)", "min": 0.5, "max": 5, "step": 0.1, "default": 1.5},
        },
    },
    "PI-FGSM": {
        "cat": "Gradient (L∞)", "threat": "whitebox", "year": 2020, "norm": "L∞",
        "authors": "Gao et al.",
        "desc": "Patch-wise Iterative FGSM. Amplifies perturbation through project kernel for targeted transfer.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 10},
            "amplification": {"label": "Amplification", "min": 1, "max": 50, "step": 1, "default": 10},
            "pi_prob": {"label": "DI probability", "min": 0.1, "max": 1, "step": 0.1, "default": 0.7},
            "pi_kern_size": {"label": "Kernel size", "min": 3, "max": 15, "step": 2, "default": 3},
        },
    },
    "Jitter": {
        "cat": "Gradient (L∞)", "threat": "whitebox", "year": 2021, "norm": "L∞",
        "authors": "Schwinn et al.",
        "desc": "PGD with random neighbourhood sampling before each gradient step. Escapes sharp local minima.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 2000, "step": 1, "default": 50},
            "alpha": {"label": "Step multiplier (α)", "min": 0.1, "max": 10, "step": 0.1, "default": 1.0},
            "random_start": {"label": "Random start", "type": "bool", "default": True},
            "jitter_ratio": {"label": "Jitter ratio", "min": 0.01, "max": 0.5, "step": 0.01, "default": 0.1},
        },
    },

    # ── Optimization / Sparse ──────────────────────────────────────────
    "DeepFool": {
        "cat": "Optimization (L2)", "threat": "whitebox", "year": 2016, "norm": "L2",
        "authors": "Moosavi-Dezfooli et al.",
        "desc": "Finds minimal perturbation to cross the decision boundary. Iterative linearisation.",
        "params": {
            "epsilon": {"label": "Max epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 10, "max": 500, "step": 10, "default": 100},
            "overshoot": {"label": "Overshoot", "min": 0.01, "max": 0.5, "step": 0.01, "default": 0.02},
        },
    },
    "C&W (L2)": {
        "cat": "Optimization (L2)", "threat": "whitebox", "year": 2017, "norm": "L2",
        "authors": "Carlini & Wagner",
        "desc": "Optimization-based L2 attack using Adam. Highly effective, slower. Gold standard for L2.",
        "params": {
            "epsilon": {"label": "Max epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 100, "max": 5000, "step": 100, "default": 1000},
            "cw_confidence": {"label": "Confidence (κ)", "min": 0, "max": 50, "step": 1, "default": 0},
            "cw_lr": {"label": "Learning rate", "min": 0.001, "max": 0.1, "step": 0.001, "default": 0.01},
            "cw_c": {"label": "Constant (c)", "min": 0.1, "max": 100, "step": 0.1, "default": 1.0},
        },
    },
    "FAB": {
        "cat": "Optimization (L2)", "threat": "whitebox", "year": 2020, "norm": "L∞/L2/L1",
        "authors": "Croce & Hein",
        "desc": "Fast Adaptive Boundary: finds closest adversarial by projecting onto decision boundary.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 10, "max": 500, "step": 10, "default": 100},
            "fab_alpha_max": {"label": "Max step (α_max)", "min": 0.01, "max": 1, "step": 0.01, "default": 0.1},
            "fab_eta": {"label": "Overshoot (η)", "min": 1.0, "max": 2.0, "step": 0.05, "default": 1.05},
            "fab_beta": {"label": "Backward step (β)", "min": 0.1, "max": 1, "step": 0.1, "default": 0.9},
            "n_restarts": {"label": "Restarts", "min": 1, "max": 10, "step": 1, "default": 1},
        },
    },
    "JSMA": {
        "cat": "Sparse (L0)", "threat": "whitebox", "year": 2016, "norm": "L0",
        "authors": "Papernot et al.",
        "desc": "Jacobian Saliency Map Attack. Uses Jacobian to find most impactful pixels. Changes fewest pixels.",
        "params": {
            "theta": {"label": "Perturbation step (θ)", "min": -1, "max": 1, "step": 0.1, "default": 1.0},
            "gamma": {"label": "Max pixel fraction (γ)", "min": 0.01, "max": 0.5, "step": 0.01, "default": 0.1},
        },
    },
    "EAD (Elastic Net)": {
        "cat": "Sparse (L1)", "threat": "whitebox", "year": 2018, "norm": "L1+L2",
        "authors": "Chen et al.",
        "desc": "Elastic-Net Attack: L1+L2 regularised C&W variant. Produces sparse perturbations via ISTA.",
        "params": {
            "epsilon": {"label": "Max epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 50, "max": 2000, "step": 50, "default": 200},
            "cw_lr": {"label": "Learning rate", "min": 0.001, "max": 0.1, "step": 0.001, "default": 0.01},
            "cw_c": {"label": "Constant (c)", "min": 0.1, "max": 100, "step": 0.1, "default": 1.0},
            "cw_confidence": {"label": "Confidence (κ)", "min": 0, "max": 50, "step": 1, "default": 0},
            "ead_beta": {"label": "L1 weight (β)", "min": 0.0001, "max": 0.1, "step": 0.0001, "default": 0.001},
            "ead_rule": {"label": "Decision rule", "type": "select", "options": ["EN", "L1"], "default": "EN"},
        },
    },
    "SparseFool": {
        "cat": "Sparse (L1)", "threat": "whitebox", "year": 2019, "norm": "L1",
        "authors": "Modas et al.",
        "desc": "Sparse DeepFool: finds minimal L1 perturbation by keeping only most salient pixel changes.",
        "params": {
            "epsilon": {"label": "Max epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 10, "max": 500, "step": 10, "default": 100},
            "overshoot": {"label": "Overshoot", "min": 0.01, "max": 0.5, "step": 0.01, "default": 0.02},
            "sf_lambda": {"label": "Sparsity (λ)", "min": 0.1, "max": 10, "step": 0.1, "default": 1.0},
        },
    },

    # ── Transfer-Optimised ─────────────────────────────────────────────
    "SSA": {
        "cat": "Transfer", "threat": "both", "year": 2022, "norm": "L∞",
        "authors": "Long et al.",
        "desc": "Spectrum Simulation Attack: augments in DCT frequency domain to prevent model-specific overfitting.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 20},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "n_spectrum": {"label": "Spectrum samples", "min": 5, "max": 50, "step": 5, "default": 20},
            "rho_spectrum": {"label": "Keep ratio (ρ)", "min": 0.1, "max": 1, "step": 0.1, "default": 0.5},
        },
    },
    "Admix": {
        "cat": "Transfer", "threat": "both", "year": 2021, "norm": "L∞",
        "authors": "Wang et al.",
        "desc": "Admix: mixes random images into input during attack to prevent overfitting to source model.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 20},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "n_mix": {"label": "Mix copies", "min": 1, "max": 10, "step": 1, "default": 5},
            "mix_ratio": {"label": "Mix ratio (η)", "min": 0.05, "max": 0.5, "step": 0.05, "default": 0.2},
        },
    },
    "BSR": {
        "cat": "Transfer", "threat": "both", "year": 2024, "norm": "L∞",
        "authors": "Wang et al.",
        "desc": "Block Shuffle & Rotation: randomly shuffles and rotates image blocks for transfer robustness.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 20},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "n_block": {"label": "Blocks per axis", "min": 2, "max": 8, "step": 1, "default": 3},
            "rotation_range": {"label": "Rotation (°)", "min": 0, "max": 45, "step": 5, "default": 10},
        },
    },
    "EMI-FGSM": {
        "cat": "Transfer", "threat": "both", "year": 2021, "norm": "L∞",
        "authors": "Wang et al. (BMVC 2021)",
        "desc": "Enhanced Momentum FGSM: averages gradients along the path between previous and current perturbation via linear interpolation sampling, stabilising momentum for better transfer.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "n_sample": {"label": "Path samples", "min": 3, "max": 21, "step": 2, "default": 11},
            "sampling_range": {"label": "Sampling range", "min": 0.1, "max": 1.0, "step": 0.1, "default": 0.3},
        },
    },
    "PGN": {
        "cat": "Transfer", "threat": "both", "year": 2023, "norm": "L∞",
        "authors": "Ge et al. (NeurIPS 2023)",
        "desc": "Penalizing Gradient Norm: adds a gradient-norm penalty to seek flat local maxima in the loss landscape. Flat maxima generalise better across models, boosting black-box transfer.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "pgn_lambda": {"label": "λ (penalty weight)", "min": 0.1, "max": 5.0, "step": 0.1, "default": 1.0},
        },
    },
    "RAP": {
        "cat": "Transfer", "threat": "both", "year": 2022, "norm": "L∞",
        "authors": "Qin et al. (NeurIPS 2022)",
        "desc": "Reverse Adversarial Perturbation: bi-level min-max optimisation that seeks flat loss regions by first finding a reverse perturbation that hurts the attack, then overcoming it.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "rap_epsilon": {"label": "Inner ε", "min": 0.005, "max": 0.1, "step": 0.005, "default": 0.02},
            "rap_steps": {"label": "Inner steps", "min": 1, "max": 20, "step": 1, "default": 5},
            "late_start": {"label": "Late start ratio", "min": 0, "max": 0.5, "step": 0.1, "default": 0.0},
        },
    },

    # ── Feature-Level ─────────────────────────────────────────────────
    "FIA": {
        "cat": "Feature-Level", "threat": "both", "year": 2021, "norm": "L∞",
        "authors": "Wang et al. (ICCV 2021)",
        "desc": "Feature Importance-aware Attack: computes aggregate gradient from random feature-drop masks to identify object-aware important features, then attacks only those features for better transfer.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "n_drop": {"label": "# drop samples", "min": 5, "max": 50, "step": 5, "default": 30},
            "drop_rate": {"label": "Drop rate", "min": 0.1, "max": 0.7, "step": 0.1, "default": 0.3},
        },
    },
    "NAA": {
        "cat": "Feature-Level", "threat": "both", "year": 2022, "norm": "L∞",
        "authors": "Zhang et al. (CVPR 2022)",
        "desc": "Neuron Attribution-based Attack: uses path-integrated gradients to compute neuron-level attribution at an intermediate layer, then weights attack gradients by these attributions.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "n_ig_steps": {"label": "IG steps", "min": 5, "max": 30, "step": 5, "default": 10},
        },
    },

    # ── ViT / Attention ───────────────────────────────────────────────
    "TGR": {
        "cat": "ViT/Attention", "threat": "both", "year": 2023, "norm": "L∞",
        "authors": "Zhang et al. (CVPR 2023)",
        "desc": "Token Gradient Regularization: regularises per-token gradients during ViT backprop to prevent extreme tokens from dominating, reducing surrogate-specific gradient spikes.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
        },
    },
    "PNA": {
        "cat": "ViT/Attention", "threat": "both", "year": 2022, "norm": "L∞",
        "authors": "Wei et al. (AAAI 2022)",
        "desc": "Pay No Attention: skips attention gradients during ViT backprop and applies PatchOut (random token dropping) for input diversity. Bypasses model-specific attention patterns.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "pna_patchout": {"label": "PatchOut keep prob", "min": 0.3, "max": 1.0, "step": 0.1, "default": 0.7},
        },
    },

    # ── Backward Propagation ──────────────────────────────────────────
    "LinBP": {
        "cat": "Backward-Prop", "threat": "both", "year": 2020, "norm": "L∞",
        "authors": "Guo et al. (NeurIPS 2020)",
        "desc": "Linear Backpropagation: linearises ReLU/GELU during backprop by removing zero-clipping, letting gradients flow through regardless of activation sign. Prevents surrogate-specific gradient loss.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
        },
    },
    "SGM": {
        "cat": "Backward-Prop", "threat": "both", "year": 2020, "norm": "L∞",
        "authors": "Wu et al. (ICLR 2020)",
        "desc": "Skip Gradient Method: amplifies gradients through skip/residual connections by decaying residual-branch gradients. Makes gradients less overfitted to residual block specifics.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "sgm_gamma": {"label": "Skip decay (γ)", "min": 0.1, "max": 1.0, "step": 0.1, "default": 0.5},
        },
    },

    # ── Integrated Gradients ──────────────────────────────────────────
    "MIG": {
        "cat": "Integrated-Grad", "threat": "both", "year": 2022, "norm": "L∞",
        "authors": "Ma et al. (ICLR 2022)",
        "desc": "Momentum Integrated Gradients: combines path-integrated gradients (from scaled baselines to current point) with momentum, providing a more generalised gradient signal.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "n_ig": {"label": "IG steps", "min": 5, "max": 30, "step": 5, "default": 10},
        },
    },

    # ── Momentum & Gradient (Advanced) ────────────────────────────────
    "SMI-FGSM": {
        "cat": "Transfer", "threat": "both", "year": 2022, "norm": "L∞",
        "authors": "Wang et al.",
        "desc": "Spatial Momentum FGSM: dual momentum — temporal across iterations + spatial across image regions. ~10% average improvement in transfer rate.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "spatial_decay": {"label": "Spatial decay", "min": 0.1, "max": 1.0, "step": 0.1, "default": 0.6},
            "smi_blocks": {"label": "# blocks", "min": 2, "max": 6, "step": 1, "default": 3},
        },
    },
    "GNP": {
        "cat": "Transfer", "threat": "both", "year": 2023, "norm": "L∞",
        "authors": "Wu et al. (ICME 2023)",
        "desc": "Gradient Norm Penalty: penalises the gradient norm to drive optimisation toward flat loss regions. Plug-in enhancement for any gradient method.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "gnp_lambda": {"label": "λ (penalty weight)", "min": 0.1, "max": 5.0, "step": 0.1, "default": 1.0},
        },
    },
    "DRA": {
        "cat": "Transfer", "threat": "both", "year": 2023, "norm": "L∞",
        "authors": "Huang et al.",
        "desc": "Direction-Aggregated Attack: aggregates gradient directions from multiple augmented views with cosine-weighted selection. 94.6% ASR on adversarially trained models.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "dra_views": {"label": "# views", "min": 3, "max": 10, "step": 1, "default": 5},
        },
    },

    # ── Input Transformation (Advanced) ───────────────────────────────
    "SIA": {
        "cat": "Input-Transform", "threat": "both", "year": 2023, "norm": "L∞",
        "authors": "Wang et al. (ICCV 2023)",
        "desc": "Structure Invariant Attack: splits image into blocks, applies independent random affine + colour transforms per block while preserving spatial structure.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "sia_blocks": {"label": "Block grid (s)", "min": 2, "max": 6, "step": 1, "default": 3},
        },
    },
    "S4ST": {
        "cat": "Input-Transform", "threat": "both", "year": 2024, "norm": "L∞",
        "authors": "Liu et al.",
        "desc": "Strong Self-transferable faSt Simple Scale Transformation: dimensionally consistent scaling + block-wise operations. SOTA targeted transfer (77.7% ASR, +17.2% over prior).",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "n_aug": {"label": "# augmented copies", "min": 1, "max": 10, "step": 1, "default": 5},
            "scale_lo": {"label": "Scale min", "min": 0.3, "max": 1.0, "step": 0.1, "default": 0.5},
            "scale_hi": {"label": "Scale max", "min": 1.0, "max": 2.0, "step": 0.1, "default": 1.5},
        },
    },
    "RDIM": {
        "cat": "Input-Transform", "threat": "both", "year": 2021, "norm": "L∞",
        "authors": "Zou et al.",
        "desc": "Resized-Diverse-Inputs Method: combines resized diverse inputs with diversity-ensemble. Wider resize range than DIM for more aggressive input diversity. 93% ASR.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "n_diverse": {"label": "# diverse copies", "min": 1, "max": 10, "step": 1, "default": 5},
            "resize_lo": {"label": "Resize min", "min": 0.3, "max": 1.0, "step": 0.1, "default": 0.5},
            "resize_hi": {"label": "Resize max", "min": 1.0, "max": 2.0, "step": 0.1, "default": 1.3},
        },
    },
    "StAdv": {
        "cat": "Input-Transform", "threat": "both", "year": 2018, "norm": "Flow",
        "authors": "Xiao et al. (ICLR 2018)",
        "desc": "Spatially Transformed Adversarial: spatial/geometric deformations (flow field) instead of pixel perturbations. Harder to defend with Lp-based defenses.",
        "params": {
            "epsilon": {"label": "Max displacement", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 50, "max": 2000, "step": 50, "default": 500},
            "flow_lr": {"label": "Flow learning rate", "min": 0.001, "max": 0.05, "step": 0.001, "default": 0.005},
            "flow_reg": {"label": "Smoothness weight", "min": 0.01, "max": 0.5, "step": 0.01, "default": 0.05},
        },
    },

    # ── Loss & Logit Optimization ─────────────────────────────────────
    "Logit Loss": {
        "cat": "Loss-Optim", "threat": "both", "year": 2021, "norm": "L∞",
        "authors": "Zhao et al. (NeurIPS 2021)",
        "desc": "Simply maximizes the target logit: L = -Z_t. Avoids gradient vanishing of cross-entropy. Constant gradient magnitude regardless of confidence.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 2000, "step": 1, "default": 300},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
        },
    },
    "Logit Calibration": {
        "cat": "Loss-Optim", "threat": "both", "year": 2023, "norm": "L∞",
        "authors": "He et al. (TIFS 2023)",
        "desc": "Temperature-scaled cross-entropy + adaptive margin to prevent logit saturation. Addresses gradient vanishing through calibrated temperature.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 2000, "step": 1, "default": 300},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "lc_temperature": {"label": "Temperature (T)", "min": 1.0, "max": 10.0, "step": 0.5, "default": 3.0},
            "lc_margin": {"label": "Margin", "min": 0.0, "max": 5.0, "step": 0.5, "default": 0.0},
        },
    },
    "Po+Trip": {
        "cat": "Loss-Optim", "threat": "both", "year": 2020, "norm": "L∞",
        "authors": "Li et al. (CVPR 2020)",
        "desc": "Poincaré distance + triplet loss: self-adaptive gradient magnitude pushing away from source and pulling toward target in hyperbolic space.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "potrip_margin": {"label": "Triplet margin", "min": 0.1, "max": 2.0, "step": 0.1, "default": 0.3},
        },
    },

    # ── Feature-Level (Advanced) ──────────────────────────────────────
    "ILA": {
        "cat": "Feature-Level", "threat": "both", "year": 2019, "norm": "L∞",
        "authors": "Huang et al. (ICCV 2019)",
        "desc": "Intermediate Level Attack: two-stage — standard attack then fine-tune by maximizing intermediate-layer perturbation in the found direction.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "ila_ratio": {"label": "ILA stage ratio", "min": 0.1, "max": 0.9, "step": 0.1, "default": 0.5},
        },
    },
    "ILPD": {
        "cat": "Feature-Level", "threat": "both", "year": 2023, "norm": "L∞",
        "authors": "Li et al. (NeurIPS 2023)",
        "desc": "Intermediate-Level Perturbation Decay: single-stage ILA alternative. Perturbation decay balances adversarial direction and intermediate magnitude. +10.07% over ILA.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "ilpd_gamma": {"label": "Decay γ", "min": 0.05, "max": 1.0, "step": 0.05, "default": 0.3},
        },
    },

    # ── Model Augmentation / Ensemble ─────────────────────────────────
    "Ghost Networks": {
        "cat": "Ensemble", "threat": "both", "year": 2020, "norm": "L∞",
        "authors": "Li et al. (AAAI 2020)",
        "desc": "Creates virtual model variants from one network via dense dropout erosion. Simulates ensemble from a single surrogate model.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "ghost_dropout": {"label": "Ghost dropout rate", "min": 0.01, "max": 0.3, "step": 0.01, "default": 0.1},
            "n_ghost": {"label": "# ghost copies", "min": 2, "max": 10, "step": 1, "default": 5},
        },
    },
    "CWA": {
        "cat": "Ensemble", "threat": "both", "year": 2024, "norm": "L∞",
        "authors": "Chen et al. (ICLR 2024)",
        "desc": "Common Weakness Attack: targets flatness of loss landscape + closeness to local optima. Simulates ensemble via model augmentation for single-model settings.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "cwa_n_aug": {"label": "# model augments", "min": 2, "max": 10, "step": 1, "default": 3},
            "cwa_flat_weight": {"label": "Flatness weight", "min": 0.1, "max": 2.0, "step": 0.1, "default": 0.5},
        },
    },

    # ── ViT / Attention (Advanced) ────────────────────────────────────
    "SU": {
        "cat": "ViT/Attention", "threat": "both", "year": 2023, "norm": "L∞",
        "authors": "Wei et al. (CVPR 2023)",
        "desc": "Self-Universality: optimises perturbation universal across different regions of a single image. Feature similarity between global and random crops. +12%.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "su_crops": {"label": "# crops", "min": 1, "max": 8, "step": 1, "default": 3},
            "su_crop_ratio": {"label": "Crop ratio", "min": 0.4, "max": 0.9, "step": 0.1, "default": 0.7},
            "su_weight": {"label": "SU loss weight", "min": 0.1, "max": 2.0, "step": 0.1, "default": 0.5},
        },
    },
    "DA": {
        "cat": "ViT/Attention", "threat": "both", "year": 2025, "norm": "L∞",
        "authors": "Wei et al. (IJCV 2025)",
        "desc": "Dilated Attention Attack: maximises attention maps of target class from multiple intermediate layers with dynamic linear augmentation. Works on CNNs + ViTs.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "da_n_aug": {"label": "# augments", "min": 1, "max": 8, "step": 1, "default": 3},
            "da_strength": {"label": "Augment strength", "min": 0.1, "max": 1.0, "step": 0.1, "default": 0.3},
        },
    },

    # ── Integrated Gradients (Advanced) ───────────────────────────────
    "IG-FGSM": {
        "cat": "Integrated-Grad", "threat": "both", "year": 2021, "norm": "L∞",
        "authors": "Qi et al.",
        "desc": "Integrated Gradients FGSM: replaces single-point gradients with path-integral from baseline. Smoother, less model-specific gradients. Predecessor to MIG.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "n_ig": {"label": "IG steps", "min": 5, "max": 30, "step": 5, "default": 10},
        },
    },

    # ── Defense-Aware ─────────────────────────────────────────────────
    "SASD": {
        "cat": "Defense-Aware", "threat": "both", "year": 2024, "norm": "L∞",
        "authors": "Chen et al. (CVPR 2024)",
        "desc": "Sharpness-Aware Self-Distillation: improves surrogate during attack via SAM-style perturbation + self-distillation. Flat loss landscape + knowledge transfer. +12.2%.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 1, "max": 500, "step": 1, "default": 50},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "sasd_rho": {"label": "SAM ρ", "min": 0.01, "max": 0.2, "step": 0.01, "default": 0.05},
            "sasd_temp": {"label": "Distillation T", "min": 1.0, "max": 10.0, "step": 0.5, "default": 4.0},
        },
    },

    # ── Black-Box ──────────────────────────────────────────────────────
    "Square Attack": {
        "cat": "Black-Box", "threat": "blackbox", "year": 2020, "norm": "L∞/L2",
        "authors": "Andriushchenko et al.",
        "desc": "Score-based random search with square-shaped colour patches. NO gradients needed. Query-efficient.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "n_queries": {"label": "Max queries", "min": 100, "max": 50000, "step": 100, "default": 5000},
            "p_init": {"label": "Initial square size", "min": 0.1, "max": 1, "step": 0.1, "default": 0.8},
        },
    },
    "SPSA": {
        "cat": "Black-Box", "threat": "blackbox", "year": 2018, "norm": "L∞",
        "authors": "Uesato et al.",
        "desc": "Estimates gradients via random perturbation pairs. Works with only model output probabilities.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 10, "max": 1000, "step": 10, "default": 100},
            "spsa_delta": {"label": "Estimation delta (δ)", "min": 0.001, "max": 0.1, "step": 0.001, "default": 0.01},
            "spsa_lr": {"label": "Learning rate", "min": 0.001, "max": 0.1, "step": 0.001, "default": 0.01},
            "nb_sample": {"label": "Samples per step", "min": 16, "max": 512, "step": 16, "default": 128},
        },
    },
    "One Pixel": {
        "cat": "Black-Box", "threat": "blackbox", "year": 2019, "norm": "L0",
        "authors": "Su, Vargas & Kouichi",
        "desc": "Differential evolution to find best 1-10 pixels to change. Famous L0 attack. No gradients.",
        "params": {
            "pixels": {"label": "Pixels to change", "min": 1, "max": 10, "step": 1, "default": 5},
            "popsize": {"label": "Population size", "min": 50, "max": 1000, "step": 50, "default": 400},
            "iterations": {"label": "Generations", "min": 10, "max": 300, "step": 10, "default": 75},
        },
    },
    "Boundary Attack": {
        "cat": "Black-Box", "threat": "blackbox", "year": 2018, "norm": "L2",
        "authors": "Brendel et al.",
        "desc": "Decision-based: starts from adversarial noise, walks along boundary toward clean image. Hard-label.",
        "params": {
            "epsilon": {"label": "Max epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Steps", "min": 100, "max": 20000, "step": 100, "default": 5000},
            "boundary_delta": {"label": "Orthogonal step (δ)", "min": 0.001, "max": 0.1, "step": 0.001, "default": 0.01},
            "boundary_step": {"label": "Toward-original step", "min": 0.001, "max": 0.1, "step": 0.001, "default": 0.01},
        },
    },
    "HopSkipJump": {
        "cat": "Black-Box", "threat": "blackbox", "year": 2020, "norm": "L2",
        "authors": "Chen et al.",
        "desc": "Improved boundary attack with gradient estimation via binary search + Monte Carlo sampling.",
        "params": {
            "epsilon": {"label": "Max epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Steps", "min": 10, "max": 500, "step": 10, "default": 50},
            "hsj_init_evals": {"label": "Initial eval samples", "min": 50, "max": 500, "step": 50, "default": 100},
            "hsj_max_evals": {"label": "Max eval samples", "min": 100, "max": 5000, "step": 100, "default": 1000},
            "hsj_gamma": {"label": "Step scale (γ)", "min": 0.1, "max": 5, "step": 0.1, "default": 1.0},
        },
    },

    # ── Physical ───────────────────────────────────────────────────────
    "Adversarial Patch": {
        "cat": "Physical", "threat": "both", "year": 2017, "norm": "Patch",
        "authors": "Brown et al.",
        "desc": "Optimises a small patch that causes misclassification when placed on any image. Works physically.",
        "params": {
            "iterations": {"label": "Iterations", "min": 50, "max": 2000, "step": 50, "default": 500},
            "patch_lr": {"label": "Learning rate", "min": 0.001, "max": 0.1, "step": 0.001, "default": 0.01},
            "patch_ratio": {"label": "Patch size (fraction)", "min": 0.05, "max": 0.5, "step": 0.05, "default": 0.1},
        },
    },
    "UAP": {
        "cat": "Physical", "threat": "whitebox", "year": 2017, "norm": "L2",
        "authors": "Moosavi-Dezfooli et al.",
        "desc": "Universal Adversarial Perturbation: single image-agnostic noise that fools the model on most inputs.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 10, "max": 500, "step": 10, "default": 100},
            "overshoot": {"label": "Overshoot", "min": 0.01, "max": 0.5, "step": 0.01, "default": 0.02},
        },
    },

    # ── Meta / Ensemble ────────────────────────────────────────────────
    "AutoAttack": {
        "cat": "Meta", "threat": "both", "year": 2020, "norm": "L∞/L2",
        "authors": "Croce & Hein",
        "desc": "THE gold standard. Ensemble of APGD-CE, APGD-DLR, FAB, Square Attack. Parameter-free robustness eval.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Budget per sub-attack", "min": 20, "max": 500, "step": 10, "default": 100},
            "aa_version": {"label": "Version", "type": "select", "options": ["standard", "fast"], "default": "standard"},
        },
    },

    # ── Research (SarabCraft) ──────────────────────────────────────────
    "SarabCraft R1": {
        "cat": "Research", "threat": "both", "year": 2026, "norm": "L∞",
        "authors": "SarabCraft Research",
        "desc": "SarabCraft's first in-house transfer-focused image attack. Standard mode prioritizes strong single-model transfer, while multi-image transfer mode unlocks broader bank-building strategies for harder evaluations.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 50, "max": 2000, "step": 50, "default": 300},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "kernel_size": {"label": "TI kernel size", "min": 3, "max": 15, "step": 2, "default": 5},
            "cfm_mix_prob": {"label": "CFM mix probability", "min": 0, "max": 0.5, "step": 0.01, "default": 0.1},
            "cfm_mix_upper": {"label": "CFM mix upper bound", "min": 0.1, "max": 1, "step": 0.05, "default": 0.75},
            "r1_multi_image_strategy": {"label": "Multi-image strategy", "type": "select", "options": ["tile_shuffle", "progressive", "self_aug_bank", "self_mix"], "default": "tile_shuffle", "hiddenInInfo": True},
            "r1_multi_image_count": {"label": "Image count", "min": 2, "max": 30, "step": 1, "default": 10, "hiddenInInfo": True},
        },
    },

    # ── CFM Paper Baseline (CVPR 2023) ────────────────────────────────
    "CFM Paper (CVPR 2023)": {
        "cat": "Transfer", "threat": "both", "year": 2023, "norm": "L∞",
        "authors": "Byun et al.",
        "desc": "Exact Config 578 from the CFM paper: CFM + RDI + MI + TI + Logit Loss. Strongest single-model transfer baseline prior to PHANTOM. Uses Resized Diverse Input (RDI) augmentation.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 50, "max": 2000, "step": 50, "default": 300},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "kernel_size": {"label": "TI kernel size", "min": 3, "max": 15, "step": 2, "default": 5},
            "cfm_mix_prob": {"label": "CFM mix prob", "min": 0, "max": 0.5, "step": 0.01, "default": 0.1},
            "cfm_mix_upper": {"label": "CFM mix upper", "min": 0.1, "max": 1, "step": 0.05, "default": 0.75},
        },
    },

    # ── Composite Baseline (TA-Bench) ─────────────────────────────────
    "NewBackend (TA-Bench)": {
        "cat": "Transfer", "threat": "both", "year": 2023, "norm": "L∞",
        "authors": "Qin et al. (NeurIPS 2023)",
        "desc": "UN+PI+DI+TI+NI+MI — strongest composite baseline from the Transfer-Attack Benchmark. Six combined techniques for maximum transfer.",
        "params": {
            "epsilon": {"label": "Epsilon (/255)", "min": 1, "max": 100, "step": 1, "default": 16},
            "iterations": {"label": "Iterations", "min": 10, "max": 2000, "step": 10, "default": 300},
            "momentum_decay": {"label": "Momentum (μ)", "min": 0, "max": 1, "step": 0.1, "default": 1.0},
            "nb_di_prob": {"label": "DI probability", "min": 0.1, "max": 1, "step": 0.1, "default": 0.5},
            "nb_di_resize_rate": {"label": "DI resize rate", "min": 0.5, "max": 1, "step": 0.05, "default": 0.9},
            "nb_ti_len": {"label": "TI translation (px)", "min": 0, "max": 15, "step": 1, "default": 3},
            "nb_npatch": {"label": "PI # patches", "min": 16, "max": 256, "step": 16, "default": 128},
            "nb_grid_scale": {"label": "PI grid scale", "min": 4, "max": 32, "step": 4, "default": 16},
            "nb_enable_un": {"label": "Uniform Noise (UN)", "type": "bool", "default": True},
            "nb_enable_pi": {"label": "Patch Interaction (PI)", "type": "bool", "default": True},
            "nb_enable_di": {"label": "Diverse Input (DI)", "type": "bool", "default": True},
            "nb_enable_ti": {"label": "Translation Inv. (TI)", "type": "bool", "default": True},
            "nb_enable_ni": {"label": "Nesterov (NI)", "type": "bool", "default": True},
        },
    },
}

AVAILABLE_ATTACKS = {k: v["desc"] for k, v in ATTACK_REGISTRY.items()}

# Backward-compatible simple list for audio attacks
AVAILABLE_ATTACKS_SIMPLE = list(ATTACK_REGISTRY.keys())


# ============================================================================
# AUDIO MODELS & ATTACKS
# ============================================================================

AVAILABLE_AUDIO_MODELS = {
    "[AST] Speech Commands (MIT)": "MIT/ast-finetuned-speech-commands-v2",
    "[AST] AudioSet (MIT)": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "[Wav2Vec2] Emotion Recognition": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    "[HuBERT] Speech Commands (SUPERB)": "superb/hubert-base-superb-ks",
    "[Wav2Vec2] Language ID (Meta)": "facebook/mms-lid-126",
}

PRELOADED_AUDIO_MODEL = None

AVAILABLE_AUDIO_ATTACKS = {
    "FGSM":        "Fast Gradient Sign Method — Goodfellow 2015",
    "I-FGSM (BIM)":"Basic Iterative Method — Kurakin 2017",
    "PGD":         "Projected Gradient Descent — Madry 2018",
    "MI-FGSM":     "Momentum Iterative FGSM — Dong 2018",
    "DeepFool":    "Minimal Perturbation — Moosavi-Dezfooli 2016",
    "C&W (L2)":    "Carlini-Wagner L2 Attack — Carlini & Wagner 2017",
}

# ============================================================================
# ASR (SPEECH-TO-TEXT) MODELS & ATTACKS
# ============================================================================

AVAILABLE_ASR_MODELS = {
    "[Whisper] Base.en (OpenAI)": "openai/whisper-base.en",
    "[Whisper] Small.en (OpenAI)": "openai/whisper-small.en",
    "[Whisper] Base (OpenAI, multilingual)": "openai/whisper-base",
    "[Whisper] Small (OpenAI, multilingual)": "openai/whisper-small",
}

AVAILABLE_TRANSCRIPTION_ATTACKS = {
    "Targeted Transcription": "Force ASR to output an attacker-chosen transcript — Carlini & Wagner 2018",
    "Hidden Command": "Embed voice command in carrier audio (music/noise) — CommanderSong 2018",
    "Universal Muting": "Prepend universal noise segment that mutes/overrides Whisper — Vyas et al. EMNLP 2024",
    "Psychoacoustic": "Hearing-threshold masked perturbations, truly imperceptible — Qin et al. ICML 2019",
    "Over-the-Air Robust": "Perturbations that survive speaker→air→mic playback — Schönherr et al. 2020",
    "Speech Jamming": "Denial-of-service: degrade ASR accuracy to gibberish — untargeted/band-noise",
}
