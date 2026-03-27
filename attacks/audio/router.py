"""
Audio attack router: dispatches gradient-based attacks on audio waveforms.

The AudioModelWrapper makes model(waveform) return .logits, so all existing
gradient-based attacks work unchanged. This router handles audio-specific
parameter scaling (epsilon is in raw amplitude, not /255) and delegates
to the same classic attack implementations.
"""

from attacks.image.classic.fgsm import targeted_fgsm
from attacks.image.classic.ifgsm import targeted_bim
from attacks.image.classic.pgd import targeted_pgd
from attacks.image.classic.mi_fgsm import targeted_mi_fgsm
from attacks.image.classic.deepfool import targeted_deepfool
from attacks.image.classic.cw import targeted_cw


def run_audio_attack_method(attack_name, model_wrapper, waveform, target_class,
                            epsilon, iterations, params=None):
    """
    Route to the appropriate attack for audio waveforms.

    Args:
        attack_name: attack method name
        model_wrapper: AudioModelWrapper (forward takes [1, samples], returns .logits)
        waveform: tensor [1, num_samples] on device
        target_class: int target class index
        epsilon: perturbation magnitude in raw amplitude (0.0 - 1.0 range)
        iterations: number of attack iterations
        params: dict of extra parameters
    """
    if params is None:
        params = {}

    alpha = params.get('alpha', 1.0)
    momentum_decay = params.get('momentum_decay', 1.0)
    random_start = params.get('random_start', True)
    overshoot = params.get('overshoot', 0.02)
    cw_confidence = params.get('cw_confidence', 0.0)
    cw_lr = params.get('cw_lr', 0.01)
    cw_c = params.get('cw_c', 1.0)

    if attack_name == "FGSM":
        return targeted_fgsm(model_wrapper, waveform, target_class, epsilon)

    elif attack_name == "I-FGSM (BIM)":
        return targeted_bim(model_wrapper, waveform, target_class, epsilon,
                            iterations, alpha)

    elif attack_name == "PGD":
        return targeted_pgd(model_wrapper, waveform, target_class, epsilon,
                            iterations, alpha, random_start)

    elif attack_name == "MI-FGSM":
        return targeted_mi_fgsm(model_wrapper, waveform, target_class, epsilon,
                                iterations, momentum_decay, alpha)

    elif attack_name == "DeepFool":
        return targeted_deepfool(model_wrapper, waveform, target_class, epsilon,
                                 iterations, overshoot)

    elif attack_name == "C&W (L2)":
        return targeted_cw(model_wrapper, waveform, target_class, epsilon,
                           iterations, cw_c, cw_lr, cw_confidence)

    else:
        raise ValueError(f"Unknown audio attack method: {attack_name}. Supported: FGSM, I-FGSM (BIM), PGD, MI-FGSM, DeepFool, C&W (L2)")
