"""
AutoAttack — Croce & Hein, ICML 2020
"Reliable evaluation of adversarial robustness with an ensemble of
diverse parameter-free attacks"
https://arxiv.org/abs/2003.01690

Ensemble of complementary attacks: APGD-CE, APGD-DLR, FAB, Square.
THE gold standard for robustness evaluation. Parameter-free.
"""

import torch
import torch.nn.functional as F
from config import device

from attacks.classic.apgd import targeted_apgd
from attacks.classic.fab import targeted_fab
from attacks.blackbox.square import targeted_square


def targeted_autoattack(model, img_tensor, target_class, epsilon,
                        iterations=100, version='standard'):
    """
    AutoAttack: runs APGD-CE, APGD-DLR, FAB, Square sequentially.
    Returns the first successful adversarial example.
    version: 'standard' (all 4), 'fast' (APGD-CE + Square only).
    """
    target_t = torch.tensor([target_class], dtype=torch.long, device=device)

    attacks = []
    if version == 'fast':
        attacks = [
            ('APGD-CE', lambda: targeted_apgd(model, img_tensor, target_class,
                                               epsilon, iterations, loss_type='ce')),
            ('Square', lambda: targeted_square(model, img_tensor, target_class,
                                               epsilon, n_queries=iterations * 10)),
        ]
    else:
        iters_each = max(iterations // 4, 20)
        attacks = [
            ('APGD-CE', lambda: targeted_apgd(model, img_tensor, target_class,
                                               epsilon, iters_each, loss_type='ce')),
            ('APGD-DLR', lambda: targeted_apgd(model, img_tensor, target_class,
                                                epsilon, iters_each, loss_type='dlr')),
            ('FAB', lambda: targeted_fab(model, img_tensor, target_class,
                                         epsilon, iters_each)),
            ('Square', lambda: targeted_square(model, img_tensor, target_class,
                                               epsilon, n_queries=iters_each * 20)),
        ]

    best_adv = img_tensor.clone()
    for name, attack_fn in attacks:
        print(f"[AutoAttack] Running {name}...", flush=True)
        try:
            adv = attack_fn()
            with torch.no_grad():
                logits = model(adv).logits
                if logits.argmax(dim=1).item() == target_class:
                    print(f"[AutoAttack] {name} succeeded!", flush=True)
                    return adv.detach()
            best_adv = adv
        except Exception as e:
            print(f"[AutoAttack] {name} failed: {e}", flush=True)

    print("[AutoAttack] No single attack succeeded, returning best attempt.", flush=True)
    return best_adv.detach()
