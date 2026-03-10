"""
SPSA (Simultaneous Perturbation Stochastic Approximation) — Uesato et al., 2018
"Adversarial Risk and the Dangers of Evaluating Against Weak Attacks"
https://arxiv.org/abs/1802.05666

Score-based black-box attack that estimates gradients using random
perturbation pairs. Works with only output probabilities.
"""

import torch
import torch.nn.functional as F
from config import device


def targeted_spsa(model, img_tensor, target_class, epsilon,
                  iterations=100, delta=0.01, lr=0.01, nb_sample=128,
                  max_batch_size=64):
    """
    SPSA: estimate gradients via random perturbation differences.
    delta: perturbation magnitude for gradient estimation.
    nb_sample: number of samples for gradient approximation.
    """
    x0 = img_tensor.clone().detach().to(device)
    adv = x0.clone()
    target_t = torch.tensor([target_class], dtype=torch.long, device=device)
    flat_shape = x0.numel()

    for t in range(int(iterations)):
        est_grad = torch.zeros(flat_shape, device=device)

        n_batches = (nb_sample + max_batch_size - 1) // max_batch_size
        for batch_i in range(n_batches):
            bs = min(max_batch_size, nb_sample - batch_i * max_batch_size)
            v = torch.sign(torch.randn(bs, flat_shape, device=device))

            x_plus = (adv.view(1, -1) + delta * v).view(bs, *x0.shape[1:]).clamp(0., 1.)
            x_minus = (adv.view(1, -1) - delta * v).view(bs, *x0.shape[1:]).clamp(0., 1.)

            with torch.no_grad():
                logits_plus = model(x_plus).logits
                logits_minus = model(x_minus).logits

            loss_plus = F.cross_entropy(logits_plus, target_t.expand(bs), reduction='none')
            loss_minus = F.cross_entropy(logits_minus, target_t.expand(bs), reduction='none')

            diff = (loss_plus - loss_minus).view(bs, 1) / (2 * delta)
            est_grad += (diff * v).sum(dim=0)

        est_grad /= nb_sample

        with torch.no_grad():
            adv_flat = adv.view(-1) - lr * est_grad.sign()
            adv = adv_flat.view(x0.shape)
            adv = x0 + torch.clamp(adv - x0, -epsilon, epsilon)
            adv = adv.clamp(0., 1.)

    return adv.detach()
