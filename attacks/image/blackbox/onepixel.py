"""
One Pixel Attack — Su, Vargas & Kouichi, 2019
"One pixel attack for fooling deep neural networks"
https://arxiv.org/abs/1710.08864

L0 black-box attack using differential evolution to find optimal
pixels to modify. Changes as few as 1-5 pixels.
"""

import torch
import numpy as np
from config import device


def targeted_onepixel(model, img_tensor, target_class, epsilon=None,
                      pixels=5, popsize=400, iterations=75, F_de=0.5, CR=1.0):
    """
    Differential evolution to find best n pixels to change.
    pixels: number of pixels to modify (1-10).
    popsize: DE population size.
    F_de: DE mutation factor.
    CR: DE crossover rate.
    """
    x0 = img_tensor.clone().detach().to(device)
    B, C, H, W = x0.shape
    n_pix = int(pixels)
    dim = n_pix * (2 + C)  # each pixel: (x, y, r, g, b)

    bounds_lo = np.zeros(dim)
    bounds_hi = np.zeros(dim)
    for p in range(n_pix):
        base = p * (2 + C)
        bounds_lo[base] = 0
        bounds_hi[base] = W - 1
        bounds_lo[base + 1] = 0
        bounds_hi[base + 1] = H - 1
        for c in range(C):
            bounds_lo[base + 2 + c] = 0.
            bounds_hi[base + 2 + c] = 1.

    population = np.random.rand(popsize, dim)
    for d in range(dim):
        population[:, d] = bounds_lo[d] + population[:, d] * (bounds_hi[d] - bounds_lo[d])

    def _apply_perturbation(individual):
        img = x0.clone()
        for p in range(n_pix):
            base = p * (2 + C)
            px = int(np.clip(individual[base], 0, W - 1))
            py = int(np.clip(individual[base + 1], 0, H - 1))
            for c in range(C):
                img[0, c, py, px] = float(np.clip(individual[base + 2 + c], 0., 1.))
        return img

    def _fitness(pop):
        scores = np.zeros(len(pop))
        with torch.no_grad():
            for idx in range(0, len(pop), 32):
                batch_imgs = torch.cat([_apply_perturbation(pop[i])
                                        for i in range(idx, min(idx + 32, len(pop)))], dim=0)
                logits = model(batch_imgs).logits
                probs = F.softmax(logits, dim=1)
                target_probs = probs[:, target_class].cpu().numpy()
                scores[idx:idx + len(target_probs)] = target_probs
        return scores

    import torch.nn.functional as F

    fitness = _fitness(population)
    best_idx = np.argmax(fitness)
    best = population[best_idx].copy()
    best_score = fitness[best_idx]

    for gen in range(int(iterations)):
        for i in range(popsize):
            idxs = [j for j in range(popsize) if j != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = a + F_de * (b - c)
            for d in range(dim):
                mutant[d] = np.clip(mutant[d], bounds_lo[d], bounds_hi[d])

            crossover = np.random.rand(dim) < CR
            if not crossover.any():
                crossover[np.random.randint(dim)] = True
            trial = np.where(crossover, mutant, population[i])

            trial_score = _fitness(trial[np.newaxis])[0]
            if trial_score > fitness[i]:
                population[i] = trial
                fitness[i] = trial_score
                if trial_score > best_score:
                    best = trial.copy()
                    best_score = trial_score

        result = _apply_perturbation(best)
        with torch.no_grad():
            if model(result).logits.argmax(dim=1).item() == target_class:
                return result.detach()

    return _apply_perturbation(best).detach()
