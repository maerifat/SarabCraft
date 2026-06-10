"""
SCTC (Spatially-Calibrated Token Competition) objective.
========================================================

This module isolates the *novel* part of SCTC: the calibrated per-spatial-token
target-competition loss term that distinguishes it from the S4ST baseline.

Faithful reference (the experiment that produced the paper's SOTA numbers):
``lab/targeted_transfer_bench/proto_rapa/results_pod/h2h_final/run_h2h.py``

In that reference the surrogate is a fixed ResNet-50, so the spatial grid is its
``layer4`` output and the classifier head is its ``fc`` weight. The app, however,
wraps arbitrary HuggingFace / timm / torchvision models, so we cannot hard-code
those two modules. Instead, ``SctcGridTap`` discovers, per model, the last 4D
spatial feature map and the final classifier (a ``Linear`` head, or a ``1x1``
``Conv2d`` head), and applies that head convolutionally at every spatial token.

If a usable (grid, head) pair cannot be found for a given surrogate (e.g. a pure
ViT whose final block does not expose a 4D grid), the tap reports
``available == False`` and the caller should fall back to the plain logit /
margin objective so nothing breaks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def nce_calibrate(logits, temp, dim=1):
    """Zero-mean / unit-std / temperature-scaled calibration across classes.

    Identical to the reference ``nce_calibrate`` in ``run_h2h.py``: this puts the
    1000-way logit vector at every location on a fair z-scored scale so that no
    single high-magnitude token dominates the competition.
    """
    mu = logits.mean(dim=dim, keepdim=True)
    sigma = logits.std(dim=dim, keepdim=True).clamp(min=1e-6)
    return (logits - mu) / (sigma * temp)


def _unwrap_inner(model):
    """Best-effort descent to the real backbone.

    The app wraps models several layers deep (cancel wrapper -> pixel wrapper ->
    HF/timm/torchvision module). We unwrap common container attributes so the
    hook attaches to actual ``Conv2d`` / ``Linear`` modules.
    """
    seen = set()
    cur = model
    for _ in range(8):
        if id(cur) in seen or cur is None:
            break
        seen.add(id(cur))
        nxt = None
        for attr in ("wrapped_model", "hf_model", "model", "module"):
            cand = getattr(cur, attr, None)
            if isinstance(cand, nn.Module):
                nxt = cand
                break
        if nxt is None:
            break
        cur = nxt
    return cur


def _find_classifier_head(inner):
    """Return (weight, bias) of the final classifier as a [num_classes, C] matrix.

    Supports a trailing ``Linear`` head (the common CNN/ViT case) and a trailing
    ``1x1 Conv2d`` head. Returns ``(None, None)`` when no usable head is found.
    """
    last_linear = None
    last_conv1x1 = None
    for mod in inner.modules():
        if isinstance(mod, nn.Linear):
            last_linear = mod
        elif isinstance(mod, nn.Conv2d) and mod.kernel_size == (1, 1):
            last_conv1x1 = mod

    if last_linear is not None:
        w = last_linear.weight.detach()                       # [num_classes, C]
        b = last_linear.bias.detach() if last_linear.bias is not None else None
        return w, b
    if last_conv1x1 is not None:
        w = last_conv1x1.weight.detach().flatten(1)            # [num_classes, C]
        b = last_conv1x1.bias.detach() if last_conv1x1.bias is not None else None
        return w, b
    return None, None


class SctcGridTap:
    """Taps the last 4D spatial feature map of a wrapped model without altering
    its forward path, and exposes the (detached) classifier head so the caller
    can build per-spatial-token logits.

    Only attaches when the tapped conv is a genuine *pre-classifier* feature grid
    (CNN-style: deep in the network, channel count matching the classifier head's
    input). Architectures without such a grid — notably pure ViTs, whose only
    ``Conv2d`` is an early patch-embedding that does not feed the head — report
    ``available == False`` so the caller falls back to the plain logit objective.

    Usage::

        tap = SctcGridTap(model)
        if tap.available:
            logits = model(x)                  # forward populates the grid
            term = tap.token_term(targets, temp)
        tap.remove()
    """

    def __init__(self, model):
        self.available = False
        self._handle = None
        self._grid = {}
        self.fc_w = None
        self.fc_b = None

        inner = _unwrap_inner(model)
        if inner is None:
            return

        self.fc_w, self.fc_b = _find_classifier_head(inner)
        if self.fc_w is None:
            return

        expected_c = self.fc_w.shape[1]
        candidates = [m for m in inner.modules()
                      if isinstance(m, nn.Conv2d) and m.out_channels == expected_c]
        if not candidates:
            return

        # Channel match alone is ambiguous: a ViT's patch-embed conv can share the
        # head width yet sit at the very FRONT of the network (its grid is not a
        # pre-classifier feature map). Trace leaf-module execution order on a dummy
        # forward and require the candidate conv to fire in the LAST quarter of the
        # network — the regime where it is genuinely the deep grid feeding the head.
        target = candidates[-1]
        order = self._trace_leaf_order(inner)
        if order is not None:
            pos = order.get(id(target))
            if pos is None or pos < 0.6 * order["__count__"]:
                return  # CNN-style deep grid not found -> fall back

        self._target_module = target
        self._expected_c = expected_c
        self._handle = self._target_module.register_forward_hook(self._hook)
        self.available = True

    @staticmethod
    def _trace_leaf_order(inner):
        """Return {id(module): execution_index, '__count__': n} for leaf modules,
        or None if a dummy forward is not possible."""
        order = {}
        counter = {"i": 0}
        handles = []

        def mk(mod):
            def fn(m, i, o):
                if id(m) not in order:
                    order[id(m)] = counter["i"]
                    counter["i"] += 1
            return fn

        leaves = [m for m in inner.modules() if not list(m.children())]
        for m in leaves:
            handles.append(m.register_forward_hook(mk(m)))
        try:
            inner.eval()
            with torch.no_grad():
                dev = next(inner.parameters()).device
                inner(torch.zeros(1, 3, 224, 224, device=dev))
        except Exception:
            for h in handles:
                h.remove()
            return None
        for h in handles:
            h.remove()
        if counter["i"] == 0:
            return None
        order["__count__"] = counter["i"]
        return order

    def _hook(self, mod, inp, out):
        if isinstance(out, torch.Tensor) and out.dim() == 4 and out.shape[1] == self._expected_c:
            self._grid["f"] = out

    def grid(self):
        return self._grid.get("f")

    def token_term(self, targets, temp):
        """Calibrated per-spatial-token target competition (the novel SCTC term).

        Mirrors ``sctc_token_term`` in the reference experiment:
          1. apply the classifier head at every spatial location (1x1 conv),
          2. calibrate the 1000-way logits per location,
          3. log-softmax so classes compete at each token,
          4. reward the target's log-prob at every token, averaged spatially.

        Returns a scalar to be *maximised* (summed over the batch), or ``None``
        if the grid was not populated / channel count mismatched.
        """
        grid = self.grid()
        if grid is None:
            return None
        b, c, h, w = grid.shape
        if c != self.fc_w.shape[1]:
            return None
        weight = self.fc_w.to(grid.device).view(self.fc_w.shape[0], c, 1, 1)
        bias = self.fc_b.to(grid.device) if self.fc_b is not None else None
        loc_logits = F.conv2d(grid, weight, bias=bias)         # [B, num_classes, H, W]
        z = nce_calibrate(loc_logits, temp, dim=1)
        logp = F.log_softmax(z, dim=1)
        tgt = targets.view(b, 1, 1, 1).expand(b, 1, h, w)
        tgt_logp = logp.gather(1, tgt).squeeze(1)              # [B, H, W]
        return tgt_logp.mean(dim=[1, 2]).sum()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._grid.clear()
