"""The gradient of a score with respect to a living weight. Measurement only.

Brian's ruling of 2026-11-16, 10:29 (``docs/DECISIONS.md``): measurement
code may take the gradient of a score with respect to a living weight, and
may step a living weight by it inside a measurement, when everything is
restored afterwards and nothing is kept. No living weight is trained by
gradient, in training or in a life; that rule stands.

The guards, which are part of the ruling, and where each is held:

1. Measurement code only; never an ``nn.Parameter`` or an optimizer. A
   living weight stays a buffer. Here it is put on the graph for the length
   of one ``with`` block (``requires_grad_`` on the buffer itself) and taken
   off in a ``finally``. Nothing in this module makes a Parameter or
   imports an optimizer, and nothing in the training path imports this
   module (tests/test_living_weight_gradient.py holds both).
2. Refused in training mode: ``living_weights_on_the_graph`` raises.
3. The model is bit-identical afterwards, and no buffer is left requiring a
   gradient (same tests). A caller that steps a weight restores it itself;
   ``adaptation_eval`` does, and its test says so.

And plasticity is frozen for as long as a weight is on the graph: a layer
that rewrites itself during the pass is not a fixed function to
differentiate. On the frozen path a living layer computes
``x @ (weight + recalled_episode).T`` and writes nothing
(``living_layer_pc.py``), so the gradient with respect to ``weight`` is that
of a fixed function.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch
import torch.nn as nn

from luthi.v2.living_layer_pc import PredictiveCodingLayer
from luthi.v2.plasticity import freeze_plasticity


def living_layers(model: nn.Module) -> list[PredictiveCodingLayer]:
    return [m for m in model.modules() if isinstance(m, PredictiveCodingLayer)]


@contextlib.contextmanager
def living_weights_on_the_graph(model: nn.Module) -> Iterator[list[PredictiveCodingLayer]]:
    """Every living weight under ``model`` requires a gradient inside the
    block, with plasticity frozen, and none does after it."""
    layers = living_layers(model)
    if not layers:
        raise RuntimeError("this model has no living layer; there is no living "
                           "weight to take a gradient for")
    training = [type(m).__name__ for m in (model, *layers) if m.training]
    if training:
        raise RuntimeError(
            "a gradient with respect to a living weight is a measurement and is "
            f"refused in training mode ({', '.join(sorted(set(training)))} "
            "in training mode); call .eval() first")
    already = [i for i, layer in enumerate(layers) if layer.weight.requires_grad]
    if already:
        raise RuntimeError(f"living weight(s) {already} already require a gradient; "
                           "something left them on the graph")
    with freeze_plasticity(model):
        try:
            for layer in layers:
                layer.weight.requires_grad_(True)
            yield layers
        finally:
            for layer in layers:
                layer.weight.requires_grad_(False)


def _clear(model: nn.Module) -> None:
    clear = getattr(model, "clear_forward_cache", None)
    if callable(clear):
        clear()


def score_and_its_gradient(loss_module: nn.Module, tail: torch.Tensor,
                           ) -> tuple[dict, list[torch.Tensor]]:
    """The frozen score of ``tail`` ([n, L] token windows) and the gradient
    of its ``nmse`` with respect to each living weight, in module order.

    ``nmse`` is the quantity E1 judges, as ``adaptation_eval._score`` takes
    it: the prediction loss over the variance of the target block. Both
    forwards go through the living weights, and so does the target and its
    variance, and a read changes all of them; so nothing is detached here.
    (The training loss detaches the target. This is not the training loss.)

    The text path of ``JEPALoss.compute_modality_loss`` is repeated here
    because that method returns its prediction loss detached. The two are
    held equal by a test, and a caller with ``_score`` in hand should compare.
    """
    model = loss_module.online_encoder
    if loss_module.training:
        raise RuntimeError("the loss module is in training mode; a gradient with "
                           "respect to a living weight is refused there")
    seq_len = tail.shape[1]
    ctx_len = int(seq_len * loss_module.context_fraction)
    if ctx_len < 1 or ctx_len >= seq_len:
        raise ValueError(f"bad context split: seq_len={seq_len}, ctx_len={ctx_len}")
    with living_weights_on_the_graph(model) as layers, torch.enable_grad():
        context = model.encode(text_tokens=tail[:, :ctx_len],
                               causal=False)["per_modality"]["text"]
        full = model.encode(text_tokens=tail, causal=False)["per_modality"]["text"]
        target = full[:, ctx_len:, :]
        positions = torch.arange(ctx_len, seq_len, device=tail.device,
                                 ).unsqueeze(0).expand(tail.shape[0], -1)
        predicted = loss_module.predictor(context, positions, loss_module.action_token)
        l_pred = (predicted - target).pow(2).mean()
        var = (target - target.mean(dim=(0, 1), keepdim=True)).pow(2).mean()
        if float(var.detach()) <= 1e-12:
            raise RuntimeError(f"the target's variance is {float(var.detach()):.3e}; "
                               "nmse has no gradient worth reading")
        nmse = l_pred / var
        if not torch.isfinite(nmse.detach()):
            raise RuntimeError(f"the score is {float(nmse.detach())}")
        grads = torch.autograd.grad(nmse, [layer.weight for layer in layers])
    _clear(model)
    for i, g in enumerate(grads):
        if not torch.isfinite(g).all():
            raise RuntimeError(f"the gradient for living layer {i} is not finite")
    return ({"l_pred": float(l_pred), "nmse": float(nmse)},
            [g.detach() for g in grads])


def step_living_weights_down_the_loss(loss_module: nn.Module, window: torch.Tensor, *,
                                      rate: float) -> list[float]:
    """One step by hand: every living weight moves by ``-rate`` times the
    gradient of the training loss of ``window`` ([1, L]). Returns each
    layer's step as a share of its weight's norm.

    The weights are left stepped. The caller restores them."""
    model = loss_module.online_encoder
    if loss_module.training:
        raise RuntimeError("the loss module is in training mode; a step of a "
                           "living weight by gradient is refused there")
    with living_weights_on_the_graph(model) as layers, torch.enable_grad():
        r = loss_module.compute_modality_loss("text", {"text_tokens": window})
        loss = r["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"step of the living weights: the loss is {float(loss)} "
                               f"at rate {rate}; nothing after it would mean anything")
        grads = torch.autograd.grad(loss, [layer.weight for layer in layers])
    shares = []
    with torch.no_grad():
        for layer, g in zip(layers, grads):
            if not torch.isfinite(g).all():
                raise RuntimeError("step of the living weights: a gradient is not finite")
            shares.append(float(rate * g.double().norm() / layer.weight.double().norm()))
            layer.weight.add_(g, alpha=-rate)
    _clear(model)
    return shares


__all__ = ["living_layers", "living_weights_on_the_graph",
           "score_and_its_gradient", "step_living_weights_down_the_loss"]
