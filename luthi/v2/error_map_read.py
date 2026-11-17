"""A read that moves living weights by a fixed map of the prediction's error.
Measurement only. (Brief 8, Part 2, 2026-11-17.)

``docs/research/2026-11-17_a-write-that-hears-the-prediction-error.md``
(chat Fable) took the helpful direction for a living weight apart. Its
learning part (the target held still) is nearly an offset: for one window,
the gradient for a layer's weight is close to

    (the summed gradient for its context tokens' outputs) x (their mean input)

and the first factor is one vector a window. The model has one vector in
hand with no backward pass: the mean error of its prediction over the target
tokens. On July seed 42 one fixed linear map from the second to the first,
fitted on other documents, gives it with a cosine of about 0.65.

This module is that as a way of reading, so that E1's harness can ask what
it buys:

``fit_error_maps``  on windows the read will never see, with a backward
    pass (for the layers' OUTPUTS; no gradient is taken for a living weight
    here), fit for each living layer the ridge map of the probe.

``read_by_error_map``  for each window read, from the forward pass alone and
    with plasticity frozen (so every other way a read reaches the score is
    off): the prediction's mean error ``e`` and each living layer's mean
    input ``x`` over the context tokens; then, by hand,

        weight -= rate * outer(wanted_mean + F (e - e_mean), x)

    No backward pass during the read.

Units. The map is fitted to the gradient of the score's own quantity with
the target held (prediction loss over the target's variance), and the
window's gradient for the weight is ``outer(summed output gradient, MEAN
input)``. The brief's wording has the summed inputs; with a map fitted to
the summed output gradient that would be the gradient times the number of
context tokens, so the mean is used. M2 steps down the TRAINING loss, whose
scale is not this one's. So "the same rate" is not the same step: each
step's size is recorded as a share of the weight, as M2's is.

Three fitted kinds, for the comparison the brief asks for:

    "map"       as above
    "shuffled"  the map fitted with the errors shuffled between windows:
                does the error's content matter, or only its common part?
    "common"    no map: every window moves the weight by the fitted mean

Four kinds that need nothing fitted (addenda 1 and 2 to brief 8, after
Part 1 and Opus's review of 2026-11-17). They take nothing from a gradient
anywhere:

    "error"           weight -= rate * outer(e, x): the first arm. ``e`` is
                      the mean over the target tokens of (the prediction
                      minus what was then perceived); ``x`` the layer's mean
                      input over the context tokens.
    "error_turned"    the same with the sign turned. It should hurt, on the
                      held score (``adaptation_eval``'s ``held_score``).
    "error_shuffled"  ``e`` taken from a window of another document.
    "bias"            the weight is not touched. Each living layer's OUTPUT
                      is shifted by a vector that moves, a window, by what
                      "error" at the same rate would add to the output of
                      the layer's mean input: -rate * (x . x) * e. If it does
                      what "error" does, the weight is adding nothing to a
                      shift (Opus). The addendum writes the shift as
                      -rate * e; the (x . x) is there so that one rate means
                      the same shift for both kinds.

Under Brian's ruling of 2026-11-16 a measurement may step a living weight
by hand when everything is restored afterwards and nothing is kept. The
weights are left stepped here; the caller restores them (``adaptation_eval``
does). Refused in training mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from luthi.v2.living_weight_gradient import living_layers
from luthi.v2.plasticity import freeze_plasticity

KINDS = ("map", "shuffled", "common")                                  # fitted
UNFITTED = ("error", "error_turned", "error_shuffled", "bias")         # nothing fitted


@dataclass
class ErrorMaps:
    kind: str
    e_mean: torch.Tensor                 # [d]
    wanted_mean: list[torch.Tensor]      # per living layer, [out]
    maps: list[torch.Tensor]             # per living layer, [out, d]
    fitted_on: int
    ridge: float
    check: dict                          # on fit windows held back from the fit


def unfitted(kind: str) -> ErrorMaps:
    """What ``read_by_error_map`` takes for a kind that needs nothing fitted."""
    if kind not in UNFITTED:
        raise ValueError(f"kind must be one of {UNFITTED}; got {kind!r}")
    return ErrorMaps(kind=kind, e_mean=torch.zeros(0), wanted_mean=[], maps=[],
                     fitted_on=0, ridge=0.0, check={})


class LivingBias:
    """A vector added to each living layer's output, for the "bias" kind.
    Zero until a read moves it. ``remove()`` takes the hooks away."""

    def __init__(self, model: nn.Module) -> None:
        self.layers = living_layers(model)
        self.vectors = [torch.zeros(layer.weight.shape[0], dtype=torch.float64,
                                    device=layer.weight.device) for layer in self.layers]
        self._handles = [layer.register_forward_hook(self._shift(i))
                         for i, layer in enumerate(self.layers)]

    def _shift(self, i: int):
        def hook(module, inputs, output):
            return output + self.vectors[i].to(output.dtype)
        return hook

    def zero(self) -> None:
        for v in self.vectors:
            v.zero_()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def _clear(model: nn.Module) -> None:
    clear = getattr(model, "clear_forward_cache", None)
    if callable(clear):
        clear()


def _refuse_training(loss_module: nn.Module) -> None:
    model = loss_module.online_encoder
    if loss_module.training or model.training:
        raise RuntimeError("a read by error map steps living weights by hand; it is a "
                           "measurement and is refused in training mode")


def _window_signals(loss_module: nn.Module, window: torch.Tensor, *, wanted: bool,
                    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor] | None]:
    """For one window ([1, L]), plasticity frozen: the prediction's mean
    error over the target tokens; each living layer's mean input over the
    context tokens; and, with ``wanted``, the summed gradient of the score
    (target held) for each layer's context-token outputs."""
    model = loss_module.online_encoder
    layers = living_layers(model)
    seq_len = window.shape[1]
    ctx_len = int(seq_len * loss_module.context_fraction)
    if ctx_len < 1 or ctx_len >= seq_len:
        raise ValueError(f"bad context split: seq_len={seq_len}, ctx_len={ctx_len}")
    index = {id(layer): i for i, layer in enumerate(layers)}
    calls: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    def hook(module, inputs, output):
        if wanted:
            output.retain_grad()
        calls.append((index[id(module)], inputs[0].detach(), output))

    handles = [layer.register_forward_hook(hook) for layer in layers]
    try:
        with freeze_plasticity(model), torch.set_grad_enabled(wanted):
            context = model.encode(text_tokens=window[:, :ctx_len],
                                   causal=False)["per_modality"]["text"]
            n_context_calls = len(calls)
            full = model.encode(text_tokens=window, causal=False)["per_modality"]["text"]
            target = full[:, ctx_len:, :].detach()
            positions = torch.arange(ctx_len, seq_len, device=window.device,
                                     ).unsqueeze(0).expand(window.shape[0], -1)
            predicted = loss_module.predictor(context, positions, loss_module.action_token)
            error = (predicted - target).detach()
            if n_context_calls != len(layers) or len(calls) != 2 * len(layers):
                raise RuntimeError(
                    f"{len(layers)} living layers were called {len(calls)} times; a read by "
                    "error map expects each once for the context and once for the window")
            grads = None
            if wanted:
                var = (target - target.mean(dim=(0, 1), keepdim=True)).pow(2).mean()
                held = (predicted - target).pow(2).mean() / var
                grads = torch.autograd.grad(held, [c[2] for c in calls[:n_context_calls]])
    finally:
        for h in handles:
            h.remove()
    _clear(model)
    e = error.reshape(-1, error.shape[-1]).double().mean(0)
    if not torch.isfinite(e).all():
        raise RuntimeError("the prediction's error is not finite")
    x_mean: list = [None] * len(layers)
    summed: list = [None] * len(layers)
    for k, (i, x, _) in enumerate(calls[:n_context_calls]):
        x_mean[i] = x.reshape(-1, x.shape[-1]).double().mean(0)
        if grads is not None:
            summed[i] = grads[k].detach().reshape(-1, grads[k].shape[-1]).double().sum(0)
    return e, x_mean, (summed if wanted else None)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm() * b.norm())
    return float((a * b).sum()) / denom if denom > 0 else float("nan")


def fit_error_maps(loss_module: nn.Module, windows: torch.Tensor, *, kind: str = "map",
                   ridge: float = 1e-2, held_back: float = 0.25, seed: int = 0) -> ErrorMaps:
    """Fit on ``windows`` ([n, L]), one window at a time. The last
    ``held_back`` of them are not fitted on: the map is read on them, and
    that reading is returned with it. The model is not changed."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}; got {kind!r}")
    _refuse_training(loss_module)
    n = windows.shape[0]
    n_fit = n - int(round(n * held_back))
    if n_fit < 8 or n - n_fit < 2:
        raise ValueError(f"{n} windows are too few to fit on and hold some back")
    errors, wanted = [], []
    for w in windows:
        e, _, summed = _window_signals(loss_module, w.unsqueeze(0), wanted=True)
        errors.append(e)
        wanted.append(summed)
    n_layers = len(wanted[0])
    E = torch.stack(errors)
    E_fit, E_check = E[:n_fit], E[n_fit:]
    e_mean = E_fit.mean(0)
    A = E_fit - e_mean
    if kind == "shuffled":
        perm = torch.randperm(n_fit, generator=torch.Generator().manual_seed(seed)).to(A.device)
        A = A[perm]
    lam = ridge * float((A * A).sum() / A.shape[0])
    gram = A @ A.T + lam * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    wanted_mean, maps, check = [], [], []
    for layer in range(n_layers):
        U = torch.stack([w[layer] for w in wanted])
        U_fit, U_check = U[:n_fit], U[n_fit:]
        u_mean = U_fit.mean(0)
        if kind == "common":
            F = torch.zeros(U.shape[1], E.shape[1], dtype=U.dtype, device=U.device)
        else:
            F = (U_fit - u_mean).T @ torch.linalg.solve(gram, A)      # [out, d], through the dual
        given = u_mean + (E_check - e_mean) @ F.T
        cos = sorted(_cosine(a, b) for a, b in zip(given, U_check))
        removed = sorted(_cosine(a - u_mean, b - u_mean) for a, b in zip(given, U_check))
        sizes = sorted(float(a.norm() / b.norm()) for a, b in zip(given, U_check))
        check.append({
            "cosine": cos[len(cos) // 2],
            "cosine_common_part_removed": (None if kind == "common"
                                           else removed[len(removed) // 2]),
            "size_given_over_wanted": sizes[len(sizes) // 2],
        })
        wanted_mean.append(u_mean)
        maps.append(F)
    return ErrorMaps(kind=kind, e_mean=e_mean, wanted_mean=wanted_mean, maps=maps,
                     fitted_on=n_fit, ridge=ridge,
                     check={"windows_held_back": n - n_fit, "median_by_layer": check})


@torch.no_grad()
def read_by_error_map(loss_module: nn.Module, windows: torch.Tensor, maps: ErrorMaps, *,
                      rate: float, repeats: int = 1,
                      step_shares: list[list[float]] | None = None,
                      errors_from: torch.Tensor | None = None,
                      bias: LivingBias | None = None,
                      before_each=None) -> None:
    """Read ``windows`` ([n, L]), one at a time, moving every living weight
    after each by the rule in the module docstring. No backward pass. The
    weights are left moved; the caller restores them.

    ``errors_from``: for "error_shuffled", another document's windows; window
    k takes its error from window k of these. ``bias``: for "bias", the
    shifts to move. ``before_each(window)``: called before each window's
    step, for a read in context (the caller makes the plastic forward there).

    Each step's size is appended to ``step_shares`` as a share of the
    layer's weight; for "bias", as a share of the layer's output for its
    mean input."""
    _refuse_training(loss_module)
    layers = living_layers(loss_module.online_encoder)
    kind = maps.kind
    if kind in KINDS and len(layers) != len(maps.maps):
        raise ValueError(f"the maps are for {len(maps.maps)} living layers; "
                         f"this model has {len(layers)}")
    if kind not in KINDS + UNFITTED:
        raise ValueError(f"unknown kind {kind!r}")
    if (kind == "error_shuffled") != (errors_from is not None):
        raise ValueError("errors_from is given for 'error_shuffled' and for no other kind")
    if (kind == "bias") != (bias is not None):
        raise ValueError("a LivingBias is given for 'bias' and for no other kind")
    if errors_from is not None and errors_from.shape[0] < windows.shape[0]:
        raise ValueError("errors_from has fewer windows than are read")
    for _ in range(repeats):
        for k, w in enumerate(windows):
            if before_each is not None:
                before_each(w)
            e, x_mean, _ = _window_signals(loss_module, w.unsqueeze(0), wanted=False)
            if errors_from is not None:
                e, _, _ = _window_signals(loss_module, errors_from[k].unsqueeze(0), wanted=False)
            shares = []
            for i, (layer, x) in enumerate(zip(layers, x_mean)):
                if kind in KINDS:
                    given = maps.wanted_mean[i] + maps.maps[i] @ (e - maps.e_mean)
                elif kind == "error_turned":
                    given = -e
                else:
                    given = e
                if given.shape[0] != layer.weight.shape[0]:
                    raise RuntimeError(
                        f"the error is {given.shape[0]} wide and living layer {i}'s output "
                        f"is {layer.weight.shape[0]}; this rule needs them equal")
                if kind == "bias":
                    move = given * float(x @ x)
                    if not torch.isfinite(move).all():
                        raise RuntimeError("a step of the bias is not finite")
                    typical = (layer.weight.double() @ x).norm()
                    shares.append(float(rate * move.norm() / typical))
                    bias.vectors[i].add_(move, alpha=-rate)
                    continue
                step = torch.outer(given, x)
                if not torch.isfinite(step).all():
                    raise RuntimeError("a step by the error is not finite")
                shares.append(float(rate * step.norm() / layer.weight.double().norm()))
                layer.weight.add_(step.to(layer.weight.dtype), alpha=-rate)
            if step_shares is not None:
                step_shares.append(shares)


def step_share_at_rate_one(loss_module: nn.Module, windows: torch.Tensor) -> list[float]:
    """For setting a rate from size: the median, over ``windows``, of one
    "error" step at rate 1 as a share of each living layer's weight. Nothing
    is moved."""
    _refuse_training(loss_module)
    layers = living_layers(loss_module.online_encoder)
    by_layer: list[list[float]] = [[] for _ in layers]
    with torch.no_grad():
        for w in windows:
            e, x_mean, _ = _window_signals(loss_module, w.unsqueeze(0), wanted=False)
            for i, (layer, x) in enumerate(zip(layers, x_mean)):
                by_layer[i].append(float(e.norm() * x.norm() / layer.weight.double().norm()))
    return [sorted(v)[len(v) // 2] for v in by_layer]


def windows_outside_documents(total: int, documents: Sequence[tuple[int, int]],
                              count: int) -> list[int]:
    """``count`` window indices, spread evenly, that lie in none of
    ``documents`` (each a ``(first, past the last)`` range). Raises when
    there are not that many: a fit on fewer windows than asked for would
    pass for the one asked for."""
    used = torch.zeros(total, dtype=torch.bool)
    for first, stop in documents:
        if not 0 <= first < stop <= total:
            raise ValueError(f"a document's windows {first}:{stop} are outside 0:{total}")
        used[first:stop] = True
    free = (~used).nonzero().flatten().tolist()
    if len(free) < count:
        raise ValueError(f"{len(free)} held-out windows are in no document; {count} were asked for")
    return [free[(k * len(free)) // count] for k in range(count)]


__all__ = ["ErrorMaps", "KINDS", "UNFITTED", "LivingBias", "fit_error_maps",
           "read_by_error_map", "step_share_at_rate_one", "unfitted",
           "windows_outside_documents"]
