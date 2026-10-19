"""Instruments for the living channel itself (2026-10-19).

Every gauge before this module read the REPRESENTATION (rank, chorus,
offset, probe) or the substrate's own counters (update_ema, duty). None
read what the living change IS. These do, and each one exists because of
a specific way the channel could be alive in name only:

* :func:`living_change_spectrum` -- the shape of the accumulated living
  change ``weight - set_point`` (homeostasis pulls weight toward set_point,
  and set_point itself adapts at ~1e-6/step, so the difference is the
  living displacement from home). Under the mean-field write every write's
  output side is the batch-mean output, so the change is ~rank-1 along it;
  ``top_share`` near 1 with ``out_align`` near 1 is that signature, readable
  from any checkpoint without training.
* :func:`write_diversity` -- whether DIFFERENT inputs produce DIFFERENT
  writes from the same state. A write rule that answers every input the
  same way is decoration, however large its updates. Measured by deep-
  copying the model, running one self-modifying forward per input, and
  comparing the resulting weight deltas.
* :func:`batch_mean_stationarity` -- how much the statistic a batch-level
  rule consumes actually moves between batches. Cosine near 1 means the
  rule sees the same input every step, whatever documents are in the batch.

All functions are read-only on the model passed in (write_diversity works
on deep copies) and never raise into a training run: the runner wraps them.
"""

from __future__ import annotations

import copy
import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from luthi.v2.living_layer_pc import PredictiveCodingLayer
from luthi.v2.plasticity import freeze_plasticity


def _entropy_rank(energy: torch.Tensor) -> float:
    total = float(energy.sum())
    if not total > 0.0:
        return float("nan")
    p = energy / total
    return float(math.exp(float(-(p * torch.log(p.clamp(min=1e-30))).sum())))


@torch.no_grad()
def living_change_spectrum(layer: PredictiveCodingLayer) -> dict:
    """Spectrum of the living displacement ``weight - set_point``.

    Returns ``{"top_share", "eff_rank", "rel_norm", "out_align"}``:

    * ``top_share`` -- largest singular value's share of the displacement's
      energy (sum of squared singular values). 1.0 = rank-1.
    * ``eff_rank`` -- exp(entropy) of the energy distribution.
    * ``rel_norm`` -- ||weight - set_point||_F / ||set_point||_F, the size of
      the living change relative to the weight's home.
    * ``out_align`` -- |cos| between the top left singular vector and the
      layer's last batch-mean output (``_last_output_mean``); NaN until a
      self-modifying forward has run. Near 1 = the change lives along the
      direction every mean-field write shares.
    """
    nan = float("nan")
    d = (layer.weight.detach().float() - layer.set_point.detach().float())
    sp_norm = float(layer.set_point.detach().float().norm())
    d_norm = float(d.norm())
    if not d_norm > 0.0:
        return {"top_share": nan, "eff_rank": nan,
                "rel_norm": 0.0 if sp_norm > 0 else nan, "out_align": nan}
    u, s, _ = torch.linalg.svd(d, full_matrices=False)
    energy = s.pow(2)
    out = {
        "top_share": float(energy[0] / energy.sum()),
        "eff_rank": _entropy_rank(energy),
        "rel_norm": d_norm / sp_norm if sp_norm > 0 else nan,
        "out_align": nan,
    }
    om = getattr(layer, "_last_output_mean", None)
    if om is not None and float(om.norm()) > 0.0:
        omf = om.detach().float().to(u.device)
        out["out_align"] = float((u[:, 0] @ (omf / omf.norm())).abs())
    return out


def living_layers(model: nn.Module) -> list[tuple[str, PredictiveCodingLayer]]:
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, PredictiveCodingLayer)]


@torch.no_grad()
def write_diversity(
    model: nn.Module,
    inputs: Sequence[dict],
    *,
    encode_kwargs: dict | None = None,
) -> dict:
    """Mean pairwise cosine between the weight writes that different inputs
    produce from the SAME starting state, per living layer.

    ``inputs`` is a sequence of keyword dicts for ``model.encode`` (e.g.
    ``{"text_tokens": t}``). For each input a deep copy of ``model`` runs
    one self-modifying forward in the model's current train/eval mode; the
    per-layer weight delta is recorded. Returns
    ``{layer_name: mean_pairwise_cosine}`` plus ``"_mean"``. Values near 1
    mean the write ignores what it was given.
    """
    kw = dict(encode_kwargs or {"causal": False})
    deltas: list[list[torch.Tensor]] = []
    names = [n for n, _ in living_layers(model)]
    if not names:
        return {"_mean": float("nan")}
    for inp in inputs:
        m = copy.deepcopy(model)
        before = [l.weight.detach().clone() for _, l in living_layers(m)]
        m.encode(**inp, **kw)
        deltas.append([
            (l.weight.detach() - b).flatten().float()
            for (_, l), b in zip(living_layers(m), before)
        ])
        del m
    out: dict = {}
    for li, name in enumerate(names):
        sims = []
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                a, b = deltas[i][li], deltas[j][li]
                den = float(a.norm() * b.norm())
                sims.append(float(a @ b) / den if den > 0 else float("nan"))
        finite = [s for s in sims if s == s]
        out[name] = sum(finite) / len(finite) if finite else float("nan")
    vals = [v for k, v in out.items() if v == v]
    out["_mean"] = sum(vals) / len(vals) if vals else float("nan")
    return out


@torch.no_grad()
def batch_mean_stationarity(
    model: nn.Module, inputs: Iterable[dict], *, encode_kwargs: dict | None = None,
) -> dict:
    """Mean pairwise cosine of each living layer's batch-mean INPUT across
    ``inputs`` -- the statistic the mean-field write and the legacy episode
    context consume. Runs under :func:`freeze_plasticity` (mutates nothing).
    """
    kw = dict(encode_kwargs or {"causal": False})
    captured: dict[str, list[torch.Tensor]] = {}
    hooks = []
    for name, layer in living_layers(model):
        def _hook(mod, args, _out, name=name):
            x = args[0].detach().reshape(-1, mod.in_features).float()
            captured.setdefault(name, []).append(x.mean(dim=0))
        hooks.append(layer.register_forward_hook(_hook))
    try:
        with freeze_plasticity(model):
            for inp in inputs:
                model.encode(**inp, **kw)
    finally:
        for h in hooks:
            h.remove()
    out: dict = {}
    for name, vecs in captured.items():
        v = torch.stack(vecs)
        v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-12)
        c = v @ v.T
        n = c.shape[0]
        out[name] = float((c.sum() - c.diagonal().sum()) / (n * (n - 1))) if n > 1 else float("nan")
    vals = [x for x in out.values() if x == x]
    out["_mean"] = sum(vals) / len(vals) if vals else float("nan")
    return out


__all__ = [
    "living_change_spectrum",
    "living_layers",
    "write_diversity",
    "batch_mean_stationarity",
]
