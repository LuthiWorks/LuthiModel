"""Inference-time adaptation eval (2026-10-19).

The README states the project's near-term goal as *structural change without
a gradient step, at inference, with behavioral consequence*. Every capability
number in the record before this module was measured under
``freeze_plasticity`` -- a frozen snapshot -- so the goal had never been
measured at all. This module measures it, on existing checkpoints, with no
training.

Protocol, per held-out "document" (a run of consecutive held-out windows,
which are contiguous text):

    A  frozen      score the document's TAIL windows; no reading.
    B  own         read the document's HEAD windows with plasticity ON
                   (B=1, one window at a time, in order), then score the
                   tail frozen.
    C  other       read a DIFFERENT document's head with plasticity ON,
                   then score this document's tail frozen.

The model is restored exactly (state_dict + living extra state) between
conditions. Reported per document and summarized:

    B - A   any change reading caused (adaptation + generic drift)
    C - A   generic drift: change from reading ANYTHING
    B - C   document-specific adaptation: what reading THIS document bought
            beyond reading some document. This is the number the claim
            needs; B - A alone cannot distinguish "learned from what it read"
            from "reading perturbs the weights".

Scores are the JEPA objective's own held-out quantities (``l_pred`` and the
target-variance-normalized ``nmse``), computed exactly as
``eval_heldout.heldout_latent_prediction`` does, plus the target-block
effective rank so that an "improvement" produced by the representation
degenerating during reading is visible rather than flattering.

Positive control: a dead-FFN arm has no living state, so A, B and C must be
BITWISE equal. ``summary["exact_null"]`` reports it; for a dead arm anything
else means the harness is broken, not the model.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from luthi.living_extra_state import (
    apply_living_extra_state,
    collect_living_extra_state,
)
from luthi.v2.plasticity import freeze_plasticity


@dataclass
class _Snapshot:
    state: dict
    extra: dict


def _snapshot(model: nn.Module) -> _Snapshot:
    return _Snapshot(
        state={k: v.detach().clone() for k, v in model.state_dict().items()},
        extra=copy.deepcopy(collect_living_extra_state(model)),
    )


def _restore(model: nn.Module, snap: _Snapshot) -> None:
    model.load_state_dict(snap.state)
    apply_living_extra_state(model, snap.extra, source="adaptation_eval restore")
    clear = getattr(model, "clear_forward_cache", None)
    if callable(clear):
        clear()


def _eff_rank(x: torch.Tensor) -> float:
    flat = x.detach().float().reshape(-1, x.shape[-1])
    flat = flat - flat.mean(dim=0, keepdim=True)
    if flat.shape[0] < 2:
        return float("nan")
    cov = (flat.t() @ flat) / (flat.shape[0] - 1)
    sv = torch.linalg.svdvals(cov).clamp(min=1e-12)
    p = sv / sv.sum()
    return float(math.exp(float(-(p * torch.log(p)).sum())))


@torch.no_grad()
def _score(loss_module: nn.Module, tail: torch.Tensor) -> dict:
    """Frozen JEPA score of ``tail`` ([n, L] token windows)."""
    was_training = loss_module.training
    loss_module.eval()
    try:
        with freeze_plasticity(loss_module.online_encoder):
            r = loss_module.compute_modality_loss("text", {"text_tokens": tail})
    finally:
        if was_training:
            loss_module.train()
    l_pred = float(r["l_pred"].item())
    target = r["target_latents"][:, r["ctx_len"]:, :]
    centered = target - target.mean(dim=(0, 1), keepdim=True)
    var = float(centered.pow(2).mean().item())
    return {
        "l_pred": l_pred,
        "nmse": l_pred / max(var, 1e-12),
        "target_eff_rank": _eff_rank(target),
    }


@torch.no_grad()
def _read(model: nn.Module, head: torch.Tensor, *, repeats: int,
          read_train_mode: bool) -> None:
    """Self-modifying reads of ``head`` ([n, L]), one window at a time."""
    was_training = model.training
    model.train(read_train_mode)
    try:
        for _ in range(repeats):
            for w in head:
                model.encode(text_tokens=w.unsqueeze(0), causal=False)
                clear = getattr(model, "clear_forward_cache", None)
                if callable(clear):
                    clear()
    finally:
        model.train(was_training)


def _summ(vals: Sequence[float]) -> dict:
    v = [x for x in vals if x == x]
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "se": float("nan"), "t": float("nan"), "n": 0}
    mean = sum(v) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in v) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = float("nan")
    t = mean / se if se and se == se and se > 0 else float("nan")
    return {"mean": mean, "se": se, "t": t, "n": n}


def holdout_documents(
    windows: torch.Tensor, *, n_read: int, n_score: int, n_docs: int,
    phase: int = 0,
) -> list[torch.Tensor]:
    """Cut consecutive held-out windows into ``n_docs`` non-overlapping
    documents of ``n_read + n_score`` windows, spread evenly over the tail.
    Raises if the holdout is too short -- a silent short eval would fake a
    precise one.

    ``phase`` (2026-10-19) shifts every document by ``phase`` document
    lengths inside its stride, so phases 0, 1, 2... are mutually DISJOINT
    document sets. E1 counts a checkpoint only when its effect replicates on
    a disjoint set (registry, 2026-10-19 amendment). Phase 0 is the original
    selection, bit for bit."""
    per = n_read + n_score
    total = windows.shape[0]
    if n_docs * per > total:
        raise ValueError(
            f"holdout has {total} windows; {n_docs} documents of {per} windows "
            f"need {n_docs * per}. Lower n_docs or n_read/n_score."
        )
    stride = total // n_docs
    if phase < 0 or (phase + 1) * per > stride:
        raise ValueError(
            f"phase {phase} does not fit: each of the {n_docs} documents has a "
            f"stride of {stride} windows and needs {(phase + 1) * per} for "
            f"phase {phase}. Lower n_docs or the phase."
        )
    off = phase * per
    return [windows[i * stride + off : i * stride + off + per] for i in range(n_docs)]


def adaptation_eval(
    loss_module: nn.Module,
    documents: Sequence[torch.Tensor],
    *,
    n_read: int,
    read_repeats: int = 1,
    control_shift: int | None = None,
    read_train_mode: bool = False,
    rate_scale: float | None = None,
) -> dict:
    """Run A / B / C (see module docstring) over ``documents``.

    Args:
        loss_module: a JEPALoss wrapping the encoder (predictor loaded).
        documents: list of ``[n_read + n_score, L]`` token tensors, each a
            run of consecutive (contiguous) held-out windows.
        n_read: how many leading windows of each document are read.
        read_repeats: passes over the read windows (the README's "first
            exposure vs tenth" encounter question).
        control_shift: document offset for the C condition; default
            ``len(documents) // 2`` (far apart in the holdout).
        read_train_mode: run reads in train mode, which also runs the
            top-down sweep. Default False = the deployment regime
            (``sanctuary_interface.generate`` runs in eval mode).
        rate_scale: if given, set every living layer's ``rate_scale`` for the
            reads (e.g. the taper floor a run ended at). Restored after.

    Returns ``{"per_doc": [...], "summary": {...}, "settings": {...}}``.
    """
    model = loss_module.online_encoder
    d = len(documents)
    if d < 2:
        raise ValueError("adaptation_eval needs at least 2 documents (C reads another)")
    shift = control_shift if control_shift is not None else max(1, d // 2)
    if shift % d == 0:
        raise ValueError("control_shift must not map a document to itself")

    from luthi.v2.living_layer_pc import PredictiveCodingLayer
    living = [m for m in model.modules() if isinstance(m, PredictiveCodingLayer)]
    prior_scales = [m.rate_scale for m in living]
    if rate_scale is not None:
        for m in living:
            m.rate_scale = float(rate_scale)

    base = _snapshot(model)
    per_doc = []
    try:
        for i, doc in enumerate(documents):
            head, tail = doc[:n_read], doc[n_read:]
            if tail.shape[0] < 1:
                raise ValueError("each document needs at least one score window")
            other_head = documents[(i + shift) % d][:n_read]

            a = _score(loss_module, tail)

            _read(model, head, repeats=read_repeats, read_train_mode=read_train_mode)
            b = _score(loss_module, tail)
            _restore(model, base)

            _read(model, other_head, repeats=read_repeats,
                  read_train_mode=read_train_mode)
            c = _score(loss_module, tail)
            _restore(model, base)

            per_doc.append({
                "doc": i,
                "A": a, "B": b, "C": c,
                "exact_null": (a == b == c),
            })
    finally:
        _restore(model, base)
        for m, s in zip(living, prior_scales):
            m.rate_scale = s

    summary: dict = {"n_docs": d, "exact_null": all(p["exact_null"] for p in per_doc)}
    for metric in ("l_pred", "nmse", "target_eff_rank"):
        summary[metric] = {
            "A_mean": _summ([p["A"][metric] for p in per_doc])["mean"],
            "B_minus_A": _summ([p["B"][metric] - p["A"][metric] for p in per_doc]),
            "C_minus_A": _summ([p["C"][metric] - p["A"][metric] for p in per_doc]),
            "B_minus_C": _summ([p["B"][metric] - p["C"][metric] for p in per_doc]),
        }
    return {
        "per_doc": per_doc,
        "summary": summary,
        "settings": {
            "n_read": n_read, "n_score": int(documents[0].shape[0]) - n_read,
            "read_repeats": read_repeats, "control_shift": shift,
            "read_train_mode": read_train_mode, "rate_scale": rate_scale,
            "living_layers": len(living),
        },
    }


__all__ = ["adaptation_eval", "holdout_documents"]
