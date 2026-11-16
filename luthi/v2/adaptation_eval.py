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

Opt-in switches (2026-11-16, exploratory; each is off by default and the
default path is bit-identical to before -- tests/test_adaptation_eval.py):

    homeostatic_decay    the pull toward home during the reads, e.g. 0.
    read_consolidation   False: consolidation cannot fire during the reads.
                         A read changes a living weight three ways: the
                         write (scaled by ``rate_scale``), the pull toward
                         home, and consolidation, which fires on its own
                         trigger at ``pc_rate`` unscaled by ``rate_scale``.
                         On the July runs it first fires around the 15th
                         window read and is the largest of the three in the
                         last layer.
    read_episode_writes  False: no episode is written during the reads. A
                         read reaches the score by a fourth way that changes
                         no weight: it writes episodes (each block's
                         ``episode_store``, and the layer's own store), and
                         the frozen score recalls them. Not scaled by
                         ``rate_scale`` either. With this off the block
                         store is frozen (it still recalls) and the layer's
                         own episode write is skipped.

"The write" here is everything ``rate_scale`` scales: the weight's update
and the prediction's learning (``pred_learning_rate`` is scaled by it too).
A read with only the write on therefore includes both.

The ways do not add up. On 768 seed 97 at four windows the write alone
gives B minus C of +5.5e-7, and the write's part of a whole read (everything
on, minus everything but the write) is +2.1e-4, nearly all of it in one
document where the write changes what the episode stores then hold (Opus's
review of 2026-11-16). To say what the write contributes to a read, take
both.

With ``rate_scale=0``, ``homeostatic_decay=0`` and the two switches above
off, a read changes nothing the score reads (tests/test_adaptation_eval.py).
Other living state still moves (``precision``, ``error_acc``, ``update_ema``);
the frozen score does not read it.

    read_what="tail"     a repeat read: B reads the windows it is then
                         scored on, C reads another document's scored
                         windows. The easiest case a fast weight can have.
    read_by_gradient     the plastic read is replaced by plain SGD steps on
                         the training loss of the windows read, plasticity
                         frozen. A positive control for the harness: does
                         B - C show document-specific change when something
                         is known to be learning from the document?
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
    # Only for a read by gradient, which changes the predictor as well as
    # the encoder: the whole loss module's state (it contains the encoder's).
    loss_state: dict | None = None


def _snapshot(model: nn.Module, loss_module: nn.Module | None = None) -> _Snapshot:
    return _Snapshot(
        state={k: v.detach().clone() for k, v in model.state_dict().items()},
        extra=copy.deepcopy(collect_living_extra_state(model)),
        loss_state=(
            None if loss_module is None else
            {k: v.detach().clone() for k, v in loss_module.state_dict().items()}
        ),
    )


def _restore(model: nn.Module, snap: _Snapshot,
             loss_module: nn.Module | None = None) -> None:
    if snap.loss_state is not None:
        if loss_module is None:
            raise RuntimeError("this snapshot holds the loss module's state; "
                               "pass the loss module to restore it")
        loss_module.load_state_dict(snap.loss_state)
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


def _no_episode(*args, **kwargs) -> None:
    """Stands in for a layer's ``_store_episode`` while episode writes are
    held off."""


@torch.no_grad()
def _read(model: nn.Module, head: torch.Tensor, *, repeats: int,
          read_train_mode: bool, consolidation: bool = True,
          episode_writes: bool = True) -> None:
    """Self-modifying reads of ``head`` ([n, L]), one window at a time.

    ``consolidation=False`` takes each living layer's consolidation tracker
    away for the duration, so the trigger neither fires nor advances, and
    puts the same tracker back after.

    ``episode_writes=False`` freezes each block's ``EpisodeStore`` for the
    duration (it recalls and does not write) and skips each living layer's
    own episode write, and undoes both after."""
    from luthi.episode_store import EpisodeStore
    from luthi.v2.living_layer_pc import PredictiveCodingLayer

    was_training = model.training
    model.train(read_train_mode)
    held: list[tuple[nn.Module, object]] = []
    frozen_stores: list[tuple[nn.Module, bool]] = []
    silenced: list[nn.Module] = []
    if not consolidation:
        for m in model.modules():
            if getattr(m, "_consolidation_tracker", None) is not None:
                held.append((m, m._consolidation_tracker))
                m._consolidation_tracker = None
    if not episode_writes:
        for m in model.modules():
            if isinstance(m, EpisodeStore):
                frozen_stores.append((m, m._plasticity_frozen))
                m._plasticity_frozen = True
            elif isinstance(m, PredictiveCodingLayer):
                # An instance attribute shadows the method; deleting it
                # afterwards uncovers the method again.
                m._store_episode = _no_episode
                silenced.append(m)
    try:
        for _ in range(repeats):
            for w in head:
                model.encode(text_tokens=w.unsqueeze(0), causal=False)
                clear = getattr(model, "clear_forward_cache", None)
                if callable(clear):
                    clear()
    finally:
        for m, tracker in held:
            m._consolidation_tracker = tracker
        for m, prev in frozen_stores:
            m._plasticity_frozen = prev
        for m in silenced:
            del m._store_episode
        model.train(was_training)


def gradient_read_parameters(loss_module: nn.Module) -> list[nn.Parameter]:
    """What a read by gradient steps: the encoder's and the predictor's
    trained parameters. A living weight is a buffer, not a parameter, so no
    living weight is among them."""
    return [p for p in loss_module.online_encoder.parameters() if p.requires_grad] + [
        p for p in loss_module.predictor.parameters() if p.requires_grad
    ]


def _read_by_gradient(loss_module: nn.Module, windows: torch.Tensor, *,
                      rate: float, steps: int, repeats: int) -> None:
    """Plain SGD on the training loss of ``windows`` ([n, L]), one window at
    a time, ``steps`` steps per window, with plasticity frozen and the
    module in eval mode (no top-down sweep, no running statistics moved).
    No optimizer object: ``p -= rate * grad`` and nothing else, so there is
    no optimizer state to restore."""
    params = gradient_read_parameters(loss_module)
    was_training = loss_module.training
    loss_module.eval()
    try:
        with freeze_plasticity(loss_module.online_encoder), torch.enable_grad():
            for _ in range(repeats):
                for w in windows:
                    for _ in range(steps):
                        r = loss_module.compute_modality_loss(
                            "text", {"text_tokens": w.unsqueeze(0)},
                        )
                        loss = r["loss"]
                        if not torch.isfinite(loss):
                            raise RuntimeError(
                                f"read by gradient: the loss is {float(loss)} "
                                f"at rate {rate}; nothing after it would mean anything"
                            )
                        grads = torch.autograd.grad(loss, params, allow_unused=True)
                        with torch.no_grad():
                            for p, g in zip(params, grads):
                                if g is not None:
                                    p.add_(g, alpha=-rate)
                        clear = getattr(loss_module.online_encoder,
                                        "clear_forward_cache", None)
                        if callable(clear):
                            clear()
    finally:
        if was_training:
            loss_module.train()


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
    homeostatic_decay: float | None = None,
    read_consolidation: bool = True,
    read_episode_writes: bool = True,
    read_what: str = "head",
    read_by_gradient: float | None = None,
    gradient_steps: int = 1,
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
        homeostatic_decay: if given, set every living layer's pull toward
            home for the reads (0 = off). Restored after. Scoring is frozen,
            so the pull acts nowhere else.
        read_consolidation: False = consolidation cannot fire during reads.
        read_episode_writes: False = no episode is written during reads
            (block stores frozen, the layer's own write skipped).
        read_what: "head" (default) reads the document's leading ``n_read``
            windows and scores the rest. "tail" is a repeat read: B reads
            the windows it is scored on, C reads the other document's
            scored windows, and the leading windows are not used.
        read_by_gradient: if given, the plastic read is replaced by SGD at
            this rate on the training loss of the windows read (see
            ``_read_by_gradient``). The whole loss module is snapshotted
            and restored around every read.
        gradient_steps: SGD steps per window read.

    Returns ``{"per_doc": [...], "summary": {...}, "settings": {...}}``.
    """
    model = loss_module.online_encoder
    d = len(documents)
    if d < 2:
        raise ValueError("adaptation_eval needs at least 2 documents (C reads another)")
    shift = control_shift if control_shift is not None else max(1, d // 2)
    if shift % d == 0:
        raise ValueError("control_shift must not map a document to itself")

    if read_what not in ("head", "tail"):
        raise ValueError(f"read_what must be 'head' or 'tail', not {read_what!r}")
    if gradient_steps < 1:
        raise ValueError("gradient_steps must be at least 1")
    by_gradient = read_by_gradient is not None

    from luthi.v2.living_layer_pc import PredictiveCodingLayer
    living = [m for m in model.modules() if isinstance(m, PredictiveCodingLayer)]
    prior_scales = [m.rate_scale for m in living]
    prior_decays = [m.homeostatic_decay for m in living]
    if rate_scale is not None:
        for m in living:
            m.rate_scale = float(rate_scale)
    if homeostatic_decay is not None:
        for m in living:
            m.homeostatic_decay = float(homeostatic_decay)

    held = loss_module if by_gradient else None
    base = _snapshot(model, held)

    def read(windows: torch.Tensor) -> None:
        if by_gradient:
            _read_by_gradient(loss_module, windows, rate=float(read_by_gradient),
                              steps=gradient_steps, repeats=read_repeats)
        else:
            _read(model, windows, repeats=read_repeats,
                  read_train_mode=read_train_mode,
                  consolidation=read_consolidation,
                  episode_writes=read_episode_writes)

    per_doc = []
    try:
        for i, doc in enumerate(documents):
            head, tail = doc[:n_read], doc[n_read:]
            if tail.shape[0] < 1:
                raise ValueError("each document needs at least one score window")
            other = documents[(i + shift) % d]
            if read_what == "tail":
                own_read, other_read = tail, other[n_read:]
            else:
                own_read, other_read = head, other[:n_read]

            a = _score(loss_module, tail)

            read(own_read)
            b = _score(loss_module, tail)
            _restore(model, base, held)

            read(other_read)
            c = _score(loss_module, tail)
            _restore(model, base, held)

            per_doc.append({
                "doc": i,
                "A": a, "B": b, "C": c,
                "exact_null": (a == b == c),
            })
    finally:
        _restore(model, base, held)
        for m, s in zip(living, prior_scales):
            m.rate_scale = s
        for m, decay in zip(living, prior_decays):
            m.homeostatic_decay = decay

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
            "homeostatic_decay": homeostatic_decay,
            "read_consolidation": read_consolidation,
            "read_episode_writes": read_episode_writes,
            "read_what": read_what,
            "read_by_gradient": read_by_gradient,
            "gradient_steps": gradient_steps if by_gradient else None,
            "gradient_parameters": (
                sum(p.numel() for p in gradient_read_parameters(loss_module))
                if by_gradient else None
            ),
        },
    }


__all__ = ["adaptation_eval", "gradient_read_parameters", "holdout_documents"]
