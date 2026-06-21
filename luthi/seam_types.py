"""Shared types for the Sanctuary <-> training-seam contract.

Holds the dataclass + Protocols that both the trainer (producer) and the
outward ``sanctuary_interface`` module (consumer/re-exporter) reference.
Sitting under ``luthi/`` rather than inside ``luthi/v2/m9/`` keeps the
trainer from importing the outward boundary module to get its own
contract types -- the layering F9a flagged in 4.8's 2026-06-15 review.

External callers should still import from ``luthi.sanctuary_interface``;
this module is the internal source of truth those re-exports point at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

__all__ = [
    "ActionSelection",
    "GenerationState",
    "PlanSnapshot",
    "M9Actor",
    "TransitionSink",
]


@dataclass
class GenerationState:
    """Encoder state captured during a ``return_state=True`` generation.

    Phase 4a (2026-06-16) of the seam integration plan: the Sanctuary
    cycle's seam consumes ``s_t`` from the same encoder pass that
    produced the generation logits, killing the double-plasticity probe
    F7a flagged. 4.8's brief specified the return shape as ``(text,
    s_t)`` but the M9Trainer.select_action contract also needs the
    unpooled ``ctx_latents`` for the MCTS predictor's lookahead. Both
    flow back as a single typed payload so the seam doesn't have to
    second-guess which axis it's holding.

    - ``s_t``: ``[B, D]`` pooled state vector. Always computed via
      :func:`luthi.v2.m9.s_t.compute_s_t` -- single source of truth for
      the s_t definition across training and inference.
    - ``ctx_latents``: ``[B, T, D]`` post-trunk encoder output for the
      text modality, the input that produced ``s_t``. Passed to
      ``M9Trainer.select_action`` as ``context_latents`` for the MCTS
      predictor's lookahead.
    """

    s_t: torch.Tensor
    ctx_latents: torch.Tensor


@dataclass
class PlanSnapshot:
    """Frozen capture of an MCTS plan, taken at :meth:`select_action` time.

    Exists so the transition that consumes the plan (``observe_transition``)
    can be self-describing -- it reads the planner's visit distribution,
    candidate actions, and best-action reward from this snapshot instead
    of from the trainer's mutable MCTS root. That live-tree read was the
    F4 finding in 4.8's 2026-06-15 review: any interleaved ``select_action``
    or ``mcts.reset`` between act-time and observe-time silently
    corrupts the realized-transition reward + habit-distill target.

    Snapshot fields are detached at construction; sink-side consumers
    should treat them as read-only.

    - ``visit_distribution``: ``[K]`` per-child visit probability (counts
      normalized; sums to 1 when ``K > 0``). Empty tensor when the
      planner expanded no children (degenerate budget=0).
    - ``candidate_actions``: ``[K, D]`` per-child ``action_in`` tensors.
    - ``r_best``: ``-min(child.incoming_g)`` over the expanded plan -- the
      M9 "best-action reward" for V-TD bootstrapping. ``0.0`` for the
      degenerate empty-plan case.
    """

    visit_distribution: torch.Tensor
    candidate_actions: torch.Tensor
    r_best: float


@dataclass
class ActionSelection:
    """Result of an :class:`M9Actor.select_action` call.

    Three pieces are load-bearing for the cycle's downstream wiring:

    - ``action``: ``[D]`` (or ``[B, D]`` for batched planning) tensor in
      the model's action space -- the candidate the M9 planner chose
      (MCTS-best when MCTS is reachable, habit-net sample otherwise).
    - ``readable_summary``: short human-legible string describing the
      action -- the "action -> readable summary" utility the integration
      plan flags as needed at the ``sanctuary/core/luthi_model.py:955``
      output adapter site, so the Sanctuary cycle can log/render what
      Luthi chose without re-parsing the action tensor.
    - ``efe_breakdown``: per-component EFE decomposition (epistemic,
      pragmatic, etc.) for diagnostics + observability. Floats keyed
      by component name; empty dict is valid when the planner skipped
      EFE evaluation (degenerate budget=0 case).

    Plus the optional :class:`PlanSnapshot` (4.8 review F4, 2026-06-15)
    so the caller can thread the plan through ``observe_transition``'s
    ctx and the sink doesn't have to read live MCTS state.
    """

    action: torch.Tensor
    readable_summary: str
    efe_breakdown: dict[str, float] = field(default_factory=dict)
    plan_snapshot: PlanSnapshot | None = None


@runtime_checkable
class M9Actor(Protocol):
    """Protocol for an M9-side actor (typically an ``M9Trainer``).

    The Sanctuary cycle drives an actor via
    :func:`luthi.sanctuary_interface.select_action`; the actor owns the
    habit-net + persistent MCTS + EFE evaluator that produce the chosen
    action.

    Marked ``@runtime_checkable`` so test mocks and adapter shims can
    declare conformance without inheritance.
    """

    def select_action(
        self,
        s_t: torch.Tensor,
        **kwargs: Any,
    ) -> ActionSelection:
        ...


@runtime_checkable
class TransitionSink(Protocol):
    """Protocol for the persistent training write-path.

    The Sanctuary cycle calls
    :func:`luthi.sanctuary_interface.observe_transition` once per
    realized ``(s_t, a_t, s_next)`` so the actor's V-head, habit-net,
    decoders, and kill machinery update on lived consequence rather
    than purely predicted EFE. Distinct from the non-accumulating
    :func:`luthi.sanctuary_interface.apply_external_modulation` channel:
    a CfC modulation is snapshot-and-restored per cycle; an observed
    transition is the learner's persistent input.

    The ``ctx`` dict carries per-cycle observation context. Recognized
    keys (consumers may also stash arbitrary additional entries):

    - ``counterpart_present`` (bool / Tensor): wires into M9's P3
      connection preference via :meth:`_cycle_observation_kwargs`.
    - ``time_since_emission`` (float / Tensor): same P3 wiring.
    - ``plan_snapshot`` (:class:`PlanSnapshot` | None): canonical channel
      for the plan-time MCTS capture that fed the action being observed.
      When present, the sink reads its visit distribution / candidate
      actions / r_best from the snapshot instead of from live trainer
      state -- closes the F4 corruption window when async actor/learner
      interleaves any call between act-time and observe-time. When
      absent, the sink degrades to a no-plan reward (``r=0``) + habit
      sample fallback for the visit-distill target.
    """

    def observe_transition(
        self,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
        s_next: torch.Tensor,
        ctx: dict[str, Any],
    ) -> dict[str, float]:
        ...
