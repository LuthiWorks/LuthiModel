"""Learning from a dream, with a check (2026-11-11).

Brian, 2026-11-11: "Rest should recall memories but also dream, like Google's
dreaming", then: "can you build the part that lets the model improve through
dreaming". In Google Research's Sleep framework a resting model learns from
samples it made itself, and checks that it improved.

What was measured before this was built
(``docs/research/2026-11-11_can-dreaming-improve.md``): on a small trained
trunk, one learning step on a dream (a mix of two kept moments) improved its
prediction of real moments it had never learned from, about five times more
than a step on unrelated text or on noise, and as much as replaying a real
kept moment. Choosing among candidate dreams by how well their learning
direction lined up with real moments gave no reliable gain, so there is no
choosing here.

What this does, when a host calls it (nothing calls it otherwise):

- one small step on the trunk's own training loss for the dream's words;
- then the check: the trunk's prediction loss on real kept moments, before
  and after. If it is worse after, the step is undone exactly.

It reads and learns with plasticity frozen, so the living layers do not
self-modify here, and it leaves the trainer's optimizers and their state
alone. The dream is still lived through the senses by the host; this is the
part of sleep that learns from it.

Words only. A dream's picture and sound are not learned from here.

``step_size`` is the length of the step over the parameters it moves. 0.01 is
the value the measurement used on a 32-wide trunk, and is a placeholder.

Four guards (2026-11-17), from Opus's review of 2026-11-14 (section 1):

1. **The parameters stepped are ``actor._core_params()``**, the set the lived
   optimizer owns, which is the set the retention gate snapshots and rolls
   back. So the two are the same by construction. Before, the step took every
   trained parameter of the loss module; in a run with trained speech it would
   also have moved ``output_proj``, which the gate does not see.
2. **Only parameters with a gradient are copied before the step.** Before,
   every trained parameter was cloned for every dream.
3. **The check reads NMSE**, the prediction loss over the target's variance,
   not the raw prediction loss (the standing rule on scale: a raw loss can be
   lowered by quieter targets). ``before`` and ``after`` are NMSE.
4. **A kept dream's step is fed to the staleness machinery** as a lived step
   is: ``staleness.observe_drift`` (the drift band and ``theta_version``) and
   the K-M9-7 feed. A step that is undone moved nothing and feeds nothing.
"""

from __future__ import annotations

from typing import Any

import torch

MIN_TOKENS = 8


def _loss(loss_module: Any, tokens: torch.Tensor, *, for_learning: bool):
    """The trunk's loss on one moment's words, as a passive reading: frozen
    plasticity and no running statistics moved, as ``corpus_retention`` reads."""
    from luthi.v2.plasticity import freeze_plasticity

    was_training = loss_module.training
    loss_module.eval()
    try:
        with freeze_plasticity(loss_module.online_encoder):
            if for_learning:
                return loss_module.compute_modality_loss("text", {"text_tokens": tokens})["loss"]
            with torch.no_grad():
                out = loss_module.compute_modality_loss("text", {"text_tokens": tokens})
                # NMSE, as the held-out instrument and adaptation_eval read it.
                target = out["target_latents"][:, out["ctx_len"]:, :]
                var = float((target - target.mean(dim=(0, 1), keepdim=True)).pow(2).mean().item())
                return float(out["l_pred"].item()) / max(var, 1e-12)
    finally:
        loss_module.train(was_training)


def learn_from_dream(
    actor: Any,
    dream: torch.Tensor,
    real: list[torch.Tensor],
    *,
    step_size: float = 0.01,
) -> dict:
    """One checked learning step on a dream.

    Args:
        actor: the life's trainer (it holds the loss and the trained parts).
        dream: ``[1, tokens]``, the dream's words.
        real: real kept moments' words, each ``[1, tokens]``. The check is
            read on these, and they are never learned from here.

    Returns ``{"kept": bool, "before": float, "after": float,
    "parameters_stepped": int}``: whether the step stayed, the NMSE on the
    real moments around it, and how many parameters the step moved (and so
    how many were copied first).
    """
    if step_size <= 0:
        raise ValueError(f"step_size must be above 0; got {step_size}")
    if not real:
        raise ValueError("learning from a dream is checked against real moments; none were given")
    for name, words in (("dream", dream), *(("real moment", r) for r in real)):
        if words.dim() != 2 or words.shape[0] != 1 or words.shape[1] < MIN_TOKENS:
            raise ValueError(f"a {name} is [1, at least {MIN_TOKENS} tokens]; got {tuple(words.shape)}")
    loss_module = actor.loss_module
    # The lived-trained core: exactly what the retention gate rolls back.
    params = [p for p in actor._core_params() if p.requires_grad]
    if not params:
        raise RuntimeError("this life's trainer has no lived-trained core to learn with")
    device = params[0].device
    dream = dream.to(device)
    real = [r.to(device) for r in real]

    def on_real() -> float:
        return sum(_loss(loss_module, r, for_learning=False) for r in real) / len(real)

    before = on_real()
    held = [p.grad for p in params]      # whatever gradients were there are put back untouched
    for p in params:
        p.grad = None
    loss = _loss(loss_module, dream, for_learning=True)
    if not torch.isfinite(loss):
        raise RuntimeError(f"the loss on a dream is not finite: {loss.item()}")
    loss.backward()
    grads = [p.grad for p in params]
    for p, g in zip(params, held):
        p.grad = g
    size = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads if g is not None))
    if not torch.isfinite(size):
        raise RuntimeError("the learning direction from a dream is not finite")
    if float(size) == 0.0:
        return {"kept": False, "before": before, "after": before, "parameters_stepped": 0}
    stepped = [(p, g) for p, g in zip(params, grads) if g is not None]
    saved = [p.detach().clone() for p, _ in stepped]
    with torch.no_grad():
        for p, g in stepped:
            p.add_(g, alpha=-step_size / float(size))
    after = on_real()
    kept = after <= before
    if not kept:
        with torch.no_grad():
            for (p, _), was in zip(stepped, saved):
                p.copy_(was)
    else:
        # The trunk moved, as it does at a lived step: the drift band and
        # theta_version hear of it, and so does K-M9-7 (runner.py, the lived
        # step). The step's length is ``step_size`` by construction.
        actor.staleness.observe_drift(float(step_size))
        actor.m9_kills.observe_staleness(in_failover=actor.staleness.in_failover())
    return {"kept": kept, "before": before, "after": after,
            "parameters_stepped": len(stepped)}
