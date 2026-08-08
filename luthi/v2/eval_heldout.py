"""Held-out evaluation for the JEPA program (2026-07-15).

The two metrics every pre-registered criterion reads
(docs/research/2026-07-15_falsification-preregistration.md; protocol
docs/research/living-weights-experiments.md, JEPA edition):

  * held-out latent-prediction error — ``compute_modality_loss``'s
    ``l_pred`` over the leakage-gapped holdout tail;
  * linear-probe accuracy — a next-token linear readout trained on
    FROZEN latents from training batches, evaluated on holdout latents.

Non-negotiable discipline, enforced here rather than hoped for:

1. **Evaluation must not change the model.** The living substrate
   self-modifies on ANY forward — an unguarded eval pass would write
   living state from held-out data (evaluation contaminating the
   subject). Every forward here runs under ``freeze_plasticity`` +
   ``torch.no_grad()`` + ``loss_module.eval()`` (the last also stops the
   projection heads' BatchNorm running stats from drinking holdout
   statistics). ``tests/test_heldout_eval.py`` pins bitwise-identical
   state across an eval call.
2. **The probe is a readout, not an objective.** The encoder is frozen;
   only the Linear(d_model → vocab) trains. Next-token readout is a
   measurement of what the representation carries — the LM *objective*
   stays retired.
3. **Positive control before any null** (§1 of the protocol): the probe
   API exposes a shuffled-labels floor so every probe result ships with
   its own chance-level calibration.
"""

from __future__ import annotations

import contextlib
import math
from typing import Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from luthi.v2.plasticity import freeze_plasticity


@contextlib.contextmanager
def _eval_guard(loss_module: nn.Module) -> Iterator[None]:
    """eval() + frozen plasticity + restore prior mode on exit."""
    was_training = loss_module.training
    loss_module.eval()
    try:
        with freeze_plasticity(loss_module.online_encoder), torch.no_grad():
            yield
    finally:
        if was_training:
            loss_module.train()


def heldout_latent_prediction(
    loss_module: nn.Module,
    batches: Iterable[dict],
    modality: str = "text",
    max_batches: int | None = None,
) -> dict:
    """Mean held-out latent-prediction error for one modality.

    ``batches`` is any iterable of modality-input dicts (typically
    ``loader.holdout_batches(modality, batch_size)``). Returns
    ``{"l_pred_mean", "l_sigreg_mean", "n_batches"}`` — ``l_pred_mean``
    is THE pre-registered number; ``l_sigreg_mean`` rides along as a
    collapse-side diagnostic, never a criterion input.
    """
    l_preds: list[float] = []
    l_sigregs: list[float] = []
    nmses: list[float] = []
    l_ntps: list[float] = []
    with _eval_guard(loss_module):
        for i, batch in enumerate(batches):
            if max_batches is not None and i >= max_batches:
                break
            result = loss_module.compute_modality_loss(modality, batch)
            l_pred = float(result["l_pred"].item())
            l_preds.append(l_pred)
            l_sigregs.append(float(result["l_sigreg"].item()))
            # NMSE (blind amendment 2026-07-16): error normalized by the
            # target block's own per-dim variance, so arms with different
            # latent scales are comparable ("what fraction of its own
            # signal's structure does the model fail to capture"). Raw
            # l_pred mechanically favors quieter latent spaces.
            target_block = result["target_latents"][:, result["ctx_len"]:, :]
            centered = target_block - target_block.mean(dim=(0, 1), keepdim=True)
            target_var = float(centered.pow(2).mean().item())
            nmses.append(l_pred / max(target_var, 1e-12))
            # LLM-JEPA (2026-08-08 spec §3): the generative gauge. Held-out
            # perplexity over 32k classes cannot be flattered by a collapsed
            # target the way NMSE can -- it is the one capability number in
            # this harness degeneracy has no route to game. None when the
            # NTP term is off, so no existing arm's record changes.
            if result.get("l_ntp") is not None:
                l_ntps.append(float(result["l_ntp"].item()))
    n = len(l_preds)
    out = {
        "l_pred_mean": (sum(l_preds) / n) if n else float("nan"),
        "nmse_mean": (sum(nmses) / n) if n else float("nan"),
        "l_sigreg_mean": (sum(l_sigregs) / n) if n else float("nan"),
        "n_batches": n,
    }
    if l_ntps:
        mean_ntp = sum(l_ntps) / len(l_ntps)
        out["l_ntp_mean"] = mean_ntp
        # exp of mean cross-entropy in nats. Guarded: a diverged run can
        # produce an l_ntp large enough to overflow float, and an eval
        # diagnostic must never be the thing that kills a run.
        out["perplexity"] = math.exp(mean_ntp) if mean_ntp < 700 else float("inf")
    return out


def _collect_next_token_pairs(
    loss_module: nn.Module,
    batches: Iterable[dict],
    max_batches: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen full-sequence latents at position t paired with token t+1.

    Returns (latents [N, D], targets [N]). Text-only for round 1 — the
    probe task needs discrete labels, which text supplies natively.
    """
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    with _eval_guard(loss_module):
        for i, batch in enumerate(batches):
            if i >= max_batches:
                break
            tokens = batch["text_tokens"]
            result = loss_module.online_encoder.encode(
                text_tokens=tokens, causal=False,
            )
            latents = result["per_modality"]["text"]  # [B, L, D]
            # Collected to CPU: the probe is a tiny linear readout, and
            # training it on the accelerator buys nothing while inviting
            # mixed-device graphs (the DirectML shakeout found exactly
            # that: a CPU probe fed DML latents dies in addmm backward).
            xs.append(
                latents[:, :-1, :].reshape(-1, latents.shape[-1]).cpu()
            )
            ys.append(tokens[:, 1:].reshape(-1).cpu())
    if not xs:
        raise ValueError("no batches supplied to the probe collector")
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def fit_next_token_probe(
    loss_module: nn.Module,
    train_batches: Iterable[dict],
    vocab_size: int,
    *,
    max_batches: int = 20,
    epochs: int = 3,
    lr: float = 1e-2,
    seed: int = 0,
) -> nn.Linear:
    """Train the linear readout on frozen latents from TRAINING batches.

    The encoder never updates (latents are collected under the eval
    guard, detached). Deterministic given (seed, batches).
    """
    x, y = _collect_next_token_pairs(loss_module, train_batches, max_batches)
    d_model = x.shape[-1]
    torch.manual_seed(seed)
    probe = nn.Linear(d_model, vocab_size)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    probe.train()
    for _ in range(epochs):
        perm = torch.randperm(x.shape[0])
        for start in range(0, x.shape[0], 4096):
            idx = perm[start : start + 4096]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(x[idx]), y[idx])
            loss.backward()
            opt.step()
    probe.eval()
    return probe


def probe_accuracy(
    loss_module: nn.Module,
    probe: nn.Linear,
    heldout_batches: Iterable[dict],
    *,
    max_batches: int = 20,
    shuffled_label_floor: bool = False,
    seed: int = 0,
) -> dict:
    """Top-1/top-5 next-token readout accuracy on held-out latents.

    ``shuffled_label_floor=True`` scores the SAME predictions against
    permuted targets — the §1 positive-control floor. Report both; a
    probe number without its floor is not admissible for a null.
    """
    x, y = _collect_next_token_pairs(loss_module, heldout_batches, max_batches)
    if shuffled_label_floor:
        gen = torch.Generator().manual_seed(seed)
        y = y[torch.randperm(y.shape[0], generator=gen)]
    with torch.no_grad():
        logits = probe(x)
        top1 = (logits.argmax(dim=-1) == y).float().mean().item()
        k = min(5, logits.shape[-1])
        topk = logits.topk(k, dim=-1).indices
        top5 = (topk == y.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return {"top1": top1, "top5": top5, "n_pairs": int(y.shape[0])}


__all__ = [
    "heldout_latent_prediction",
    "fit_next_token_probe",
    "probe_accuracy",
]
