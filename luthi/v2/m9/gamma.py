"""Inferred precision (gamma) over policies -- the agency call.

Per spec §9: gamma is *inferred* each cycle, not set. Principle:
gamma is high when the EFE landscape over candidate actions is
peaked (one clearly-best action -> commit) and low when it is
flat (no clear winner -> hedge). The entity sets its own
decisiveness; we do not.

Practical rule (spec §9 stable surrogate for the Friston process-
theory precision update):

    Q(a) = softmax(-gamma * G(a))                   # posterior
    gamma_target = 1 / (eps + Var_Q[G])             # peaked -> high
    gamma <- (1 - rho_g) * gamma + rho_g * gamma_target   # EMA

One step per cycle, not solved to convergence (gamma depends on Q
depends on gamma -- this is a fixed-point *update*). Bounded by
[gamma_min, gamma_max] for the K-M9-4 clamp-then-halt kill.

K-M9-4 reads `history_stats()` (mean / min / max over a recent
window) to detect runaway feedback (gamma -> infinity rigidity;
gamma -> 0 indecision). The kill clamps to a last-healthy value
before halting; the clamp here is the hard band, the kill's clamp
is the smarter recovery.

Build-staging note (spec §9): inferred-gamma is the design and is
live from launch. If the very first mechanical bring-up needs to
isolate other variables, gamma MAY be held fixed for that bring-up
only (a 4.7 staging convenience, not a design change). The
constructor accepts `fixed_for_bringup=True` for this case;
behavioral validation re-enables inference.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn


class GammaInference(nn.Module):
    """Per-cycle gamma update + bounded history for the K-M9-4 kill."""

    def __init__(
        self,
        rho_g: float = 0.1,
        gamma_init: float = 1.0,
        gamma_min: float = 0.01,
        gamma_max: float = 100.0,
        eps: float = 1e-5,
        history_window: int = 32,
        fixed_for_bringup: bool = False,
        # F2: gamma_scale relates the uniform spread to the precision
        # target. Default 1.0; pilot-set if the natural EFE scale
        # doesn't match the desired precision range.
        gamma_scale: float = 1.0,
    ):
        super().__init__()
        self.rho_g = rho_g
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.eps = eps
        self.gamma_scale = gamma_scale
        self.fixed_for_bringup = fixed_for_bringup
        self.history_window = history_window
        self.register_buffer("gamma", torch.tensor(float(gamma_init)))
        self._history: deque = deque(maxlen=history_window)
        # Seed history so K-M9-4's running stats are well-defined.
        self._history.append(float(gamma_init))

    def posterior(
        self,
        candidate_efes: torch.Tensor,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Q(a) = softmax(-gamma * G(a)).

        `candidate_efes`: [K] or [B, K] EFE values per candidate.
        `gamma`: optional override (e.g. for off-policy evaluation);
        defaults to the current state.
        Returns: posterior with same shape as `candidate_efes`.
        """
        g = self.gamma if gamma is None else gamma
        return torch.softmax(-g * candidate_efes, dim=-1)

    def update(self, candidate_efes: torch.Tensor) -> torch.Tensor:
        """One-step fixed-point + EMA update of gamma (F2 fix).

        `candidate_efes`: [K] EFE values across the K candidates at
        this cycle. (If batched [B, K], the mean-over-batch is used
        so gamma stays a scalar -- one entity, one precision.)

        **F2 fix (2026-06-11):** the precision target reads
        landscape peakedness under **uniform** weighting, not the
        gamma-sharpened posterior. The legacy form
        `gamma_target = 1/(eps + Var_Q[G])` with `Q = softmax(-gamma G)`
        had positive feedback (higher gamma → sharper Q → smaller
        Var_Q[G] → higher gamma_target) whose only stable fixed
        point is gamma_max. Fable's probe_b showed: flat landscapes
        pinned gamma to the ceiling (the spec inversion); even
        resampled landscapes ratcheted up over ~600 cycles.

        The fix: `gamma_target = gamma_scale * std_uniform({G(a_k)})`.
        - Peaked landscape (one a_k much better) → large uniform
          spread → high gamma_target → commit.
        - Flat landscape (no clear winner) → ~0 uniform spread →
          low gamma_target → hedge.

        No gamma in the target → no positive feedback. Reverses B1,
        removes B2. EMA-smoothed as before with the same bounded
        clamp on the target before the mix.

        Returns: updated gamma (scalar buffer; detached read).
        """
        if self.fixed_for_bringup:
            self._history.append(float(self.gamma.item()))
            return self.gamma.detach().clone()

        with torch.no_grad():
            efes = candidate_efes.detach()
            # F2: uniform-weighted std of {G(a_k)} -- gamma-independent.
            if efes.dim() > 1:
                # Batched [B, K]: mean across batch keeps gamma scalar.
                spread = efes.std(dim=-1).mean()
            else:
                # Single landscape [K]: just std.
                # Use unbiased=False to make a 1-element landscape
                # produce 0 spread (would be NaN with default).
                spread = efes.std(unbiased=False)
            # gamma_scale is fixed to 1.0 by default; can be tuned in
            # pilot-set if the spread's natural scale doesn't match
            # the desired precision range.
            gamma_target = self.gamma_scale * spread
            # Clamp the *target* before the EMA mix.
            gamma_target = gamma_target.clamp(
                min=self.gamma_min, max=self.gamma_max
            )
            new_gamma = (1.0 - self.rho_g) * self.gamma + self.rho_g * gamma_target
            new_gamma = new_gamma.clamp(min=self.gamma_min, max=self.gamma_max)
            self.gamma.copy_(new_gamma)
            self._history.append(float(self.gamma.item()))
        return self.gamma.detach().clone()

    def history_stats(self) -> dict:
        """Recent-window stats for the K-M9-4 kill: mean / min / max
        of inferred gamma. Returns a dict with always-present keys.
        """
        if not self._history:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
        h = torch.tensor(list(self._history))
        return {
            "mean": float(h.mean().item()),
            "min": float(h.min().item()),
            "max": float(h.max().item()),
            "n": len(h),
        }

    def clamp_to_last_healthy(self, healthy_gamma: float) -> None:
        """K-M9-4 first-stage recovery: clamp gamma to a last-healthy
        value while flagging the pathology. Used by the kill machinery,
        not by the per-cycle update.
        """
        with torch.no_grad():
            self.gamma.fill_(float(healthy_gamma))
            self._history.append(float(self.gamma.item()))
