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
            # F-A round-2 fix: dimensionless peakedness target.
            #
            # Round-1 used raw std(G), which is NOT scale-invariant --
            # Fable's H1 showed identical-shape × 50 EFE landscape →
            # × 50 gamma, decisiveness tracking absolute cost magnitude
            # rather than landscape shape. With P3 unbounded, growing
            # silence-time inflated EFE → gamma → ceiling → K-M9-4
            # false-positive halt (H2 at ~20 s).
            #
            # Replacement: `peakedness = (G_2nd − G_min) / (std + eps)`,
            # the normalized best-vs-second-best gap (4.8's round-2
            # spec). Numerator and denominator both scale identically
            # with any linear EFE rescaling, so the ratio is
            # dimensionless: × 50 EFE leaves gamma unchanged.
            #
            # Captures "one clearly best action vs flat landscape"
            # exactly: large gap (best stands out) → high gamma
            # (commit); small gap (best is tied with second) → low
            # gamma (hedge). Better than CoV (std/|mean|) which
            # blows up at mean ≈ 0.
            if efes.dim() > 1:
                # Batched [B, K]: compute per-row and average.
                gap = self._normalized_gap_batched(efes)
            else:
                gap = self._normalized_gap(efes)
            gamma_target = self.gamma_scale * gap
            # Clamp the target before the EMA mix.
            gamma_target = gamma_target.clamp(
                min=self.gamma_min, max=self.gamma_max
            )
            new_gamma = (1.0 - self.rho_g) * self.gamma + self.rho_g * gamma_target
            new_gamma = new_gamma.clamp(min=self.gamma_min, max=self.gamma_max)
            self.gamma.copy_(new_gamma)
            self._history.append(float(self.gamma.item()))
        return self.gamma.detach().clone()

    def _normalized_gap(self, efes: torch.Tensor) -> torch.Tensor:
        """Dimensionless peakedness: (median − G_min) / (std + eps).

        `efes`: [K] candidate costs. EFE is cost so the "winner" is
        the lowest G. Measures how far the winner sits BELOW the
        typical candidate, in units of spread.

        - Single peaked winner: large (winner well below median).
        - Flat: small (winner ≈ median).
        - Bimodal (cluster of winners + cluster of losers, e.g. silent
          vs active candidates under sustained P3 pressure): bounded,
          because median tracks the inter-cluster gap and std tracks
          the cluster spread; both scale identically with any linear
          rescaling of EFE.

        Both numerator and denominator scale linearly with the EFE
        magnitude, so the ratio is dimensionless: rescaling EFE by
        any positive constant leaves gamma_target unchanged. (4.8's
        round-2 recommendation was `(G_2nd − G_min) / std`; we use
        `(median − G_min) / std` because the 2nd-min form collapses
        to ≈ 0 on bimodal landscapes -- exactly the regime the
        sustained-silence probe_h H2 exercises -- and would push
        gamma toward gamma_min there. Median is robust to both
        single-winner and bimodal cases.)
        """
        if efes.numel() < 2:
            return torch.tensor(0.0, device=efes.device, dtype=efes.dtype)
        # Use quantile-0.5 (interpolated midpoint) rather than
        # torch.median: torch.median returns the LOWER middle element
        # for even-length tensors, which on a bimodal landscape with
        # K=8 (4 active + 4 silent) gives the max of the active
        # cluster -- right next to G_min, so (median - G_min) → ~0
        # and gamma → gamma_min. Interpolated median lands BETWEEN
        # the two clusters and gives the stable ~1.0 ratio the
        # formula is designed to produce.
        med = torch.quantile(efes, 0.5)
        g_min = efes.min()
        std = efes.std(unbiased=False)
        return (med - g_min) / (std + self.eps)

    def _normalized_gap_batched(self, efes: torch.Tensor) -> torch.Tensor:
        """Batched [B, K] -> scalar (mean over B). See `_normalized_gap`."""
        if efes.shape[-1] < 2:
            return torch.tensor(0.0, device=efes.device, dtype=efes.dtype)
        med = torch.quantile(efes, 0.5, dim=-1)  # [B]
        g_min = efes.min(dim=-1).values          # [B]
        std = efes.std(dim=-1, unbiased=False)   # [B]
        per_row = (med - g_min) / (std + self.eps)
        return per_row.mean()

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
