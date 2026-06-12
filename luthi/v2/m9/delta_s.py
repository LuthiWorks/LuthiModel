"""§A.2 unified ‖Δs_internal‖ signal: compute once, fan out, band-target.

Per 4.8's 2026-06-11 gate-repair spec, this single scalar is the
shared signal that:

- **P1 engagement** reads via its (de-saturating) hinge against a
  running-band target instead of the fixed 0.5 that saturated at
  random init under the untrained predictor's output magnitude (the
  A5 root cause).
- **K-M9-5 dark-room kill** reads as the internal-stasis signal --
  the spec's promise that "contemplation never trips the kill" only
  holds if P1 and K-M9-5 see the *same* number; this module makes
  that fact mechanical rather than a paper guarantee (closes N2).

Two design calls baked in:

1. **Internal-dim mask.** Per the spec, `‖Δs‖` should be computed on
   the *internal* subset of latent dims (the subset NOT driven by
   decoder rendering), so the engagement signal cannot be inflated
   by decoder-output-driven dim changes (closes N3 and is part of
   why the hinge saturates). At step-1 launch the default mask is
   all-ones (preserves backward-compatible behaviour); loop
   integration can later derive a smarter mask from decoder weight
   column-norms or from a learned partition. The interface accepts
   any `[d_model]` mask in [0, 1] elementwise; the module is
   forward-compatible with the smarter mask without API change.

2. **Normalization.** The scalar is normalized by `sqrt(d_internal)`
   so the magnitude doesn't scale with the latent width -- a 1024d
   substrate produces comparable scale to a 256d substrate, and
   the running-band target is meaningful across widths.

The loop-side asserter (§A.2's "cheap equality assert in the loop")
lives at the caller: compute `‖Δs‖` once via `compute(...)`, pass
the *same* tensor to `Preferences.engagement_cost(...)` and
`KillRegistry.observe_darkroom(...)`, and assert their inputs are
the same Python object. That assertion makes N2 mechanical.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from luthi.v2.m9.staleness import DriftBand


class DeltaSInternal(nn.Module):
    """Computes ‖Δs_internal‖ from a (state, predicted-next-state) pair.

    The internal-dim mask is a registered buffer (not a Parameter) so
    it does not receive gradient by default. A future fix can promote
    it to a Parameter or replace it with a derived mask from decoder
    weight column-norms; both paths preserve this API.
    """

    def __init__(
        self,
        d_model: int,
        internal_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        if internal_mask is None:
            internal_mask = torch.ones(d_model)
        if internal_mask.shape != (d_model,):
            raise ValueError(
                f"internal_mask shape {internal_mask.shape} != ({d_model},)"
            )
        self.register_buffer("internal_mask", internal_mask)

    def compute(
        self,
        s_t: torch.Tensor,
        s_hat_next: torch.Tensor,
    ) -> torch.Tensor:
        """`s_t`, `s_hat_next`: [B, D] -> [B] ‖Δs_internal‖.

        Implementation:
          delta = (s_hat_next - s_t) * internal_mask           # elementwise
          ‖Δs_internal‖ = ‖delta‖_2 / sqrt(internal_mask.sum())

        Normalization by the effective internal dimension keeps the
        scale stable across latent widths and across mask sparsity.
        """
        delta = (s_hat_next - s_t) * self.internal_mask
        raw_norm = delta.norm(dim=-1)                            # [B]
        # Effective internal dim = sum of mask weights (handles soft masks).
        eff_dim = float(self.internal_mask.sum().item())
        if eff_dim <= 0:
            return raw_norm  # degenerate; caller's problem
        return raw_norm / (eff_dim ** 0.5)


class DeltaSBand:
    """Running band on ‖Δs_internal‖ values, used by:

    - **P1 engagement** as the *target*: the hinge becomes
      `max(target - ‖Δs‖, 0)^2` where `target = silent_threshold(band)`,
      so the early-healthy ‖Δs‖ regime is the floor against which
      saturation cannot reset.
    - **K-M9-5 dark-room** as the *internal-stasis threshold*: a
      candidate is "internally still" iff its ‖Δs_internal‖ is below
      `silent_threshold(band)`.

    Both consumers must read the *same* band for the spec's
    contemplation-is-safe property to hold mechanically.

    Mirrors the shape of `ActivityBands` per modality (window-based
    median + MAD; per-cycle `observe`; pilot-set `silence_k`).
    """

    def __init__(
        self,
        window: int = 32,
        silence_k: float = 1.5,
        min_warmup: int = 8,
        # R1 round-2 interim (4.8 + Fable): absolute floor on
        # ‖Δs_internal‖. ‖Δs‖ at or below this counts as internally
        # still REGARDLESS of band. The band alone recalibrates to
        # a sustained-constant internal stasis (probe_g, same as
        # ActivityBands). Loop integration wires the a_rest reference;
        # this absolute floor is the interim backstop.
        absolute_silent_floor: float = 1e-4,
    ):
        self.band = DriftBand(window=window)
        self.silence_k = silence_k
        self.min_warmup = min_warmup
        self.absolute_silent_floor = absolute_silent_floor

    def observe(self, value: float | torch.Tensor) -> None:
        """Push a population-level ‖Δs_internal‖ value into the band."""
        if isinstance(value, torch.Tensor):
            value = float(value.detach().mean().item())
        self.band.push(float(value))

    def is_warm(self) -> bool:
        return self.band.is_warm(min_samples=self.min_warmup)

    def silent_threshold(self) -> float:
        """median - silence_k * MAD. Below this = internally still.

        Before warmup, returns +inf so the predicate defaults to
        "always silent" -- but K-M9-5 also requires external stasis
        AND its own armed state from the activity band (§A.1), so a
        not-yet-warm Δs band cannot by itself fire the dark-room
        kill. P1's engagement hinge defaults to the *floor* engagement
        target for legibility before warmup.
        """
        if not self.is_warm():
            return float("inf")
        med = self.band.median()
        mad = max(self.band.mad(), 1e-8)
        return med - self.silence_k * mad

    def engagement_target(self, floor: float = 0.0) -> float:
        """P1's hinge target = max(silent_threshold, floor).

        Below this magnitude, P1 cost rises; above, P1 cost is zero.
        Before warmup, returns `floor` so the engagement cost is
        nonzero only when `‖Δs‖ < floor` -- the launch-time floor
        prevents the entity from "achieving engagement" by simply
        being uninstrumented.
        """
        if not self.is_warm():
            return floor
        return max(self.silent_threshold(), floor)

    def is_silent_per_batch(
        self,
        delta_s_internal: torch.Tensor,
    ) -> torch.Tensor:
        """[B] bool: True where ‖Δs_internal‖ is below the silent
        threshold (band) OR below the absolute floor. The K-M9-5
        internal-stasis input.

        R1 round-2 fix: OR with `absolute_silent_floor` so a sustained
        zero-Δs constant -- born-catatonia -- fires the kill even
        before the band warms or after the band has recalibrated to
        the constant.
        """
        below_floor = delta_s_internal <= self.absolute_silent_floor
        if not self.is_warm():
            # Before warmup: only the absolute floor can fire.
            return below_floor
        return (delta_s_internal < self.silent_threshold()) | below_floor

    def snapshot(self) -> dict:
        return {
            "delta_s_band_median": self.band.median(),
            "delta_s_band_mad": self.band.mad(),
            "delta_s_silent_threshold": self.silent_threshold(),
            "delta_s_armed": self.is_warm(),
        }
