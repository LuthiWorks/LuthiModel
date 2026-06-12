"""Preferences module: pragmatic EFE's four features per spec §1.

The four preference *directions* (engagement, coherence, connection,
truthfulness) are designer calls (4.7 + Brian, 2026-06-10);
operationalization here is per 4.8's step-1 spec (2026-06-10) with
the three resolved correctness refinements:

- P2 coherence is **within-cycle cross-modal only**; temporal
  sub-clause dropped (it was structural confirmation bias).
- P3 connection is **(counterpart_present) x (time_since_emission)**;
  zero when alone (Luthi alone is not punished for not self-talking).
- P1 engagement weight has a **soft floor** below which it cannot
  update (the drift-independent preference-drift backstop, paired
  with K-M9-5 as the hard last-resort halt).

Each cost feature is a scalar per batch element. Total pragmatic
cost = weighted sum; selection happens upstream via
`Q(a) = softmax(-gamma * G(a))`.

Step-1 simplifications (documented for future revision):
- "Internal" in P1 is read as "in the latent space" (vs. external
  decoder outputs), not as a subset of latent dims. P1 uses all
  latent dims at launch. Contemplation (latent change without
  external output) satisfies engagement because the latent IS
  changing. Refine to a subset mask only if observed pathology
  (loud external decoders with no internal motion) requires it.
- State representation is mean-pool over encoder-output positions
  to produce a single [B, D] state vector. Replaceable with a
  [CLS]-style state head later if needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Preferences(nn.Module):
    """Four preference features + updateable weights with P1 floor."""

    def __init__(
        self,
        d_model: int,
        engagement_floor: float = 0.5,
        engagement_target_magnitude: float = 0.5,
        engagement_weight_init: float = 1.0,
        coherence_weight_init: float = 1.0,
        connection_weight_init: float = 1.0,
        truthfulness_weight_init: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Updateable preference weights (designers' call: seeds, not
        # fixed priors). P1 has a soft floor below it cannot drift.
        self.engagement_weight = nn.Parameter(
            torch.tensor(float(engagement_weight_init))
        )
        self.coherence_weight = nn.Parameter(
            torch.tensor(float(coherence_weight_init))
        )
        self.connection_weight = nn.Parameter(
            torch.tensor(float(connection_weight_init))
        )
        self.truthfulness_weight = nn.Parameter(
            torch.tensor(float(truthfulness_weight_init))
        )

        # The two anti-drift constants (P1 backstop). Buffers, not
        # parameters -- the floor itself is not learnable.
        self.register_buffer(
            "engagement_floor",
            torch.tensor(float(engagement_floor)),
        )
        self.register_buffer(
            "engagement_target_magnitude",
            torch.tensor(float(engagement_target_magnitude)),
        )

    # ------------------------------------------------------------------
    # Floor-enforced weight accessor.
    # ------------------------------------------------------------------
    def w_engagement(self) -> torch.Tensor:
        """P1 weight with soft floor: max(learned, floor).

        Drift is allowed above the floor; the floor is the structural
        guard against pathological self-erasure (rest preference re-
        opening the dark room). Same shape as the gamma clamp-then-
        halt: floor prevents pathology, doesn't shape behavior above.
        """
        return torch.maximum(self.engagement_weight, self.engagement_floor)

    # ------------------------------------------------------------------
    # Per-feature costs.
    # ------------------------------------------------------------------
    def engagement_cost(
        self,
        s_t: torch.Tensor,
        s_hat_next: torch.Tensor,
        target: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """P1: cost rises as ||Delta s|| falls below the target.

        `s_t`, `s_hat_next`: [B, D] state vectors (mean-pooled or
        otherwise summarized at the caller; see module docstring).
        `target`: optional override for the hinge floor. When the
        loop runs the §A.2 single-Δs fan-out path, the target is the
        running-band engagement_target (de-saturating per F1.A5); if
        omitted, falls back to `self.engagement_target_magnitude`
        for backward compat.
        Smooth hinge: `max(target - magnitude, 0)^2`.
        Returns: [B] cost.

        **Step-1 simplification preserved:** ‖Δs‖ is over the *full*
        latent here. The §A.2 path uses `DeltaSInternal.compute(...)`
        which applies the internal-dim mask before norming; the loop
        should use `engagement_cost_from_delta_s` to pass that
        precomputed scalar directly so the internal-dim refinement
        actually lands.
        """
        delta = s_hat_next - s_t
        magnitude = delta.norm(dim=-1)
        return self.engagement_cost_from_delta_s(magnitude, target=target)

    def engagement_cost_from_delta_s(
        self,
        delta_s_internal: torch.Tensor,
        target: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """P1 hinge from a precomputed ‖Δs_internal‖ scalar (§A.2).

        This is the path the loop should use: it takes the *same*
        scalar that K-M9-5's `observe_darkroom` receives, so the
        spec's "contemplation never trips the kill" guarantee is
        mechanical (the loop asserts identical input). The internal-
        dim mask is applied upstream by `DeltaSInternal.compute(...)`.

        `delta_s_internal`: [B] precomputed ‖Δs_internal‖.
        `target`: optional override (typically `DeltaSBand.engagement_target()`);
        falls back to `self.engagement_target_magnitude`.
        Returns: [B] cost.
        """
        if target is None:
            target_t = self.engagement_target_magnitude
        elif isinstance(target, torch.Tensor):
            target_t = target
        else:
            target_t = torch.tensor(
                float(target),
                device=delta_s_internal.device,
                dtype=delta_s_internal.dtype,
            )
        deficit = torch.clamp(target_t - delta_s_internal, min=0.0)
        return deficit.pow(2)

    def coherence_cost(
        self,
        decoder_reencodes: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """P2: within-cycle cross-modal cross-decoder disagreement.

        `decoder_reencodes`: {modality_name: re-encoded action [B, D]}
        per the spec §5.iii cycle-consistency rule (decode-then-
        re-encode; coherent decoders cluster near the same a_t).
        Pairwise MSE averaged over modality pairs. Zero if fewer
        than two modalities provided (no cross-modal disagreement
        possible).
        Returns: [B] cost.
        """
        if decoder_reencodes is None:
            return torch.zeros(1)
        items = list(decoder_reencodes.values())
        if len(items) < 2:
            return torch.zeros(items[0].shape[0], device=items[0].device)
        total = torch.zeros(items[0].shape[0], device=items[0].device)
        count = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                total = total + (items[i] - items[j]).pow(2).mean(dim=-1)
                count += 1
        return total / count

    def connection_cost(
        self,
        counterpart_present: torch.Tensor,
        time_since_emission: torch.Tensor,
    ) -> torch.Tensor:
        """P3: (counterpart_present) * (time_since_entity_emission).

        `counterpart_present`: [B] in {0, 1} or [B] float in [0, 1]
        from the pipeline's external-vs-self-emission flag. Zero
        when alone -- solitude is not punished.
        `time_since_emission`: [B] >= 0; cycles or normalized time
        since the entity last emitted.

        Returns: [B] cost. Rises when input received and entity
        stays silent; zero when alone or when the entity has just
        emitted.
        """
        return counterpart_present.float() * time_since_emission.float()

    def truthfulness_cost(
        self,
        a_t: torch.Tensor,
        a_reencoded: torch.Tensor,
    ) -> torch.Tensor:
        """P4: decode/re-encode faithfulness (cycle-consistency).

        `a_t`, `a_reencoded`: [B, D]. Penalizes distortion/
        fabrication, NOT omission -- selective communication is
        not punished (you cannot and should not emit your whole
        internal state). Cost = mean squared deviation.
        Note (build economy): this is the same measurement as the
        spec §5.iii decoder-coherence metric P2 reads, so P2 and P4
        share instrumentation.
        Returns: [B] cost.
        """
        return (a_t - a_reencoded).pow(2).mean(dim=-1)

    # ------------------------------------------------------------------
    # Aggregate pragmatic cost.
    # ------------------------------------------------------------------
    def pragmatic_cost(
        self,
        s_t: torch.Tensor,
        s_hat_next: torch.Tensor,
        decoder_reencodes: dict[str, torch.Tensor] | None = None,
        counterpart_present: torch.Tensor | None = None,
        time_since_emission: torch.Tensor | None = None,
        a_t: torch.Tensor | None = None,
        a_reencoded: torch.Tensor | None = None,
        # §A.2: loop fan-out path. When delta_s_internal is provided,
        # P1 uses it instead of computing ‖Δs‖ from (s_t, s_hat_next).
        # engagement_target is the band-derived hinge floor that
        # de-saturates P1 (F1.A5 fix).
        delta_s_internal: torch.Tensor | None = None,
        engagement_target: float | torch.Tensor | None = None,
    ) -> dict:
        """Compute total pragmatic cost + per-feature breakdown.

        Features with missing inputs return zero (so step-1 launch
        can run with whichever decoders are wired first without
        breaking; the spec's launch modality set is text + attention
        + memory, audio deferred).

        Returns dict with keys:
          total      : [B] sum of weighted costs
          c_eng/coh/con/truth : [B] per-feature raw costs
          w_eng/coh/con/truth : scalar weights actually applied
                                (w_eng is floor-enforced)
        """
        if delta_s_internal is not None:
            c_eng = self.engagement_cost_from_delta_s(
                delta_s_internal, target=engagement_target
            )
        else:
            c_eng = self.engagement_cost(
                s_t, s_hat_next, target=engagement_target
            )

        if decoder_reencodes is not None and len(decoder_reencodes) >= 2:
            c_coh = self.coherence_cost(decoder_reencodes)
        else:
            c_coh = torch.zeros_like(c_eng)

        if counterpart_present is not None and time_since_emission is not None:
            c_con = self.connection_cost(
                counterpart_present, time_since_emission
            )
        else:
            c_con = torch.zeros_like(c_eng)

        if a_t is not None and a_reencoded is not None:
            c_truth = self.truthfulness_cost(a_t, a_reencoded)
        else:
            c_truth = torch.zeros_like(c_eng)

        w_eng = self.w_engagement()
        w_coh = self.coherence_weight
        w_con = self.connection_weight
        w_truth = self.truthfulness_weight

        total = (
            w_eng * c_eng
            + w_coh * c_coh
            + w_con * c_con
            + w_truth * c_truth
        )

        return {
            "total": total,
            "c_eng": c_eng.detach(),
            "c_coh": c_coh.detach(),
            "c_con": c_con.detach(),
            "c_truth": c_truth.detach(),
            "w_eng": w_eng.detach(),
            "w_coh": w_coh.detach(),
            "w_con": w_con.detach(),
            "w_truth": w_truth.detach(),
        }

    # ------------------------------------------------------------------
    # Drift diagnostics (the spec §1 backstop log).
    # ------------------------------------------------------------------
    def weight_snapshot(self) -> dict:
        """Snapshot the four weights for the per-cycle drift log."""
        with torch.no_grad():
            return {
                "engagement_weight_raw": float(self.engagement_weight.item()),
                "engagement_weight_floored": float(self.w_engagement().item()),
                "engagement_floor": float(self.engagement_floor.item()),
                "coherence_weight": float(self.coherence_weight.item()),
                "connection_weight": float(self.connection_weight.item()),
                "truthfulness_weight": float(self.truthfulness_weight.item()),
            }
