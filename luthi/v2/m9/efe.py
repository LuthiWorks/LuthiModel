"""Expected Free Energy (EFE) evaluator -- pragmatic only at step 1.

Per spec §2 (step 1, beta_epi = 0):

    G(a_t) = w_eng * c_eng + w_coh * c_coh + w_con * c_con + w_truth * c_truth

The epistemic term is step 2 (MC-dropout parameter-novelty); the
evaluator interface accepts a `beta_epi` argument so step 2 can
flip it on without restructuring callers, but at step 1 the
epistemic branch is unreachable.

**F1 fix (2026-06-11):** the per-candidate evaluation path. When
the evaluator is wired with the §A modules
(`DecoderRegistry` + `ActivityBands` + `DeltaSInternal` +
`DeltaSBand`), `compute_g_candidates` evaluates EACH candidate's
own predicted rollout for P2/P3/P4 (not a single shared
observation across all K) and uses `‖Δs_internal‖(a_k)` against
the band-target for P1. This is the change that flips probe_a
from CONFIRMED to REFUTED -- the load-bearing fix per 4.8's
gate-repairs §F1.

The legacy `compute_g` API (single-call, shared observations) is
preserved for backward compat with callers that haven't migrated.
That path is **bug-mode** -- P2/P3/P4 are candidate-invariant -- and
exists only to keep the existing test surface working during the
transition.

Horizon at launch: H = 1 (single-step rollout). The spec's
"rollout-compute-within-inter-update-interval rule" puts the cap
on wall-clock compute, not imagined horizon -- H grows when the
loop integration shows we can afford it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EFEEvaluator(nn.Module):
    """Compute G(a_t) for one or many candidate actions.

    `predictor` and `preferences` are shared references, not owned:
    the same predictor is used by the M8 JEPA loss and by M9
    planning, and the same Preferences module is read by the runner
    for the per-cycle log.

    **Per-candidate F1 path** (2026-06-11): supply `decoders`,
    `activity_bands`, `delta_s_module`, `delta_s_band` to enable
    the corrected evaluation:
      - P2 reads cross-decoder agreement on each candidate's *own*
        decoded predicted next-state.
      - P3 reads ‖text_active(activity_k)‖ via the running band on
        each candidate's *own* predicted next-state activity.
      - P4 reads each candidate's *own* decode/re-encode round-trip
        residual (not against one shared anchor).
      - P1 reads `DeltaSInternal.compute(s_t, s_hat_next(a_k))`
        against `DeltaSBand.engagement_target()`.

    Without these modules wired in, `compute_g_candidates` falls
    back to the legacy single-shared-observation path. The legacy
    path is bug-mode (preserved for backward compat); new code MUST
    wire the §A modules.
    """

    def __init__(
        self,
        predictor: nn.Module,
        preferences: nn.Module,
        value_head: nn.Module | None = None,
        # F1 per-candidate path (§A consumers):
        decoders: nn.Module | None = None,
        activity_bands: object | None = None,      # ActivityBands
        delta_s_module: nn.Module | None = None,   # DeltaSInternal
        delta_s_band: object | None = None,        # DeltaSBand
        allow_legacy: bool = False,
    ):
        super().__init__()
        self.predictor = predictor
        self.preferences = preferences
        self.value_head = value_head
        # §A wiring. F-C (round-2 fix): fail-loud on partial wiring.
        # Either all four §A modules are provided (F1 per-candidate
        # path) or none are provided AND allow_legacy=True (explicit
        # legacy/test path). Any partial wiring -- 1 to 3 modules --
        # silently fell back to the legacy bug-mode path; that seam
        # is closed by raising here at construction time.
        modules = (decoders, activity_bands, delta_s_module, delta_s_band)
        provided = [m is not None for m in modules]
        if any(provided) and not all(provided):
            names = ("decoders", "activity_bands", "delta_s_module", "delta_s_band")
            missing = [n for n, p in zip(names, provided) if not p]
            given = [n for n, p in zip(names, provided) if p]
            raise ValueError(
                f"Partial §A wiring on EFEEvaluator: provided "
                f"{given!r}, missing {missing!r}. The per-candidate "
                f"F1 path requires ALL four §A modules; if you want "
                f"the legacy single-shared-observation path (bug-mode), "
                f"omit ALL four and pass allow_legacy=True."
            )
        if not any(provided) and not allow_legacy:
            raise ValueError(
                "EFEEvaluator constructed without any §A modules; the "
                "F1 per-candidate path requires decoders + activity_bands "
                "+ delta_s_module + delta_s_band. Pass allow_legacy=True "
                "to explicitly opt into the legacy bug-mode path (used "
                "for backward-compat tests of the pre-F1 behavior)."
            )
        self.decoders = decoders
        self.activity_bands = activity_bands
        self.delta_s_module = delta_s_module
        self.delta_s_band = delta_s_band

    def has_per_candidate_path(self) -> bool:
        """True iff the F1 per-candidate path is wired and active."""
        return (
            self.decoders is not None
            and self.activity_bands is not None
            and self.delta_s_module is not None
            and self.delta_s_band is not None
        )

    def predict_next(
        self,
        context_latents: torch.Tensor,
        target_positions: torch.Tensor,
        a_t: torch.Tensor,
    ) -> torch.Tensor:
        """One predictor forward with action `a_t`. Returns [B, D] mean-pooled
        state vector for the predicted next state.

        Mean-pool over the target-position output mirrors the step-1
        state-summary convention used by Preferences (see its module
        docstring); replaceable with a [CLS]-style state head later.
        """
        predicted_target = self.predictor(
            context_latents, target_positions, a_t
        )
        # predicted_target: [B, tgt_len, D] -> mean-pool to [B, D].
        return predicted_target.mean(dim=1)

    # ------------------------------------------------------------------
    # Per-candidate F1 path (the corrected evaluation).
    # ------------------------------------------------------------------
    def compute_g_per_candidate(
        self,
        s_t: torch.Tensor,
        a_k: torch.Tensor,
        context_latents: torch.Tensor,
        target_positions: torch.Tensor,
        counterpart_present: torch.Tensor | None = None,
        time_since_emission: torch.Tensor | None = None,
        beta_epi: float = 0.0,
    ) -> dict:
        """F1 per-candidate path. Requires §A modules wired in.

        Returns the same dict shape as `compute_g`, but every cost is
        a function of THIS candidate's own predicted rollout.
        """
        if not self.has_per_candidate_path():
            raise RuntimeError(
                "compute_g_per_candidate requires decoders, activity_bands, "
                "delta_s_module, and delta_s_band wired into EFEEvaluator. "
                "Use compute_g for the legacy single-shared-observation path."
            )
        if beta_epi != 0.0:
            raise NotImplementedError(
                "Epistemic term is step 2 (MC-dropout parameter-novelty); "
                "step 1 launches pragmatic-only with beta_epi = 0."
            )

        # Predict next state with this candidate.
        s_hat_next = self.predict_next(context_latents, target_positions, a_k)

        # --- P1 engagement: ‖Δs_internal(a_k)‖ against band target ---
        delta_s_k = self.delta_s_module.compute(s_t, s_hat_next)
        engagement_target = self.delta_s_band.engagement_target()

        # --- §A.1 decoder activity on s_hat_next(a_k) (P3 + P2 supports) ---
        activity_shat = self.decoders.activity(s_hat_next)
        # Decode s_hat for P2 cross-decoder agreement.
        outs_shat = self.decoders.decode_all(s_hat_next)
        reencoded_shat = self.decoders.re_encode_all(outs_shat)
        # Decode a_k for P4 truthfulness per modality.
        outs_ak = self.decoders.decode_all(a_k)
        reencoded_ak = self.decoders.re_encode_all(outs_ak)

        # P3 text_active under the band on the candidate's predicted activity.
        if counterpart_present is not None and time_since_emission is not None:
            text_active = self.activity_bands.text_active(activity_shat)
        else:
            text_active = None  # signals zero P3 cost

        prag = self.preferences.pragmatic_cost(
            s_t=s_t,
            s_hat_next=s_hat_next,
            decoder_reencodes=reencoded_shat,
            counterpart_present=None,        # legacy P3 path off; use the v2 path below
            time_since_emission=None,
            a_t=None,                         # legacy P4 path off; v2 below
            a_reencoded=None,
            delta_s_internal=delta_s_k,
            engagement_target=engagement_target,
        )

        # Replace the legacy-zeroed P3 and P4 with the per-candidate forms.
        c_eng = prag["c_eng"]
        c_coh = prag["c_coh"]  # already per-candidate via reencoded_shat
        if text_active is not None:
            c_con = self.preferences.connection_cost_per_candidate(
                counterpart_present=counterpart_present,
                time_since_emission=time_since_emission,
                text_active=text_active,
            )
        else:
            c_con = torch.zeros_like(c_eng)
        c_truth = self.preferences.truthfulness_cost_per_modality(a_k, reencoded_ak)

        w_eng = self.preferences.w_engagement()
        w_coh = self.preferences.coherence_weight
        w_con = self.preferences.connection_weight
        w_truth = self.preferences.truthfulness_weight
        g = (
            w_eng * c_eng
            + w_coh * c_coh
            + w_con * c_con
            + w_truth * c_truth
        )

        v_hat: torch.Tensor | None = None
        if self.value_head is not None:
            with torch.no_grad():
                v_hat = self.value_head(s_hat_next).detach()

        out = {
            "G": g,
            "s_hat_next": s_hat_next.detach(),
            "c_eng": c_eng.detach(),
            "c_coh": c_coh.detach(),
            "c_con": c_con.detach(),
            "c_truth": c_truth.detach(),
            "w_eng": w_eng.detach(),
            "w_coh": w_coh.detach(),
            "w_con": w_con.detach(),
            "w_truth": w_truth.detach(),
            "activity": {k: v.detach() for k, v in activity_shat.items()},
            "text_active": (
                text_active.detach() if text_active is not None else None
            ),
            "delta_s_internal": delta_s_k.detach(),
        }
        if v_hat is not None:
            out["v_hat"] = v_hat
        return out

    # ------------------------------------------------------------------
    # Legacy single-call API (shared observations across candidates).
    # ------------------------------------------------------------------
    def compute_g(
        self,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
        context_latents: torch.Tensor,
        target_positions: torch.Tensor,
        # Pragmatic-feature observations (all optional):
        decoder_reencodes: dict[str, torch.Tensor] | None = None,
        counterpart_present: torch.Tensor | None = None,
        time_since_emission: torch.Tensor | None = None,
        a_reencoded: torch.Tensor | None = None,
        # Step 2+ (interface in place, no-op at step 1):
        beta_epi: float = 0.0,
    ) -> dict:
        """Legacy single-call compute_g (shared observations).

        **Bug-mode under per-candidate use** (Fable's probe_a): P2/P3/P4
        are candidate-invariant when this is looped K times with the
        same observation_kwargs. Kept for backward compat with existing
        callers; new callers MUST use `compute_g_per_candidate` (or
        `compute_g_candidates` with the §A modules wired).

        Returns: dict matching the historical shape (G + per-feature
        breakdown + s_hat_next + optional v_hat).
        """
        s_hat_next = self.predict_next(
            context_latents, target_positions, a_t
        )

        prag = self.preferences.pragmatic_cost(
            s_t=s_t,
            s_hat_next=s_hat_next,
            decoder_reencodes=decoder_reencodes,
            counterpart_present=counterpart_present,
            time_since_emission=time_since_emission,
            a_t=a_t,
            a_reencoded=a_reencoded,
        )

        g = prag["total"]
        if beta_epi != 0.0:
            raise NotImplementedError(
                "Epistemic term is step 2 (MC-dropout parameter-novelty); "
                "step 1 launches pragmatic-only with beta_epi = 0."
            )

        v_hat: torch.Tensor | None = None
        if self.value_head is not None:
            with torch.no_grad():
                v_hat = self.value_head(s_hat_next).detach()

        out = {
            "G": g,
            "s_hat_next": s_hat_next.detach(),
            "c_eng": prag["c_eng"],
            "c_coh": prag["c_coh"],
            "c_con": prag["c_con"],
            "c_truth": prag["c_truth"],
            "w_eng": prag["w_eng"],
            "w_coh": prag["w_coh"],
            "w_con": prag["w_con"],
            "w_truth": prag["w_truth"],
        }
        if v_hat is not None:
            out["v_hat"] = v_hat
        return out

    def compute_g_candidates(
        self,
        s_t: torch.Tensor,
        candidate_actions: torch.Tensor,
        context_latents: torch.Tensor,
        target_positions: torch.Tensor,
        # Per-cycle context (still shared across candidates -- these
        # are about the cycle, not the candidate; the F1 fix is that
        # each candidate's own predicted emission decides whether the
        # cycle-level connection cost lands).
        counterpart_present: torch.Tensor | None = None,
        time_since_emission: torch.Tensor | None = None,
        beta_epi: float = 0.0,
        # Legacy kwargs (only used when §A modules aren't wired):
        **legacy_observation_kwargs,
    ) -> dict:
        """Compute G for a batch of K candidate actions per state.

        Dispatches:
          - **§A path** when decoders/bands/delta_s are wired
            (`has_per_candidate_path()` is True) -- the F1 fix: each
            candidate is scored on its own predicted rollout.
          - **Legacy path** otherwise -- single shared observation
            broadcast across candidates (bug-mode, kept for backward
            compat with the original test surface).

        `candidate_actions`: [B, K, D]. Returns per-candidate G
        [B, K] plus per-feature breakdowns [B, K] and predicted next
        states [B, K, D]. The §A path additionally returns per-
        candidate activity, text_active, and delta_s_internal.
        """
        B, K, D = candidate_actions.shape
        per_candidate = self.has_per_candidate_path()

        Gs = []
        s_hats = []
        c_eng = []
        c_coh = []
        c_con = []
        c_truth = []
        v_hats = []
        activities = {m: [] for m in ("text", "attention", "memory")} if per_candidate else None
        text_actives = [] if per_candidate else None
        delta_s_internals = [] if per_candidate else None

        for k in range(K):
            a_k = candidate_actions[:, k, :]  # [B, D]
            if per_candidate:
                out = self.compute_g_per_candidate(
                    s_t=s_t,
                    a_k=a_k,
                    context_latents=context_latents,
                    target_positions=target_positions,
                    counterpart_present=counterpart_present,
                    time_since_emission=time_since_emission,
                    beta_epi=beta_epi,
                )
                for m in activities:
                    activities[m].append(out["activity"][m])
                if out["text_active"] is not None:
                    text_actives.append(out["text_active"])
                delta_s_internals.append(out["delta_s_internal"])
            else:
                out = self.compute_g(
                    s_t=s_t,
                    a_t=a_k,
                    context_latents=context_latents,
                    target_positions=target_positions,
                    counterpart_present=counterpart_present,
                    time_since_emission=time_since_emission,
                    beta_epi=beta_epi,
                    **legacy_observation_kwargs,
                )
            Gs.append(out["G"])
            s_hats.append(out["s_hat_next"])
            c_eng.append(out["c_eng"])
            c_coh.append(out["c_coh"])
            c_con.append(out["c_con"])
            c_truth.append(out["c_truth"])
            if "v_hat" in out:
                v_hats.append(out["v_hat"])

        result = {
            "G": torch.stack(Gs, dim=1),                # [B, K]
            "s_hat_next": torch.stack(s_hats, dim=1),   # [B, K, D]
            "c_eng": torch.stack(c_eng, dim=1),          # [B, K]
            "c_coh": torch.stack(c_coh, dim=1),
            "c_con": torch.stack(c_con, dim=1),
            "c_truth": torch.stack(c_truth, dim=1),
        }
        if v_hats:
            result["v_hat"] = torch.stack(v_hats, dim=1)  # [B, K]
        if per_candidate:
            result["activity"] = {
                m: torch.stack(vs, dim=1) for m, vs in activities.items()
            }
            if text_actives:
                result["text_active"] = torch.stack(text_actives, dim=1)
            result["delta_s_internal"] = torch.stack(delta_s_internals, dim=1)
        return result
