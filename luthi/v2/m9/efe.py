"""Expected Free Energy (EFE) evaluator -- pragmatic only at step 1.

Per spec §2 (step 1, beta_epi = 0):

    G(a_t) = w_eng * c_eng + w_coh * c_coh + w_con * c_con + w_truth * c_truth

The epistemic term is step 2 (MC-dropout parameter-novelty); the
evaluator interface accepts a `beta_epi` argument so step 2 can
flip it on without restructuring callers, but at step 1 the
epistemic branch is unreachable.

Horizon at launch: H = 1 (single-step rollout: predict s_hat from
(s_t, a_t), evaluate preferences on the (s_t, s_hat) transition).
The spec's "rollout-compute-within-inter-update-interval rule" puts
the cap on wall-clock compute, not imagined horizon -- H grows when
the loop integration shows we can afford it. The interface accepts
arbitrary horizon so the same code carries forward.

Inputs are threaded through optionally: features with missing
observations contribute zero per `Preferences.pragmatic_cost`'s
zero-on-missing semantics, so MCTS leaf scoring can run on
partial decoder availability (e.g., evaluating G during early
text-decoder-only bring-up before attention/memory decoders land).
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
    """

    def __init__(
        self,
        predictor: nn.Module,
        preferences: nn.Module,
        value_head: nn.Module | None = None,
    ):
        super().__init__()
        self.predictor = predictor
        self.preferences = preferences
        self.value_head = value_head  # optional; used by MCTS leaf bootstrap

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
        """Compute G(a_t) for a candidate action under the current
        substrate state.

        Returns:
            dict with keys:
              G          : [B] total EFE (pragmatic only at step 1)
              c_eng/coh/con/truth : [B] per-feature pragmatic costs
              w_eng/coh/con/truth : scalar weights actually applied
                                     (w_eng is floor-enforced)
              s_hat_next : [B, D] predicted next-state (detached)
              v_hat      : [B] V(s_hat_next), if value_head provided
                            (detached; for MCTS leaf bootstrap)
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
            # Step 2+: add epistemic. Not reachable at step 1; placeholder
            # so the interface is stable.
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
        **observation_kwargs,
    ) -> dict:
        """Compute G for a batch of K candidate actions per state.

        `candidate_actions`: [B, K, D]. Returns per-candidate
        G [B, K] plus the per-feature breakdowns [B, K] and the
        predicted next states [B, K, D].

        Implementation: loops over K and concatenates the per-batch
        results. For larger K this should batch through the predictor
        in one shot; the loop is cheap at K ~ 10-20 (the spec's
        habit-net proposal size) and trades simplicity for the
        engineering cost of a batched predictor variant.
        """
        B, K, D = candidate_actions.shape
        Gs = []
        s_hats = []
        c_eng = []
        c_coh = []
        c_con = []
        c_truth = []
        v_hats = []
        for k in range(K):
            a_k = candidate_actions[:, k, :]  # [B, D]
            out = self.compute_g(
                s_t=s_t,
                a_t=a_k,
                context_latents=context_latents,
                target_positions=target_positions,
                **observation_kwargs,
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
        return result
