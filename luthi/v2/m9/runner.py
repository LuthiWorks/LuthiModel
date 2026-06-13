"""M9 step-1 runner: composes JEPATrainer with the M9 head training loop.

Per `docs/research/2026-06-11_m9-step1-training-integration-spec.md`:

- **Extend, don't fork.** `M9Trainer` subclasses `JEPATrainer` so the
  data loop, per-modality cadence, pilot-set framework, and
  checkpoint/resume machinery are reused verbatim.
- **Two-phase step.** Core M8 update bit-identical (same encoder +
  predictor + projection_heads + SIGReg). Then a second
  backward/step on the **separate M9 optimizer** for the heads
  (V-TD, habit-distill, decoder cycle-consistency) over **detached
  latents** -- stop-grad discipline keeps M9 head gradients out of
  the JEPA representation while the head training is unverified.
- **Synchronous at step 1.** The CC cycle (perceive → predict →
  plan[habit+MCTS] → act[decode] → consolidate[plasticity]) and
  `train_step` run in one loop. Async actor/learner is a later
  optimization.
- **Diagnostics + kills reuse the existing framework.** New M9
  metric keys plug into `_compute_and_log_diagnostics` and
  `_advance_pilot_state`/`_observe_stationary`/`_observe_trending`
  -- K-M9-1..9 are new entries in the kill-criteria scheduler, not
  new machinery.
- **Checkpoint extends symmetrically.** New keys: V head +
  V-target, habit net, decoders, m9 optimizer, gamma scalar,
  preference weights. The persistent MCTS tree is **not persisted**
  (cold-rebuild on resume, self-heals via recency-decay).
- **Stop-grad isolation.** The M9 heads receive `.detach()`'d
  encoder/predictor latents at every head input; the spec calls
  this out as load-bearing for M8 stability while the M9
  interaction is unverified.

Scope of this first slice: the trainer skeleton -- composition,
construction, two-phase train_step, checkpoint/resume extension.
MCTS planning, a_rest reference, theta_version stamping, and the
M9 kill wiring land in follow-up slices.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer

from luthi.v2.jepa_loss import JEPALoss
from luthi.v2.jepa_runner import (
    JEPATrainer,
    ModalitySampler,
    MultimodalDataLoader,
    RunnerConfig,
)
from luthi.v2.m9.activity_bands import ActivityBandConfig, ActivityBands
from luthi.v2.m9.decoders import (
    AttentionDecoder,
    DecoderRegistry,
    MemoryDecoder,
    TextDecoder,
)
from luthi.v2.m9.delta_s import DeltaSBand, DeltaSInternal
from luthi.v2.m9.efe import EFEEvaluator
from luthi.v2.m9.gamma import GammaInference
from luthi.v2.m9.habit_net import HabitNet
from luthi.v2.m9.instrumentation import ActionLog, MIProbe
from luthi.v2.m9.kills import KillRegistry
from luthi.v2.m9.mcts import MCTS
from luthi.v2.m9.preferences import Preferences
from luthi.v2.m9.rest_action import RestActionNet
from luthi.v2.m9.staleness import StalenessConfig, StalenessManager
from luthi.v2.m9.value_head import ValueHead

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# M9Config
# ---------------------------------------------------------------------------


@dataclass
class M9Config:
    """Step-1 M9 hyperparameters. Layered alongside the M8 `RunnerConfig`
    so M9-specific knobs don't bloat the M8 surface.

    All pilot-set values; the round-2 audit flagged a handful as
    needing real-workload calibration (connection_max_silence,
    value_abs_ceiling, absolute_silent_floor). Defaults match the
    pilot values that round-2 probes verified.
    """

    # Head learning rate (separate optimizer over V + habit + decoders).
    head_lr: float = 1e-3
    # V target network update rate (Polyak averaging per training step).
    v_target_polyak: float = 0.005
    # Reward discount for V-TD bootstrap.
    discount: float = 0.99
    # Habit distillation: visit-weighted MLE temperature on MCTS visits.
    habit_distill_temperature: float = 1.0
    # Decoder cycle-consistency loss weight.
    decoder_cycle_consistency_weight: float = 1.0
    # Rest-action loss weight: minimize ‖predict_next(s_t, a_rest) - s_t‖_internal.
    # The RestActionNet is zero-init so the early loss is uninformative;
    # weight ramps in over training as a_rest accumulates context-dependent
    # structure.
    rest_action_weight: float = 1.0
    # MCTS planning budget per cycle (simulations to expand per train_step).
    mcts_budget_per_cycle: int = 8
    # MCTS progressive-widening params.
    mcts_widening_alpha: float = 0.5
    mcts_widening_c: float = 1.0
    mcts_c_puct: float = 1.0
    mcts_max_depth: int = 1
    # Habit-net K candidates per planning step.
    habit_n_candidates: int = 8
    # MI probe ridge regularization for the trunk-vs-target linear probe.
    mi_probe_ridge_lambda: float = 1e-2
    # gamma inference defaults.
    gamma_init: float = 1.0
    gamma_min: float = 0.01
    gamma_max: float = 100.0
    gamma_rho: float = 0.1
    gamma_scale: float = 1.0
    # Preferences defaults.
    engagement_floor: float = 0.5
    engagement_target_magnitude: float = 0.5
    connection_max_silence: float = 50.0
    # ActivityBands defaults (R1 round-2 floor + R3 round-2 emission scale).
    activity_band_window: int = 32
    activity_silence_k: float = 1.5
    activity_absolute_silent_floor: float = 1e-3
    activity_emission_signal_scale: float = 0.1
    # DeltaSBand defaults.
    delta_s_band_window: int = 32
    delta_s_silence_k: float = 1.5
    delta_s_absolute_silent_floor: float = 1e-4
    # Staleness defaults.
    staleness_refresh_scale: float = 10.0
    staleness_alpha_refresh_min: float = 0.1
    # Action log JSONL filename (relative to run_dir).
    action_log_filename: str = "m9_action_log.jsonl"
    # gamma inference can be held FIXED during the very first mechanical
    # bring-up (per spec §9 staging note). Behavioural validation
    # requires inferred-gamma; this is a 4.7 staging convenience only.
    gamma_fixed_for_bringup: bool = False


# ---------------------------------------------------------------------------
# M9Trainer
# ---------------------------------------------------------------------------


class M9Trainer(JEPATrainer):
    """JEPATrainer extended with the M9 step-1 head training loop.

    Construction mirrors `JEPATrainer` and adds the M9 component bag.
    The M9 components are owned by this trainer:
      - ValueHead `V` + target Polyak copy `V_target`
      - HabitNet
      - DecoderRegistry (text reuses M8 `output_proj`; attention,
        memory new)
      - GammaInference
      - KillRegistry (M9 kills; M8 kills continue through the
        inherited pilot-set framework)
      - ActivityBands + DeltaSInternal + DeltaSBand (§A round-1 +
        round-2 absolute floors)
      - Preferences module
      - EFEEvaluator (the planning surface; MCTS is wired in a
        follow-up slice)
      - StalenessManager (cross-cycle tree machinery)
      - MIProbe + ActionLog (instrumentation)

    The trainer has TWO optimizers (per spec §3 actor/learner split):
      `optimizer` (inherited): JEPA core -- encoder + predictor +
        projection_heads. Bit-identical M8 dynamics.
      `m9_optimizer`: the M9 heads -- V + habit_net + decoders.
        Trains over **detached** encoder/predictor latents so M9
        head gradients cannot reshape the JEPA representation.

    `train_step` runs the two phases sequentially:
      1. super().train_step(modality, batch)  → M8 core update
      2. self._m9_head_step(modality, raw)    → M9 head update on
                                                 detached latents
    """

    def __init__(
        self,
        loss_module: JEPALoss,
        optimizer: Optimizer,
        sampler: ModalitySampler,
        data_loader: MultimodalDataLoader,
        config: RunnerConfig,
        run_dir: Path,
        m9_config: Optional[M9Config] = None,
    ):
        super().__init__(
            loss_module=loss_module,
            optimizer=optimizer,
            sampler=sampler,
            data_loader=data_loader,
            config=config,
            run_dir=run_dir,
        )
        self.m9_config = m9_config or M9Config()
        d_model = loss_module.online_encoder.d_model
        vocab_size = loss_module.online_encoder.vocab_size

        # ---- M9 module bag ----
        # Value head + Polyak target copy (V_target is not gradient-
        # trained; updated via Polyak averaging from V on each step).
        self.v_head = ValueHead(d_model=d_model)
        self.v_target = ValueHead(d_model=d_model)
        self.v_target.load_state_dict(self.v_head.state_dict())
        for p in self.v_target.parameters():
            p.requires_grad = False

        # Habit network -- Fountas-style amortized proposal.
        self.habit_net = HabitNet(d_model=d_model)

        # Rest-action network -- spec §6.i `a_rest(s_t)`. Zero-init so
        # early-cycle a_rest is near zero (a "do nothing" prior); the
        # M9 head training minimizes ‖predict_next(s_t, a_rest) - s_t‖
        # so it learns context-dependent minimal-self-change actions.
        # Feeds the rest-reference plumbing in ActivityBands /
        # DeltaSBand (which replaces the round-2 absolute-floor backstop
        # once RestActionNet has trained).
        self.rest_action = RestActionNet(d_model=d_model)

        # Decoders. Text decoder reuses the M8 `output_proj` head per
        # spec §1 (frozen / low-LR at launch).
        self.decoders = DecoderRegistry(
            text=TextDecoder(
                output_proj=loss_module.online_encoder.output_proj,
                d_model=d_model,
                vocab_size=vocab_size,
            ),
            attention=AttentionDecoder(d_model=d_model, n_modalities=3),
            memory=MemoryDecoder(d_model=d_model),
        )

        # Preferences module.
        self.preferences = Preferences(
            d_model=d_model,
            engagement_floor=self.m9_config.engagement_floor,
            engagement_target_magnitude=self.m9_config.engagement_target_magnitude,
            connection_max_silence=self.m9_config.connection_max_silence,
        )

        # §A modules: activity bands (R1 round-2 absolute floor), Δs band
        # (R1 round-2 absolute floor), Δs computation module.
        self.activity_bands = ActivityBands(
            config=ActivityBandConfig(
                window=self.m9_config.activity_band_window,
                silence_k=self.m9_config.activity_silence_k,
                absolute_silent_floor=self.m9_config.activity_absolute_silent_floor,
                emission_signal_scale=self.m9_config.activity_emission_signal_scale,
            )
        )
        self.delta_s_module = DeltaSInternal(d_model=d_model)
        self.delta_s_band = DeltaSBand(
            window=self.m9_config.delta_s_band_window,
            silence_k=self.m9_config.delta_s_silence_k,
            absolute_silent_floor=self.m9_config.delta_s_absolute_silent_floor,
        )

        # EFE evaluator -- the F1 per-candidate path is the only one
        # production callers should use, enforced by F-C round-2.
        self.efe = EFEEvaluator(
            predictor=loss_module.predictor,
            preferences=self.preferences,
            value_head=self.v_head,
            decoders=self.decoders,
            activity_bands=self.activity_bands,
            delta_s_module=self.delta_s_module,
            delta_s_band=self.delta_s_band,
        )

        # gamma + kill registry + staleness manager + MI probe + action log.
        self.gamma = GammaInference(
            rho_g=self.m9_config.gamma_rho,
            gamma_init=self.m9_config.gamma_init,
            gamma_min=self.m9_config.gamma_min,
            gamma_max=self.m9_config.gamma_max,
            gamma_scale=self.m9_config.gamma_scale,
            fixed_for_bringup=self.m9_config.gamma_fixed_for_bringup,
        )
        self.m9_kills = KillRegistry()
        self.staleness = StalenessManager(
            StalenessConfig(
                staleness_refresh_scale=self.m9_config.staleness_refresh_scale,
                alpha_refresh_min=self.m9_config.staleness_alpha_refresh_min,
                # F-D loop-integration: stale-node identification reads
                # from `self.theta_version` (one tick per cycle) instead
                # of `mcts.sim_counter` (one tick per simulation). The
                # right unit for "Q is from an older theta."
                staleness_uses_theta_version=True,
            )
        )
        self.mi_probe = MIProbe(
            ridge_lambda=self.m9_config.mi_probe_ridge_lambda
        )
        self.action_log = ActionLog(
            self.run_dir / self.m9_config.action_log_filename
        )

        # Persistent MCTS. The tree is single-state per spec §3 at
        # step 1 (one entity, one current state). reset() is called
        # each train_step against the cycle's s_t; plan_budget runs
        # mcts_budget_per_cycle simulations against that root. The
        # tree is intentionally NOT persisted across resume (spec
        # §5: cold-rebuild, self-heals via recency-decay).
        self.mcts = MCTS(
            habit_net=self.habit_net,
            efe_evaluator=self.efe,
            value_head=self.v_head,
            widening_alpha=self.m9_config.mcts_widening_alpha,
            widening_c=self.m9_config.mcts_widening_c,
            c_puct=self.m9_config.mcts_c_puct,
            max_depth=self.m9_config.mcts_max_depth,
        )

        # ---- M9 optimizer ----
        # Separate Adam over the M9 heads. V_target is not trained
        # directly. Text decoder's wrapped `output_proj` is NOT included
        # here -- it continues to train via the M8 core (low-LR /
        # frozen behavior per spec; the wrapping just preserves the
        # interface, not the param ownership).
        m9_params = list(self.v_head.parameters()) \
            + list(self.habit_net.parameters()) \
            + list(self.rest_action.parameters()) \
            + list(self.decoders.attention.parameters()) \
            + list(self.decoders.memory.parameters()) \
            + list(self.decoders.text.intensity_head.parameters()) \
            + list(self.decoders.text.reencode_head.parameters()) \
            + list(self.preferences.parameters()) \
            + list(self.delta_s_module.parameters())
        self.m9_optimizer = Adam(m9_params, lr=self.m9_config.head_lr)

        # ---- Step-1 training-story metrics ----
        # Last-cycle realized transition; used by V-TD for the bootstrap.
        # Filled in by `train_step` after the M8 forward.
        self._last_state: Optional[torch.Tensor] = None
        self._last_action: Optional[torch.Tensor] = None
        self._last_reward: Optional[float] = None

    # ------------------------------------------------------------------
    # Two-phase train_step.
    # ------------------------------------------------------------------
    def train_step(self, modality: str, batch: dict) -> dict:
        """Phase 1: M8 core update (super). Phase 2: M9 head update on
        detached latents.

        Returns the M8 step result dict augmented with M9 sub-losses
        under the `m9` key. Existing M8 diagnostics + kill plumbing
        sees the same shape as before, so the inherited framework
        continues to work unmodified.
        """
        # ---- Phase 1: M8 core update ----
        m8_result = super().train_step(modality, batch)

        # ---- Phase 2: M9 head update on detached latents ----
        m9_losses = self._m9_head_step(modality, m8_result["raw"])
        m8_result["m9"] = m9_losses

        # ---- Polyak update of V_target ----
        # Done outside the gradient step; just an exponential moving
        # average toward V's parameters. K-M9-3 backstops divergence.
        with torch.no_grad():
            tau = self.m9_config.v_target_polyak
            for p, p_t in zip(self.v_head.parameters(), self.v_target.parameters()):
                p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

        return m8_result

    def _m9_head_step(self, modality: str, raw: dict) -> dict:
        """Train V, habit, and decoders on **detached** latents using
        live MCTS planning as the visit-distill + reward source.

        Per spec §1 stop-grad discipline: encoder/predictor latents
        are .detach()'d at every M9 head input; the M9 gradient
        cannot flow back into the JEPA representation while the
        head training is unverified.
        """
        # Pool the encoder context to a [B, D] state vector (mean over
        # context positions, mirroring the EFEEvaluator step-1
        # convention). DETACHED -- stop-grad discipline.
        s_t = raw["online_context_latents"].detach().mean(dim=1)  # [B, D]
        s_hat_next = raw["predicted_target"].detach().mean(dim=1)  # [B, D]
        # MCTS planning is single-state per spec §3 (one entity, one
        # state); train on batch element 0 as the cycle's "current"
        # state. Other batch elements still benefit from M8 core
        # training; their M9 head signal is the cycle-consistency
        # path which is per-element.
        s_root = s_t[0].detach()                            # [D]
        ctx_for_mcts = raw["online_context_latents"].detach()  # [B, ctx, D]
        # MCTS reads a single context window; use batch[0]'s slice.
        ctx_single = ctx_for_mcts[0:1]                       # [1, ctx, D]
        tgt_positions = self._target_positions_for(
            raw["ctx_len"]
        ).to(s_root.device)

        # ---- MCTS plan-budget ----
        # Reset to current state and run the per-cycle budget. The
        # tree persists across train_steps for the spec's amortized
        # planning (advance_root would carry the chosen subtree
        # forward at inference); at training we reset each step so
        # the visit distribution is rooted at THIS batch's state.
        self.mcts.reset(s_root, ctx_single, tgt_positions)
        # F-D loop-integration: synchronize the MCTS stamping source
        # with the staleness manager's theta_version so node Q values
        # carry the cycle-units stamp (one tick per weight update),
        # not the sim-units stamp (one tick per simulation). This is
        # what makes "Q is from an older theta" mean what staleness
        # actually wants to ask.
        self.mcts.current_theta_version = self.staleness.theta_version
        self.mcts.plan_budget(
            budget=self.m9_config.mcts_budget_per_cycle,
            observation_kwargs=self._cycle_observation_kwargs(),
        )

        self.m9_optimizer.zero_grad(set_to_none=True)

        # ---- Habit visit-distill (replaces placeholder) ----
        # The MCTS root expanded K_actual children, each from a
        # habit-net sample. Compute the visit-weighted log-prob of
        # those actions under the *current* habit net (gradient
        # flows through habit-net forward → log-prob → loss).
        children = self.mcts.root.children
        if children:
            actions_k = torch.stack(
                [c.action_in for c in children], dim=0
            ).unsqueeze(0)                                  # [1, K, D]
            visits_k = torch.tensor(
                [c.N for c in children], dtype=torch.float32,
                device=s_root.device,
            )                                                # [K]
            visit_target = visits_k / visits_k.sum().clamp(min=1.0)
            log_p_habit = self.habit_net.log_prob(
                s_root.unsqueeze(0), actions_k
            )                                                # [1, K]
            habit_loss = -(visit_target * log_p_habit[0]).sum()

            # ---- gamma update: read the K candidate EFEs ----
            child_g = torch.tensor(
                [c.incoming_g for c in children if c.incoming_g is not None],
                dtype=torch.float32, device=s_root.device,
            )
            if child_g.numel() >= 2:
                self.gamma.update(child_g)
            # ---- Reward = -G_best (lowest G = best candidate) ----
            r_best = float(-child_g.min().item()) if child_g.numel() > 0 else 0.0
        else:
            # No children expanded (degenerate budget = 0 edge). Fall
            # back to the original placeholder pair so training stays
            # well-defined.
            sample = self.habit_net.sample(
                s_t, K=self.m9_config.habit_n_candidates
            )
            habit_loss = -sample["log_prob"].mean()
            r_best = 0.0

        # ---- V-TD: r_best + gamma * V_target(s_{t+1}) ----
        # r_best is the realized-cycle reward (negative EFE of the
        # MCTS-best action). Broadcast across the batch since at
        # this step the same cycle reward applies; per-batch-element
        # rewards arrive when batched MCTS lands later.
        r_t = torch.full(
            (s_t.shape[0],), r_best, device=s_t.device, dtype=s_t.dtype,
        )
        with torch.no_grad():
            v_target_next = self.v_target(s_hat_next)
            td_target = r_t + self.m9_config.discount * v_target_next
        v_pred = self.v_head(s_t)
        v_loss = (v_pred - td_target).pow(2).mean()

        # ---- Decoder cycle-consistency (per-element, unchanged) ----
        outs = self.decoders.decode_all(s_hat_next)
        reencoded = self.decoders.re_encode_all(outs)
        dec_loss = sum(
            (s_hat_next - r).pow(2).mean()
            for r in reencoded.values()
        ) / max(1, len(reencoded))
        dec_loss = dec_loss * self.m9_config.decoder_cycle_consistency_weight

        # ---- a_rest reference + rest-action loss ----
        # Per spec §6.i: a_rest(s_t) = "predict minimal self-change".
        # Train RestActionNet to minimize ‖predict_next(s_t, a_rest) - s_t‖
        # along the internal-dim axis. Once trained, s_rest gives the
        # per-state silence reference that the bands consume via
        # is_silent(rest_activity_value=...) / is_silent_per_batch(
        # rest_delta_s=...) -- the context-dependent stasis floor that
        # replaces the round-2 absolute_silent_floor.
        #
        # Stop-grad: the rest forward goes through the M8 predictor.
        # The predictor's params are in the M8 optimizer (not in
        # m9_optimizer), so m9_optimizer.step() won't update them; any
        # gradient that accumulates on loss_module params is wiped
        # below (see `for p in self.loss_module.parameters(): p.grad =
        # None`) so the M9 path provably cannot reshape M8.
        a_rest_t = self.rest_action(s_t)                            # [B, D]
        ctx_full = raw["online_context_latents"].detach()           # [B, ctx, D]
        tgt_full = self._target_positions_for(raw["ctx_len"]).to(
            s_t.device
        ).expand(s_t.shape[0], -1)
        s_rest = self.efe.predict_next(ctx_full, tgt_full, a_rest_t)  # [B, D]
        rest_delta_s = self.delta_s_module.compute(s_t, s_rest)       # [B]
        rest_loss = rest_delta_s.mean() * self.m9_config.rest_action_weight

        # Detached references for the band observations + classifiers.
        with torch.no_grad():
            rest_activity = self.decoders.activity(s_rest.detach())
            realized_activity = self.decoders.activity(s_hat_next)
            realized_delta_s = self.delta_s_module.compute(s_t, s_hat_next)

        # Push the realized cycle's signals into the per-modality and
        # ‖Δs‖ bands. The bands' silence thresholds calibrate to the
        # observed distribution; the rest_reference path gives the
        # per-state silence floor.
        self.activity_bands.observe(realized_activity)
        self.delta_s_band.observe(realized_delta_s)

        # Stasis checks read with the a_rest reference. Captured as
        # snapshot scalars for the M9 diagnostics; the kill plumbing
        # for K-M9-5 lands when the M9 kills land in the pilot-set
        # framework (task #24).
        external_silent_mask = self.activity_bands.external_stasis(
            realized_activity, rest_activity=rest_activity,
        )
        internal_silent_mask = self.delta_s_band.is_silent_per_batch(
            realized_delta_s, rest_delta_s=rest_delta_s,
        )
        external_silent_frac = float(external_silent_mask.float().mean().item())
        internal_silent_frac = float(internal_silent_mask.float().mean().item())

        total_m9 = v_loss + habit_loss + dec_loss + rest_loss
        total_m9.backward()
        self.m9_optimizer.step()

        # ---- M9 stop-grad enforcement ----
        # The rest forward routes through the M8 predictor's params.
        # m9_optimizer doesn't own those params so it doesn't update
        # them, but the backward leaves residual `.grad` on them. Wipe
        # the M8 loss_module grads so the next M8 step starts clean
        # and so this contract is explicit at the code level instead of
        # relying on the next call to optimizer.zero_grad to scrub.
        for p in self.loss_module.parameters():
            if p.grad is not None:
                p.grad = None

        return {
            "v_loss": float(v_loss.detach().item()),
            "habit_loss": float(habit_loss.detach().item()),
            "decoder_loss": float(dec_loss.detach().item()),
            "rest_loss": float(rest_loss.detach().item()),
            "total": float(total_m9.detach().item()),
            "mcts_tree_size": self.mcts.tree_stats()["size"],
            "mcts_root_visits": self.mcts.tree_stats()["root_visits"],
            "r_best": r_best,
            "gamma": float(self.gamma.gamma.detach().item()),
            "external_silent_frac": external_silent_frac,
            "internal_silent_frac": internal_silent_frac,
            "k_m9_5_armed": self.activity_bands.k_m9_5_armed(),
            "rest_delta_s_mean": float(rest_delta_s.detach().mean().item()),
        }

    def _target_positions_for(
        self,
        ctx_len: int,
    ) -> torch.Tensor:
        """Build the predictor's target-position queries for the
        MCTS-internal predict_next() call.

        MCTS rollout is a one-step lookahead from the cycle's state,
        so we query a single target position one step past the
        context: position `ctx_len`. predict_next() mean-pools over
        the target-position axis; a single-position query gives a
        deterministic [B, D] next-state estimate without averaging
        over a multi-step future. (The full-tgt-block convention
        belongs to the M8 loss, where every target position has a
        teacher signal; MCTS has no such teacher per-position.)
        """
        return torch.tensor([[ctx_len]], dtype=torch.long)

    def _cycle_observation_kwargs(self) -> dict:
        """Per-cycle observations for the EFE evaluator's per-candidate
        path. At this slice we provide a minimal "alone" cycle context
        (no counterpart present) so P3 contributes zero -- the real
        loop-side context plumbing (counterpart_present from the
        sensorium, time_since_emission from the action log) lands in
        a follow-up slice. The keys are positional in the EFE
        evaluator's API; the empty dict here means decoders + bands
        still drive P1/P2/P4 but P3 stays at zero cost.
        """
        return {}

    # ------------------------------------------------------------------
    # Checkpoint + resume (extended schema).
    # ------------------------------------------------------------------
    def _checkpoint(self, reason: str) -> None:
        """Extend M8's checkpoint with the M9 state.

        New keys (per spec §5):
          v_head_state_dict, v_target_state_dict
          habit_net_state_dict
          decoder_state_dicts (attention, memory; text rides M8)
          m9_optimizer_state_dict
          gamma (scalar)
          preference_weights (snapshot for forward-compat)

        Persistent MCTS tree is intentionally NOT persisted (cold
        rebuild on resume; self-heals via recency-decay).
        """
        # Run M8's checkpoint first, capturing the path it wrote.
        super()._checkpoint(reason)
        ckpt_dir = self.run_dir / "checkpoints"
        slot_path = sorted(ckpt_dir.glob("ckpt_*.pt"))[-1]

        # Re-load the M8 state, attach the M9 keys, atomic-rewrite.
        existing = torch.load(slot_path, weights_only=False)
        existing["m9_v_head_state_dict"] = self.v_head.state_dict()
        existing["m9_v_target_state_dict"] = self.v_target.state_dict()
        existing["m9_habit_net_state_dict"] = self.habit_net.state_dict()
        existing["m9_decoder_attention_state_dict"] = self.decoders.attention.state_dict()
        existing["m9_decoder_memory_state_dict"] = self.decoders.memory.state_dict()
        existing["m9_decoder_text_intensity_state_dict"] = self.decoders.text.intensity_head.state_dict()
        existing["m9_decoder_text_reencode_state_dict"] = self.decoders.text.reencode_head.state_dict()
        existing["m9_preferences_state_dict"] = self.preferences.state_dict()
        existing["m9_delta_s_state_dict"] = self.delta_s_module.state_dict()
        existing["m9_rest_action_state_dict"] = self.rest_action.state_dict()
        existing["m9_optimizer_state_dict"] = self.m9_optimizer.state_dict()
        existing["m9_gamma"] = float(self.gamma.gamma.detach().item())

        tmp_path = slot_path.with_suffix(".pt.tmp")
        with open(tmp_path, "wb") as f:
            torch.save(existing, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, slot_path)

    def resume(self, ckpt_path: Path) -> None:
        """Extend M8's resume with M9 state restoration.

        Tolerant of missing M9 keys (loading an M8-only checkpoint
        into an M9Trainer leaves the M9 heads at their fresh init).
        """
        super().resume(ckpt_path)
        state = torch.load(ckpt_path, weights_only=False)
        if "m9_v_head_state_dict" in state:
            self.v_head.load_state_dict(state["m9_v_head_state_dict"])
        if "m9_v_target_state_dict" in state:
            self.v_target.load_state_dict(state["m9_v_target_state_dict"])
        if "m9_habit_net_state_dict" in state:
            self.habit_net.load_state_dict(state["m9_habit_net_state_dict"])
        if "m9_decoder_attention_state_dict" in state:
            self.decoders.attention.load_state_dict(state["m9_decoder_attention_state_dict"])
        if "m9_decoder_memory_state_dict" in state:
            self.decoders.memory.load_state_dict(state["m9_decoder_memory_state_dict"])
        if "m9_decoder_text_intensity_state_dict" in state:
            self.decoders.text.intensity_head.load_state_dict(state["m9_decoder_text_intensity_state_dict"])
        if "m9_decoder_text_reencode_state_dict" in state:
            self.decoders.text.reencode_head.load_state_dict(state["m9_decoder_text_reencode_state_dict"])
        if "m9_preferences_state_dict" in state:
            self.preferences.load_state_dict(state["m9_preferences_state_dict"])
        if "m9_delta_s_state_dict" in state:
            self.delta_s_module.load_state_dict(state["m9_delta_s_state_dict"])
        if "m9_rest_action_state_dict" in state:
            self.rest_action.load_state_dict(state["m9_rest_action_state_dict"])
        if "m9_optimizer_state_dict" in state:
            self.m9_optimizer.load_state_dict(state["m9_optimizer_state_dict"])
        if "m9_gamma" in state:
            with torch.no_grad():
                self.gamma.gamma.fill_(float(state["m9_gamma"]))
