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
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer

from luthi.seam_types import ActionSelection, PlanSnapshot
from luthi.v2.m9.s_t import compute_s_t

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
    # Text LM loss weight (spec §1): the M9 path adds a real token-level
    # signal (output_proj over predicted_target -> cross-entropy against
    # ground-truth target tokens) so the text decoder isn't trained only
    # on the cycle-consistency reencode path. output_proj is shared with
    # M8 but JEPALoss doesn't use it, so the M9 LM signal is the only
    # gradient source for output_proj. Text-only at step 1 -- audio and
    # vision get equivalents in step 2.
    text_lm_weight: float = 1.0
    # Tolerance for "best_action == a_rest" detection in the action log
    # (spec §6.iii). When the MCTS-chosen action is within this L2
    # distance of a_rest(s_t), the rest-vs-active flag is set per
    # MCTS visit-distribution concentration. Small constant in action
    # space; calibratable.
    rest_match_tolerance: float = 0.5
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
    # Gamma update warmup. During the first `gamma_warmup_cycles`
    # train_steps, gamma stays at gamma_init and is NOT inferred from
    # candidate EFE spread. Rationale: early-training candidate EFEs
    # are dominated by encoder-init noise, not by real precision
    # structure; inferring gamma against that noise drives it to
    # extremes (sustained-run smoke test showed gamma collapsing to
    # the lower clamp within 70 cycles on random-token data). With
    # warmup, gamma waits for the encoder to stabilize before reading
    # the landscape. The K-M9-4 kill therefore also stays healthy
    # during warmup (it observes gamma=gamma_init, never at clamp).
    gamma_warmup_cycles: int = 100
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
            + list(self.delta_s_module.parameters()) \
            + list(loss_module.online_encoder.output_proj.parameters())
        # NB: output_proj is *also* in M8's optimizer (it's part of
        # loss_module.parameters()) but JEPALoss doesn't exercise it,
        # so M8 optimizer.step() is a no-op for output_proj. The M9
        # text-LM loss is the only gradient source; m9_optimizer is
        # the only thing that actually moves output_proj. After the
        # M9 step, the stop-grad scrub wipes loss_module grads so
        # the next M8 zero_grad starts clean.
        self.m9_optimizer = Adam(m9_params, lr=self.m9_config.head_lr)

        # ---- Step-1 training-story metrics ----
        # Last-cycle realized transition; used by V-TD for the bootstrap.
        # Filled in by `train_step` after the M8 forward.
        self._last_state: Optional[torch.Tensor] = None
        self._last_action: Optional[torch.Tensor] = None
        self._last_reward: Optional[float] = None

        # ---- External-actor seam (2026-06-15 integration plan, Phase 2) ----
        # When the Sanctuary cycle drives this trainer via select_action /
        # observe_transition, it stashes its per-cycle ctx (counterpart_present,
        # time_since_emission, ...) here so _cycle_observation_kwargs can
        # surface the keys to the EFE evaluator. None during corpus-only
        # train_step() runs -> _cycle_observation_kwargs returns {} the way
        # it did before this seam existed.
        self._current_cycle_ctx: Optional[dict] = None
        # One-shot warning gate for the F4-residual snapshot-threading
        # regression check (4.8 round-2 review, 2026-06-15). See
        # observe_transition for the firing condition.
        self._observe_no_snapshot_warned: bool = False

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
        # ---- Phase 1: M8 core update (with ‖Δθ‖ measurement) ----
        # Snapshot M8 params pre-update so we can compute the
        # parameter-space delta the M8 step produces. This drives the
        # staleness machinery's drift band -- the manager spike-tests
        # `‖Δθ‖` against a running median+MAD to detect plasticity
        # spikes (plan §4.v). Per-step clone is O(|params|) memory;
        # acceptable at step-1 scale, revisit if we move to large
        # models (gradient-norm proxy or subset sampling).
        param_snapshot = [
            p.detach().clone()
            for p in self.loss_module.parameters()
            if p.requires_grad
        ]
        m8_result = super().train_step(modality, batch)
        delta_theta_norm = self._compute_delta_theta_norm(param_snapshot)
        # Cache for the action-log writer in _m9_head_step.
        self._last_delta_theta_norm = delta_theta_norm
        # observe_drift increments theta_version (cycle counter), pushes
        # into the drift band, and runs the F3 C1 event-driven recovery
        # detector. The runner is now the manager's per-cycle driver.
        self.staleness.observe_drift(delta_theta_norm)

        # ---- Phase 2: M9 head update on detached latents ----
        m9_losses = self._m9_head_step(modality, batch, m8_result["raw"])
        m8_result["m9"] = m9_losses
        m9_losses["delta_theta_norm"] = delta_theta_norm
        m9_losses["theta_version"] = self.staleness.theta_version

        # ---- Polyak update of V_target ----
        # Done outside the gradient step; just an exponential moving
        # average toward V's parameters. K-M9-3 backstops divergence.
        with torch.no_grad():
            tau = self.m9_config.v_target_polyak
            for p, p_t in zip(self.v_head.parameters(), self.v_target.parameters()):
                p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

        return m8_result

    def _m9_head_step(
        self,
        modality: str,
        batch: dict,
        raw: dict,
    ) -> dict:
        """Train V, habit, and decoders on **detached** latents using
        live MCTS planning as the visit-distill + reward source.

        Per spec §1 stop-grad discipline: encoder/predictor latents
        are .detach()'d at every M9 head input; the M9 gradient
        cannot flow back into the JEPA representation while the
        head training is unverified.
        """
        # Pool the encoder context to a [B, D] state vector via the
        # canonical helper (Phase 4a, 2026-06-16): same s_t definition
        # the inference seam uses, so the V-head sees one distribution
        # across training and Sanctuary-driven inference. Detach is
        # inside compute_s_t -- stop-grad discipline preserved.
        s_t = compute_s_t(raw["online_context_latents"])           # [B, D]
        s_hat_next = compute_s_t(raw["predicted_target"])          # [B, D]
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
            # Warmup gate: don't infer gamma from candidate EFE spread
            # while the encoder is still in noise-dominated init. See
            # M9Config.gamma_warmup_cycles docstring for the rationale.
            if (
                child_g.numel() >= 2
                and self.global_step >= self.m9_config.gamma_warmup_cycles
            ):
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

        # ---- MI probe (K-M9-6 baseline; step 1 observes only) ----
        # Pair pooled context with first target-block position as the
        # canonical (trunk, target). Tiny-batch runs return 0.0 (the
        # estimator's n>=4 guard); production batches are large enough
        # to produce a real signal. The probe writes into its own
        # running band -- step 1 sets the band, step 2's K-M9-6 reads
        # it as the guard-then-kill trigger.
        target_latents = raw["target_latents"]
        ctx_len = raw["ctx_len"]
        trunk_pooled = raw["online_context_latents"].mean(dim=1)  # [B, D]
        if target_latents.shape[1] > ctx_len:
            target_first = target_latents[:, ctx_len, :]          # [B, D]
        else:
            target_first = target_latents.mean(dim=1)
        mi_signal = self.mi_probe.observe(
            trunk_pooled.detach(), target_first.detach()
        )

        # ---- Rest-selected vs rest-defaulted (spec §6.iii) ----
        # Compare the MCTS-chosen best action against a_rest at the
        # cycle's state. If they match and the visit distribution is
        # concentrated (top_share > 0.5), rest was *selected* over
        # alternatives. If they match but the planner had no clear
        # winner, rest was *defaulted to*. Structural distinction the
        # spec asks the action log to expose so we can tell "deciding
        # to be silent" from "no intent."
        best_action = self.mcts.best_action()
        if best_action is not None:
            a_rest_root = a_rest_t[0].detach()
            rest_match = (
                (best_action - a_rest_root).norm().item()
                < self.m9_config.rest_match_tolerance
            )
        else:
            rest_match = False
        top_share = self.mcts.tree_stats().get("top_share", 0.0)
        rest_selected = bool(rest_match and top_share > 0.5)
        rest_defaulted = bool(rest_match and not rest_selected)

        # ---- Text LM loss (spec §1) ----
        # Decode predicted-target latents to vocab logits via output_proj
        # and cross-entropy against the ground-truth target tokens. Text
        # only at step 1; other modalities skip this term (their decoders
        # train via cycle-consistency only until step 2 wires equivalents).
        if modality == "text" and "text_tokens" in batch:
            tgt_tokens = batch["text_tokens"][:, ctx_len:].to(s_t.device)
            # predicted_target detached so encoder/predictor stop-grad
            # holds; output_proj gets gradient from this loss path
            # (output_proj is in m9_optimizer, see __init__ comment).
            pred_latents = raw["predicted_target"].detach()
            logits = self.loss_module.online_encoder.output_proj(pred_latents)
            text_lm_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tgt_tokens.reshape(-1),
            ) * self.m9_config.text_lm_weight
        else:
            text_lm_loss = torch.zeros((), device=s_t.device)

        total_m9 = v_loss + habit_loss + dec_loss + rest_loss + text_lm_loss
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

        # ---- M9 kill observations ----
        # Each kill observe_* is called once per cycle with the right
        # signal. The KillRegistry maintains its own running bands
        # internally (same DriftBand machinery the spec calls "pilot-
        # set"); the per-kill state machines (HEALTHY/FLAGGED/FIRED)
        # advance per the §7 thresholds.
        with torch.no_grad():
            v_mean = float(v_pred.detach().mean().item())
        gamma_value = float(self.gamma.gamma.detach().item())
        # K-M9-2 MCTS entropy. visit_distribution may be empty if no
        # children were expanded; observe_mcts_entropy handles that.
        visit_dist = self.mcts.visit_distribution()
        if visit_dist.numel() > 0:
            self.m9_kills.observe_mcts_entropy(visit_dist)
        # K-M9-3 V divergence (|V| ceiling backstops the bootstrap).
        self.m9_kills.observe_value(v_mean)
        # K-M9-4 gamma divergence (clamp-proximity + band).
        self.m9_kills.observe_gamma(gamma_value)
        # K-M9-5 dark-room / catatonia. Use the all-batch reduction:
        # silent iff EVERY batch element is silent (conservative; a
        # single-entity cycle has only one realized state, and we
        # don't want partial-batch silence to drive the kill).
        internal_silent_all = bool(internal_silent_mask.all().item())
        external_silent_all = bool(external_silent_mask.all().item())
        self.m9_kills.observe_darkroom_v2(
            internal_silent=internal_silent_all,
            external_silent=external_silent_all,
            is_armed=self.activity_bands.k_m9_5_armed(),
        )
        # K-M9-7 staleness runaway. The failover state machine
        # advances after sustained failover; at this slice failover
        # itself is gated by the consistency-history breach check,
        # which fires only once the staleness manager has accumulated
        # enough drift-vs-cached deviations to trip.
        self.staleness.update_failover_state()
        self.m9_kills.observe_staleness(in_failover=self.staleness.in_failover())

        kill_states = {k: v.value for k, v in self.m9_kills.states().items()}

        m9_out = {
            "v_loss": float(v_loss.detach().item()),
            "habit_loss": float(habit_loss.detach().item()),
            "decoder_loss": float(dec_loss.detach().item()),
            "rest_loss": float(rest_loss.detach().item()),
            "text_lm_loss": float(text_lm_loss.detach().item()),
            "total": float(total_m9.detach().item()),
            "mcts_tree_size": self.mcts.tree_stats()["size"],
            "mcts_root_visits": self.mcts.tree_stats()["root_visits"],
            "mcts_top_share": top_share,
            "r_best": r_best,
            "gamma": gamma_value,
            "external_silent_frac": external_silent_frac,
            "internal_silent_frac": internal_silent_frac,
            "k_m9_5_armed": self.activity_bands.k_m9_5_armed(),
            "rest_delta_s_mean": float(rest_delta_s.detach().mean().item()),
            "mi_signal": mi_signal,
            "rest_selected": rest_selected,
            "rest_defaulted": rest_defaulted,
            "kill_states": kill_states,
        }

        # ---- ActionLog write (spec §11.ii) ----
        # One JSONL record per cycle. Fields cover the decision context
        # so 4.7 can step through at debug time and reconstruct what the
        # entity decided and why. Tensors are converted to scalars/lists
        # by the log's _to_jsonable helper.
        with torch.no_grad():
            v_s_mean = float(self.v_head(s_t).mean().item())
        self.action_log.write({
            "cycle": self.global_step,
            "modality": modality,
            "sim_counter": self.mcts.sim_counter,
            "theta_version": self.staleness.theta_version,
            "delta_theta_norm": float(getattr(self, "_last_delta_theta_norm", 0.0)),
            "s_t_summary": {
                "mean": float(s_t.mean().item()),
                "norm": float(s_t.norm().item()),
            },
            "best_action_summary": (
                {
                    "norm": float(best_action.norm().item()),
                    "dist_to_a_rest": (
                        float((best_action - a_rest_t[0].detach()).norm().item())
                    ),
                }
                if best_action is not None
                else None
            ),
            "gamma": float(self.gamma.gamma.detach().item()),
            "v_s": v_s_mean,
            "r_best": r_best,
            "tree_stats": self.mcts.tree_stats(),
            "mi_probe": self.mi_probe.snapshot(),
            "rest_selected": rest_selected,
            "rest_defaulted": rest_defaulted,
            "external_silent_frac": external_silent_frac,
            "internal_silent_frac": internal_silent_frac,
            "k_m9_5_armed": self.activity_bands.k_m9_5_armed(),
            "staleness": self.staleness.snapshot(),
            "activity_bands": self.activity_bands.snapshot(),
            "delta_s_band": self.delta_s_band.snapshot(),
            "kill_states": kill_states,
        })

        return m9_out

    def _compute_delta_theta_norm(
        self,
        param_snapshot: list,
    ) -> float:
        """L2 norm of `theta_after - theta_before` across all M8
        trainable parameters. Drives the staleness manager's drift
        band each cycle.

        Returns float(0.0) for the degenerate empty-params case.
        """
        delta_sq = 0.0
        current_params = [
            p for p in self.loss_module.parameters() if p.requires_grad
        ]
        for before, after in zip(param_snapshot, current_params):
            delta_sq += float((after.detach() - before).pow(2).sum().item())
        return math.sqrt(delta_sq)

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
        path.

        Reads from ``self._current_cycle_ctx`` when the Sanctuary cycle
        has installed one (via select_action / observe_transition);
        returns {} when the trainer is running corpus-only train_step(),
        preserving the inert P3 behavior the corpus path expects.

        Scalar P3 wiring keys ``counterpart_present`` /
        ``time_since_emission`` are promoted to ``[1]`` tensors here so
        the EFE evaluator API receives the shape it documents (the keys
        are tensor-positional). 2026-06-15 seam plan Phase 2.
        """
        ctx = self._current_cycle_ctx
        if not ctx:
            return {}
        out: dict = {}
        if "counterpart_present" in ctx:
            cp = ctx["counterpart_present"]
            if not isinstance(cp, torch.Tensor):
                cp = torch.tensor([float(cp)], dtype=torch.float32)
            out["counterpart_present"] = cp
        if "time_since_emission" in ctx:
            ts = ctx["time_since_emission"]
            if not isinstance(ts, torch.Tensor):
                ts = torch.tensor([float(ts)], dtype=torch.float32)
            out["time_since_emission"] = ts
        return out

    # ------------------------------------------------------------------
    # External-actor seam: M9Actor + TransitionSink Protocol satisfaction.
    # See 2026-06-15 sanctuary-training-seam-integration-plan.md Phase 2.
    # ------------------------------------------------------------------
    def select_action(
        self,
        s_t: torch.Tensor,
        *,
        context_latents: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
        cycle_ctx: Optional[dict] = None,
        budget: Optional[int] = None,
    ) -> ActionSelection:
        """Run the M9 act path (habit-net + plan-budget MCTS) for state s_t.

        Satisfies :class:`luthi.sanctuary_interface.M9Actor`. The
        Sanctuary cycle calls this once per cycle to obtain the action
        the substrate selects; the returned :class:`ActionSelection`
        carries the chosen action tensor, a readable summary, and the
        per-component EFE decomposition for diagnostics.

        Args:
            s_t: ``[D]`` or ``[B, D]`` encoder state. MCTS is single-state
                per spec ?3 -- batched s_t plans against ``s_t[0]``.
            context_latents: Unpooled encoder output ``[B, T, D]`` (or
                ``[T, D]``) for the EFE predictor's lookahead inside MCTS.
                Sanctuary obtains this from ``encode_state(..., pool=False)``.
            target_positions: Optional ``[1, num_targets]`` positions for the
                predictor's target queries. Defaults to a single position one
                step past the context (the M9 head-step convention).
            cycle_ctx: Optional per-cycle observation dict (counterpart_present,
                time_since_emission). Installed for the duration of the plan.
            budget: Override the MCTS budget for this call; defaults to
                ``m9_config.mcts_budget_per_cycle``.
        """
        device = s_t.device
        s_root = s_t[0] if s_t.dim() == 2 else s_t
        if context_latents.dim() == 2:
            ctx_single = context_latents.unsqueeze(0)
        else:
            ctx_single = context_latents[0:1]
        if target_positions is None:
            ctx_len = ctx_single.shape[1]
            target_positions = self._target_positions_for(ctx_len).to(device)

        plan_budget = (
            budget if budget is not None else self.m9_config.mcts_budget_per_cycle
        )

        # Install per-cycle ctx for both plan-time and breakdown-time
        # observation kwarg derivation. Restored on exit so corpus-driven
        # train_step() calls continue to see ``_current_cycle_ctx = None``.
        prev_ctx = self._current_cycle_ctx
        if cycle_ctx is not None:
            self._current_cycle_ctx = cycle_ctx
        try:
            self.mcts.reset(s_root, ctx_single, target_positions)
            self.mcts.current_theta_version = self.staleness.theta_version
            self.mcts.plan_budget(
                budget=plan_budget,
                observation_kwargs=self._cycle_observation_kwargs(),
            )

            best_action = self.mcts.best_action()
            tree_stats = self.mcts.tree_stats()

            if best_action is None:
                # Degenerate budget=0 (or expansion failure). Fall back to a
                # habit-net sample so the cycle still gets a defined action;
                # the EFE breakdown stays empty -- there is no scored child
                # to decompose. The PlanSnapshot is built empty so an
                # observe_transition that consumes it falls cleanly into
                # the no-plan path.
                with torch.no_grad():
                    sample = self.habit_net.sample(
                        s_root.unsqueeze(0), K=1,
                    )
                    best_action = sample["candidates"][0, 0].detach()
                summary = "action(habit-fallback, top_share=0.000)"
                empty_snapshot = PlanSnapshot(
                    visit_distribution=torch.zeros(0),
                    candidate_actions=torch.zeros(0),
                    r_best=0.0,
                )
                return ActionSelection(
                    action=best_action,
                    readable_summary=summary,
                    efe_breakdown={},
                    plan_snapshot=empty_snapshot,
                )

            # Decompose the chosen action's EFE per-component. The MCTS
            # children store only the scalar G; re-call compute_g on the
            # best action to surface the c_eng / c_coh / c_con / c_truth
            # breakdown the diagnostics callers want.
            obs_kwargs = self._cycle_observation_kwargs()
            with torch.no_grad():
                g_out = self.efe.compute_g(
                    s_t=s_root.unsqueeze(0),
                    a_t=best_action.unsqueeze(0),
                    context_latents=ctx_single,
                    target_positions=target_positions,
                    **obs_kwargs,
                )

            with torch.no_grad():
                a_rest = self.rest_action(s_root.unsqueeze(0))[0].detach()
            dist_to_rest = float((best_action - a_rest).norm().item())
            top_share = float(tree_stats.get("top_share", 0.0))
            summary = (
                f"action(norm={best_action.norm().item():.3f}, "
                f"dist_to_rest={dist_to_rest:.3f}, "
                f"top_share={top_share:.3f})"
            )
            efe_breakdown = {
                "total": float(g_out["G"].item()),
                "engagement_cost": float(g_out["c_eng"].item()),
                "coherence_cost": float(g_out["c_coh"].item()),
                "connection_cost": float(g_out["c_con"].item()),
                "truthfulness_cost": float(g_out["c_truth"].item()),
            }

            # F4 fix (4.8 review 2026-06-15): freeze the plan now, so
            # the transition that consumes it is self-describing.
            # observe_transition reads from this snapshot via
            # ctx["plan_snapshot"], not from live self.mcts.root.
            children = self.mcts.root.children
            visit_dist = self.mcts.visit_distribution().detach()
            candidate_actions = torch.stack(
                [c.action_in for c in children], dim=0,
            ).detach()
            child_g_list = [
                c.incoming_g for c in children
                if c.incoming_g is not None
            ]
            r_best = float(-min(child_g_list)) if child_g_list else 0.0
            plan_snapshot = PlanSnapshot(
                visit_distribution=visit_dist,
                candidate_actions=candidate_actions,
                r_best=r_best,
            )

            return ActionSelection(
                action=best_action,
                readable_summary=summary,
                efe_breakdown=efe_breakdown,
                plan_snapshot=plan_snapshot,
            )
        finally:
            self._current_cycle_ctx = prev_ctx

    def observe_transition(
        self,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
        s_next: torch.Tensor,
        ctx: dict[str, Any],
    ) -> dict[str, float]:
        """Run an M9 head update against a realized ``(s_t, a_t, s_next)``.

        Satisfies :class:`luthi.sanctuary_interface.TransitionSink`. The
        Sanctuary cycle calls this once per realized cycle so V/habit
        update on lived consequence rather than purely predicted EFE.

        Scope at Phase 2: V-TD bootstrap on ``s_next`` plus habit-distill
        from the most recent MCTS tree, both run on the separate M9
        optimizer. The richer decoder cycle-consistency / text-LM / rest
        terms continue to train through the corpus ``train_step()`` path;
        wiring them onto external transitions requires ``ctx`` to carry
        target_latents / text_tokens, which the seam plan flags as a
        Phase 3 extension once Sanctuary actually supplies them.

        Args:
            s_t, a_t, s_next: ``[D]`` or ``[B, D]`` tensors -- the realized
                transition. ``a_t`` is typically the ``ActionSelection.action``
                returned by :meth:`select_action`; ``s_next`` is
                ``encode_state`` applied to the post-action observation.
            ctx: Per-cycle observation dict. Recognized keys include
                ``counterpart_present``, ``time_since_emission`` (consumed
                by the EFE evaluator's per-candidate path during the next
                select_action call), plus arbitrary caller-supplied entries.

        Returns:
            Metrics dict: ``v_loss``, ``habit_loss``, ``r_best`` (the
            **predicted** M9 reward = ``-min(child G)`` taken from the
            previous select_action's MCTS tree -- planner estimate,
            **not** the realized consequence of ``s_next``; realized-
            reward grounding is Phase 4 work per the 2026-06-15
            integration plan ?3), ``theta_version`` (M8 substrate
            version, unchanged by this M9-only update).

        Phase-3 grounding posture: ``s_next`` enters learning *only*
        through ``v_target(s_next)`` in the TD bootstrap -- the value
        head now sees realized next-states, but the immediate reward
        signal it bootstraps against is still planner-estimated. Phase 4
        substitutes a reward derived from ``s_next`` itself (e.g.
        preference-cost evaluated on the realized transition).
        """
        # Reshape singletons to [B, D] for uniform downstream handling.
        if s_t.dim() == 1:
            s_t = s_t.unsqueeze(0)
        if s_next.dim() == 1:
            s_next = s_next.unsqueeze(0)
        if a_t.dim() == 1:
            a_t = a_t.unsqueeze(0)

        s_t = s_t.detach()
        s_next = s_next.detach()
        a_t = a_t.detach()

        prev_ctx = self._current_cycle_ctx
        self._current_cycle_ctx = ctx
        try:
            self.m9_optimizer.zero_grad(set_to_none=True)

            # F4 fix (4.8 review 2026-06-15): the planning state that
            # produced this transition's a_t arrives via the snapshot in
            # ctx, NOT via live self.mcts.root. The live tree is whatever
            # the most recent select_action left -- which, the moment
            # the async actor/learner split lands, is the *next* cycle's
            # plan rather than the plan that this transition realizes.
            # The transition is self-describing.
            snapshot: PlanSnapshot | None = ctx.get("plan_snapshot")

            # F4-residual (4.8 round-2 review 2026-06-15): warn once if
            # the caller forgot to thread the snapshot but the trainer
            # has a non-degenerate plan live. The no-snapshot path
            # degrades silently to r=0 + random habit target; that's the
            # intended degenerate behavior, but it's also the signature
            # of a future threading regression. Cheap insurance: catch
            # the regression at the first cycle it bites.
            if (
                snapshot is None
                and not self._observe_no_snapshot_warned
                and self.mcts.root is not None
                and self.mcts.root.children
            ):
                logger.warning(
                    "observe_transition called with no plan_snapshot in "
                    "ctx, but the trainer has a non-degenerate MCTS plan "
                    "(%d children). Falling back to r=0 + random habit "
                    "target; this is likely a threading regression in the "
                    "actor->sink path. Logged once per trainer.",
                    len(self.mcts.root.children),
                )
                self._observe_no_snapshot_warned = True

            # ---- M9 reward from the captured plan ----
            # Still planner-estimated; realized-reward grounding is
            # Phase 4 work per the integration plan ?3 (sequenced after
            # F7b s_t-convention alignment per 4.8's 2026-06-15 callback).
            r_best = snapshot.r_best if snapshot is not None else 0.0

            r_t = torch.full(
                (s_t.shape[0],), r_best,
                device=s_t.device, dtype=s_t.dtype,
            )

            # ---- V-TD bootstrap on the realized s_next ----
            with torch.no_grad():
                v_target_next = self.v_target(s_next)
                td_target = r_t + self.m9_config.discount * v_target_next
            v_pred = self.v_head(s_t)
            v_loss = (v_pred - td_target).pow(2).mean()

            # ---- Habit visit-distill against the captured plan ----
            if (
                snapshot is not None
                and snapshot.visit_distribution.numel() > 0
            ):
                actions_k = snapshot.candidate_actions.to(
                    device=s_t.device, dtype=s_t.dtype,
                ).unsqueeze(0)  # [1, K, D]
                visit_target = snapshot.visit_distribution.to(
                    device=s_t.device, dtype=s_t.dtype,
                )
                # log_prob expects [B, ...]; route via batch element 0
                # because MCTS is single-state per spec ?3.
                log_p_habit = self.habit_net.log_prob(
                    s_t[0:1], actions_k,
                )
                habit_loss = -(visit_target * log_p_habit[0]).sum()
            else:
                # No plan supplied (caller didn't run select_action, or
                # the plan was degenerate budget=0). Fall back to a
                # sample-based negative log-likelihood so the path stays
                # well-defined.
                sample = self.habit_net.sample(
                    s_t, K=self.m9_config.habit_n_candidates,
                )
                habit_loss = -sample["log_prob"].mean()

            total_m9 = v_loss + habit_loss
            total_m9.backward()
            self.m9_optimizer.step()

            # ---- Polyak update of V_target (same as corpus path) ----
            with torch.no_grad():
                tau = self.m9_config.v_target_polyak
                for p, p_t in zip(
                    self.v_head.parameters(),
                    self.v_target.parameters(),
                ):
                    p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

            # ---- Stop-grad scrub on M8 params ----
            # Mirrors the corpus _m9_head_step's scrub so any backward()
            # that touches loss_module params (e.g. the rest forward
            # routes through the M8 predictor in the corpus path; this
            # trimmed path doesn't, but the contract is preserved).
            for p in self.loss_module.parameters():
                if p.grad is not None:
                    p.grad = None

            return {
                "v_loss": float(v_loss.item()),
                "habit_loss": float(habit_loss.item()),
                "r_best": r_best,
                # M8 substrate version is unchanged by an M9-only update --
                # observe_drift is deliberately NOT called here; that
                # belongs to the corpus train_step() path where the M8
                # forward actually shifts the predictor.
                "theta_version": self.staleness.theta_version,
            }
        finally:
            self._current_cycle_ctx = prev_ctx

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
