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
from luthi.v2.m9.preferences import Preferences
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
            )
        )
        self.mi_probe = MIProbe(
            ridge_lambda=self.m9_config.mi_probe_ridge_lambda
        )
        self.action_log = ActionLog(
            self.run_dir / self.m9_config.action_log_filename
        )

        # ---- M9 optimizer ----
        # Separate Adam over the M9 heads. V_target is not trained
        # directly. Text decoder's wrapped `output_proj` is NOT included
        # here -- it continues to train via the M8 core (low-LR /
        # frozen behavior per spec; the wrapping just preserves the
        # interface, not the param ownership).
        m9_params = list(self.v_head.parameters()) \
            + list(self.habit_net.parameters()) \
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
        """Train V, habit, and decoders on **detached** latents.

        At this slice we wire the gradient paths and stop-grad
        discipline; the planning-time MCTS visit targets and the
        realized-action negative-EFE rewards land in follow-up
        slices when the cycle loop drives the trainer. For now the
        head training uses zero-reward / identity habit / cycle-
        consistency-only signals so the integration plumbing is
        exercised end-to-end without changing M8 behaviour.
        """
        # Pool the encoder context to a [B, D] state vector (mean over
        # context positions, mirroring the EFEEvaluator step-1
        # convention). All tensors are DETACHED here -- the M9 head
        # gradients must not flow back into the encoder or predictor.
        s_t = raw["online_context_latents"].detach().mean(dim=1)  # [B, D]
        # Predicted next-state from the M8 predictor's target-block
        # output, mean-pooled. The realized action at step 1 is the
        # zero action_token (M8 stub); MCTS-driven actions land next
        # slice.
        s_hat_next = raw["predicted_target"].detach().mean(dim=1)  # [B, D]

        self.m9_optimizer.zero_grad(set_to_none=True)

        # --- V TD: r_t + gamma * V_target(s_{t+1}) ---
        # At step 1 the reward signal is the negative EFE of the
        # cycle's realized action; until MCTS is wired we use the
        # placeholder r_t = 0 so V trains toward a stable fixed point
        # rather than chasing an unbounded signal. K-M9-3 absolute
        # ceiling backstops divergence either way.
        r_t = torch.zeros(s_t.shape[0], device=s_t.device)
        with torch.no_grad():
            v_target_next = self.v_target(s_hat_next)
            td_target = r_t + self.m9_config.discount * v_target_next
        v_pred = self.v_head(s_t)
        v_loss = (v_pred - td_target).pow(2).mean()

        # --- Habit distill: cross-entropy of habit prior over MCTS
        # visit distribution. Until MCTS is wired, use a placeholder
        # log_prob loss = - entropy(habit_dist) so the head gets
        # gradient but does not chase a degenerate target. ---
        sample = self.habit_net.sample(s_t, K=self.m9_config.habit_n_candidates)
        # Use mean negative log-prob as a placeholder; real habit
        # distillation uses MCTS visit-weighted MLE.
        habit_loss = -sample["log_prob"].mean()

        # --- Decoder cycle-consistency: ‖a_t - encode(decode(a_t))‖
        # per modality. At step 1 we use s_hat_next as the candidate
        # action (the realized action under action-space (c) before
        # MCTS lands proper a_t selection). ---
        outs = self.decoders.decode_all(s_hat_next)
        reencoded = self.decoders.re_encode_all(outs)
        dec_loss = sum(
            (s_hat_next - r).pow(2).mean()
            for r in reencoded.values()
        ) / max(1, len(reencoded))
        dec_loss = dec_loss * self.m9_config.decoder_cycle_consistency_weight

        total_m9 = v_loss + habit_loss + dec_loss
        total_m9.backward()
        self.m9_optimizer.step()

        return {
            "v_loss": float(v_loss.detach().item()),
            "habit_loss": float(habit_loss.detach().item()),
            "decoder_loss": float(dec_loss.detach().item()),
            "total": float(total_m9.detach().item()),
        }

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
        if "m9_optimizer_state_dict" in state:
            self.m9_optimizer.load_state_dict(state["m9_optimizer_state_dict"])
        if "m9_gamma" in state:
            with torch.no_grad():
                self.gamma.gamma.fill_(float(state["m9_gamma"]))
