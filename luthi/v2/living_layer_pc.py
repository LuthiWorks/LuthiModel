"""PredictiveCodingLayer: a living layer with PC error-driven updates.

The v2 substrate replaces v1's Hebbian self-modification with hierarchical
predictive coding (Whittington-Bogacz variant). The forward pass IS still
learning, but updates are driven by prediction error rather than
input-output correlation.

Two-tier memory: fast episode store (retrieval, identical to v1) plus
consolidation into PC weights (M4 work — see consolidation.py).

See `docs/V2_IMPLEMENTATION_PLAN.md` for the architectural specification
and `docs/LUTHI_V2_PREDICTIVE_CODING_BRIEF.md` for the design rationale.
"""

import torch
import torch.nn as nn


class PredictiveCodingLayer(nn.Module):
    """A linear layer whose weights self-modify via predictive coding.

    Each forward pass:
      1. Episodic recall: optionally blend a context-matched weight
         snapshot into the active weight.
      2. Linear computation: output = input @ weight_snapshot.T.
      3. PC self-modification: update weight, prediction, set_point,
         momentum, update_ema, precision, error_acc — all locally,
         no gradient flow.
      4. Episode storage if salience exceeds threshold.

    Living-state buffers are registered as buffers, not parameters. The
    optimizer never touches them. The layer trains itself.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pc_rate: float = 0.001,
        pred_learning_rate: float = 0.0001,
        homeostatic_decay: float = 0.001,
        set_point_adapt_rate: float = 1e-6,
        momentum_decay: float = 0.99,
        update_ema_decay: float = 0.99,
        precision_ema_decay: float = 0.999,
        precision_min: float = 0.1,
        precision_max: float = 10.0,
        prediction_clamp: float = 10.0,
        salience_threshold: float = 0.1,
        num_episodes: int = 32,
        context_dim: int = 64,
        episode_blend: float = 0.1,
        compressed_episodes: bool = False,
        consolidation_enabled: bool = False,
        consolidation_window: int = 1000,
        consolidation_trigger_window: int = 100,
        consolidation_threshold_factor: float = 0.5,
        consolidation_rate_factor: float = 0.1,
        consolidation_style: str = "gradient",
        consolidation_attractor_passes: int = 1,
        sparse_threshold: float = 0.0,
        sparse_warmup_steps: int = 500,
        inference_steps_per_forward: int = 1,
        buffer_dtypes: dict[str, torch.dtype] | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pc_rate = pc_rate
        self.pred_learning_rate = pred_learning_rate
        self.homeostatic_decay = homeostatic_decay
        self.set_point_adapt_rate = set_point_adapt_rate
        self.momentum_decay = momentum_decay
        self.update_ema_decay = update_ema_decay
        self.precision_ema_decay = precision_ema_decay
        self.precision_min = precision_min
        self.precision_max = precision_max
        self.prediction_clamp = prediction_clamp
        self.salience_threshold = salience_threshold
        self.num_episodes = num_episodes
        self.context_dim = context_dim
        self.episode_blend = episode_blend
        self.consolidation_enabled = consolidation_enabled
        self.consolidation_rate_factor = consolidation_rate_factor
        # Salvatori-style attractor consolidation (2026-05-14). "gradient"
        # is the original M4 pathway (pull weight toward stored snapshot).
        # "attractor" is the Salvatori 2023 pathway (replay stored input
        # pattern through pc_self_modify so the pattern becomes a local
        # energy minimum). "both" runs gradient first, then attractor —
        # gradient places the weight near the stored regime, then
        # attractor reinforces the input as a fixed point of that regime.
        # No silent fallback: invalid value raises here, not at forward
        # time. Future instances reading this: do not add a default-case
        # else clause that picks a style — the choice should always be
        # explicit.
        valid_styles = ("gradient", "attractor", "both")
        if consolidation_style not in valid_styles:
            raise ValueError(
                f"consolidation_style must be one of {valid_styles}, "
                f"got {consolidation_style!r}"
            )
        self.consolidation_style = consolidation_style
        if consolidation_attractor_passes < 1:
            raise ValueError(
                f"consolidation_attractor_passes must be >= 1, got "
                f"{consolidation_attractor_passes}"
            )
        self.consolidation_attractor_passes = int(
            consolidation_attractor_passes
        )

        # Sparse PC gating (lit-followup 2026-05-13). When sparse_threshold
        # > 0, output rows with error_acc[j] below the threshold skip the
        # weight update for that row. Analog of v1's spiking gate in
        # continuous error space. Bandwidth saving is proportional to the
        # fraction of outputs gated off. The warmup window prevents the
        # bootstrap-deadlock (error_acc starts at 0; if we gated
        # immediately, nothing would ever learn and error_acc would never
        # grow). After warmup_steps the gate engages.
        self.sparse_threshold = float(sparse_threshold)
        self.sparse_warmup_steps = int(sparse_warmup_steps)
        # Step counter as a non-persistent attribute (not a buffer — we
        # don't checkpoint it; recovery from checkpoint resets warmup).
        self._sparse_step_count: int = 0

        # iPC interleaved inference + update (Salvatori et al. 2022,
        # arXiv:2212.00720). With inference_steps_per_forward=T>1, the
        # forward repeats the matmul + self-mod cycle T times within
        # one external forward call. Classical PC = T=1 (current
        # default, behavior unchanged). Reported in the iPC paper to
        # consistently improve convergence over the "infer to
        # completion, then update" classical schedule.
        # Constraint: iPC mode is incompatible with gradient checkpointing
        # because the weight evolves T times within the forward; the
        # cached snapshot for recompute would not reproduce the original
        # trajectory. Forward raises if recomputing and T > 1.
        if inference_steps_per_forward < 1:
            raise ValueError(
                f"inference_steps_per_forward must be >= 1, got "
                f"{inference_steps_per_forward}"
            )
        self.inference_steps_per_forward = int(inference_steps_per_forward)

        if consolidation_enabled:
            from luthi.v2.consolidation import ConsolidationTracker
            self._consolidation_tracker = ConsolidationTracker(
                window_size=consolidation_window,
                trigger_window=consolidation_trigger_window,
                threshold_factor=consolidation_threshold_factor,
            )
        else:
            self._consolidation_tracker = None
        self._consolidation_fire_count: int = 0

        self._buffer_dtype_overrides: dict[str, torch.dtype] = dict(
            buffer_dtypes or {}
        )

        # Compressed episode storage: episode_values stored as INT8 with a
        # per-episode FP32 scale. Reduces episode-store memory by 4x vs FP32.
        # The 2026-05-10 audit flagged that at 4096d × 36 blocks × 64
        # episodes, FP32 episode_values is ~150 GB — infeasible. INT8 brings
        # that to ~36 GB, still large but tractable. Production scale will
        # additionally need low-rank delta compression (rank-r SVD of
        # (snapshot - baseline)) — left as a Phase 5 follow-up since M5
        # pilot scale (≤256d) fits comfortably without it. The INT8 code
        # path is inherited from v1's Ablation C design, already in
        # `_store_episode` and `_recall_episode`.
        if compressed_episodes:
            self._buffer_dtype_overrides.setdefault(
                "episode_values", torch.int8
            )

        # --- Core weight state (all buffers, not parameters) ---

        weight = torch.empty(
            out_features, in_features, dtype=self._buf_dtype("weight")
        )
        nn.init.kaiming_uniform_(weight)
        self.register_buffer("weight", weight)

        # Top-down prediction matrix. Initialized to zero so that
        # output @ prediction starts at zero, making the initial pred_error
        # equal to the actual input mean — a clean starting signal that the
        # PC update can immediately drive toward zero.
        self.register_buffer(
            "prediction",
            torch.zeros(
                out_features, in_features,
                dtype=self._buf_dtype("prediction"),
            ),
        )

        self.register_buffer(
            "set_point",
            weight.clone().to(dtype=self._buf_dtype("set_point")),
        )

        self.register_buffer(
            "momentum",
            torch.zeros(
                out_features, in_features,
                dtype=self._buf_dtype("momentum"),
            ),
        )

        # Metaplasticity tracker. Refinement 5 verifies the v1 ratio-check
        # semantics still hold under PC dynamics.
        self.register_buffer(
            "update_ema",
            torch.ones(
                out_features, in_features,
                dtype=self._buf_dtype("update_ema"),
            )
            * 1e-4,
        )

        # Per-input error reliability — self-organizes toward 1/error_variance.
        self.register_buffer(
            "precision",
            torch.ones(
                in_features, dtype=self._buf_dtype("precision")
            ),
        )

        # Per-output running prediction-error magnitude.
        # Refinement 2: salience for episode storage = mean(error_acc).
        self.register_buffer(
            "error_acc",
            torch.zeros(
                out_features, dtype=self._buf_dtype("error_acc")
            ),
        )

        # Per-input plasticity. The buffer-table in V2_IMPLEMENTATION_PLAN.md
        # does not list plasticity, but the plan's two-layer top-down design
        # (decision 3) modulates "the lower block's plasticity and set_point."
        # Without plasticity the modulation channel collapses to set_point
        # only, weakening the design. Retained as [in_features] (matching
        # v1's free-win refactor) so apply_top_down has both channels to act on.
        self.register_buffer(
            "plasticity",
            torch.ones(
                in_features, dtype=self._buf_dtype("plasticity")
            ),
        )

        # --- Layer-level episode store (identical to v1) ---

        proj = torch.randn(in_features, context_dim) / (in_features ** 0.5)
        self.register_buffer("context_proj", proj)

        self.register_buffer(
            "episode_contexts",
            torch.zeros(
                num_episodes, context_dim,
                dtype=self._buf_dtype("episode_contexts"),
            ),
        )
        self.register_buffer(
            "episode_values",
            torch.zeros(
                num_episodes, out_features, in_features,
                dtype=self._buf_dtype("episode_values"),
            ),
        )
        self.register_buffer(
            "episode_scales",
            torch.ones(
                num_episodes, dtype=self._buf_dtype("episode_scales")
            ),
        )
        self.register_buffer("episode_saliences", torch.zeros(num_episodes))
        self.register_buffer(
            "episode_count", torch.tensor(0, dtype=torch.long)
        )
        # Mean input pattern at episode write time. Used by Salvatori-style
        # attractor consolidation (`consolidate_layer_attractor`) to re-
        # present stored input through the layer's PC dynamics so the
        # stored pattern becomes a local minimum of the prediction-error
        # energy. Always allocated regardless of consolidation_style; the
        # cost is num_episodes * in_features * 4 bytes (32 KB at
        # 32 episodes / 256d). Gradient-replay consolidation ignores it.
        self.register_buffer(
            "episode_inputs",
            torch.zeros(
                num_episodes, in_features,
                dtype=self._buf_dtype("episode_inputs"),
            ),
        )

        # Cache of the most recent per-input prediction error from the
        # PC self-modification step. Used by hybrid_block_pc.top_down_pass
        # to propagate prediction error from this block to the one below.
        # Non-persistent — recomputed every forward pass, not checkpointed.
        self._last_pred_error: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _buf_dtype(self, name: str) -> torch.dtype:
        return self._buffer_dtype_overrides.get(name, torch.float32)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse)
        for name, target_dtype in self._buffer_dtype_overrides.items():
            buf = self._buffers.get(name, None)
            if buf is not None and buf.dtype != target_dtype:
                self._buffers[name] = buf.to(dtype=target_dtype)
        return self

    def _compute_context(self, x_flat: torch.Tensor) -> torch.Tensor:
        ctx = x_flat.mean(dim=0) @ self.context_proj
        return ctx / (ctx.norm() + 1e-8)

    def _recall_episode(self, context: torch.Tensor) -> torch.Tensor | None:
        if self.episode_count == 0:
            return None
        n = self.episode_count.item()
        stored = self.episode_contexts[:n]
        sims = torch.mm(stored, context.unsqueeze(-1)).squeeze(-1)
        best_idx = sims.argmax()
        best_sim = sims[best_idx]
        if best_sim < 0.5:
            return None
        if self.episode_values.dtype == torch.int8:
            recalled = (
                self.episode_values[best_idx].to(self.weight.dtype)
                * self.episode_scales[best_idx]
            )
        else:
            recalled = self.episode_values[best_idx]
        delta = recalled - self.weight
        return delta * best_sim * self.episode_blend

    def _store_episode(
        self,
        context: torch.Tensor,
        salience: float,
        input_pattern: torch.Tensor,
    ) -> None:
        """Write one episode if salience exceeds threshold or beats the
        weakest stored episode. `input_pattern` ([in_features]) is the
        mean input vector for the current batch; saved into
        `episode_inputs` for use by Salvatori-style attractor
        consolidation. The weight snapshot, context, and salience are
        also written as before.
        """
        if salience < self.salience_threshold:
            return
        n = self.episode_count.item()
        if n < self.num_episodes:
            idx = n
            self.episode_count.add_(1)
        else:
            min_idx = self.episode_saliences[:n].argmin()
            if salience <= self.episode_saliences[min_idx]:
                return
            idx = min_idx.item()
        self.episode_contexts[idx] = context
        if self.episode_values.dtype == torch.int8:
            snapshot = self.weight.detach()
            scale = (
                snapshot.abs().max().clamp(min=1e-8) / 127.0
            ).to(self.episode_scales.dtype)
            quantized = (
                (snapshot / scale).round().clamp(-128, 127).to(torch.int8)
            )
            self.episode_values[idx] = quantized
            self.episode_scales[idx] = scale
        else:
            self.episode_values[idx] = self.weight.detach().clone()
        self.episode_inputs[idx] = input_pattern.detach().to(
            self.episode_inputs.dtype
        )
        self.episode_saliences[idx] = salience

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        if x.dim() == 3:
            batch, seq_len, _ = x.shape
            x_flat = x.reshape(-1, self.in_features)
        elif x.dim() == 2:
            x_flat = x
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        # Skip PC self-modification during gradient-checkpoint recomputation.
        # Mirrors v1's LivingLayerV6 guard — without this, enabling gradient
        # checkpointing for v2 would fire pc_self_modify twice per training
        # step (once on the original forward, once on the checkpoint replay),
        # double-applying every weight/prediction update and corrupting all
        # living state. Hits the cached recall context from the original
        # forward so the recomputed activation is bit-identical.
        from luthi.grad_checkpoint import is_recomputing
        recomputing = is_recomputing()

        # iPC: gradient checkpointing isn't compatible with T>1 because
        # the weight evolves within the forward and the cached snapshot
        # can't reproduce the trajectory. Fail loud rather than silently
        # produce wrong gradients.
        if recomputing and self.inference_steps_per_forward > 1:
            raise RuntimeError(
                "iPC (inference_steps_per_forward > 1) is incompatible "
                "with gradient checkpointing: the weight evolves within "
                "the forward, so the recompute path cannot reproduce the "
                "original trajectory. Disable one or the other."
            )

        if recomputing:
            weight_snapshot = self._fwd_weight_snapshot
            episode_delta = self._fwd_episode_delta
            context = None
        else:
            with torch.no_grad():
                context = self._compute_context(x_flat)
                episode_delta = self._recall_episode(context)

            # The clone is essential: the PC update below modifies self.weight
            # in-place, which would break autograd version-tracking when
            # backward() needs this tensor for gradients flowing through
            # surrounding (attention) parameters. The 2026-05-10 audit
            # flagged the per-forward allocation cost (~67 MB at 4096d,
            # ~2.4 GB transient across 36 blocks) — real but unavoidable
            # without an autograd refactor (custom Function or
            # buffer-rotation pattern). Closed as "investigated, required
            # for correctness; reduction is a future architectural rework,
            # not a bugfix." Mirrors v1's LivingLayerV6 pattern.
            weight_snapshot = self.weight.clone()
            # Cache for any subsequent recompute pass.
            self._fwd_weight_snapshot = weight_snapshot
            self._fwd_episode_delta = episode_delta

        if episode_delta is not None:
            weight_snapshot = weight_snapshot + episode_delta

        output = x_flat @ weight_snapshot.T

        if recomputing:
            # Recompute path: skip self-modification entirely. The original
            # forward already mutated buffers; replaying here would corrupt them.
            if len(input_shape) == 3:
                output = output.reshape(batch, seq_len, self.out_features)
            return output

        # NaN-safety: if the forward produced non-finite values, skip PC
        # self-modification entirely so living buffers don't consume corrupt
        # data. Without this guard, a NaN/Inf in the input or weights
        # propagates into weight / prediction / set_point / momentum / etc.,
        # persisting beyond the bad batch even after the trainer skips its
        # optimizer step. One host sync per layer per batch — small relative
        # to the PC update math.
        if not torch.isfinite(output).all():
            # Cache a safe zero pred_error so the inter-block top-down sweep
            # sees a valid (if no-op) signal when self-mod was skipped.
            self._last_pred_error = torch.zeros(
                self.in_features, device=output.device,
            )
            if len(input_shape) == 3:
                output = output.reshape(batch, seq_len, self.out_features)
            return output

        with torch.no_grad():
            from luthi.v2.pc_ops import pc_self_modify

            T = self.inference_steps_per_forward
            for inner_step in range(T):
                # Recompute output for inner_step > 0 since the weight
                # has changed since the last iteration. For the first
                # iteration we already have `output` from the matmul above.
                if inner_step > 0:
                    # Use self.weight directly (mutated by previous inner
                    # step's self_modify). We don't need a fresh snapshot
                    # inside the loop — backward() flows through the
                    # initial weight_snapshot only.
                    output = x_flat @ self.weight.T

                # Compute the sparse gate for this inner step (gate state
                # depends on error_acc which evolves across inner steps).
                sparse_gate: torch.Tensor | None = None
                if (
                    self.sparse_threshold > 0.0
                    and self._sparse_step_count >= self.sparse_warmup_steps
                ):
                    sparse_gate = (self.error_acc > self.sparse_threshold).to(
                        dtype=self.weight.dtype
                    )

                salience, pred_error = pc_self_modify(
                    self.weight, self.prediction, self.set_point,
                    self.momentum, self.update_ema, self.precision,
                    self.error_acc, self.plasticity,
                    x_flat, output,
                    self.pc_rate, self.pred_learning_rate,
                    self.homeostatic_decay, self.set_point_adapt_rate,
                    self.momentum_decay, self.update_ema_decay,
                    self.precision_ema_decay,
                    self.precision_min, self.precision_max,
                    self.prediction_clamp,
                    sparse_gate=sparse_gate,
                )

            # Recompute final output after the last inner step so the
            # returned activation reflects the post-self-mod weight state.
            # When T=1, this matches the classical PC behavior — the
            # returned output is computed from the snapshotted weight
            # (before the single self-mod), preserving backward() semantics.
            # When T>1, the inner loop's last output is what we want.
            # (For T=1 we keep `output` from the initial matmul above —
            # see the `if inner_step > 0` branch, which means the T=1
            # case never recomputes and is bit-identical to the prior
            # implementation.)

            self._sparse_step_count += 1
            self._last_pred_error = pred_error
            # Mean input pattern: matches the actual_input used inside
            # pc_self_modify so the stored pattern is the same signal the
            # PC dynamics were trying to predict at this step.
            input_pattern = x_flat.mean(dim=0).detach()
            self._store_episode(context, salience, input_pattern)

            if self._consolidation_tracker is not None:
                # Audit 2026-05-11 fix: feed mean-squared-error magnitude
                # (a scalar per step) rather than pred_error.var() (which
                # is variance ACROSS input dimensions for a single step —
                # spatial, not temporal). The ConsolidationTracker holds
                # a rolling window of these scalars and compares the
                # latest to the frozen warmup baseline — so the right
                # input is "how big is the error right now" relative to
                # "how big was it during warmup". Spatial variance gave
                # the wrong signal (uniformly-large error reads as low
                # spatial variance → spurious consolidation triggers).
                err_magnitude = pred_error.pow(2).mean().item()
                if self._consolidation_tracker.step(err_magnitude):
                    # Dispatch to gradient, attractor, or both. Order
                    # for "both" is gradient first (places weight near
                    # stored regime), then attractor (reinforces the
                    # stored input as a fixed point of that regime).
                    from luthi.v2.consolidation import (
                        consolidate_layer,
                        consolidate_layer_attractor,
                    )
                    if self.consolidation_style in ("gradient", "both"):
                        consolidate_layer(
                            self,
                            consolidation_rate_factor=self.consolidation_rate_factor,
                        )
                    if self.consolidation_style in ("attractor", "both"):
                        consolidate_layer_attractor(
                            self,
                            consolidation_rate_factor=self.consolidation_rate_factor,
                            n_replay_passes=self.consolidation_attractor_passes,
                        )
                    self._consolidation_fire_count += 1

        if len(input_shape) == 3:
            output = output.reshape(batch, seq_len, self.out_features)
        return output

    # ------------------------------------------------------------------
    # Top-down modulation (two-layer per decision 3)
    # ------------------------------------------------------------------

    def apply_top_down(self, signal) -> None:
        """Two-channel top-down: prediction-driven modulation of plasticity
        and set_point.

        Signal carries:
          - prediction_error: [in_features], nudges set_point.
          - salience: [in_features], modulates plasticity.
          - modulation_strength: scalar.

        Per V2_IMPLEMENTATION_PLAN.md decision 3. Refinement 3's M2
        isolation tests verify the two channels do their jobs independently
        without destructive compounding when joint.
        """
        if signal.salience.shape[-1] != self.in_features:
            raise ValueError(
                f"TopDownSignal.salience size {signal.salience.shape[-1]} "
                f"does not match in_features {self.in_features}."
            )
        if signal.prediction_error.shape[-1] != self.in_features:
            raise ValueError(
                f"TopDownSignal.prediction_error size "
                f"{signal.prediction_error.shape[-1]} does not match "
                f"in_features {self.in_features}."
            )

        with torch.no_grad():
            strength = signal.modulation_strength

            self.plasticity.mul_(1.0 - 0.01 * strength).add_(
                signal.salience * 0.01 * strength
            )
            # Floor relaxed from 0.1 → 0.01 on 2026-05-16 to support
            # the plasticity-partition design direction (see
            # docs/research/2026-05-16_plasticity-partitions-design.md).
            # The new floor allows weights to be 10× more stable than
            # the previous minimum — enough to express identity-anchor
            # behavior (per-step updates an order of magnitude smaller)
            # while preserving a nonzero learning rate so weights are
            # never fully frozen. Full freezing was considered (floor
            # = 0.0) but rejected: it leaves no baseline learning rate
            # if the top-down recovery signal is weak or absent, and
            # exact-zero updates can distort downstream metaplasticity
            # tracking. Biological precedent agrees — even the most
            # stable synapses retain nonzero plasticity. Upper bound
            # 10.0 retained as a safety guard against runaway
            # plasticity until there's a specific motivation to raise it.
            self.plasticity.clamp_(0.01, 10.0)

            error_signal = signal.prediction_error.unsqueeze(0)
            self.set_point.add_(
                error_signal * self.set_point_adapt_rate * 10.0 * strength
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def aliveness(self) -> dict[str, float]:
        return {
            "weight_mean": self.weight.mean().item(),
            "weight_std": self.weight.std().item(),
            "prediction_norm": self.prediction.norm().item(),
            "set_point_drift": (
                (self.weight - self.set_point).abs().mean().item()
            ),
            "momentum_magnitude": self.momentum.abs().mean().item(),
            "update_ema_mean": self.update_ema.mean().item(),
            "precision_mean": self.precision.mean().item(),
            "precision_min": self.precision.min().item(),
            "precision_max": self.precision.max().item(),
            "error_acc_mean": self.error_acc.mean().item(),
            "error_acc_max": self.error_acc.max().item(),
            "episodes_stored": self.episode_count.item(),
        }

    def clear_forward_cache(self) -> None:
        """Release the snapshots kept for gradient-checkpoint replay.

        Audit 2026-05-11 fix: `_fwd_weight_snapshot` and `_fwd_episode_delta`
        are cached on the layer so a gradient-checkpoint recompute can
        replay the same forward bit-identically. Between training steps
        they're stale and would otherwise hold ~67 MB per layer at 4096d
        (~2.4 GB across 36 blocks). Trainers should call this after
        optimizer.step() to free that memory. `_last_pred_error` (also
        non-persistent) is dropped similarly so cross-block top-down
        sweep state from one batch doesn't leak into the next.
        """
        self._fwd_weight_snapshot = None
        self._fwd_episode_delta = None
        self._last_pred_error = None

    def non_feedforward_signal(self, x: torch.Tensor) -> float:
        """Mean absolute difference between two consecutive identical-input
        forward passes. Positive value confirms the layer is not feedforward.
        """
        with torch.no_grad():
            out1 = self.forward(x)
            out2 = self.forward(x)
            return (out2 - out1).abs().mean().item()
