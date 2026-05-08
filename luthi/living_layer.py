"""LivingLayerV6: A self-modifying linear layer.

Core component of the LWM (Living Weight Model) architecture. The forward
pass IS learning. Weights carry per-parameter history, plasticity, momentum,
excitability, and context-gated retrieval. Processing changes the processor
— same input produces different output on consecutive passes.

This is neither feedforward nor recurrent. It is a third kind of computation.
"""

import torch
import torch.nn as nn


class LivingLayerV6(nn.Module):
    """A linear layer whose weights self-modify during every forward pass.

    Three modes of adaptation, all local, none requiring gradient flow:
      1. Hebbian self-modification — temporal dynamics, always active
      2. Error-directed learning — convergence, when error signal provided
      3. Layer-level episodic recall — context-gated weight blending

    Weights are registered as buffers, not parameters. The optimizer never
    touches them. The layer trains itself.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hebb_rate: float = 0.001,
        error_rate: float = 0.001,
        homeostatic_decay: float = 0.001,
        set_point_adapt_rate: float = 1e-6,
        momentum_decay: float = 0.99,
        input_mag_decay: float = 0.99,
        salience_threshold: float = 0.5,
        excitability_min: float = 0.3,
        excitability_max: float = 3.0,
        num_episodes: int = 32,
        context_dim: int = 64,
        episode_blend: float = 0.1,
        eval_hebb_fraction: float = 0.33,
        update_ema_decay: float = 0.99,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hebb_rate = hebb_rate
        self.error_rate = error_rate
        self.homeostatic_decay = homeostatic_decay
        self.set_point_adapt_rate = set_point_adapt_rate
        self.momentum_decay = momentum_decay
        self.input_mag_decay = input_mag_decay
        self.salience_threshold = salience_threshold
        self.excitability_min = excitability_min
        self.excitability_max = excitability_max
        self.num_episodes = num_episodes
        self.context_dim = context_dim
        self.episode_blend = episode_blend
        self.eval_hebb_fraction = eval_hebb_fraction
        self.update_ema_decay = update_ema_decay

        # --- Core weight state (all buffers, not parameters) ---

        # Weight matrix: initialized with Kaiming uniform
        weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(weight)
        self.register_buffer("weight", weight)

        # Homeostatic set point: where weights rest when not driven
        self.register_buffer("set_point", weight.clone())

        # Momentum: running average of recent weight updates
        self.register_buffer("momentum", torch.zeros(out_features, in_features))

        # Input magnitude running average: for synaptic scaling (V5)
        self.register_buffer("input_avg_mag", torch.ones(in_features))

        # Excitability accumulator: drives sigmoid -> effective excitability.
        # Per-output-neuron: every column in the original [out, in] layout was
        # mathematically identical (the Hebbian-step update broadcasts
        # salience_per_dim across the input axis; Sanctuary modulation adds a
        # scalar). Stored as [out_features] and broadcast at use time.
        self.register_buffer("excitability_acc", torch.zeros(out_features))

        # Per-input-feature plasticity: learning rate multiplier shared across
        # all output rows for a given input dimension. Every row in the
        # original [out, in] layout was mathematically identical (the only
        # update site, apply_top_down, broadcasts a per-in_features signal
        # across the output axis). Stored as [in_features] and broadcast at
        # use time.
        self.register_buffer("plasticity", torch.ones(in_features))

        # --- Layer-level episode store ---

        # Random projection for context compression (fixed, not learned)
        proj = torch.randn(in_features, context_dim) / (in_features**0.5)
        self.register_buffer("context_proj", proj)

        # Episode storage
        self.register_buffer(
            "episode_contexts", torch.zeros(num_episodes, context_dim)
        )
        self.register_buffer(
            "episode_values", torch.zeros(num_episodes, out_features, in_features)
        )
        self.register_buffer("episode_saliences", torch.zeros(num_episodes))
        self.register_buffer("episode_count", torch.tensor(0, dtype=torch.long))

        # --- Metaplasticity: per-weight update history ---
        # Running average of Hebbian update magnitudes. Each weight tracks
        # how much it typically changes, enabling self-regulation: unusual
        # spikes get dampened, normal updates pass through.
        self.register_buffer(
            "update_ema", torch.ones(out_features, in_features) * 1e-4
        )

        # --- Cached input for error-directed learning ---
        self._cached_input: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_update_gate(self) -> torch.Tensor | None:
        """Return a per-weight gate for activity-dependent updates.

        Base class returns None (all updates ungated). Subclasses override
        to provide selective gating (e.g., spike mask for spiking layers).

        When not None, the returned tensor has shape [out_features, in_features]
        with values in [0, 1]. Gated operations use delta formulation so
        elements with gate=0 are frozen (retain their last state).
        """
        return None

    def _excitability_factor(self) -> torch.Tensor:
        """Compute effective excitability from accumulator via sigmoid.

        Maps the unbounded accumulator to [excitability_min, excitability_max].
        The accumulator retains full history even when effective excitability
        saturates — this is intentional metadata.
        """
        span = self.excitability_max - self.excitability_min
        return self.excitability_min + span * torch.sigmoid(self.excitability_acc)

    def _compute_context(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Compute a context signature from input via random projection.

        Args:
            x_flat: [N, in_features] flattened input.

        Returns:
            [context_dim] context vector (L2-normalized).
        """
        # Mean pool across the batch/sequence dimension, then project
        ctx = x_flat.mean(dim=0) @ self.context_proj  # [context_dim]
        # L2 normalize for cosine similarity later
        return ctx / (ctx.norm() + 1e-8)

    def _recall_episode(self, context: torch.Tensor) -> torch.Tensor | None:
        """Retrieve the best-matching episode's weight snapshot.

        Returns the stored weight delta (episode_value - current_weight) scaled
        by match quality, or None if no episodes stored or no good match.
        """
        if self.episode_count == 0:
            return None

        n = self.episode_count.item()
        stored = self.episode_contexts[:n]  # [n, context_dim]

        # Cosine similarity (contexts are already L2-normalized)
        sims = torch.mm(stored, context.unsqueeze(-1)).squeeze(-1)  # [n]
        best_idx = sims.argmax()
        best_sim = sims[best_idx]

        # Only recall if similarity exceeds threshold
        if best_sim < 0.5:
            return None

        # Return the stored weight snapshot blended by similarity
        recalled_weights = self.episode_values[best_idx]  # [out, in]
        delta = recalled_weights - self.weight
        return delta * best_sim * self.episode_blend

    def _store_episode(self, context: torch.Tensor, salience: float) -> None:
        """Store the current weight configuration as a layer-level episode.

        Uses salience-based pruning when at capacity: replaces the lowest-
        salience episode if the new one is more salient.
        """
        if salience < self.salience_threshold:
            return

        n = self.episode_count.item()

        if n < self.num_episodes:
            # Room available — just append
            idx = n
            self.episode_count.add_(1)
        else:
            # At capacity — replace lowest salience if we're more salient
            min_idx = self.episode_saliences[:n].argmin()
            if salience <= self.episode_saliences[min_idx]:
                return  # Not salient enough to replace anything
            idx = min_idx.item()

        self.episode_contexts[idx] = context
        self.episode_values[idx] = self.weight.detach().clone()
        self.episode_saliences[idx] = salience

    # ------------------------------------------------------------------
    # Forward pass — the core of living computation
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Hebbian self-modification and episodic recall.

        The act of processing changes the processor. This is not feedforward.

        Args:
            x: Input tensor. Supports:
               - [batch, in_features]
               - [batch, seq_len, in_features]

        Returns:
            Output tensor with the same leading dimensions and out_features
            as the last dimension.
        """
        # Handle both 2D and 3D input
        input_shape = x.shape
        if x.dim() == 3:
            batch, seq_len, _ = x.shape
            x_flat = x.reshape(-1, self.in_features)  # [B*S, in]
        elif x.dim() == 2:
            x_flat = x  # [B, in]
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        n_samples = x_flat.shape[0]

        from luthi.grad_checkpoint import is_recomputing
        recomputing = is_recomputing()

        if recomputing:
            # Replay using the snapshot from the original forward so the
            # recomputed activations are bit-identical. self.weight may
            # have already been mutated by the original-forward Hebbian.
            weight_snapshot = self._fwd_weight_snapshot
            episode_delta = self._fwd_episode_delta
            # Skip cache update + context recompute — use originals.
            context = None
        else:
            # Cache input for error-directed learning (detached from graph)
            self._cached_input = x_flat.detach()

            # --- Episodic recall: context-gated weight blending ---
            with torch.no_grad():
                context = self._compute_context(x_flat)
                episode_delta = self._recall_episode(context)

            # Snapshot weights for computation. The clone is essential: the
            # Hebbian update below modifies self.weight in-place, which would
            # break autograd's version tracking when backward() needs this
            # tensor to compute gradients for upstream (attention) parameters.
            weight_snapshot = self.weight.clone()
            # Save for any subsequent recompute pass (gradient checkpointing).
            self._fwd_weight_snapshot = weight_snapshot
            self._fwd_episode_delta = episode_delta

        if episode_delta is not None:
            weight_snapshot = weight_snapshot + episode_delta

        # --- Linear computation ---
        output = x_flat @ weight_snapshot.T  # [N, out]

        # --- Self-modification (no gradient flow) ---
        # Gated: only fires on the original forward, never on recompute.
        # The recompute path must observe identical state to be a faithful
        # replay; firing Hebbian twice would double the self-modification
        # rate and corrupt gradients computed from the recomputed graph.
        if not recomputing:
            with torch.no_grad():
                from luthi.fused_ops import living_self_modify

                gate = self._get_update_gate()
                salience_scalar = living_self_modify(
                    self.weight, self.set_point, self.momentum,
                    self.input_avg_mag, self.excitability_acc, self.plasticity,
                    self.update_ema, x_flat, output, gate,
                    self.hebb_rate, self.homeostatic_decay,
                    self.set_point_adapt_rate, self.momentum_decay,
                    self.input_mag_decay, self.salience_threshold,
                    self.excitability_min, self.excitability_max,
                    self.update_ema_decay, self.eval_hebb_fraction,
                    self.training,
                )

                # Store episode if salient enough
                self._store_episode(context, salience_scalar)

        # Reshape output to match input shape
        if len(input_shape) == 3:
            output = output.reshape(batch, seq_len, self.out_features)

        return output

    # ------------------------------------------------------------------
    # Top-down modulation (backward pass)
    # ------------------------------------------------------------------

    def apply_top_down(self, signal) -> None:
        """Modulate self-modification parameters based on downstream feedback.

        Called during the top-down backward pass. Adjusts this layer's
        plasticity and homeostatic set points based on what later blocks
        found important and unexpected.

        This modifies state for the NEXT forward pass — it does not
        change the current pass's output.

        Both ``salience`` and ``prediction_error`` on the signal live in
        **input-dimension space** (see TopDownSignal docstring). They are
        vectors of length ``in_features``. Plasticity is now stored as a
        per-input-dim vector (`[in_features]`), so the salience signal maps
        directly onto its axis with no broadcast needed. Set point remains
        per-weight `[out, in]` and the prediction-error signal is broadcast
        across the output axis (every weight fed by input dim ``i`` drifts
        equally), preserving the original homeostatic semantics.

        Args:
            signal: TopDownSignal with salience, prediction_error, and
                modulation_strength fields. Both vectors must have
                length ``self.in_features``.
        """
        # Fail loud if the signal's shape contract is violated. Without
        # this, a wrong-axis signal would silently push set_points and
        # plasticity in the wrong direction — a bug that wouldn't crash,
        # just slowly corrupt the homeostatic state across training.
        if signal.salience.shape[-1] != self.in_features:
            raise ValueError(
                f"TopDownSignal.salience has size {signal.salience.shape[-1]} "
                f"on its last axis, but this layer expects in_features="
                f"{self.in_features}. Top-down signals live in input-dim space."
            )
        if signal.prediction_error.shape[-1] != self.in_features:
            raise ValueError(
                f"TopDownSignal.prediction_error has size "
                f"{signal.prediction_error.shape[-1]} on its last axis, but "
                f"this layer expects in_features={self.in_features}. "
                f"Top-down signals live in input-dim space."
            )

        with torch.no_grad():
            strength = signal.modulation_strength

            # 1. Plasticity modulation: boost the learning rate for inputs
            # whose dimension was important downstream. Plasticity is
            # `[in_features]`, signal.salience is `[in_features]` — the
            # axes match directly.
            self.plasticity.mul_(1.0 - 0.01 * strength).add_(
                signal.salience * 0.01 * strength
            )
            # Clamp plasticity to prevent runaway or death.
            self.plasticity.clamp_(0.1, 10.0)

            # 2. Set point drift: nudge homeostatic targets along the
            # per-input-dim error pattern. Each input dim's error
            # (over- or under-utilized relative to downstream
            # expectation) shifts the resting state of every weight
            # connected to that input. Output rows all move together —
            # this is the right semantics because the signal indexes
            # input dimensions, not output dimensions (see TopDownSignal).
            error_signal = signal.prediction_error.unsqueeze(0)  # [1, in_features]
            self.set_point.add_(
                error_signal * self.set_point_adapt_rate * 10.0 * strength
            )

    # ------------------------------------------------------------------
    # Error-directed learning (V6)
    # ------------------------------------------------------------------

    def apply_error(self, output_error: torch.Tensor) -> None:
        """Apply error-directed local learning.

        Each weight adjusts based on the correlation between its input and
        the error it contributed to. Mathematically equivalent to the gradient
        for a single linear layer, but framed as a purely local operation.

        Biologically plausible: a neuron observes its own input (presynaptic)
        and receives a signal about downstream error (postsynaptic response).

        Args:
            output_error: [N, out_features] or [batch, seq_len, out_features]
                          error signal at this layer's output.
        """
        if self._cached_input is None:
            raise RuntimeError(
                "apply_error called before forward. No cached input available."
            )

        # Flatten if 3D
        if output_error.dim() == 3:
            output_error = output_error.reshape(-1, self.out_features)

        n_samples = self._cached_input.shape[0]

        with torch.no_grad():
            # Local error-directed update: input.T @ error / n_samples
            # This is the gradient for a single linear layer, computed locally
            update = (output_error.T @ self._cached_input) / n_samples  # [out, in]

            # Per-element clipping: no single weight gets a massive update.
            # Prevents large gradients (especially in later blocks near the
            # loss) from destabilizing the living weights.
            update = update.clamp(-1.0, 1.0)

            self.weight.sub_(update * self.error_rate)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def aliveness(self) -> dict[str, float]:
        """Return diagnostic metrics about the layer's living state.

        Useful for monitoring during training and debugging.
        """
        exc = self._excitability_factor()
        drift = (self.weight - self.set_point).abs().mean().item()

        return {
            "weight_mean": self.weight.mean().item(),
            "weight_std": self.weight.std().item(),
            "set_point_drift": drift,
            "excitability_mean": exc.mean().item(),
            "excitability_min": exc.min().item(),
            "excitability_max": exc.max().item(),
            "momentum_magnitude": self.momentum.abs().mean().item(),
            "input_avg_mag_mean": self.input_avg_mag.mean().item(),
            "episodes_stored": self.episode_count.item(),
            "update_ema_mean": self.update_ema.mean().item(),
        }

    def non_feedforward_signal(self, x: torch.Tensor) -> float:
        """Measure how much output differs on two consecutive identical passes.

        A positive value confirms the layer is not feedforward.
        Returns the mean absolute difference between two consecutive outputs.
        """
        with torch.no_grad():
            # We need to be careful — forward() modifies state, so two calls
            # SHOULD produce different outputs. That's what we're measuring.
            out1 = self.forward(x)
            out2 = self.forward(x)
            return (out2 - out1).abs().mean().item()
