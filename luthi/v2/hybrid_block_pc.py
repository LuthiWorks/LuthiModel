"""PredictiveCodingBlock: attention + PC living FFN + episode store.

The v2 building block. Same residual structure as v1's HybridBlock; the
FFN is a `PredictiveCodingLayer` (PC dynamics) instead of a `LivingLayerV6`
(v1's Hebbian dynamics). Attention and the block-level episode store are
unchanged — Decision 9 reuses them as-is.

The top-down sweep is two-channel per Decision 3:
  - prediction channel: pred_error flows down as the next block's signal,
    sourced from this block's PC layer.
  - modulation channel: applies salience-driven plasticity adjust and
    pred_error-driven set_point adjust to the layer.

`top_down_pass` accepts `apply_modulation` and `apply_prediction` flags
so the M2 isolation tests (refinement 3) can run each channel alone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from luthi.attention import ScalarAttention
from luthi.episode_store import EpisodeStore
from luthi.v2.living_layer_pc import PredictiveCodingLayer
from luthi.v2.backward_pass_pc import (
    TopDownSignal,
    compute_block_top_down,
    mask_signal,
)


class PredictiveCodingBlock(nn.Module):
    """One block of the v2 architecture.

    Stack N of these for a complete v2 model. Attention trains via standard
    backpropagation. PC living FFN trains itself via prediction-error
    self-modification. Episode store accumulates context-gated experiences.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        ffn_expansion: int = 1,
        pc_rate: float = 0.001,
        pred_learning_rate: float = 0.0001,
        homeostatic_decay: float = 0.001,
        set_point_adapt_rate: float = 1e-6,
        num_episodes: int = 64,
        episode_blend: float = 0.3,
        compressed_episodes: bool = False,
        consolidation_enabled: bool = False,
        consolidation_style: str = "gradient",
        consolidation_attractor_passes: int = 1,
        mu_pc_enabled: bool = False,
        mu_pc_exponent: float = 0.5,
        mu_pc_rate_power: float = 0.0,
        n_blocks_total: int = 1,
        buffer_dtypes: dict[str, torch.dtype] | None = None,
        dead_ffn: bool = False,
        # Run-3 plumb-through (2026-07-17): the inverted-U learning gain
        # (built 2026-07-05, layer-level; never reachable from the model
        # until now) and the episode recall gate.
        learning_gain_enabled: bool = False,
        relative_trust: bool = False,
        # Episode-store admission/retention fix (2026-07-27), opt-in per arm.
        adaptive_episodes: bool = False,
        adaptive_recall: bool = False,
        homeostatic_band_enabled: bool = False,
        drive_mode: str = "raw",
        learning_gain_rise: float = 2.0,
        learning_gain_cap: float = 3.0,
        episode_recall_threshold: float = 0.5,
    ):
        super().__init__()
        # Dead-encoder arm (Experiment 1 / JEPA pilot control, 2026-07-15):
        # dead_ffn=True swaps the PC living layer for a standard trainable
        # nn.Linear and removes the block-level episode store — the same
        # trunk with the living channel OFF, isolating exactly the variable
        # the matched-capacity comparison needs. Construction-time guard:
        # living-specific machinery explicitly enabled alongside dead_ffn
        # is a contradiction, not a configuration.
        if dead_ffn and consolidation_enabled:
            raise ValueError(
                "dead_ffn=True with consolidation_enabled=True: a dead FFN "
                "has no living weights to consolidate into. Pick one."
            )
        self.dead_ffn = bool(dead_ffn)
        self.d_model = d_model
        self.n_heads = n_heads
        self.ffn_expansion = ffn_expansion
        self.ffn_inner_dim = d_model * ffn_expansion
        self.mu_pc_enabled = mu_pc_enabled
        self.mu_pc_exponent = float(mu_pc_exponent)
        # Depth-μP residual scaling. Original μPC spec is 1/√L
        # (exponent = 0.5, Innocenti et al. 2025). The exponent is
        # exposed as a tunable knob added 2026-05-16 after M6's depth
        # sweep at 128d surfaced that μPC's signal attenuation interacts
        # poorly with PC's living-weights property at depth (NFF shrank
        # from 5.77e-3 at L=4 to ~2e-3 at L=12). Lower exponent =
        # milder attenuation per block:
        #   exponent = 0.5 → 1/√L  (original μPC, residual ÷3.46 at L=12)
        #   exponent = 0.25 → 1/L^¼ (milder, residual ÷1.86 at L=12)
        #   exponent = 0.0 → 1.0   (no attenuation; equivalent to mu_pc_enabled=False
        #                            for the residual path, though init still scales
        #                            unless that's also disabled)
        # See `docs/research/2026-05-16_plasticity-partitions-design.md`
        # for the design discussion that led to this knob.
        if mu_pc_enabled:
            self.residual_scale = 1.0 / (n_blocks_total ** self.mu_pc_exponent)
        else:
            self.residual_scale = 1.0

        # muPC rate balancing (2026-07-30). `residual_scale` multiplies this
        # block's output into the residual stream:
        #     x = x + residual_scale * attn_out
        #     x = x + residual_scale * ffn_out
        # so by the chain rule every BACKPROP-trained parameter in the block
        # (attention, up/down projections) receives a gradient scaled by
        # `residual_scale`. The living FFN does NOT: it self-modifies locally
        # inside the forward pass, from its own input and output, before that
        # multiplication is applied. Its update magnitude is set by `pc_rate`
        # alone and is completely unaffected.
        #
        # muPC (Innocenti et al. 2025) is derived for a network that is PC
        # THROUGHOUT, where the residual scaling and init scaling together give
        # depth-independent dynamics because everything scales together. This
        # trunk is a hybrid -- backprop attention, local-PC FFN -- so applying
        # muPC's attenuation scales one half of a two-speed system and leaves
        # the other at full rate. The deeper the trunk, the smaller
        # `residual_scale`, and the more the local PC substrate overpowers the
        # global objective.
        #
        # Measured consequences at 2026-07-30, all consistent with this:
        #   * depth 8 with muPC (s=0.5946) collapses in training to
        #     within-batch cosine 0.970 -- the encoder stops distinguishing
        #     its inputs;
        #   * the same arm with muPC off (s=1.0) trains to 0.0111;
        #   * depth 4 (s=0.7071) is healthy -- the imbalance is mild;
        #   * the collapse is total at BLOCK 0, because block 0's backprop path
        #     is attenuated exactly like every other block's.
        #
        # `mu_pc_rate_power` scales the PC rates by
        # `residual_scale ** mu_pc_rate_power`:
        #
        #   power =  0.0  factor 1.0        OFF. Bit-identical to every run
        #                                   before 2026-07-30.
        #   power = +1.0  factor s          ATTENUATE, matching the backprop
        #                                   path's gradient scaling.
        #   power = -1.0  factor 1/s        AMPLIFY, compensating for it.
        #
        # MEASURED 2026-07-30, and it refuted the reasoning that motivated
        # power=+1. The prediction was that matching the PC rate to the
        # backprop attenuation would restore a two-speed balance and prevent
        # the collapse. It made the collapse WORSE: offset dominance 0.5657
        # unbalanced -> 0.8277 attenuated, centered cosine 0.5087 -> 0.1121.
        #
        # The three-point ordering rules out the ratio account entirely:
        #
        #   residual_scale  PC factor  offset dominance
        #        1.0           1.0          0.21     (muPC off)
        #        0.595         1.0          0.57     (power 0, unbalanced)
        #        0.595         0.595        0.83     (power +1, attenuated)
        #
        # muPC-off and power=+1 have the SAME PC/backprop ratio and land at
        # 0.21 vs 0.83, so the ratio is not what drives it. What orders these
        # is TOTAL attenuation: damping either path worsens the offset and
        # damping both is worst.
        #
        # That fits a different mechanism. The block computes
        # `x = x_0 + s * sum(f)`, and the embedding `x_0` is NOT scaled by s.
        # If x_0 carries a batch-constant component -- and the positional
        # embedding is identical across sequences by construction -- then
        # shrinking s raises that constant's share of the representation.
        # Less signal from the blocks, the same constant underneath.
        #
        # power=-1 tests the prediction that account makes: amplifying the PC
        # rate should give the blocks MORE learned, input-dependent structure
        # per unit of attenuation, and drive offset dominance back down.
        self.mu_pc_rate_power = float(mu_pc_rate_power)
        self._mu_pc_rate_factor = (
            float(self.residual_scale ** self.mu_pc_rate_power)
            if (mu_pc_enabled and self.mu_pc_rate_power != 0.0)
            else 1.0
        )
        self._n_blocks_total = n_blocks_total

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # n_heads=4 default per the 2026-05-10 audit — single-head couldn't
        # attend to multiple independent subspaces at scale. With d_model=128
        # this gives d_head=32; at d_model=256 it gives d_head=64; both
        # are standard MHA sizes. Override n_heads if the caller wants
        # different splitting (must divide d_model evenly).
        self.attention = ScalarAttention(d_model, n_heads=n_heads)

        # FFN expansion (2026-05-10 audit): standard transformer pattern is
        # d_model → expansion*d_model → d_model with a nonlinearity. The
        # original PC living FFN was d_model → d_model linear — the audit
        # called this a fundamental capacity bottleneck. With ffn_expansion=1
        # we preserve the original behavior (no projections, no GELU); with
        # > 1 the PC layer operates in the expanded space sandwiched between
        # standard trainable projections.
        if ffn_expansion == 1:
            self.up_proj: nn.Module = nn.Identity()
            self.down_proj: nn.Module = nn.Identity()
            inner_dim = d_model
        else:
            self.up_proj = nn.Linear(d_model, self.ffn_inner_dim, bias=False)
            self.down_proj = nn.Linear(self.ffn_inner_dim, d_model, bias=False)
            inner_dim = self.ffn_inner_dim

        if self.dead_ffn:
            # The control arm: a plain trainable linear in the PC layer's
            # position (same shape, backprop-trained, no self-modification,
            # no episodes). Attribute keeps the `living_ffn` name so the
            # forward paths are shared — isinstance checks distinguish it
            # everywhere the difference matters (freeze_plasticity's type
            # sweep already skips it for free).
            self.living_ffn = nn.Linear(inner_dim, inner_dim, bias=False)
        else:
            self.living_ffn = PredictiveCodingLayer(
                inner_dim, inner_dim,
                pc_rate=pc_rate * self._mu_pc_rate_factor,
                pred_learning_rate=(
                    pred_learning_rate * self._mu_pc_rate_factor
                ),
                homeostatic_decay=homeostatic_decay,
                set_point_adapt_rate=set_point_adapt_rate,
                num_episodes=num_episodes,
                episode_blend=episode_blend,
                compressed_episodes=compressed_episodes,
                consolidation_enabled=consolidation_enabled,
                consolidation_style=consolidation_style,
                consolidation_attractor_passes=consolidation_attractor_passes,
                buffer_dtypes=buffer_dtypes,
                learning_gain_enabled=learning_gain_enabled,
                relative_trust=relative_trust,
                adaptive_episodes=adaptive_episodes,
                adaptive_recall=adaptive_recall,
                homeostatic_band_enabled=homeostatic_band_enabled,
                drive_mode=drive_mode,
                learning_gain_rise=learning_gain_rise,
                learning_gain_cap=learning_gain_cap,
                episode_recall_threshold=episode_recall_threshold,
            )

        # Apply Depth-μP re-init AFTER all sub-modules constructed. We
        # don't modify ScalarAttention or PredictiveCodingLayer internals —
        # just overwrite their weight tensors with Gaussian samples at
        # `std = 1/√(fan_in · L)`. This is the simplest implementation of
        # the parameterization (Innocenti et al. 2025) and decouples μPC
        # from the sub-modules' default init logic.
        if mu_pc_enabled:
            self._apply_mu_pc_init(n_blocks_total)
        if self.dead_ffn:
            # No block-level episode store either: the episode machinery is
            # living-channel state, and the control isolates the whole
            # living channel, not just the PC update rule.
            self.episode_store = None
        else:
            self.episode_store = EpisodeStore(
                d_model,
                num_episodes=num_episodes,
                blend_factor=episode_blend,
                recall_similarity_threshold=episode_recall_threshold,
            )

        # Cached state for the top-down sweep — the block's input is needed
        # to compute local salience in compute_block_top_down.
        self._block_input: torch.Tensor | None = None

    def _apply_mu_pc_init(self, L: int) -> None:
        """Re-initialize all trainable linears and the PC weight buffer
        with `Normal(0, 1/(fan_in^0.5 · L^exponent))` per Innocenti et
        al. 2025. The exponent on L is `self.mu_pc_exponent` (default
        0.5 = original μPC; lower values give larger initial weights,
        which compensates for milder residual attenuation at the same
        depth). The fan_in scaling (always sqrt) is standard Kaiming-
        style and is kept fixed; only the depth scaling is tunable.
        """
        import math

        def _mupc_normal_(weight: torch.Tensor, fan_in: int) -> None:
            std = 1.0 / (math.sqrt(fan_in) * (L ** self.mu_pc_exponent))
            with torch.no_grad():
                weight.normal_(mean=0.0, std=std)

        # Attention projections: q/k/v/o are all d_model -> d_model.
        for proj in (
            self.attention.q_proj,
            self.attention.k_proj,
            self.attention.v_proj,
            self.attention.o_proj,
        ):
            _mupc_normal_(proj.weight, proj.in_features)

        # FFN expansion projections (only present when ffn_expansion > 1).
        if isinstance(self.up_proj, nn.Linear):
            _mupc_normal_(self.up_proj.weight, self.up_proj.in_features)
        if isinstance(self.down_proj, nn.Linear):
            _mupc_normal_(self.down_proj.weight, self.down_proj.in_features)

        # PC layer's weight buffer (or the dead arm's plain Linear weight).
        # set_point tracks weight on the living path only.
        _mupc_normal_(self.living_ffn.weight, self.living_ffn.in_features)
        if not self.dead_ffn:
            with torch.no_grad():
                self.living_ffn.set_point.copy_(self.living_ffn.weight)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv_cache: bool = False,
    ):
        block_input = x
        self._block_input = block_input

        if return_kv_cache:
            attn_out, new_kv = self.attention(
                self.norm1(x),
                causal=causal,
                kv_cache=kv_cache,
                return_kv_cache=True,
            )
        else:
            attn_out = self.attention(
                self.norm1(x), causal=causal, kv_cache=kv_cache,
            )
            new_kv = None
        # Depth-μP: residual contribution per block scaled by 1/√L. With
        # default L=1 the factor is 1.0 and behavior matches the un-scaled
        # block exactly.
        x = x + self.residual_scale * attn_out

        ffn_in = self.norm2(x)
        # FFN body: up-project, nonlinearity, PC living layer in expanded
        # space, down-project. When ffn_expansion=1 the projections are
        # nn.Identity and GELU operates directly on d_model — equivalent
        # to the original "Linear with GELU" path the v2 block had before
        # this refactor, except GELU was implicit in the linearity of the
        # PC layer. With expansion > 1, GELU lives in the expanded space
        # (standard transformer FFN pattern).
        if self.ffn_expansion == 1:
            ffn_out = self.living_ffn(ffn_in)
        else:
            up = self.up_proj(ffn_in)
            up = F.gelu(up)
            up = self.living_ffn(up)
            ffn_out = self.down_proj(up)
        x = x + self.residual_scale * ffn_out

        if self.episode_store is not None:
            x = self.episode_store(block_input, x)

        if return_kv_cache:
            return x, new_kv
        return x

    def top_down_pass(
        self,
        signal: TopDownSignal,
        apply_modulation: bool = True,
        apply_prediction: bool = True,
    ) -> TopDownSignal:
        """Backward sweep through this block.

        Args:
            signal: TopDownSignal from the block above (or initial signal).
            apply_modulation: When True (default), apply the signal to the
                living FFN — salience modulates plasticity, prediction_error
                modulates set_point. False = M2 prediction-only mode: the
                block still produces a downstream signal but does not modify
                its own living state. Refinement 3 isolation test.
            apply_prediction: When True (default), use this block's PC layer
                pred_error as the prediction_error of the downstream signal
                (the prediction channel of Decision 3's two-layer top-down).
                False = M2 modulation-only mode: downstream prediction_error
                falls back to v1's salience-difference heuristic, isolating
                the modulation channel from this block's PC contribution.

        Returns:
            Refined TopDownSignal for the block below.
        """
        if apply_modulation and self.ffn_expansion == 1 and not self.dead_ffn:
            self.living_ffn.apply_top_down(signal)
        # When ffn_expansion > 1, the PC layer operates in the expanded
        # inner space (inner_dim = d_model * expansion). The TopDownSignal
        # carries d_model-shaped vectors, so apply_top_down's shape contract
        # wouldn't match. For pilot/M5, fall back to the v1-style heuristic
        # cross-block signal (compute_block_top_down with pc_pred_error=None)
        # — the within-layer PC dynamics still drive their own learning, we
        # just don't try to propagate the inner pred_error across the
        # block boundary. A future improvement: learn a projection from
        # inner-space pred_error back to d_model space for the cross-block
        # signal. Out of scope for the current audit.

        if self.ffn_expansion == 1 and not self.dead_ffn:
            pc_pred_error = (
                self.living_ffn._last_pred_error if apply_prediction else None
            )
        else:
            # expansion > 1 (see comment above) or dead arm (no PC layer,
            # no pred_error) — fall back to the v1-style heuristic signal.
            pc_pred_error = None

        if self._block_input is None:
            raise RuntimeError(
                "PredictiveCodingBlock.top_down_pass called before forward; "
                "no cached block_input available for downstream signal."
            )

        return compute_block_top_down(
            signal, self._block_input, pc_pred_error=pc_pred_error
        )

    def aliveness(self) -> dict[str, float]:
        if self.dead_ffn:
            # The dead arm reports honestly: no living signals to fake.
            # Consumers aggregate by key presence (_substrate_health_metrics
            # skips absent keys), so a sparse dict is safe by construction.
            return {"dead_ffn": 1.0}
        living = self.living_ffn.aliveness()
        episodes = self.episode_store.stats()
        living["output_episodes_stored"] = episodes["episodes_stored"]
        living["output_episodes_mean_salience"] = episodes["mean_salience"]
        return living


__all__ = ["PredictiveCodingBlock", "TopDownSignal", "mask_signal"]
