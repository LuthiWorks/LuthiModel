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
        buffer_dtypes: dict[str, torch.dtype] | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.ffn_expansion = ffn_expansion
        self.ffn_inner_dim = d_model * ffn_expansion

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

        self.living_ffn = PredictiveCodingLayer(
            inner_dim, inner_dim,
            pc_rate=pc_rate,
            pred_learning_rate=pred_learning_rate,
            homeostatic_decay=homeostatic_decay,
            set_point_adapt_rate=set_point_adapt_rate,
            num_episodes=num_episodes,
            episode_blend=episode_blend,
            compressed_episodes=compressed_episodes,
            consolidation_enabled=consolidation_enabled,
            buffer_dtypes=buffer_dtypes,
        )
        self.episode_store = EpisodeStore(
            d_model,
            num_episodes=num_episodes,
            blend_factor=episode_blend,
        )

        # Cached state for the top-down sweep — the block's input is needed
        # to compute local salience in compute_block_top_down.
        self._block_input: torch.Tensor | None = None

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
        x = x + attn_out

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
        x = x + ffn_out

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
        if apply_modulation and self.ffn_expansion == 1:
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

        if self.ffn_expansion == 1:
            pc_pred_error = (
                self.living_ffn._last_pred_error if apply_prediction else None
            )
        else:
            pc_pred_error = None  # see comment above

        if self._block_input is None:
            raise RuntimeError(
                "PredictiveCodingBlock.top_down_pass called before forward; "
                "no cached block_input available for downstream signal."
            )

        return compute_block_top_down(
            signal, self._block_input, pc_pred_error=pc_pred_error
        )

    def aliveness(self) -> dict[str, float]:
        living = self.living_ffn.aliveness()
        episodes = self.episode_store.stats()
        living["output_episodes_stored"] = episodes["episodes_stored"]
        living["output_episodes_mean_salience"] = episodes["mean_salience"]
        return living


__all__ = ["PredictiveCodingBlock", "TopDownSignal", "mask_signal"]
