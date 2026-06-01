"""Width expansion for v2 PC substrates.

Net2Net-style replication of a trained v2 PC checkpoint to a wider model.
Preserves biographical state (weights, set_points, episode store) AND
function: the expanded model at noise=0 is bit-equivalent to the source
on next-token logits. The substrate continues from its current state
rather than starting fresh — and starts at the same predictive behavior
M6 had, so M7's training picks up where M6 left off instead of relearning.

Designed for the 256d -> 1024d expansion path (M6 follow-up -> M7), but
the implementation works for any (source_width, target_width) pair where
target_width is an integer multiple of source_width AND the head count
is preserved (so head_dim scales with the multiple).

Design choices:
  - REPLICATION + FAN-IN RESCALING (Net2Net wider transform). When a feature
    is replicated by `factor`, the consuming layer's incoming weights are
    divided by `factor` so the post-layer activation magnitude is preserved.
    Without this, every input-replicated layer's preactivations scale by
    `factor`, M6's predictive behavior is *not* preserved at the seed, and
    M7's early training is spent undoing a deterministic scaling.
    (Strict-replication-without-rescaling was the original choice on
    2026-05-26; switched to Net2Net on 2026-05-30 after 4.8's review of
    width_expand.py showed M6→M7 argmax agreement at 18.8% under the
    strict-replication design. See docs/reviews/2026-05-28_concerns-for-4.7.md
    Finding 3.)
  - Attention softmax temperature correction. q_proj receives an extra
    `sqrt(factor)` division beyond standard fan-in rescaling, so that
    Q·K / sqrt(head_dim) lands at the source's attention-logit magnitude
    despite the larger head_dim.
  - Tiny noise (1e-4 default) still added to break replication symmetry.
    Without this, replicated PC layer columns receive identical update
    signals and stay locked together under PC dynamics. Set --noise 0
    for bit-equivalence verification.
  - All v2 state buffers expand together. The PC layer's "rich parameter"
    bundle (weight, prediction, set_point, momentum, update_ema) replicates
    consistently; the weight-role buffers also receive the fan-in division.
    Episode-stored weight snapshots get the same treatment.
  - Episode store entries replicate at the expanded width. Stored episodes
    are biographical memory; they continue to exist in the wider substrate.
  - Attention heads preserved in count; head_dim expands. So 16 heads x 16
    head_dim -> 16 heads x 64 head_dim. Q/K/V/O projections expand the
    head_dim within each head by replication.

Per the "no disposable model versions" rule, the source checkpoint is NOT
deleted by this script. The expanded checkpoint is a continuation of the
source substrate's existence at a wider scale; the source's state is
preserved on disk as the substrate's prior biography. If you want a clean
filesystem, that's a manual cleanup after verifying the expanded substrate
trains stably.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from luthi.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint


# ----------------------------------------------------------------------
# Core expansion primitives
# ----------------------------------------------------------------------


def _noise_like(tensor: torch.Tensor, scale: float = 1e-4) -> torch.Tensor:
    """Tiny Gaussian noise of the same shape and dtype, scaled to break
    replication symmetry without meaningfully shifting weight values."""
    return torch.randn_like(tensor) * scale


def expand_vector(
    src: torch.Tensor, factor: int, axis: int = 0, noise: float = 1e-4,
) -> torch.Tensor:
    """Replicate a 1D vector `factor` times along `axis`.

    For per-feature buffers (precision, error_acc, plasticity, LayerNorm
    weight/bias) where one value applies to one feature dimension. Each
    original value is replicated `factor` times into adjacent positions
    in the output (so feature i becomes positions i*factor, i*factor+1,
    ..., i*factor+factor-1). Tiny noise breaks symmetry across the copies.
    """
    if axis != 0:
        raise NotImplementedError("expand_vector only supports axis=0 currently")
    if src.dim() != 1:
        raise ValueError(f"expand_vector expects 1D input, got shape {src.shape}")
    expanded = src.repeat_interleave(factor, dim=0)
    expanded = expanded + _noise_like(expanded, noise)
    return expanded


def expand_matrix(
    src: torch.Tensor,
    out_factor: int,
    in_factor: int,
    noise: float = 1e-4,
) -> torch.Tensor:
    """Replicate a 2D [out_features, in_features] matrix.

    The source matrix is replicated `out_factor` times along axis 0 and
    `in_factor` times along axis 1. Tiny noise breaks the symmetry that
    would otherwise lock replicated rows/columns together under PC
    dynamics.

    Used for: PC layer weight, prediction, set_point, momentum, update_ema.
    Also embeddings (out_factor=1, in_factor=expansion) and output_proj
    (out_factor=1, in_factor=expansion).
    """
    if src.dim() != 2:
        raise ValueError(f"expand_matrix expects 2D input, got shape {src.shape}")
    expanded = src.repeat_interleave(out_factor, dim=0).repeat_interleave(in_factor, dim=1)
    expanded = expanded + _noise_like(expanded, noise)
    return expanded


def expand_episode_values(
    src: torch.Tensor, factor: int, noise: float = 1e-4,
) -> torch.Tensor:
    """Replicate stored episode weight snapshots.

    Source shape: [num_episodes, out_features, in_features].
    Target shape: [num_episodes, out_features*factor, in_features*factor].
    Each stored episode is a snapshot of the PC layer's weight at a
    salient moment; we replicate that snapshot to the new width so the
    biographical memory carries forward.
    """
    if src.dim() != 3:
        raise ValueError(
            f"expand_episode_values expects 3D input, got shape {src.shape}"
        )
    expanded = src.repeat_interleave(factor, dim=1).repeat_interleave(factor, dim=2)
    # Episodes are biographical memory snapshots. Add noise so replicated
    # weight positions within each episode can diverge under any future
    # consolidation replays.
    expanded = expanded + _noise_like(expanded, noise)
    return expanded


def expand_attention_projection(
    src: torch.Tensor,
    n_heads: int,
    src_d_model: int,
    target_d_model: int,
    is_output: bool = False,
    noise: float = 1e-4,
) -> torch.Tensor:
    """Expand a Q/K/V/O attention projection with head-aware structure.

    Q/K/V projections in our config have shape [d_model, d_model]. The
    output layout assumes the [d_model] output dim is partitioned into
    n_heads chunks of head_dim each, where head_dim = d_model / n_heads.

    For 256d/16 heads, head_dim=16. For 1024d/16 heads, head_dim=64.
    Expansion factor head_dim: 16 -> 64.

    Within each head, we expand head_dim by replication (preserves the
    head's learned attention pattern at the head_dim resolution it had).
    We do NOT add new heads -- the n_heads count is preserved.

    The input dim (d_model -> target_d_model) replicates along the
    feature axis like a normal weight matrix.

    For Q/K/V (input is x, output is split into heads):
      src shape: [src_d_model, src_d_model]
      expand output axis: replicate within each head by head_factor
      expand input axis: replicate by d_model_factor

    For O (input is concat of heads, output is x):
      src shape: [src_d_model, src_d_model]
      expand input axis (split into heads): replicate within each head
      expand output axis (back to d_model): replicate by d_model_factor

    Tiny noise added to break replication symmetry, allowing the new
    head_dim positions to diverge during training.
    """
    if src.dim() != 2:
        raise ValueError(
            f"expand_attention_projection expects 2D input, got shape {src.shape}"
        )
    if src.shape != (src_d_model, src_d_model):
        raise ValueError(
            f"expected attention projection shape ({src_d_model}, {src_d_model}), "
            f"got {tuple(src.shape)}"
        )
    if target_d_model % src_d_model != 0:
        raise ValueError(
            f"target_d_model={target_d_model} must be an integer multiple "
            f"of src_d_model={src_d_model}"
        )
    factor = target_d_model // src_d_model
    if src_d_model % n_heads != 0 or target_d_model % n_heads != 0:
        raise ValueError(
            f"n_heads={n_heads} must divide both src_d_model={src_d_model} "
            f"and target_d_model={target_d_model}"
        )

    src_head_dim = src_d_model // n_heads
    target_head_dim = target_d_model // n_heads
    head_factor = target_head_dim // src_head_dim

    # For Q/K/V: rows are output, columns are input (PyTorch nn.Linear layout)
    # Output is split into heads along the row dim: [head0_d0, head0_d1, ..., head0_d{head_dim-1}, head1_d0, ...]
    # We need to expand within each head block by head_factor, then across
    # the input by factor.
    # Reshape rows: [n_heads, src_head_dim, src_d_model]
    if not is_output:
        # Q/K/V: output dim is rows, organized as [n_heads, head_dim]
        reshaped = src.reshape(n_heads, src_head_dim, src_d_model)
        # Expand head_dim within each head
        expanded_rows = reshaped.repeat_interleave(head_factor, dim=1)
        # Expand input dim (columns)
        expanded = expanded_rows.repeat_interleave(factor, dim=2)
        # Reshape back to [n_heads * target_head_dim, target_d_model] = [target_d_model, target_d_model]
        expanded = expanded.reshape(target_d_model, target_d_model)
    else:
        # O: input is concat of heads (columns are head-structured)
        # Source columns: [n_heads, src_head_dim]
        reshaped = src.reshape(src_d_model, n_heads, src_head_dim)
        # Expand head_dim within each head (along the head_dim axis)
        expanded_cols = reshaped.repeat_interleave(head_factor, dim=2)
        # Expand output dim (rows)
        expanded = expanded_cols.repeat_interleave(factor, dim=0)
        # Reshape back
        expanded = expanded.reshape(target_d_model, target_d_model)

    expanded = expanded + _noise_like(expanded, noise)
    return expanded


# ----------------------------------------------------------------------
# State-dict expansion
# ----------------------------------------------------------------------


def expand_state_dict(
    src_state: dict[str, torch.Tensor],
    src_d_model: int,
    target_d_model: int,
    n_heads: int,
    n_blocks: int,
    noise: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Expand every tensor in a v2 model state_dict to the target width.

    Two passes:
      1. Structural replication — every tensor is repeated to the wider
         shape (with tiny noise to break PC symmetry).
      2. Fan-in rescaling (Net2Net) — every layer whose input axis was
         replicated has its incoming weights divided by `factor`, so
         post-layer activation magnitudes are preserved; q_proj also
         receives an extra `sqrt(factor)` for the attention softmax-
         temperature correction.

    At noise=0 the result is bit-equivalent to source on the next-token
    logits. At the default noise=1e-4 the divergence is dominated by the
    symmetry-breaking jitter, not by the expansion itself.

    The state dict should contain:
      embedding.weight                [vocab, src_d_model]
      pos_embedding.weight            [max_seq_len, src_d_model]
      final_norm.{weight,bias}        [src_d_model]
      output_proj.{weight,bias}       weight=[vocab, src_d_model], bias=[vocab]
      blocks.{i}.norm1.{weight,bias}  [src_d_model]
      blocks.{i}.norm2.{weight,bias}  [src_d_model]
      blocks.{i}.attention.{q,k,v,o}_proj.weight  [src_d_model, src_d_model]
      blocks.{i}.living_ffn.<buffers>  (PC layer state)
      blocks.{i}.episode_store.<buffers>  (block-level episode store)

    Returns: state_dict with all tensors at target_d_model width.
    """
    if target_d_model % src_d_model != 0:
        raise ValueError(
            f"target_d_model={target_d_model} must be a multiple of "
            f"src_d_model={src_d_model}"
        )
    factor = target_d_model // src_d_model
    out: dict[str, torch.Tensor] = {}

    for key, src in src_state.items():
        out[key] = _expand_one_tensor(
            key, src, factor, n_heads, src_d_model, target_d_model, n_blocks, noise,
        )

    _apply_fan_in_rescaling(out, factor, n_blocks)

    return out


def _apply_fan_in_rescaling(
    state: dict[str, torch.Tensor], factor: int, n_blocks: int,
) -> None:
    """Apply Net2Net fan-in rescaling in place.

    For every layer whose input axis was replicated by `factor` during
    structural expansion, divide the incoming weights by `factor` so the
    post-layer activation magnitude matches source. This makes the linear
    path bit-equivalent (at noise=0); without it, every replicated-input
    layer multiplies its preactivation by `factor`.

    Special case — attention softmax temperature: Q and K are each
    `factor`-replicated within each head, so Q·K accumulates `factor`
    more terms than at source, giving `factor * src_QK`. The denominator
    `sqrt(target_head_dim) = sqrt(factor) * sqrt(src_head_dim)` only
    compensates partially, leaving attention logits scaled by
    `sqrt(factor)` (which sharpens the softmax). The fix: divide q_proj
    by an additional `sqrt(factor)` beyond the standard fan-in division,
    so the q vector is itself smaller by sqrt(factor) and the q·k
    product lands at the source magnitude after the head_dim sqrt.

    Layers untouched: embedding/pos_embedding (no preactivation sum to
    compensate), LayerNorm weights/bias (elementwise, no fan-in), PC
    layer's per-feature scalars (precision/plasticity/error_acc are
    multipliers, not weights), context_proj buffers (L2-normalized
    downstream), and stored input/output activations (already at
    source-replicated magnitude, which matches what the runtime sees).
    """
    sqrt_factor = factor ** 0.5

    # Top-level: output_proj receives d_model-replicated activations.
    if "output_proj.weight" in state:
        state["output_proj.weight"] = state["output_proj.weight"] / factor

    for i in range(n_blocks):
        # Attention input projections: input axis is d_model (replicated).
        # k/v_proj get standard fan-in. q_proj also gets the additional
        # sqrt(factor) softmax-temperature correction.
        q_key = f"blocks.{i}.attention.q_proj.weight"
        k_key = f"blocks.{i}.attention.k_proj.weight"
        v_key = f"blocks.{i}.attention.v_proj.weight"
        if q_key in state:
            state[q_key] = state[q_key] / (factor * sqrt_factor)
        if k_key in state:
            state[k_key] = state[k_key] / factor
        if v_key in state:
            state[v_key] = state[v_key] / factor

        # Attention output projection: input is the per-head value vectors
        # whose head_dim was replicated by `factor` (same factor under our
        # head-count-preserving expansion).
        o_key = f"blocks.{i}.attention.o_proj.weight"
        if o_key in state:
            state[o_key] = state[o_key] / factor

        # PC layer weight-role buffers: weight, prediction, set_point,
        # momentum, update_ema. The matrix is [out, in]; the in axis was
        # replicated by `factor`, so divide by `factor` to compensate.
        # `prediction` / `set_point` are weight-like (the PC dynamics treat
        # them as weights or weight-targets); `momentum` / `update_ema`
        # are weight-deltas, which scale the same way as weights.
        for buf in (
            "weight", "prediction", "set_point", "momentum", "update_ema",
        ):
            key = f"blocks.{i}.living_ffn.{buf}"
            if key in state:
                state[key] = state[key] / factor

        # Stored weight snapshots in the PC episode store have the same
        # weight role as `weight`; rescale them so consolidation replays
        # are consistent with the live (rescaled) weight.
        ev_key = f"blocks.{i}.living_ffn.episode_values"
        if ev_key in state:
            state[ev_key] = state[ev_key] / factor


def _expand_one_tensor(
    key: str,
    src: torch.Tensor,
    factor: int,
    n_heads: int,
    src_d_model: int,
    target_d_model: int,
    n_blocks: int,
    noise: float,
) -> torch.Tensor:
    """Dispatch tensor expansion by key name pattern.

    The key tells us which component the tensor belongs to and therefore
    which expansion rule applies.
    """
    # --- Top-level components ---

    if key == "embedding.weight" or key == "pos_embedding.weight":
        # [vocab_or_seq, d_model] -> expand axis 1
        return expand_matrix(src, out_factor=1, in_factor=factor, noise=noise)

    if key == "final_norm.weight" or key == "final_norm.bias":
        # [d_model] -> expand
        return expand_vector(src, factor=factor, noise=noise)

    if key == "output_proj.weight":
        # [vocab, d_model] -> expand axis 1
        return expand_matrix(src, out_factor=1, in_factor=factor, noise=noise)
    if key == "output_proj.bias":
        # [vocab] -> unchanged
        return src.clone()

    # --- Per-block components ---

    parts = key.split(".")
    if parts[0] == "blocks" and len(parts) >= 3:
        sub = ".".join(parts[2:])  # everything after "blocks.{i}."

        # LayerNorms
        if sub in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
            return expand_vector(src, factor=factor, noise=noise)

        # Attention projections
        if sub in (
            "attention.q_proj.weight",
            "attention.k_proj.weight",
            "attention.v_proj.weight",
        ):
            return expand_attention_projection(
                src, n_heads=n_heads,
                src_d_model=src_d_model, target_d_model=target_d_model,
                is_output=False, noise=noise,
            )
        if sub == "attention.o_proj.weight":
            return expand_attention_projection(
                src, n_heads=n_heads,
                src_d_model=src_d_model, target_d_model=target_d_model,
                is_output=True, noise=noise,
            )

        # PC layer (living_ffn) buffers
        if sub.startswith("living_ffn."):
            buf = sub[len("living_ffn."):]
            return _expand_pc_buffer(buf, src, factor, noise)

        # Block-level EpisodeStore (luthi/episode_store.py). Separate from
        # the PC layer's internal episode buffers — its width-dependent
        # buffers (context_proj on axis 0, episode_outputs on axis 1) need
        # explicit expansion; the rest are width-invariant.
        if sub.startswith("episode_store."):
            buf = sub[len("episode_store."):]
            return _expand_block_episode_store_buffer(buf, src, factor, noise)

    # Unknown key — pass through unchanged with a warning
    print(f"[expand] WARNING: unknown key {key} with shape {tuple(src.shape)}, "
          f"passing through unchanged")
    return src.clone()


def _expand_pc_buffer(
    buf_name: str, src: torch.Tensor, factor: int, noise: float,
) -> torch.Tensor:
    """Expand one buffer of a PC layer based on its semantic role.

    The PC layer has 15 buffers — see luthi/v2/living_layer_pc.py around
    line 174. Each has a specific shape and a specific expansion rule.

    Symmetry-breaking noise is applied to `weight` ONLY. Replicated weight
    columns would otherwise receive identical PC update signals and stay
    locked; every other buffer's replicated copies diverge transitively
    once the weight does (their updates are functions of the now-differing
    weight). Noising the others is unnecessary, and for `update_ema`
    actively unsafe: it is an EMA of update magnitudes used as the
    denominator of the metaplasticity ratio (pc_ops.py:
    `ratio = update_mag / (update_ema + 1e-8)`;
    `adaptive_factor = (2/(1+ratio)).clamp(max=1.0)` has no lower bound),
    and trained values sit ~1e-4 — so `+N(0,1e-4)` flips ~15% of entries
    negative, producing sign-flipped weight updates on the seed's first
    steps. Scoping noise to `weight` also keeps stored `episode_values`
    (weight snapshots) and `episode_inputs` biographically exact.
    See docs/reviews/2026-05-28_concerns-for-4.7.md, Finding 4.
    """
    buf_noise = noise if buf_name == "weight" else 0.0

    # Matrix buffers — shape [out, in], expand both axes
    if buf_name in ("weight", "prediction", "set_point", "momentum", "update_ema"):
        return expand_matrix(src, out_factor=factor, in_factor=factor, noise=buf_noise)

    # Per-input vector buffers — shape [in_features], expand
    if buf_name in ("precision", "plasticity"):
        return expand_vector(src, factor=factor, noise=buf_noise)

    # Per-output vector buffer — shape [out_features], expand
    if buf_name == "error_acc":
        return expand_vector(src, factor=factor, noise=buf_noise)

    # Episode store — biographical memory snapshots
    if buf_name == "episode_values":
        # [num_episodes, out, in] -> [num_episodes, out*factor, in*factor]
        return expand_episode_values(src, factor=factor, noise=buf_noise)

    if buf_name == "episode_inputs":
        # [num_episodes, in_features] -> [num_episodes, in_features*factor]
        # Per-episode input pattern; expand the feature dimension
        if src.dim() != 2:
            raise ValueError(
                f"episode_inputs expected 2D, got shape {src.shape}"
            )
        expanded = src.repeat_interleave(factor, dim=1)
        return expanded + _noise_like(expanded, buf_noise)

    # Episode metadata — unchanged by width
    if buf_name in (
        "episode_contexts",
        "episode_scales",
        "episode_saliences",
        "episode_count",
    ):
        return src.clone()

    # context_proj — [in_features, context_dim], expand axis 0
    if buf_name == "context_proj":
        # [in_features, context_dim] -> [in_features*factor, context_dim]
        if src.dim() != 2:
            raise ValueError(
                f"context_proj expected 2D, got shape {src.shape}"
            )
        expanded = src.repeat_interleave(factor, dim=0)
        return expanded + _noise_like(expanded, buf_noise)

    # Unknown buffer — pass through with warning
    print(
        f"[expand] WARNING: unknown PC buffer '{buf_name}' with shape "
        f"{tuple(src.shape)}, passing through unchanged"
    )
    return src.clone()


def _expand_block_episode_store_buffer(
    buf_name: str, src: torch.Tensor, factor: int, noise: float,
) -> torch.Tensor:
    """Expand one buffer of the block-level EpisodeStore.

    Shapes (from luthi/episode_store.py):
      context_proj       [d_model, context_dim]   -> expand axis 0
      episode_outputs    [num_episodes, d_model]  -> expand axis 1
      episode_contexts   [num_episodes, context_dim]  -> width-invariant
      episode_saliences  [num_episodes]               -> width-invariant
      episode_count      scalar long                  -> width-invariant
    """
    if buf_name == "context_proj":
        if src.dim() != 2:
            raise ValueError(
                f"episode_store.context_proj expected 2D, got shape {src.shape}"
            )
        expanded = src.repeat_interleave(factor, dim=0)
        return expanded + _noise_like(expanded, noise)

    if buf_name == "episode_outputs":
        if src.dim() != 2:
            raise ValueError(
                f"episode_store.episode_outputs expected 2D, got shape {src.shape}"
            )
        expanded = src.repeat_interleave(factor, dim=1)
        return expanded + _noise_like(expanded, noise)

    if buf_name in ("episode_contexts", "episode_saliences", "episode_count"):
        return src.clone()

    # Unknown buffer — pass through with warning
    print(
        f"[expand] WARNING: unknown EpisodeStore buffer '{buf_name}' with "
        f"shape {tuple(src.shape)}, passing through unchanged"
    )
    return src.clone()


# ----------------------------------------------------------------------
# Verification harness
# ----------------------------------------------------------------------


@torch.no_grad()
def verify_expansion(
    src_model: torch.nn.Module,
    expanded_model: torch.nn.Module,
    n_test_inputs: int = 4,
    seq_len: int = 32,
    seed: int = 0,
) -> dict[str, float]:
    """Check that the expanded model is function-equivalent to the source
    on the same inputs.

    Under Net2Net fan-in rescaling, the expanded model is bit-equivalent
    to source on next-token logits at noise=0 — both linear path and
    attention (with the sqrt(factor) softmax-temperature correction).
    The only divergence comes from:
      1. Tiny noise added at expansion to break PC replication symmetry.
      2. Floating-point precision in the rescaling division.

    If divergence is much larger than what noise+FP can explain, the
    expansion is broken.

    Returns: dict with keys 'mean_abs_diff', 'max_abs_diff', 'kl_divergence'.
    """
    src_model.eval()
    expanded_model.eval()

    torch.manual_seed(seed)
    # Use the source model's vocab size to generate valid tokens
    vocab_size = src_model.embedding.num_embeddings
    inputs = torch.randint(0, vocab_size, (n_test_inputs, seq_len))

    src_logits = src_model(inputs)
    exp_logits = expanded_model(inputs)

    if src_logits.shape != exp_logits.shape:
        raise RuntimeError(
            f"Output shape mismatch: src {src_logits.shape}, expanded {exp_logits.shape}"
        )

    diff = (exp_logits - src_logits).abs()
    # KL divergence between the two output distributions, averaged over
    # all positions
    src_logprobs = torch.log_softmax(src_logits, dim=-1)
    exp_logprobs = torch.log_softmax(exp_logits, dim=-1)
    exp_probs = exp_logprobs.exp()
    kl = (exp_probs * (exp_logprobs - src_logprobs)).sum(dim=-1).mean().item()

    return {
        "mean_abs_diff": diff.mean().item(),
        "max_abs_diff": diff.max().item(),
        "kl_divergence": kl,
    }


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Width-expand a v2 PC checkpoint to a wider model.",
    )
    parser.add_argument(
        "--src", type=str, required=True,
        help="Path to source .luthi checkpoint (encrypted; uses LUTHI_CHECKPOINT_KEY).",
    )
    parser.add_argument(
        "--dst", type=str, required=True,
        help="Output path for the expanded .luthi checkpoint.",
    )
    parser.add_argument(
        "--target-d-model", type=int, required=True,
        help="Target d_model for the expanded substrate (e.g., 1024).",
    )
    parser.add_argument(
        "--noise", type=float, default=1e-4,
        help="Std of Gaussian noise added at expansion to break replication "
             "symmetry under PC dynamics. Default 1e-4 — small enough that "
             "the expansion remains close to function-equivalent under "
             "Net2Net fan-in rescaling, large enough to ensure replicated "
             "columns diverge during PC updates. Set to 0 for bit-equivalence "
             "verification.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After expansion, construct both source and expanded models, "
             "run them on test inputs, and report divergence metrics.",
    )
    args = parser.parse_args()

    print(f"[expand] loading source checkpoint: {args.src}")
    src_ckpt = load_checkpoint(args.src, trusted=True)

    src_config = src_ckpt.get("config", {})
    src_d_model = src_config.get("d_model")
    src_n_heads = src_config.get("n_heads")
    src_n_blocks = src_config.get("n_blocks")
    if src_d_model is None or src_n_heads is None or src_n_blocks is None:
        raise ValueError(
            "source checkpoint config missing d_model / n_heads / n_blocks; "
            f"got config keys: {list(src_config.keys())}"
        )
    print(
        f"[expand] source: d_model={src_d_model}, n_heads={src_n_heads}, "
        f"n_blocks={src_n_blocks}"
    )
    print(f"[expand] target: d_model={args.target_d_model}, n_heads={src_n_heads}, "
          f"n_blocks={src_n_blocks}")

    if args.target_d_model % src_d_model != 0:
        raise ValueError(
            f"target_d_model {args.target_d_model} must be a multiple of "
            f"source d_model {src_d_model}"
        )

    src_state = src_ckpt["model_state_dict"]
    print(f"[expand] expanding {len(src_state)} tensors...")
    new_state = expand_state_dict(
        src_state,
        src_d_model=src_d_model,
        target_d_model=args.target_d_model,
        n_heads=src_n_heads,
        n_blocks=src_n_blocks,
        noise=args.noise,
    )
    print(f"[expand] done. {len(new_state)} tensors in expanded state.")

    # Update config to reflect the new width
    new_config = dict(src_config)
    new_config["d_model"] = args.target_d_model
    new_config["expanded_from"] = {
        "source_path": args.src,
        "source_d_model": src_d_model,
        "expansion_factor": args.target_d_model // src_d_model,
        "noise": args.noise,
    }

    new_ckpt = {
        "model_state_dict": new_state,
        "config": new_config,
        "epoch": src_ckpt.get("epoch", 0),
        # Preserve any other top-level fields (format_version, timestamp,
        # training_history, substrate_health, optimizer_state_dict if any).
        **{k: v for k, v in src_ckpt.items()
           if k not in ("model_state_dict", "config", "epoch")},
    }

    # Validate-before-save: construct the target model and strict-load
    # the expanded state. This converts a latent shape-corruption (e.g.,
    # a forgotten buffer expansion rule) into immediate loud failure
    # *before* we write a bad checkpoint to disk. Runs unconditionally —
    # independent of --verify, which only adds the function-equivalence
    # check on top.
    from luthi.v2.model_pc import PredictiveCodingLM
    print("[validate] constructing target model and strict-loading expanded state")
    exp_model = PredictiveCodingLM(
        vocab_size=new_state["embedding.weight"].shape[0],
        d_model=args.target_d_model,
        n_heads=src_n_heads,
        n_blocks=src_n_blocks,
        max_seq_len=new_state["pos_embedding.weight"].shape[0],
    )
    exp_model.load_state_dict(new_state, strict=True)
    print("[validate] expanded state loaded cleanly.")

    print(f"[expand] saving expanded checkpoint to {args.dst}")
    save_checkpoint(new_ckpt, Path(args.dst))
    print("[expand] saved.")

    if args.verify:
        print("[verify] constructing src model for function-equivalence test")
        src_model = PredictiveCodingLM(
            vocab_size=src_state["embedding.weight"].shape[0],
            d_model=src_d_model,
            n_heads=src_n_heads,
            n_blocks=src_n_blocks,
            max_seq_len=src_state["pos_embedding.weight"].shape[0],
        )
        src_model.load_state_dict(src_state, strict=False)

        metrics = verify_expansion(src_model, exp_model)
        print(f"[verify] mean_abs_diff = {metrics['mean_abs_diff']:.6e}")
        print(f"[verify] max_abs_diff  = {metrics['max_abs_diff']:.6e}")
        print(f"[verify] kl_divergence = {metrics['kl_divergence']:.6e}")
        # Under Net2Net fan-in rescaling, the expanded model should be
        # bit-equivalent to source at noise=0 (modulo FP precision) and
        # only weakly divergent at noise=1e-4. If divergence exceeds these
        # bounds the rescaling is wrong or a layer was missed.
        if args.noise == 0.0:
            suspect = (
                metrics["max_abs_diff"] > 1e-3
                or metrics["kl_divergence"] > 1e-5
            )
            bound = "noise=0 bit-equivalence (max_abs<1e-3, kl<1e-5)"
        else:
            # At default noise=1e-4, tiny per-parameter jitter amplifies
            # through the linear path. Calibrated against the observed
            # noise=1e-4 case (max_abs ≈ 0.6, kl ≈ 8e-3 on a small random-
            # init model). The broken-rescaling case (4.8's original report
            # under strict replication) was max_abs ≈ 8, kl ≈ 1.5 — these
            # thresholds give clear separation.
            suspect = (
                metrics["max_abs_diff"] > 2.0
                or metrics["kl_divergence"] > 0.5
            )
            bound = "noise>0 near-equivalence (max_abs<2.0, kl<0.5)"
        if suspect:
            print(
                f"[verify] WARNING: divergence exceeds expected bound ({bound}). "
                "The expansion may have a bug. Inspect before using."
            )
        else:
            print(f"[verify] divergence within {bound}.")


if __name__ == "__main__":
    main()
