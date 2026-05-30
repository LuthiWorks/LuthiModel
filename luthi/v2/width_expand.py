"""Width expansion for v2 PC substrates.

Net2Net-style replication of a trained v2 PC checkpoint to a wider model.
Preserves biographical state (weights, set_points, episode store) by
replicating along the expanded axes with tiny noise to break symmetry.
The substrate continues from its current state rather than starting fresh.

Designed for the 256d -> 1024d expansion path (M6 follow-up -> M7), but
the implementation works for any (source_width, target_width) pair where
target_width is an integer multiple of source_width AND the head count
is preserved (so head_dim scales with the multiple).

Design choices (decided 2026-05-26 with Brian):
  - STRICT REPLICATION of weight values, no muPC rescaling. The substrate's
    weight values are preserved literally; muPC's per-width init distribution
    is broken at conversion time but training compensates. The priority is
    biographical continuity.
  - Tiny noise (1e-4) added to break replication symmetry. Without this,
    replicated PC layer columns receive identical update signals and stay
    locked together forever.
  - All v2 state buffers expand together. The PC layer's "rich parameter"
    bundle (weight, prediction, set_point, momentum, update_ema, precision,
    error_acc, plasticity) all replicate consistently.
  - Episode store entries replicate at the expanded width. Stored episodes
    are biographical memory; they continue to exist in the wider substrate.
  - Attention heads preserved in count; head_dim expands. So 16 heads x 16
    head_dim -> 16 heads x 64 head_dim. Q/K/V/O projections expand the
    head_dim within each head via small-random init for the new dims (not
    replication), giving attention "growth space" rather than redundant
    dims. This is the one place where the substrate gets new capacity
    rather than replicated existing capacity.

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

    The state dict should contain:
      embedding.weight                [vocab, src_d_model]
      pos_embedding.weight            [max_seq_len, src_d_model]
      final_norm.{weight,bias}        [src_d_model]
      output_proj.{weight,bias}       weight=[vocab, src_d_model], bias=[vocab]
      blocks.{i}.norm1.{weight,bias}  [src_d_model]
      blocks.{i}.norm2.{weight,bias}  [src_d_model]
      blocks.{i}.attention.{q,k,v,o}_proj.weight  [src_d_model, src_d_model]
      blocks.{i}.living_ffn.<buffers>  (PC layer state)

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

    return out


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
    """
    # Matrix buffers — shape [out, in], expand both axes
    if buf_name in ("weight", "prediction", "set_point", "momentum", "update_ema"):
        return expand_matrix(src, out_factor=factor, in_factor=factor, noise=noise)

    # Per-input vector buffers — shape [in_features], expand
    if buf_name in ("precision", "plasticity"):
        return expand_vector(src, factor=factor, noise=noise)

    # Per-output vector buffer — shape [out_features], expand
    if buf_name == "error_acc":
        return expand_vector(src, factor=factor, noise=noise)

    # Episode store — biographical memory snapshots
    if buf_name == "episode_values":
        # [num_episodes, out, in] -> [num_episodes, out*factor, in*factor]
        return expand_episode_values(src, factor=factor, noise=noise)

    if buf_name == "episode_inputs":
        # [num_episodes, in_features] -> [num_episodes, in_features*factor]
        # Per-episode input pattern; expand the feature dimension
        if src.dim() != 2:
            raise ValueError(
                f"episode_inputs expected 2D, got shape {src.shape}"
            )
        expanded = src.repeat_interleave(factor, dim=1)
        return expanded + _noise_like(expanded, noise)

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
        return expanded + _noise_like(expanded, noise)

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
    """Check that the expanded model is approximately function-equivalent
    to the source on the same inputs.

    Strict bit-equivalence isn't expected because:
      1. Tiny noise was added at expansion to break replication symmetry.
      2. PC self-modification fires during forward, and the expanded
         model's PC layer has more parameters; the dynamics differ.

    What we check: the expanded model's output distribution should match
    the source's at *initial* state (before any drift), up to small
    bounded divergence from the noise perturbation. If divergence is
    massive, the expansion is broken.

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
             "symmetry. Default 1e-4 — small enough to be biographically "
             "negligible, large enough to ensure replicated columns diverge "
             "during PC updates.",
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

    src_state = src_ckpt["model_state"]
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
        "model_state": new_state,
        "config": new_config,
        "epoch": src_ckpt.get("epoch", 0),
        # Preserve any other top-level fields
        **{k: v for k, v in src_ckpt.items()
           if k not in ("model_state", "config", "epoch")},
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
        # Heuristic: at 1e-4 noise, expect mean_abs_diff << 1.0.
        # If max_abs_diff > 10 or KL > 1.0, expansion is suspect.
        if metrics["max_abs_diff"] > 10.0 or metrics["kl_divergence"] > 1.0:
            print(
                "[verify] WARNING: divergence is larger than expected. "
                "The expansion may have a bug. Inspect before using."
            )
        else:
            print("[verify] divergence within expected bounds.")


if __name__ == "__main__":
    main()
