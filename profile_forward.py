"""Profile the forward pass to identify optimization targets.

Measures time spent in each phase of the forward pass:
- Attention (ScalarAttention)
- Living FFN self-modification (Hebbian, metaplasticity, homeostatic)
- Episode store (context recall, blending, storage)
- Error-directed learning (post-backward)

Run:
    python profile_forward.py [--d_model 1024] [--n_blocks 2] [--spiking]

Reports per-component timing as both absolute (ms) and percentage.
"""

import argparse
import time
import torch
import torch.nn.functional as F
from contextlib import contextmanager


@contextmanager
def timer(name, results):
    """Context manager that records elapsed time in milliseconds."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    results.setdefault(name, []).append(elapsed_ms)


def profile_living_layer(layer, x, results, prefix=""):
    """Profile a single living layer's forward pass with fine-grained timing.

    This replaces the normal forward() to measure each phase separately.
    """
    input_shape = x.shape
    if x.dim() == 3:
        batch, seq_len, _ = x.shape
        x_flat = x.reshape(-1, layer.in_features)
    else:
        x_flat = x

    layer._cached_input = x_flat.detach()

    # Episodic recall
    with timer(f"{prefix}episode_recall", results):
        with torch.no_grad():
            context = layer._compute_context(x_flat)
            episode_delta = layer._recall_episode(context)

    # Weight snapshot + linear computation
    with timer(f"{prefix}linear_compute", results):
        weight_snapshot = layer.weight.clone()
        if episode_delta is not None:
            weight_snapshot = weight_snapshot + episode_delta
        output = x_flat @ weight_snapshot.T

    # Self-modification (the big block)
    with timer(f"{prefix}self_modify", results):
        with torch.no_grad():
            exc = layer._excitability_factor()
            salience_per_dim = output.abs().mean(dim=0)
            salience_scalar = salience_per_dim.mean().item()
            input_mag = x_flat.abs().mean(dim=0)
            layer.input_avg_mag.mul_(layer.input_mag_decay).add_(
                input_mag, alpha=1.0 - layer.input_mag_decay
            )
            normalized_input = x_flat / layer.input_avg_mag.clamp(min=0.01)
            weight_norm = layer.weight.abs() / layer.weight.abs().mean().clamp(min=0.01)
            per_weight_salience = weight_norm * x_flat.abs().mean(dim=0).unsqueeze(0)
            input_signal = normalized_input.mean(dim=0).unsqueeze(0)
            hebb_update = (
                input_signal * per_weight_salience * exc * layer.plasticity
                * layer.hebb_rate
            )

            # Metaplasticity
            update_mag = hebb_update.abs()
            ratio = update_mag / (layer.update_ema + 1e-8)
            adaptive_factor = (2.0 / (1.0 + ratio)).clamp(max=1.0)
            gate = layer._get_update_gate()

            if gate is not None:
                layer.update_ema.add_(
                    (update_mag - layer.update_ema)
                    * (1.0 - layer.update_ema_decay) * gate
                )
            else:
                layer.update_ema.mul_(layer.update_ema_decay).add_(
                    update_mag, alpha=1.0 - layer.update_ema_decay
                )

            if not layer.training:
                adaptive_factor = adaptive_factor * layer.eval_hebb_fraction

            if gate is not None:
                layer.weight.add_(hebb_update * adaptive_factor * gate)
            else:
                layer.weight.add_(hebb_update * adaptive_factor)

            if gate is not None:
                layer.momentum.add_(
                    (hebb_update - layer.momentum)
                    * (1.0 - layer.momentum_decay) * gate
                )
            else:
                layer.momentum.mul_(layer.momentum_decay).add_(
                    hebb_update, alpha=1.0 - layer.momentum_decay
                )

            homeostatic_force = layer.set_point - layer.weight
            layer.weight.add_(homeostatic_force, alpha=layer.homeostatic_decay)
            set_point_delta = layer.weight - layer.set_point
            layer.set_point.add_(set_point_delta, alpha=layer.set_point_adapt_rate)

            salience_broadcast = salience_per_dim.unsqueeze(1).expand_as(
                layer.excitability_acc
            )
            exc_update = torch.where(
                salience_broadcast > layer.salience_threshold,
                torch.tensor(0.01, device=x.device, dtype=x.dtype),
                torch.tensor(-0.005, device=x.device, dtype=x.dtype),
            )
            layer.excitability_acc.add_(exc_update)

    # Episode storage
    with timer(f"{prefix}episode_store", results):
        with torch.no_grad():
            layer._store_episode(context, salience_scalar)

    if len(input_shape) == 3:
        output = output.reshape(batch, seq_len, layer.out_features)
    return output


def main():
    parser = argparse.ArgumentParser(description="Profile LuthiModel forward pass")
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations (not timed)")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Timed iterations to average over")
    parser.add_argument("--spiking", action="store_true", default=False)
    parser.add_argument("--vocab_size", type=int, default=4096)
    args = parser.parse_args()

    # Device selection (same as train.py)
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: d_model={args.d_model}, n_blocks={args.n_blocks}, "
          f"batch={args.batch_size}, seq_len={args.seq_len}")
    print(f"Spiking: {args.spiking}")

    # Build model
    if args.spiking:
        from luthi.model_spiking import SpikingLuthiLM
        model = SpikingLuthiLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            max_seq_len=args.seq_len,
        ).to(device)
    else:
        from luthi.model import LuthiLM
        model = LuthiLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            max_seq_len=args.seq_len,
        ).to(device)

    param_counts = model.total_parameters()
    print(f"Trainable params:  {param_counts['trainable']:,}")
    print(f"Living buffers:    {param_counts['living_buffers']:,}")

    # Dummy data
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # ---- Warmup ----
    print(f"\nWarming up ({args.warmup} iterations)...")
    model.train()
    for _ in range(args.warmup):
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        model.apply_living_errors(expect_grad=True)
        optimizer.step()

    # ---- Profiled iterations ----
    print(f"Profiling ({args.iterations} iterations)...")
    results = {}

    for i in range(args.iterations):
        optimizer.zero_grad()

        # Full forward pass
        with timer("total_forward", results):
            logits = model(x)

        # Loss + backward
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        with timer("backward", results):
            loss.backward()

        # Error-directed living learning
        with timer("apply_living_errors", results):
            model.apply_living_errors(expect_grad=True)

        # Optimizer step
        with timer("optimizer_step", results):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    # ---- Now profile forward pass internals ----
    print(f"Profiling forward pass internals ({args.iterations} iterations)...")
    internal_results = {}

    for i in range(args.iterations):
        # Embedding
        with timer("embedding", internal_results):
            positions = torch.arange(args.seq_len, device=x.device).unsqueeze(0)
            h = model.embedding(x) + model.pos_embedding(positions)

        # Per-block profiling
        for block_idx, block in enumerate(model.blocks):
            prefix = f"block{block_idx}_"

            # Attention (norm + attention + residual)
            with timer(f"{prefix}attention", internal_results):
                attn_out = block.attention(block.norm1(h), causal=True)
                h_after_attn = h + attn_out

            # Living FFN (norm + living layer + residual)
            ffn_in = block.norm2(h_after_attn)
            with timer(f"{prefix}living_ffn", internal_results):
                ffn_out = block.living_ffn(ffn_in)
            h_after_ffn = h_after_attn + ffn_out

            # Episode store
            with timer(f"{prefix}episode_store", internal_results):
                h = block.episode_store(h, h_after_ffn)

        # Head
        with timer("head", internal_results):
            _ = model.output_proj(model.final_norm(h))

    # ---- Report ----
    print("\n" + "=" * 65)
    print("TRAINING LOOP BREAKDOWN")
    print("=" * 65)
    total_loop = sum(
        sum(v) / len(v) for k, v in results.items()
    )
    for name in ["total_forward", "backward", "apply_living_errors", "optimizer_step"]:
        vals = results[name]
        avg = sum(vals) / len(vals)
        pct = avg / total_loop * 100
        print(f"  {name:30s}  {avg:8.2f} ms  ({pct:5.1f}%)")
    print(f"  {'TOTAL':30s}  {total_loop:8.2f} ms")

    print("\n" + "=" * 65)
    print("FORWARD PASS INTERNALS (per component)")
    print("=" * 65)
    total_internal = sum(
        sum(v) / len(v) for k, v in internal_results.items()
    )
    # Group by category
    categories = {}
    for name, vals in internal_results.items():
        avg = sum(vals) / len(vals)
        if "attention" in name:
            categories.setdefault("attention", 0)
            categories["attention"] += avg
        elif "living_ffn" in name:
            categories.setdefault("living_ffn", 0)
            categories["living_ffn"] += avg
        elif "episode_store" in name:
            categories.setdefault("episode_store", 0)
            categories["episode_store"] += avg
        elif name == "embedding":
            categories["embedding"] = avg
        elif name == "head":
            categories["head"] = avg

    for cat in ["embedding", "attention", "living_ffn", "episode_store", "head"]:
        if cat in categories:
            avg = categories[cat]
            pct = avg / total_internal * 100
            print(f"  {cat:30s}  {avg:8.2f} ms  ({pct:5.1f}%)")
    print(f"  {'TOTAL':30s}  {total_internal:8.2f} ms")

    # Per-block detail
    print("\n" + "=" * 65)
    print("PER-BLOCK DETAIL")
    print("=" * 65)
    for block_idx in range(args.n_blocks):
        prefix = f"block{block_idx}_"
        print(f"\n  Block {block_idx}:")
        for component in ["attention", "living_ffn", "episode_store"]:
            key = f"{prefix}{component}"
            if key in internal_results:
                vals = internal_results[key]
                avg = sum(vals) / len(vals)
                print(f"    {component:26s}  {avg:8.2f} ms")


if __name__ == "__main__":
    main()
