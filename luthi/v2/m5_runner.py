"""M5: v2 PC vs DeadLM head-to-head training harness.

Single script that trains either `PredictiveCodingLM` (v2) or `DeadLM`
(vanilla transformer + episode store baseline) at matched config — same
d_model, n_blocks, n_heads, ffn_expansion, seq_len, batch, stride,
epochs, seed, optimizer, data, LR schedule. Per the V2 plan's M5 spec
post-strategic-shift, this is the falsification-criteria comparison
that decides whether v2 (the now-primary substrate) actually beats a
vanilla baseline.

Usage::

    python -m luthi.v2.m5_runner --arch v2 --seed 42 ...
    python -m luthi.v2.m5_runner --arch dead --seed 42 ...

Falsification criteria (per docs/V2_IMPLEMENTATION_PLAN.md M5):
- v2 convergence penalty worse than DeadLM by ≥20% at matched scale
- v2 cascade fails at depths where DeadLM succeeds
- v2 attractor dynamics indistinguishable from random control (separate
  attractor-dynamics script handles this)
- M4 STOP GATE already passed (consolidation has measurable effect)
- VRAM exceeded at equivalent parameter count

If any falsify, v2 needs revisiting; v1 ablation pipeline revives.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from luthi.data import (
    CharDataset,
    CharTokenizer,
    load_corpus,
    load_corpus_as_tensor,
    load_corpus_sample,
    load_file_list,
)
from luthi.tokenizer import BPETokenizer
from luthi.v2 import PredictiveCodingLM
from luthi.model import DeadLM


def _build_model(args, vocab_size: int, device: torch.device) -> torch.nn.Module:
    """Construct either v2 PredictiveCodingLM or DeadLM at matched config.

    Notes on matched-config fairness:
    - d_model, n_blocks, n_heads, ffn_expansion, max_seq_len all transfer
      identically to both architectures.
    - v2 turns consolidation ON for M5 (the M4 STOP GATE result is part
      of what's being validated end-to-end here).
    - DeadLM doesn't have living state, so PC-specific hyperparameters
      (pc_rate, pred_learning_rate, etc.) are silently ignored.
    """
    if args.arch == "v2":
        model = PredictiveCodingLM(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            n_heads=args.n_heads,
            ffn_expansion=args.ffn_expansion,
            max_seq_len=args.seq_len,
            pc_rate=args.pc_rate,
            pred_learning_rate=args.pred_learning_rate,
            homeostatic_decay=args.homeostatic_decay,
            set_point_adapt_rate=args.set_point_adapt_rate,
            compressed_episodes=args.compressed_episodes,
            consolidation_enabled=not args.no_consolidation,
            consolidation_style=args.consolidation_style,
            consolidation_attractor_passes=args.consolidation_attractor_passes,
            mu_pc_enabled=args.mu_pc_enabled,
            mu_pc_exponent=args.mu_pc_exponent,
        ).to(device)
        # The PC layer's sparse-gating and iPC knobs aren't wired into
        # PredictiveCodingLM's constructor (they're per-layer optimizations,
        # not model-level architectural choices). Set them post-construction
        # on every living FFN. This pattern matches the existing per-layer
        # `consolidation_*` knobs as the only way to vary them via CLI
        # without bloating the model constructor signature.
        for block in model.blocks:
            living = block.living_ffn
            living.sparse_threshold = float(args.sparse_threshold)
            living.sparse_warmup_steps = int(args.sparse_warmup_steps)
            if args.inference_steps_per_forward < 1:
                raise ValueError(
                    f"--inference-steps-per-forward must be >= 1, got "
                    f"{args.inference_steps_per_forward}"
                )
            living.inference_steps_per_forward = int(
                args.inference_steps_per_forward
            )
        return model
    elif args.arch == "dead":
        return DeadLM(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            n_heads=args.n_heads,
            ffn_expansion=args.ffn_expansion,
            max_seq_len=args.seq_len,
        ).to(device)
    else:
        raise ValueError(f"Unknown --arch {args.arch!r}; expected 'v2' or 'dead'")


def _compute_lr(args, epoch: int) -> float:
    """Cosine LR schedule with linear warmup, same as train_pc.py."""
    if args.lr_schedule == "constant":
        return args.lr
    if epoch < args.lr_warmup_epochs:
        return args.lr * (epoch + 1) / max(1, args.lr_warmup_epochs)
    progress = (epoch - args.lr_warmup_epochs) / max(
        1, args.epochs - args.lr_warmup_epochs
    )
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every_batches: int = 0,
    epoch_label: str = "",
    total_batches_estimate: int | None = None,
) -> tuple[float, int]:
    """Train one epoch.

    When `log_every_batches > 0`, prints a per-batch progress line at
    that cadence with: batch index, rolling-100-batch mean loss, current
    batch loss, wall-clock elapsed, ETA, per-block NFF/Frobenius/precision
    diagnostics for v2 (cheap; reads buffer state directly without extra
    forward passes), and running NaN count. Lines are flushed
    immediately so a tail -F observer sees them as they happen.

    `epoch_label` is prepended to each progress line (e.g., "[ep 1/40]").
    `total_batches_estimate` is used for the ETA calculation; if None,
    ETA is omitted from the line.
    """
    import collections
    import time

    model.train()
    total = 0.0
    n_batches = 0
    nan_count = 0

    # Rolling window of the last 100 batch losses for a smoothed
    # "current" loss. 100 was picked because it's large enough to
    # smooth single-batch noise but small enough to track real
    # convergence shifts (vs cumulative-average, which is dominated
    # by early batches once you're past a few thousand).
    rolling = collections.deque(maxlen=100)
    t0 = time.time()
    last_log_time = t0

    def _quick_diagnostics() -> dict:
        """Cheap-to-compute v2 buffer summaries. Reads state directly;
        no extra forward passes. Returns empty dict for DeadLM (which
        has no per-block living state)."""
        out = {}
        if not hasattr(model, "blocks"):
            return out
        pred_norms = []
        precision_means = []
        error_acc_means = []
        for block in model.blocks:
            living = getattr(block, "living_ffn", None)
            if living is None or not hasattr(living, "prediction"):
                continue
            pred_norms.append(living.prediction.norm().item())
            if hasattr(living, "precision"):
                precision_means.append(living.precision.mean().item())
            if hasattr(living, "error_acc"):
                error_acc_means.append(living.error_acc.mean().item())
        if pred_norms:
            out["pred_frob"] = sum(pred_norms) / len(pred_norms)
        if precision_means:
            out["prec"] = sum(precision_means) / len(precision_means)
        if error_acc_means:
            out["err_acc"] = sum(error_acc_means) / len(error_acc_means)
        return out

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            n_batches += 1  # count for ETA pacing even when skipped
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()
        if hasattr(model, "clear_forward_cache"):
            model.clear_forward_cache()
        loss_val = loss.item()
        total += loss_val
        rolling.append(loss_val)
        n_batches += 1

        # Per-batch progress line.
        if log_every_batches > 0 and n_batches % log_every_batches == 0:
            now = time.time()
            elapsed = now - t0
            rate = n_batches / max(elapsed, 1e-6)  # batches/sec
            mean_recent = sum(rolling) / len(rolling)
            if total_batches_estimate and total_batches_estimate > n_batches:
                remaining = total_batches_estimate - n_batches
                eta_sec = remaining / max(rate, 1e-6)
                eta_str = f" eta={eta_sec/3600:.2f}h"
                pct = 100.0 * n_batches / total_batches_estimate
                progress_str = f"[{n_batches:>7d}/{total_batches_estimate} {pct:>5.1f}%]"
            else:
                eta_str = ""
                progress_str = f"[{n_batches:>7d}]"
            diag = _quick_diagnostics()
            diag_str = " ".join(f"{k}={v:.4f}" for k, v in diag.items())
            print(
                f"{epoch_label}{progress_str} "
                f"loss(roll100)={mean_recent:.4f} loss(cur)={loss_val:.4f} "
                f"rate={rate:.2f}b/s elapsed={elapsed/3600:.2f}h{eta_str} "
                f"nans={nan_count} {diag_str}",
                flush=True,
            )
            last_log_time = now

    return (total / max(n_batches, 1)), nan_count


@torch.no_grad()
def _measure_nff(model: torch.nn.Module, x: torch.Tensor) -> float:
    """Two consecutive identical-input forward passes. >0 = non-feedforward."""
    was_training = model.training
    model.eval()
    try:
        out1 = model(x)
        out2 = model(x)
        return (out2 - out1).abs().mean().item()
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def _measure_living_diagnostics(model: torch.nn.Module) -> dict:
    """Per-epoch v2 diagnostics flagged by 4.6's 2026-05-12 audit:
    prediction Frobenius norm (mean over blocks) and precision distribution
    (mean / min / max over blocks). No-op for DeadLM (no PC layer)."""
    if not hasattr(model, "blocks"):
        return {}
    pred_norms = []
    precision_means = []
    precision_mins = []
    precision_maxes = []
    for block in model.blocks:
        living = getattr(block, "living_ffn", None)
        if living is None or not hasattr(living, "prediction"):
            continue
        pred_norms.append(living.prediction.norm().item())
        if hasattr(living, "precision"):
            precision_means.append(living.precision.mean().item())
            precision_mins.append(living.precision.min().item())
            precision_maxes.append(living.precision.max().item())
    diag = {}
    if pred_norms:
        diag["prediction_frob_mean"] = sum(pred_norms) / len(pred_norms)
    if precision_means:
        diag["precision_mean"] = sum(precision_means) / len(precision_means)
        diag["precision_min"] = min(precision_mins)
        diag["precision_max"] = max(precision_maxes)
    return diag


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n_batches = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        total += loss.item()
        n_batches += 1
    return total / max(n_batches, 1)


def _save_checkpoint(model, path: Path, epoch: int, config: dict) -> None:
    """Encrypted checkpoint when LUTHI_CHECKPOINT_KEY is set, plaintext
    with loud warning otherwise — same policy as train_pc._save_checkpoint_v2.
    """
    try:
        from luthi.checkpoint import build_checkpoint, save_checkpoint
        ckpt = build_checkpoint(model, epoch=epoch, config=config)
        encrypted_path = save_checkpoint(ckpt, path)
        print(f"[checkpoint] encrypted save: {encrypted_path}")
    except ValueError as e:
        plain_path = path.with_suffix(".pt")
        torch.save(model.state_dict(), plain_path)
        print(
            f"[checkpoint] WARNING: no LUTHI_CHECKPOINT_KEY set; saved "
            f"plaintext to {plain_path}. Reason: {e}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="M5: v2 PC vs DeadLM head-to-head training harness.",
    )
    parser.add_argument(
        "--arch", type=str, required=True, choices=["v2", "dead"],
        help="Architecture: 'v2' = PredictiveCodingLM, 'dead' = DeadLM baseline.",
    )
    parser.add_argument("--data_dir", type=str, default="corpus_build/gutenberg_100")
    parser.add_argument("--file-list", dest="file_list", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30,
                        help="M5 default 30 epochs (per V2 plan).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--ffn_expansion", type=int, default=1,
                        help="FFN expansion factor. Default 1 (no expansion) "
                             "for matched comparison with v1 baseline reference.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pc_rate", type=float, default=0.001,
                        help="v2 only; ignored for --arch dead.")
    parser.add_argument("--pred_learning_rate", type=float, default=0.0001,
                        help="v2 only; ignored for --arch dead.")
    parser.add_argument("--homeostatic_decay", type=float, default=0.001,
                        help="v2 only; ignored for --arch dead.")
    parser.add_argument("--set_point_adapt_rate", type=float, default=1e-6,
                        help="v2 only; ignored for --arch dead.")
    parser.add_argument("--compressed_episodes", action="store_true", default=False,
                        help="v2 only; ignored for --arch dead.")
    parser.add_argument(
        "--no_consolidation", action="store_true", default=False,
        help="v2 only: disable consolidation during training. By default "
             "M5 runs with consolidation ON (the M4 STOP GATE result is "
             "part of what's being validated end-to-end). Set this only "
             "for ablation comparisons.",
    )

    # ------------------------------------------------------------------
    # Phase 3G knobs (v2 only — all ignored for --arch dead).
    # See `docs/RESEARCH_LITERATURE_2026-05-13.md` and
    # `docs/RESEARCH_SALVATORI_ATTRACTOR_MEMORY.md` for the design
    # rationale on each. Defaults preserve M5 behavior — these knobs
    # default to off so existing run_*.bat scripts and reproductions of
    # M5 are unaffected.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--consolidation-style", dest="consolidation_style",
        type=str, default="gradient",
        choices=["gradient", "attractor", "both"],
        help="v2 only: consolidation replay pathway. 'gradient' = pull "
             "weight toward stored snapshot (M4 original). 'attractor' = "
             "Salvatori-style replay through pc_self_modify, making stored "
             "input patterns local energy minima. 'both' = run gradient "
             "first, then attractor. Default 'gradient' preserves M5 "
             "baseline; switch to 'attractor' or 'both' for Phase 3G "
             "ablations.",
    )
    parser.add_argument(
        "--consolidation-attractor-passes", dest="consolidation_attractor_passes",
        type=int, default=1,
        help="v2 only: number of replay passes per attractor consolidation "
             "event (consolidation_style='attractor' or 'both'). Default 1 "
             "mirrors gradient-replay's single-pass semantics. Salvatori's "
             "paper uses many passes for full convergence; the trigger fires "
             "many times across training, so each event being a single pass "
             "is usually sufficient.",
    )
    parser.add_argument(
        "--mu-pc-enabled", dest="mu_pc_enabled",
        action="store_true", default=False,
        help="v2 only: enable Depth-muP / muPC parameterization "
             "(Innocenti et al. 2025). Re-inits q/k/v/o_proj, up/down_proj, "
             "and the PC weight buffer with N(0, 1/(sqrt(fan_in)*L^exp)) "
             "where L=n_blocks and exp is set by --mu-pc-exponent. Applies "
             "1/L^exp residual scale on both attention and FFN branches. "
             "Designed to make hyperparameters transfer across depths "
             "without retuning -- the natural pairing for the M6 depth sweep.",
    )
    parser.add_argument(
        "--mu-pc-exponent", dest="mu_pc_exponent",
        type=float, default=0.5,
        help="v2 only (requires --mu-pc-enabled): exponent on L in the "
             "muPC depth-scaling formula. Default 0.5 = original Innocenti "
             "et al. spec (1/sqrt(L)). Lower values give milder attenuation: "
             "0.25 -> 1/L^0.25 (residual divided by 1.86 at L=12 instead of "
             "3.46), 0.0 -> 1.0 (no attenuation, equivalent to "
             "--mu-pc-enabled off for the residual path but init still "
             "scales). Added 2026-05-16 after M6's depth sweep at 128d "
             "surfaced muPC attenuation interacting poorly with PC's "
             "living-weights property at depth; see "
             "docs/research/2026-05-16_plasticity-partitions-design.md.",
    )
    parser.add_argument(
        "--inference-steps-per-forward", dest="inference_steps_per_forward",
        type=int, default=1,
        help="v2 only: iPC interleaved inference + update "
             "(Salvatori et al. 2024). T=1 (default) is bit-identical to "
             "classical PC. T>1 runs T inner pc_self_modify cycles per "
             "external forward call. Incompatible with gradient "
             "checkpointing (the PC layer raises a loud RuntimeError if "
             "both are enabled).",
    )
    parser.add_argument(
        "--sparse-threshold", dest="sparse_threshold",
        type=float, default=0.0,
        help="v2 only: sparse PC update gating threshold on error_acc. "
             "0.0 (default) disables gating (bit-identical to non-sparse). "
             ">0 zeroes the delta_w rows for outputs whose recent "
             "prediction-error magnitude is below the threshold. Recovers "
             "v1's spiking-gate sparsity property in continuous error space. "
             "NOTE: sparse-gating active forces the Python pc_self_modify "
             "path (C++ kernel skips when sparse_gate is not None) until "
             "the Triton kernel lands. See docs/KNOWN_INCOMPLETE.md.",
    )
    parser.add_argument(
        "--sparse-warmup-steps", dest="sparse_warmup_steps",
        type=int, default=500,
        help="v2 only: warmup window during which sparse gating does NOT "
             "engage even when --sparse-threshold > 0. Prevents the "
             "bootstrap deadlock (error_acc starts at 0; gating "
             "immediately would freeze every output forever). Default 500.",
    )

    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="runs/m5")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Defaults to <arch>_seed<seed>.")
    parser.add_argument("--tokenizer", type=str, default="bpe", choices=["char", "bpe"])
    parser.add_argument("--bpe_vocab_size", type=int, default=32000)
    parser.add_argument("--load_tokenizer", type=str, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument(
        "--probe-file-list", dest="probe_file_list", type=str, default=None,
        help="Optional path to a separate file_list.txt for held-out "
             "perplexity probe. When set, the corpus listed is tokenized "
             "with the SAME tokenizer and used as an additional eval pass "
             "at each epoch end. Results land in metrics['probe_losses']. "
             "Distinct from --val_fraction (in-distribution held-out from "
             "the training corpus): the probe is an OUT-of-training-corpus "
             "held-out set, used to track generalization to in-domain but "
             "unseen material. Recommended for any run where the training "
             "corpus is curated/staged and val_fraction's tail-of-tokens "
             "split would give a non-representative val signal.",
    )
    parser.add_argument(
        "--init-from", dest="init_from", type=str, default=None,
        help="Optional path to a .luthi checkpoint to load as starting "
             "weights before training begins. Intended for runs that "
             "continue from an existing substrate (e.g., an expanded "
             "checkpoint produced by luthi.v2.width_expand) rather than "
             "starting from scratch. The checkpoint's config is validated "
             "against the run's d_model, n_heads, n_blocks, and "
             "ffn_expansion — mismatches fail loud at load time. State "
             "is strict-loaded (no missing or unexpected keys allowed). "
             "Optimizer state is NOT loaded: this is for starting a new "
             "training run from an existing substrate's weights, not for "
             "resuming a paused run.",
    )
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["constant", "cosine"])
    parser.add_argument("--lr_warmup_epochs", type=int, default=2)
    parser.add_argument(
        "--log-every-batches", dest="log_every_batches",
        type=int, default=0,
        help="If > 0, print a progress line every N batches during "
             "training. Each line carries: batch index + percent + ETA, "
             "rolling-100-batch mean loss, current batch loss, "
             "batches/sec rate, wall-clock elapsed, NaN count, and (v2 "
             "only) cheap per-block diagnostics (prediction Frobenius "
             "norm mean, precision mean, error_acc mean). Default 0 = "
             "no per-batch logging (preserves existing per-epoch-only "
             "log behavior). Recommended for long single-epoch runs "
             "where the per-epoch granularity is too coarse to monitor.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    run_name = args.run_name or f"{args.arch}_seed{args.seed}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {output_dir}")
    print(f"[arch] {args.arch}")

    # --- Tokenizer ---
    if args.tokenizer == "bpe":
        if args.load_tokenizer:
            tokenizer = BPETokenizer.load(args.load_tokenizer)
            print(f"[tokenizer] loaded BPE from {args.load_tokenizer}")
        else:
            sample_text = load_corpus_sample(args.data_dir)
            tokenizer = BPETokenizer(vocab_size=args.bpe_vocab_size)
            tokenizer.train(sample_text)
            print(f"[tokenizer] trained BPE vocab={tokenizer.vocab_size}")
    else:
        sample_text = load_corpus(args.data_dir)
        tokenizer = CharTokenizer(sample_text)
        print(f"[tokenizer] char vocab={tokenizer.vocab_size}")

    # --- Data ---
    if args.file_list:
        file_paths = load_file_list(args.file_list)
        data = load_corpus_as_tensor(*file_paths, tokenizer=tokenizer)
    else:
        data = load_corpus_as_tensor(args.data_dir, tokenizer=tokenizer)
    print(f"[data] {data.numel():,} tokens")

    n = data.numel()
    val_fraction = max(0.0, min(0.5, args.val_fraction))
    if val_fraction > 0:
        split = int(n * (1.0 - val_fraction))
        train_data, val_data = data[:split], data[split:]
        print(f"[split] {1.0 - val_fraction:.0%} train / {val_fraction:.0%} val")
    else:
        train_data = data
        val_data = None
        print("[split] 100% train / 0% val (probe-only eval path)")

    train_ds = CharDataset(
        train_data, seq_len=args.seq_len, tokenizer=tokenizer, stride=args.stride,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    if val_data is not None:
        val_ds = CharDataset(
            val_data, seq_len=args.seq_len, tokenizer=tokenizer, stride=args.stride,
        )
        val_loader: DataLoader | None = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
        )
        print(f"[batches] train={len(train_loader)} val={len(val_loader)}")
    else:
        val_loader = None
        print(f"[batches] train={len(train_loader)} val=0")

    # --- Held-out probe (out-of-training-corpus eval) ---
    probe_loader: DataLoader | None = None
    if args.probe_file_list:
        print(f"[probe] loading from {args.probe_file_list}")
        probe_paths = load_file_list(args.probe_file_list)
        probe_data = load_corpus_as_tensor(*probe_paths, tokenizer=tokenizer)
        probe_ds = CharDataset(
            probe_data, seq_len=args.seq_len, tokenizer=tokenizer, stride=args.stride,
        )
        probe_loader = DataLoader(
            probe_ds, batch_size=args.batch_size, shuffle=False,
        )
        print(
            f"[probe] {probe_data.numel():,} tokens, "
            f"{len(probe_loader)} batches"
        )

    # --- Model ---
    model = _build_model(args, tokenizer.vocab_size, device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buffer = sum(b.numel() for b in model.buffers())
    print(f"[model] arch={args.arch} trainable={n_train:,} buffers={n_buffer:,}")

    # --- Optional init-from: load an existing substrate's state ---
    if args.init_from:
        from luthi.checkpoint import load_checkpoint
        print(f"[init-from] loading {args.init_from}")
        ckpt = load_checkpoint(args.init_from, trusted=True)
        ckpt_config = ckpt.get("config", {})
        for key, run_val in (
            ("d_model", args.d_model),
            ("n_heads", args.n_heads),
            ("n_blocks", args.n_blocks),
            ("ffn_expansion", args.ffn_expansion),
        ):
            ckpt_val = ckpt_config.get(key)
            if ckpt_val is None:
                raise ValueError(
                    f"--init-from checkpoint config missing '{key}' "
                    f"(checkpoint config keys: {list(ckpt_config.keys())})"
                )
            if ckpt_val != run_val:
                raise ValueError(
                    f"--init-from {key} mismatch: checkpoint has "
                    f"{ckpt_val!r}, run is configured for {run_val!r}"
                )
        expanded_from = ckpt_config.get("expanded_from")
        if expanded_from:
            print(
                f"[init-from] checkpoint lineage: expanded from "
                f"{expanded_from.get('source_path')} "
                f"({expanded_from.get('source_d_model')}d -> "
                f"{args.d_model}d, factor="
                f"{expanded_from.get('expansion_factor')}, "
                f"noise={expanded_from.get('noise')})"
            )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(
            f"[init-from] loaded substrate state "
            f"(source epoch={ckpt.get('epoch', 0)})"
        )

    # --- Optimizer (DirectMLAdamW for DirectML compatibility) ---
    from luthi.optimizer import DirectMLAdamW
    optimizer = DirectMLAdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )

    # --- Per-epoch NFF probe input (audit 2026-05-12 instrumentation) ---
    # Fixed across the run so NFF values are commensurable epoch-to-epoch.
    # Drawn from probe_loader if available (preferred — held-out from
    # training so NFF measures non-feedforward signal on truly novel
    # input), otherwise val_loader, otherwise train_loader.
    if probe_loader is not None:
        nff_source = probe_loader
    elif val_loader is not None:
        nff_source = val_loader
    else:
        nff_source = train_loader
    nff_probe = next(iter(nff_source))[0][:1].to(device)

    # --- Training loop ---
    train_losses: list[float] = []
    val_losses: list[float] = []
    probe_losses: list[float] = []
    nff_signals: list[float] = []
    pred_frob_norms: list[float] = []
    precision_means: list[float] = []
    precision_ranges: list[list[float]] = []  # [[min, max], ...]
    total_nan_events = 0
    best_val = float("inf")
    best_probe = float("inf")

    for epoch in range(1, args.epochs + 1):
        lr_for_epoch = _compute_lr(args, epoch - 1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_for_epoch

        train_loss, nans = _train_one_epoch(
            model, train_loader, optimizer, device,
            log_every_batches=args.log_every_batches,
            epoch_label=f"[ep {epoch+1:>3d}/{args.epochs}] ",
            total_batches_estimate=len(train_loader),
        )
        train_losses.append(train_loss)
        total_nan_events += nans

        if val_loader is not None:
            val_loss = _evaluate(model, val_loader, device)
            val_losses.append(val_loss)
            best_val = min(best_val, val_loss)
        else:
            val_loss = float("nan")

        if probe_loader is not None:
            probe_loss = _evaluate(model, probe_loader, device)
            probe_losses.append(probe_loss)
            best_probe = min(best_probe, probe_loss)
        else:
            probe_loss = float("nan")

        # Audit 2026-05-12 instrumentation: NFF + PC diagnostics per epoch.
        # For DeadLM, NFF should be ~0 (deterministic forward) and the PC
        # diagnostics return empty.
        nff = _measure_nff(model, nff_probe)
        living = _measure_living_diagnostics(model)
        nff_signals.append(nff)
        if "prediction_frob_mean" in living:
            pred_frob_norms.append(living["prediction_frob_mean"])
        if "precision_mean" in living:
            precision_means.append(living["precision_mean"])
            precision_ranges.append(
                [living["precision_min"], living["precision_max"]]
            )

        log_extra = f" nff={nff:.4e}"
        if "prediction_frob_mean" in living:
            log_extra += f" pred_frob={living['prediction_frob_mean']:.3f}"
        if "precision_mean" in living:
            log_extra += f" prec={living['precision_mean']:.3f}"
        eval_summary = ""
        if val_loader is not None:
            eval_summary += f" val={val_loss:.4f}"
        if probe_loader is not None:
            from math import exp
            try:
                probe_ppl = exp(probe_loss)
            except OverflowError:
                probe_ppl = float("inf")
            eval_summary += f" probe={probe_loss:.4f} (ppl={probe_ppl:.1f})"
        print(
            f"[epoch {epoch:3d}/{args.epochs}] "
            f"train={train_loss:.4f}{eval_summary} "
            f"lr={lr_for_epoch:.2e} nans={nans}{log_extra}"
        )

    # --- Save final checkpoint + metrics ---
    _save_checkpoint(model, output_dir / "model_final", args.epochs, vars(args))
    metrics = {
        "arch": args.arch,
        "seed": args.seed,
        "config": vars(args),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "probe_losses": probe_losses,
        "nff_signals": nff_signals,
        "prediction_frob_norms": pred_frob_norms,
        "precision_means": precision_means,
        "precision_ranges": precision_ranges,
        "best_val": best_val if val_losses else None,
        "best_probe": best_probe if probe_losses else None,
        "final_nff": nff_signals[-1] if nff_signals else 0.0,
        "nan_events": total_nan_events,
        "trainable_params": n_train,
        "buffer_params": n_buffer,
    }
    (output_dir / "results.json").write_text(json.dumps(metrics, indent=2))
    print(f"[done] saved to {output_dir}")
    summary_eval = ""
    if val_losses:
        summary_eval += f" best_val={best_val:.4f}"
    if probe_losses:
        summary_eval += f" best_probe={best_probe:.4f}"
    print(
        f"[summary] arch={args.arch} seed={args.seed}{summary_eval} "
        f"final_train={train_losses[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
