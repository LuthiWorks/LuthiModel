"""Per-block linear probe + spectrum read from a trained checkpoint.

Built 2026-08-13 for the 768x8 family's inverted-profile forensic
(docs/research/2026-08-11_768x8-family-spec.md, VERDICT section).

The question
-----------
Seed 97 ended with per-block effective rank 2.0, 2.4, 112, 210, 222, 280,
390, 403 -- blocks 0-1 at the collapse floor while 2-7 are the healthiest
the project has recorded. Effective rank is a LINEAR, second-order gauge
of the covariance spectrum. It cannot distinguish:

  (a) a high-magnitude carrier direction dominating the variance while
      the signal survives in the low-variance directions (benign: a
      downstream LayerNorm rescales and nothing is lost), from
  (b) the representation genuinely collapsing onto ~2 dimensions
      (fatal: information destroyed at block 0, and every healthy number
      downstream is variance manufactured from a 2-D manifold).

A rank number cannot answer that. A FUNCTIONAL read can: fit the same
linear next-token probe the verdict reports (eval_heldout.fit_next_token_probe)
to each block's latents in turn. If block 0's probe accuracy is comparable
to block 7's, the information survived the low-rank bottleneck and (a) is
the right reading. If it collapses to the shuffled floor at block 0, the
early blocks are destroying content and (b) is.

Reads reported per block
------------------------
  probe top1/top5       -- functional information content (the decisive read)
  shuffled floor        -- the same probe on permuted labels (chance, given
                           this label distribution); lift = top1 / floor
  effective_rank        -- exp(spectral entropy), reproducing the runner's
                           gauge as a provenance check against the tape
  top_dir_share         -- largest eigenvalue's share of total variance
  chorus_eff_rank       -- effective rank with the TOP direction removed.
                           Separates "one soloist over a healthy chorus"
                           from "there is no chorus". Cheap and decisive
                           for hypothesis (a) on the spectrum side.
  offset_dominance      -- ||mean|| / mean||row||; the project's existing
                           gauge of the collapse's measured "first act".

Block index -1 is the pre-trunk embedding stream (the input to block 0),
included as the reference ceiling: it is the information the trunk was
handed. Every later block is measured against it.

Read-only: loads a checkpoint, writes a JSON report, never trains the
model or touches the run directory.

Usage:
    python scripts/per_block_probe.py \
        --run-dir runs/jepa_pilot/probe_768_visreg_768d_seed97
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------- spectra


def _spectrum(x: torch.Tensor) -> dict:
    """Covariance spectrum reads for one block's latents.

    Identical math to jepa_runner._rank_and_top_share for the first two
    keys, so the numbers are directly comparable to the training tape.
    """
    flat = x.detach().float().reshape(-1, x.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    n = centered.shape[0]
    cov = (centered.t() @ centered) / max(n - 1, 1)
    sv = torch.linalg.svdvals(cov).clamp(min=1e-12)

    def eff_rank(s: torch.Tensor) -> float:
        p = s / s.sum()
        return float(math.exp(float(-(p * torch.log(p.clamp(min=1e-12))).sum().item())))

    out = {
        "effective_rank": eff_rank(sv),
        "top_dir_share": float((sv.max() / sv.sum()).item()),
        # Drop the single largest direction: is there a chorus behind the
        # soloist, or is the soloist the whole representation?
        "chorus_eff_rank": eff_rank(sv[1:]) if sv.numel() > 1 else 1.0,
        "chorus2_eff_rank": eff_rank(sv[2:]) if sv.numel() > 2 else 1.0,
        "offset_dominance": float(
            (flat.mean(dim=0).norm() / flat.norm(dim=1).mean().clamp(min=1e-12)).item()
        ),
        # Raw scale, recorded because it is the reason the probe MUST be
        # standardized (see _fit_probe): block latent magnitudes span
        # orders of magnitude across the stack, and an unstandardized
        # probe's convergence tracks the scale rather than the content.
        "latent_rms": float(flat.pow(2).mean().sqrt().item()),
        "latent_std_mean": float(flat.std(dim=0).mean().item()),
    }
    return out


# ------------------------------------------------------------------ model


def _rebuild(run_dir: Path, ckpt_path: Path, device):
    """Rebuild the arm from the run's OWN recorded model_kwargs.

    Deliberately NOT via scripts.jepa_pilot_driver.ARM_CONFIGS: that path
    starts from a hardcoded n_heads=4 and depends on the arm's entry to
    override it, so a rebuild is only correct while the registry and the
    run agree. pilot_result.json records the kwargs the run was actually
    constructed with, which is the provenance that matters here.
    """
    from luthi.living_extra_state import apply_living_extra_state
    from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM

    result = json.loads((run_dir / "pilot_result.json").read_text(encoding="utf-8"))
    cfg = result["config"]
    kwargs = dict(cfg["model_kwargs"])
    model = MultimodalPredictiveCodingLM(**kwargs).to(device)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["online_state_dict"])
    apply_living_extra_state(
        model, state.get("living_extra_state"),
        source=f"per_block_probe {ckpt_path.name}",
    )
    model.eval()
    return model, result, cfg, state


def _dataset(cfg: dict, seed: int):
    from luthi.v2.multimodal_data import TextDataset, TextDatasetConfig

    if cfg.get("file_list"):
        source_paths = [
            line.strip()
            for line in Path(cfg["file_list"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        source_paths = [cfg["data_dir"]]
    return TextDataset(TextDatasetConfig(
        source_paths=source_paths,
        tokenizer_path=REPO_ROOT / "corpus_build" / "tokenizer_32k.json",
        batch_size=cfg["batch_size"],
        seq_len=cfg["seq_len"],
        stride=cfg["stride"],
        base_seed=seed,
        holdout_fraction=cfg["holdout_fraction"],
    ))


# ------------------------------------------------------------- collection


@torch.no_grad()
def _encode_all_blocks(model, tokens: torch.Tensor) -> list[torch.Tensor]:
    """[emb, block0, ..., blockN-1] latents for one batch, on CPU float16.

    The embedding stream is obtained through the model's own modality
    encode + concatenate, i.e. exactly the tensor block 0 receives.
    """
    vision_emb, audio_emb, text_emb = model._encode_modality_streams(
        tokens, None, None, None, None,
    )
    h, _spans = model._concatenate_modalities(vision_emb, audio_emb, text_emb)
    out = [h.detach().to("cpu", torch.float16)]
    for block in model.blocks:
        h = block(h, causal=False)
        out.append(h.detach().to("cpu", torch.float16))
    return out


def _collect(model, batches, device, max_batches: int):
    """Returns (per_block_latents[list of [N, D] cpu f16], targets [N])."""
    per_block: list[list[torch.Tensor]] = []
    ys: list[torch.Tensor] = []
    for i, batch in enumerate(batches):
        if i >= max_batches:
            break
        tokens = batch["text_tokens"].to(device)
        blocks = _encode_all_blocks(model, tokens)
        if not per_block:
            per_block = [[] for _ in blocks]
        for bi, h in enumerate(blocks):
            # positions :-1 pair with tokens 1:, matching
            # eval_heldout._collect_next_token_pairs exactly.
            per_block[bi].append(h[:, :-1, :].reshape(-1, h.shape[-1]))
        ys.append(batch["text_tokens"][:, 1:].reshape(-1).cpu())
    if not ys:
        raise ValueError("no batches collected -- check batch counts")
    return [torch.cat(c, dim=0) for c in per_block], torch.cat(ys, dim=0)


# ----------------------------------------------------------------- probes


def _standardizer(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension mean/std from the TRAIN split, applied to both splits.

    Why this is required rather than optional (found 2026-08-13, first
    run of this script): block latent scale varies by orders of magnitude
    across the stack, and `eval_heldout`'s recipe (Adam, lr=1e-2, no
    input normalization) converges for some scales and not others. The
    unstandardized first pass read b0=0.0411, b1=0.0528, b2=0.0673,
    b3=0.0912 and then b4=0.0013 -- AT CHANCE -- which is not a
    representation property but the probe failing to optimize on a
    large-magnitude input. A linear probe's accuracy is invariant to an
    invertible affine map of its input in the limit of perfect
    optimization; standardizing is what lets the optimizer reach that
    limit, so the comparison across blocks measures content instead of
    scale. Mean/std come from TRAIN only -- computing them on the
    holdout would leak.
    """
    mu = x.float().mean(dim=0, keepdim=True)
    sd = x.float().std(dim=0, keepdim=True).clamp(min=1e-6)
    return mu, sd


def _fit_probe(x: torch.Tensor, y: torch.Tensor, vocab: int,
               epochs: int, lr: float, seed: int,
               mu: torch.Tensor, sd: torch.Tensor) -> nn.Linear:
    """eval_heldout.fit_next_token_probe's recipe (Adam, 1e-2, 3 epochs,
    4096 minibatch) on STANDARDIZED inputs -- see _standardizer."""
    torch.manual_seed(seed)
    probe = nn.Linear(x.shape[-1], vocab)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    probe.train()
    for _ in range(epochs):
        perm = torch.randperm(x.shape[0])
        for start in range(0, x.shape[0], 4096):
            idx = perm[start:start + 4096]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe((x[idx].float() - mu) / sd), y[idx])
            loss.backward()
            opt.step()
    probe.eval()
    return probe


@torch.no_grad()
def _accuracy(probe: nn.Linear, x: torch.Tensor, y: torch.Tensor,
              shuffle: bool, seed: int,
              mu: torch.Tensor, sd: torch.Tensor) -> dict:
    if shuffle:
        gen = torch.Generator().manual_seed(seed)
        y = y[torch.randperm(y.shape[0], generator=gen)]
    top1 = 0.0
    top5 = 0.0
    n = 0
    for start in range(0, x.shape[0], 8192):
        xb = (x[start:start + 8192].float() - mu) / sd
        yb = y[start:start + 8192]
        logits = probe(xb)
        top1 += (logits.argmax(dim=-1) == yb).float().sum().item()
        k = min(5, logits.shape[-1])
        topk = logits.topk(k, dim=-1).indices
        top5 += (topk == yb.unsqueeze(-1)).any(dim=-1).float().sum().item()
        n += xb.shape[0]
    return {"top1": top1 / n, "top5": top5 / n, "n_pairs": n}


# ------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="", help="checkpoint filename; default = last")
    ap.add_argument("--train-batches", type=int, default=20,
                    help="probe training batches (eval_heldout default is 20)")
    ap.add_argument("--heldout-batches", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="report path; default = run-dir/per_block_probe.json")
    args = ap.parse_args()

    from scripts.jepa_pilot_driver import _device

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    ckpts = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {run_dir}")
    ckpt_path = (run_dir / "checkpoints" / args.ckpt) if args.ckpt else ckpts[-1]
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    device = _device()
    print(f"device={device}  checkpoint={ckpt_path.name}", flush=True)

    model, result, cfg, state = _rebuild(run_dir, ckpt_path, device)
    step = state.get("global_step", "?")
    print(f"arm={result['arm']} seed={result['seed']} d_model={result['d_model']} "
          f"n_blocks={cfg['n_blocks']} ckpt_step={step}", flush=True)
    print(f"verdict reference: probe top1={result['probe']['top1']:.4f} "
          f"floor={result['probe_shuffled_floor']['top1']:.4f} "
          f"lift={result['probe']['top1'] / result['probe_shuffled_floor']['top1']:.2f}x",
          flush=True)

    ds = _dataset(cfg, result["seed"])
    vocab = ds.vocab_size()
    print(f"corpus loaded: vocab={vocab}", flush=True)

    print(f"collecting {args.train_batches} train batches...", flush=True)
    train_x, train_y = _collect(
        model, (ds.next_batch() for _ in range(args.train_batches)),
        device, args.train_batches,
    )
    print(f"collecting {args.heldout_batches} holdout batches...", flush=True)
    hold_x, hold_y = _collect(
        model, ds.holdout_batches(cfg["batch_size"]), device, args.heldout_batches,
    )
    print(f"train pairs={train_y.shape[0]}  holdout pairs={hold_y.shape[0]}  "
          f"streams={len(train_x)} (emb + {len(train_x) - 1} blocks)", flush=True)

    rows = []
    for bi in range(len(train_x)):
        label = "emb" if bi == 0 else f"b{bi - 1}"
        spec = _spectrum(hold_x[bi])
        mu, sd = _standardizer(train_x[bi])
        probe = _fit_probe(train_x[bi], train_y, vocab,
                           args.epochs, args.lr, args.seed, mu, sd)
        acc = _accuracy(probe, hold_x[bi], hold_y, False, args.seed, mu, sd)
        floor = _accuracy(probe, hold_x[bi], hold_y, True, args.seed, mu, sd)
        row = {
            "block": label,
            "probe_top1": acc["top1"],
            "probe_top5": acc["top5"],
            "floor_top1": floor["top1"],
            "lift": acc["top1"] / floor["top1"] if floor["top1"] > 0 else float("inf"),
            "n_pairs": acc["n_pairs"],
            **spec,
        }
        rows.append(row)
        print(f"  {label:>4}  top1={row['probe_top1']:.4f}  floor={row['floor_top1']:.4f}  "
              f"lift={row['lift']:6.2f}x  eff={spec['effective_rank']:7.1f}  "
              f"tds={spec['top_dir_share']:.3f}  chorus={spec['chorus_eff_rank']:7.1f}  "
              f"offset={spec['offset_dominance']:.4f}  rms={spec['latent_rms']:.3g}",
              flush=True)

    print("\n" + "=" * 104)
    print(f"{'blk':>4} {'top1':>8} {'floor':>8} {'lift':>8} {'eff_rank':>9} "
          f"{'top_share':>10} {'chorus':>9} {'chorus2':>9} {'offset':>8} {'rms':>10}")
    print("-" * 104)
    for r in rows:
        print(f"{r['block']:>4} {r['probe_top1']:>8.4f} {r['floor_top1']:>8.4f} "
              f"{r['lift']:>7.2f}x {r['effective_rank']:>9.1f} {r['top_dir_share']:>10.4f} "
              f"{r['chorus_eff_rank']:>9.1f} {r['chorus2_eff_rank']:>9.1f} "
              f"{r['offset_dominance']:>8.4f} {r['latent_rms']:>10.3g}")

    out_path = Path(args.out) if args.out else run_dir / "per_block_probe.json"
    out_path.write_text(json.dumps({
        "run_dir": str(run_dir),
        "checkpoint": ckpt_path.name,
        "ckpt_step": step if isinstance(step, int) else str(step),
        "arm": result["arm"],
        "seed": result["seed"],
        "settings": {
            "train_batches": args.train_batches,
            "heldout_batches": args.heldout_batches,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
        },
        "verdict_reference": {
            "probe_top1": result["probe"]["top1"],
            "floor_top1": result["probe_shuffled_floor"]["top1"],
        },
        "blocks": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
