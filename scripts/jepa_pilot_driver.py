"""Two-arm JEPA pilot driver — Experiment 1, JEPA edition (2026-07-15).

The falsification-critical run, merged with the M8 256d de-risking pilot
(critical-path item 1). One instrumented sweep answers three
pre-registered questions (docs/research/2026-07-15_falsification-
preregistration.md; protocol living-weights-experiments.md, JEPA
edition):

  (a) the pilot's collapse-kill thresholds (pilot-set machinery derives
      them per-run; this sweep sanity-checks the derived values);
  (b) does representation collapse behave differently when the weights
      self-modify — the dead arm is the direct control;
  (c) matched capacity: does the living arm sit above the dead arm's
      effective-capacity curve on held-out latent prediction + probes?

Arms and stages (5 seeds per condition — Brian, 2026-07-15):

  stage 1   living@256 x5  +  dead@256 x5     the matched point
  stage 2   dead@192 x5  +  dead@384 x5       the curve's shape
  stage 3   dead@512 x5                       the upper bracket

Stage 1 decides half the outcomes alone: if the living arm loses or
ties at the matched point, KF2-strong dies and stages 2-3 are
unnecessary. Bracket {192,256,384,512} pends Brian's S1 ratification.

Per run: text-only JEPA (round-1 scope), leakage-gapped 2% holdout,
end-of-epoch held-out eval (built into JEPATrainer), then a final
held-out eval + next-token linear probe WITH its shuffled-label floor.
Results land in <run_dir>/pilot_result.json — the completion marker, so
the sweep is resumable (completed runs skipped; a failed run stops the
queue loudly: a condition with fewer seeds than its siblings corrupts
the variance estimate).

Device: DirectML -> CUDA -> CPU (the train_pc pick). NOTE: the JEPA path
has only ever run on CPU (the smokes); the first pilot run doubles as
the device shakeout — watch the first hundred steps.

Usage:
  python scripts/jepa_pilot_driver.py --stage 1 --dry-run
  python scripts/jepa_pilot_driver.py --stage 1
  python scripts/jepa_pilot_driver.py --aggregate
Smoke (CPU, minutes):
  python scripts/jepa_pilot_driver.py --stage 1 --smoke
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
# The driver imports luthi in-process (unlike the retired LM driver,
# which shelled out); make it runnable from anywhere.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OUTPUT_ROOT = REPO_ROOT / "runs" / "jepa_pilot"

SEEDS = (42, 43, 44, 45, 46)

# (arm, d_model) conditions per stage. Bracket amended by Brian
# 2026-07-16: a SINGLE overshoot point (dead@512, ~4x the living FFN's
# nominal weight count) replaces the {192, 384, 512} curve; stage 3 is
# reserved for the 384 fallback the pre-registered read requires if
# dead@512 wins or ties (see the pre-registration's bracket entry).
STAGES: dict[int, list[tuple[str, int]]] = {
    1: [("living", 256), ("dead", 256)],
    2: [("dead", 512)],
    3: [("dead", 384)],  # fallback only -- run on an inconclusive stage 2
}


def _device() -> torch.device:
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _DeviceLoader:
    """Wraps a MultimodalDataLoader; delivers device-local batches."""

    def __init__(self, inner, device: torch.device):
        self._inner = inner
        self._device = device

    def _move(self, batch: dict) -> dict:
        return {k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def next_batch(self, modality: str) -> dict:
        return self._move(self._inner.next_batch(modality))

    def batch_token_count(self, modality: str, batch: dict) -> int:
        return self._inner.batch_token_count(modality, batch)

    def state_dict(self) -> dict:
        return self._inner.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._inner.load_state_dict(state)

    def holdout_batch_count(self, modality: str, batch_size: int) -> int:
        return self._inner.holdout_batch_count(modality, batch_size)

    def holdout_batches(self, modality: str, batch_size: int):
        for batch in self._inner.holdout_batches(modality, batch_size):
            yield self._move(batch)


def _run_name(arm: str, d_model: int, seed: int) -> str:
    return f"{arm}_{d_model}d_seed{seed}"


def _result_path(arm: str, d_model: int, seed: int) -> Path:
    return OUTPUT_ROOT / _run_name(arm, d_model, seed) / "pilot_result.json"


def _run_one(arm: str, d_model: int, seed: int, args) -> dict:
    from luthi.tokenizer import BPETokenizer
    from luthi.v2.eval_heldout import (
        fit_next_token_probe,
        heldout_latent_prediction,
        probe_accuracy,
    )
    from luthi.v2.jepa_loss import JEPALoss
    from luthi.v2.jepa_runner import (
        CheckpointConfig,
        EpochConfig,
        JEPATrainer,
        KillCriteriaConfig,
        LoggingConfig,
        ModalitySampler,
        RunnerConfig,
        SamplerConfig,
    )
    from luthi.v2.multimodal_data import (
        MultimodalDataLoaderImpl,
        TextDataset,
        TextDatasetConfig,
    )
    from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM

    device = _device()
    run_dir = OUTPUT_ROOT / _run_name(arm, d_model, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    text_ds = TextDataset(TextDatasetConfig(
        source_paths=[args.data_dir],
        tokenizer_path=Path(args.tokenizer),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        stride=args.stride,
        base_seed=seed,
        holdout_fraction=args.holdout_fraction,
    ))
    loader = _DeviceLoader(MultimodalDataLoaderImpl(text=text_ds), device)

    model = MultimodalPredictiveCodingLM(
        vocab_size=text_ds.vocab_size(),
        d_model=d_model,
        n_blocks=args.n_blocks,
        n_heads=4,
        ffn_expansion=1,
        max_seq_len=args.seq_len,
        backward_pass_enabled=False,  # pilot scope: match the M8 smokes
        dead_ffn=(arm == "dead"),
    ).to(device)
    loss_module = JEPALoss(online_encoder=model).to(device)

    sampler_cfg = SamplerConfig(
        corpus_sizes_tokens={"text": text_ds.tokens_per_pass()}, alpha=0.7,
    )
    gen = torch.Generator(device="cpu").manual_seed(seed)
    trainer = JEPATrainer(
        loss_module=loss_module,
        optimizer=optim.AdamW(
            [p for p in loss_module.parameters() if p.requires_grad],
            lr=args.lr,
        ),
        sampler=ModalitySampler(sampler_cfg, generator=gen),
        data_loader=loader,
        config=RunnerConfig(
            sampler=sampler_cfg,
            checkpoint=CheckpointConfig(rolling_slots=3),
            logging=LoggingConfig(heldout_eval_batches=args.heldout_batches),
            kill_criteria=KillCriteriaConfig(
                warmup_batches=args.kill_warmup,
                # Pilot-derived thresholds (calibration pass 1, 2026-07-16
                # -- docs/research/2026-07-16_jepa-pilot-calibration-pass.md).
                # The static defaults killed 10/10 healthy runs: kill-1's
                # init-window baseline fired while effective rank was
                # RISING; kill-6's 25% band around a transiently-latched
                # running min fired at the substrate's healthiest moment.
                stationary_deviation_pct=0.85,
                substrate_health_degradation_pct=1.0,
                trending_smoothing_window=9,
                substrate_health_window=10,
            ),
            epoch=EpochConfig(
                max_epochs=args.epochs,
                abort_continue_at_epoch_1=False,  # sweep runs are unattended
                max_batches_per_epoch=args.max_batches_per_epoch,
            ),
        ),
        run_dir=run_dir,
    )

    started = time.time()
    outcome = trainer.run()

    heldout = heldout_latent_prediction(
        loss_module,
        loader.holdout_batches("text", 8),
        "text",
        max_batches=args.heldout_batches,
    )
    train_batches = [loader.next_batch("text") for _ in range(args.probe_batches)]
    probe = fit_next_token_probe(
        loss_module, train_batches,
        vocab_size=text_ds.vocab_size(),
        max_batches=args.probe_batches,
    )
    heldout_list = list(loader.holdout_batches("text", 8))
    probe_real = probe_accuracy(loss_module, probe, heldout_list)
    probe_floor = probe_accuracy(
        loss_module, probe, heldout_list, shuffled_label_floor=True,
    )

    result = {
        "arm": arm,
        "d_model": d_model,
        "seed": seed,
        "outcome": outcome,
        # Collapse-admissibility (protocol section 1): a killed run's
        # numbers are reported but flagged inadmissible for comparison.
        "admissible": outcome == "completed",
        "heldout": heldout,
        "probe": probe_real,
        "probe_shuffled_floor": probe_floor,
        "wall_clock_seconds": time.time() - started,
        "config": {
            "n_blocks": args.n_blocks, "epochs": args.epochs,
            "batch_size": args.batch_size, "seq_len": args.seq_len,
            "stride": args.stride, "lr": args.lr,
            "holdout_fraction": args.holdout_fraction,
            "data_dir": str(args.data_dir),
        },
    }
    _result_path(arm, d_model, seed).write_text(json.dumps(result, indent=2))
    return result


def run(stages: list[int], args) -> int:
    plan = [
        (arm, d, s)
        for stage in stages
        for arm, d in STAGES[stage]
        for s in (SEEDS[: args.n_seeds])
    ]
    done = [c for c in plan if _result_path(*c).exists()]
    todo = [c for c in plan if not _result_path(*c).exists()]
    print(f"[jepa-pilot] plan: {len(plan)} runs "
          f"({len(done)} complete, {len(todo)} to run)")
    for arm, d_model, seed in todo:
        name = _run_name(arm, d_model, seed)
        if args.dry_run:
            print(f"  DRY-RUN: {name} (n_blocks={args.n_blocks}, "
                  f"epochs={args.epochs}, data={args.data_dir})")
            continue
        print(f"[jepa-pilot] starting {name}")
        try:
            result = _run_one(arm, d_model, seed, args)
        except Exception as e:  # noqa: BLE001 -- stop the queue loudly
            print(f"[jepa-pilot] FAILED: {name}: {type(e).__name__}: {e}")
            raise
        print(f"[jepa-pilot] {name}: outcome={result['outcome']} "
              f"heldout_l_pred={result['heldout']['l_pred_mean']:.6f} "
              f"probe_top1={result['probe']['top1']:.4f} "
              f"({result['wall_clock_seconds']/3600:.2f}h)")
    return 0


def aggregate() -> int:
    """Per-condition summary. Prints the ingredients; the curve-level
    verdict (effective-capacity placement) is the analysis doc's job."""
    conditions: dict[str, list[dict]] = {}
    for path in sorted(OUTPUT_ROOT.glob("*/pilot_result.json")):
        r = json.loads(path.read_text())
        conditions.setdefault(f"{r['arm']}_{r['d_model']}d", []).append(r)
    if not conditions:
        print("[aggregate] no completed runs")
        return 1
    summary = {}
    hdr = f"{'condition':<14} {'n':>2} {'adm':>3} {'l_pred mean':>12} {'std':>9} {'probe_top1':>10}"
    print(hdr)
    for cond, runs in sorted(conditions.items()):
        adm = [r for r in runs if r["admissible"]]
        vals = [r["heldout"]["l_pred_mean"] for r in adm]
        probes = [r["probe"]["top1"] for r in adm]
        mean = statistics.mean(vals) if vals else float("nan")
        std = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        p1 = statistics.mean(probes) if probes else float("nan")
        summary[cond] = {
            "n_total": len(runs), "n_admissible": len(adm),
            "l_pred_mean": mean, "l_pred_std": std,
            "probe_top1_mean": p1,
            "inadmissible": [
                {"seed": r["seed"], "outcome": r["outcome"]}
                for r in runs if not r["admissible"]
            ],
        }
        print(f"{cond:<14} {len(runs):>2} {len(adm):>3} {mean:>12.6f} "
              f"{std:>9.6f} {p1:>10.4f}")
        if len(adm) < len(runs):
            print(f"  NOTE: {len(runs)-len(adm)} inadmissible run(s) "
                  f"(killed) -- reported, excluded from means")
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[aggregate] written to {OUTPUT_ROOT / 'summary.json'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", type=str, default=None, help="1, 2, 3, or 'all'")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny CPU shakeout: 64d/1-block, capped steps.")
    p.add_argument("--data_dir", type=str,
                   default=str(REPO_ROOT / "corpus_build" / "gutenberg_100"))
    p.add_argument("--tokenizer", type=str,
                   default=str(REPO_ROOT / "corpus_build" / "tokenizer_32k.json"))
    p.add_argument("--n-blocks", dest="n_blocks", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--holdout-fraction", dest="holdout_fraction",
                   type=float, default=0.02)
    p.add_argument("--heldout-batches", dest="heldout_batches",
                   type=int, default=50)
    p.add_argument("--probe-batches", dest="probe_batches",
                   type=int, default=32)
    p.add_argument("--kill-warmup", dest="kill_warmup",
                   type=int, default=5000)
    p.add_argument("--max-batches-per-epoch", dest="max_batches_per_epoch",
                   type=int, default=-1)
    p.add_argument("--n-seeds", dest="n_seeds", type=int, default=len(SEEDS))
    args = p.parse_args()

    if args.smoke:
        args.epochs = 1
        args.max_batches_per_epoch = 20
        args.heldout_batches = 3
        args.probe_batches = 4
        args.n_seeds = 1
        args.kill_warmup = 10**9
        args.batch_size = 4
        args.seq_len = 64
        # Smoke shrinks the conditions too: one living + one dead at 64d.
        STAGES[1] = [("living", 64), ("dead", 64)]
        print("[jepa-pilot] SMOKE MODE: 64d, 20 steps, 1 seed, CPU-ok")

    if args.aggregate:
        return aggregate()
    if args.stage is None:
        p.error("--stage required unless --aggregate")
    stages = [1, 2, 3] if args.stage == "all" else [int(args.stage)]
    return run(stages, args)


if __name__ == "__main__":
    raise SystemExit(main())
