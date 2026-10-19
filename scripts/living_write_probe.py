"""What does the living channel of a trained checkpoint actually write?

Built 2026-10-19 (docs/research/2026-10-19_the-living-write-is-mean-field.md).
Read-only on the run directory. Four reads per living block:

  1. living-change spectrum -- top_share / eff_rank / rel_norm of
     weight - set_point, and out_align: |cos| of its top direction with the
     block's batch-mean FFN output. The mean-field signature is
     top_share ~1 AND out_align ~1.
  2. batch-mean stationarity -- mean pairwise cosine of the FFN's batch-mean
     input across independent training-size batches, and across single
     held-out sequences. ~1 means the statistic the mean-field write and the
     legacy episode context consume does not change with content.
  3. write diversity -- from the SAME state, one self-modifying forward per
     input (deep copies), cosine between the resulting weight writes, for
     training-size batches and for single documents. ~1 means different
     inputs produce the same write.
  4. block output store -- recall duty and blend share under frozen reads.
     Duty ~1 with blend share > 0 means a stored batch-mean output is being
     mixed into nearly every forward (legacy configuration).

Usage:
    python scripts/living_write_probe.py --run-dir E:\\runs\\jepa_pilot\\probe_768_visreg_768d_seed97

Pre-registered expectations for existing legacy-rule checkpoints (registry
entry E2, frozen 2026-10-19 before any production checkpoint was read):
top_share >= 0.9 and out_align >= 0.9 in every living block; training-batch
stationarity >= 0.99 in every living block; block-store recall_duty >= 0.9
with blend_share > 0.05 in every block that has a store. Write diversity is
DESCRIPTIVE (no threshold): on the CPU smoke the output side of every write
was shared while the input side varied (training-batch write cosine
0.49-0.89), so a threshold on the cosine would test the wrong thing. Any
block that misses an expectation is a finding against the diagnosis and goes
in the research log as such.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _frozen_output_means(model, inputs) -> dict[str, torch.Tensor]:
    """Batch-mean living-FFN outputs averaged over ``inputs``, under freeze."""
    from luthi.v2.living_diagnostics import living_layers
    from luthi.v2.plasticity import freeze_plasticity

    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []
    for name, layer in living_layers(model):
        def _hook(mod, _args, out, name=name):
            o = out.detach().reshape(-1, mod.out_features).float().mean(dim=0)
            sums[name] = sums.get(name, 0) + o
            counts[name] = counts.get(name, 0) + 1
        hooks.append(layer.register_forward_hook(_hook))
    try:
        with freeze_plasticity(model), torch.no_grad():
            for inp in inputs:
                model.encode(**inp, causal=False)
    finally:
        for h in hooks:
            h.remove()
    return {k: sums[k] / counts[k] for k in sums}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--batches", type=int, default=8,
                    help="independent training-size batches for stationarity")
    ap.add_argument("--write-pairs", type=int, default=4,
                    help="inputs per write-diversity comparison")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from luthi.v2.living_diagnostics import (
        batch_mean_stationarity, living_change_spectrum, living_layers,
        write_diversity,
    )
    from luthi.v2.plasticity import freeze_plasticity
    from luthi.v2.run_rebuild import rebuild_run
    from scripts.jepa_pilot_driver import _device

    device = _device()
    run = rebuild_run(args.run_dir, device=device, ckpt_name=args.ckpt or None)
    model, ds = run.model, run.text_ds
    print(f"arm={run.result['arm']} seed={run.result['seed']} "
          f"ckpt={run.checkpoint.name} step={run.global_step}", flush=True)
    layers = living_layers(model)
    if not layers:
        print("no living layers (dead arm): nothing to probe", flush=True)

    train_batches = [
        {"text_tokens": ds.next_batch()["text_tokens"].to(device)}
        for _ in range(args.batches)
    ]
    windows = ds.holdout_window_tokens()
    stride = max(1, windows.shape[0] // max(args.batches, 1))
    singles = [
        {"text_tokens": windows[i * stride : i * stride + 1].to(device)}
        for i in range(args.batches)
    ]

    # 1. spectrum, with out_align against the frozen mean FFN output.
    means = _frozen_output_means(model, train_batches)
    spectra = {}
    for name, layer in layers:
        layer._last_output_mean = means.get(name)
        spectra[name] = living_change_spectrum(layer)

    # 2. stationarity of the statistic the mean-field rule consumes.
    stat_train = batch_mean_stationarity(model, train_batches)
    stat_single = batch_mean_stationarity(model, singles)

    # 3. write diversity from the same state (deep copies; train mode as in
    #    training, which also runs the top-down sweep).
    model.train()
    div_train = write_diversity(model, train_batches[: args.write_pairs])
    div_single = write_diversity(model, singles[: args.write_pairs])
    model.eval()

    # 4. block output store activity under frozen reads.
    stores = {}
    for bi, blk in enumerate(getattr(model, "blocks", [])):
        st = getattr(blk, "episode_store", None)
        if st is None:
            continue
        st.forward_calls = st.recall_fires = st.store_writes = 0
    with freeze_plasticity(model), torch.no_grad():
        for inp in train_batches:
            model.encode(**inp, causal=False)
    for bi, blk in enumerate(getattr(model, "blocks", [])):
        st = getattr(blk, "episode_store", None)
        if st is not None:
            s = st.stats()
            stores[f"blocks.{bi}"] = {
                k: s[k] for k in ("episodes_stored", "recall_duty",
                                  "last_recall_sim", "last_blend_share")
            }

    report = {
        "run_dir": str(run.result.get("run_dir", args.run_dir)),
        "arm": run.result["arm"], "seed": run.result["seed"],
        "checkpoint": run.checkpoint.name, "global_step": run.global_step,
        "living_change_spectrum": spectra,
        "batch_mean_stationarity": {"train_batches": stat_train,
                                    "single_sequences": stat_single},
        "write_diversity": {"train_batches": div_train,
                            "single_sequences": div_single},
        "block_output_store": stores,
    }
    for name, sp in spectra.items():
        print(f"  {name:<24} top_share={sp['top_share']:.4f} "
              f"eff_rank={sp['eff_rank']:.2f} rel_norm={sp['rel_norm']:.4f} "
              f"out_align={sp['out_align']:.4f} "
              f"stat(B)={stat_train.get(name, float('nan')):.4f} "
              f"stat(1)={stat_single.get(name, float('nan')):.4f} "
              f"writecos(B)={div_train.get(name, float('nan')):.4f} "
              f"writecos(1)={div_single.get(name, float('nan')):.4f}", flush=True)
    for name, s in stores.items():
        print(f"  {name:<24} store duty={s['recall_duty']:.3f} "
              f"sim={s['last_recall_sim']:.4f} blend_share={s['last_blend_share']:.4f}",
              flush=True)
    out = Path(args.out) if args.out else Path(args.run_dir) / "living_write_probe.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
