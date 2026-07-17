"""Stage-1 verdict under the blind-amended criteria (2026-07-16).

Reloads every completed pilot run's FINAL checkpoint, recomputes the
held-out eval with the amended primary metric (NMSE — committed blind at
0fcc92a before any run's NMSE existed), and applies the pre-registered
verdict rule verbatim:

  per axis (NMSE, probe top-1): |mean difference| > pooled sigma = win
  for the better side; within = tie. KF2-strong SURVIVES only if the
  living arm wins >= 1 axis and loses none; the KILL fires if the living
  arm loses any axis or ties both (ON the curve = no advantage).

Probe numbers are taken from each run's pilot_result.json (computed at
run end on the same holdout). NMSE is recomputed here from checkpoints
because pass-2 runs predate the metric.

Writes runs/jepa_pilot/verdict.json and prints the reading.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_ROOT = REPO_ROOT / "runs" / "jepa_pilot"

from scripts.jepa_pilot_driver import _DeviceLoader, _device  # noqa: E402


def _nmse_for_run(run_dir: Path, result: dict) -> float:
    from luthi.living_extra_state import apply_living_extra_state
    from luthi.v2.eval_heldout import heldout_latent_prediction
    from luthi.v2.jepa_loss import JEPALoss
    from luthi.v2.multimodal_data import (
        MultimodalDataLoaderImpl,
        TextDataset,
        TextDatasetConfig,
    )
    from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM

    cfg = result["config"]
    device = _device()

    text_ds = TextDataset(TextDatasetConfig(
        source_paths=[cfg["data_dir"]],
        tokenizer_path=REPO_ROOT / "corpus_build" / "tokenizer_32k.json",
        batch_size=cfg["batch_size"],
        seq_len=cfg["seq_len"],
        stride=cfg["stride"],
        base_seed=result["seed"],
        holdout_fraction=cfg["holdout_fraction"],
    ))
    loader = _DeviceLoader(MultimodalDataLoaderImpl(text=text_ds), device)

    model = MultimodalPredictiveCodingLM(
        vocab_size=text_ds.vocab_size(),
        d_model=result["d_model"],
        n_blocks=cfg["n_blocks"],
        n_heads=4,
        ffn_expansion=1,
        max_seq_len=cfg["seq_len"],
        backward_pass_enabled=False,
        dead_ffn=(result["arm"] == "dead"),
    ).to(device)
    loss_module = JEPALoss(online_encoder=model).to(device)

    ckpts = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {run_dir}")
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    model.load_state_dict(state["online_state_dict"])
    apply_living_extra_state(
        model, state.get("living_extra_state"),
        source=f"verdict eval {ckpts[-1]}",
    )
    loss_module.predictor.load_state_dict(state["predictor_state_dict"])
    loss_module.projection_heads.load_state_dict(
        state["projection_heads_state_dict"],
    )

    r = heldout_latent_prediction(
        loss_module, loader.holdout_batches("text", 8), "text",
        max_batches=50,
    )
    return float(r["nmse_mean"])


def _axis(living: list[float], dead: list[float], lower_is_better: bool):
    lm, dm = statistics.mean(living), statistics.mean(dead)
    pooled = statistics.mean([statistics.stdev(living), statistics.stdev(dead)])
    diff = (dm - lm) if lower_is_better else (lm - dm)  # >0 = living better
    if abs(diff) <= pooled:
        return "tie", lm, dm, pooled
    return ("living" if diff > 0 else "dead"), lm, dm, pooled


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dead-dmodel", type=int, default=256,
                   help="Which dead size to compare against (the frozen "
                        "reads are per-point: 256 = stage 1's matched "
                        "point; 512 = the ratified overshoot; 384 = the "
                        "fallback ceiling). Never pool sizes.")
    args = p.parse_args()

    runs = []
    for path in sorted(OUTPUT_ROOT.glob("*/pilot_result.json")):
        r = json.loads(path.read_text())
        if not r["admissible"]:
            print(f"[verdict] {path.parent.name}: INADMISSIBLE ({r['outcome']}) -- excluded")
            continue
        runs.append((path.parent, r))

    living = [r for _, r in runs if r["arm"] == "living"]
    dead = [r for _, r in runs
            if r["arm"] == "dead" and r["d_model"] == args.dead_dmodel]
    print(f"[verdict] comparing living@256 vs dead@{args.dead_dmodel}")
    if len(living) < 5 or len(dead) < 5:
        print(f"[verdict] incomplete: living={len(living)}/5 dead={len(dead)}/5 admissible")
        return 1

    print("[verdict] recomputing NMSE from final checkpoints (blind metric)...")
    nmse: dict[str, list[float]] = {"living": [], "dead": []}
    for run_dir, r in runs:
        value = _nmse_for_run(run_dir, r)
        nmse[r["arm"]].append(value)
        print(f"  {run_dir.name}: nmse={value:.6f} "
              f"(raw l_pred={r['heldout']['l_pred_mean']:.6f})")

    probe = {
        "living": [r["probe"]["top1"] for r in living],
        "dead": [r["probe"]["top1"] for r in dead],
    }

    nmse_axis = _axis(nmse["living"], nmse["dead"], lower_is_better=True)
    probe_axis = _axis(probe["living"], probe["dead"], lower_is_better=False)

    wins = sum(1 for a in (nmse_axis, probe_axis) if a[0] == "living")
    losses = sum(1 for a in (nmse_axis, probe_axis) if a[0] == "dead")
    if losses > 0:
        verdict = "KILL: living arm loses >= 1 axis"
    elif wins == 0:
        verdict = "KILL: tie on both axes (ON the curve -- no advantage at matched capacity)"
    else:
        verdict = "KF2-strong SURVIVES at the matched point (bracket now decisive)"

    report = {
        "criteria": "blind amendment 0fcc92a, 2026-07-16",
        "nmse": {
            "living": nmse["living"], "dead": nmse["dead"],
            "axis": {"winner": nmse_axis[0], "living_mean": nmse_axis[1],
                     "dead_mean": nmse_axis[2], "pooled_sigma": nmse_axis[3]},
        },
        "probe_top1": {
            "living": probe["living"], "dead": probe["dead"],
            "axis": {"winner": probe_axis[0], "living_mean": probe_axis[1],
                     "dead_mean": probe_axis[2], "pooled_sigma": probe_axis[3]},
        },
        "verdict": verdict,
    }
    (OUTPUT_ROOT / "verdict.json").write_text(json.dumps(report, indent=2))

    print()
    print(f"NMSE : living {nmse_axis[1]:.6f} vs dead {nmse_axis[2]:.6f} "
          f"(pooled sigma {nmse_axis[3]:.6f}) -> {nmse_axis[0]}")
    print(f"PROBE: living {probe_axis[1]:.4f} vs dead {probe_axis[2]:.4f} "
          f"(pooled sigma {probe_axis[3]:.4f}) -> {probe_axis[0]}")
    print()
    print(f"VERDICT: {verdict}")
    print(f"[verdict] written to {OUTPUT_ROOT / 'verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
